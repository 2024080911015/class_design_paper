from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pseudo_common import CLASS_COLUMNS, normalize_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create soft pseudo labels from calibrated teacher probabilities.")
    parser.add_argument("--teacher-csv", default="models/strict_assets/graph_knn_calibrated_test_preds.csv")
    parser.add_argument(
        "--agreement-csvs",
        nargs="*",
        default=[],
        help="Optional teacher CSVs whose top1 labels must agree with --teacher-csv.",
    )
    parser.add_argument("--require-agreement", action="store_true")
    parser.add_argument("--soft-threshold", type=float, default=0.90)
    parser.add_argument("--hard-threshold", type=float, default=0.98)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--hard-min-margin", type=float, default=0.25)
    parser.add_argument("--per-class-limit", type=int, default=4000)
    parser.add_argument("--max-pseudo", type=int, default=40000)
    parser.add_argument("--output", default="pseudo_soft_labels.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_path = Path(args.teacher_csv)
    teacher_df = pd.read_csv(teacher_path)
    missing = [column for column in ["img"] + CLASS_COLUMNS if column not in teacher_df.columns]
    if missing:
        raise ValueError(f"{teacher_path} is missing columns: {missing}")

    probs = normalize_probabilities(teacher_df[CLASS_COLUMNS].to_numpy(dtype=np.float64))
    labels = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    agreement_mask = np.ones(len(teacher_df), dtype=bool)
    agreement_sources: list[str] = []
    if args.agreement_csvs:
        base = teacher_df[["img"]].copy()
        for agreement_csv in args.agreement_csvs:
            agreement_path = Path(agreement_csv)
            agreement_df = pd.read_csv(agreement_path)
            missing = [column for column in ["img"] + CLASS_COLUMNS if column not in agreement_df.columns]
            if missing:
                raise ValueError(f"{agreement_path} is missing columns: {missing}")
            if agreement_df["img"].duplicated().any():
                duplicated = agreement_df.loc[agreement_df["img"].duplicated(), "img"].head(5).tolist()
                raise ValueError(f"{agreement_path} has duplicate img rows, for example: {duplicated}")

            aligned = base.merge(agreement_df[["img"] + CLASS_COLUMNS], on="img", how="left", sort=False)
            if aligned[CLASS_COLUMNS].isna().any().any():
                missing_imgs = aligned.loc[aligned[CLASS_COLUMNS].isna().any(axis=1), "img"].head(5).tolist()
                raise ValueError(f"{agreement_path} does not cover every teacher image, for example: {missing_imgs}")

            agreement_probs = normalize_probabilities(aligned[CLASS_COLUMNS].to_numpy(dtype=np.float64))
            agreement_labels = np.argmax(agreement_probs, axis=1)
            agreement_mask &= agreement_labels == labels
            agreement_sources.append(str(agreement_path))

    selection_mask = (confidence >= args.soft_threshold) & (margin >= args.min_margin)
    if args.require_agreement:
        selection_mask &= agreement_mask

    selected = np.where(selection_mask)[0]
    if args.per_class_limit > 0 and selected.size > 0:
        limited: list[int] = []
        for class_idx in range(len(CLASS_COLUMNS)):
            class_indices = selected[labels[selected] == class_idx]
            order = np.argsort(-confidence[class_indices])
            limited.extend(class_indices[order[: args.per_class_limit]].tolist())
        selected = np.asarray(limited, dtype=np.int64)

    if args.max_pseudo > 0 and selected.size > args.max_pseudo:
        order = np.argsort(-confidence[selected])
        selected = selected[order[: args.max_pseudo]]

    selected = selected[np.lexsort((teacher_df["img"].astype(str).to_numpy()[selected], -confidence[selected]))]
    hard_pseudo = (confidence[selected] >= args.hard_threshold) & (margin[selected] >= args.hard_min_margin)

    output_columns = [
        "subject",
        "classname",
        "img",
        "label_int",
        "fold",
        "pseudo_confidence",
        "pseudo_margin",
        "pseudo_source",
        "agreement_sources",
        "teacher_agreement",
        "hard_pseudo",
    ] + CLASS_COLUMNS
    if selected.size == 0:
        out = pd.DataFrame(columns=output_columns)
        out.to_csv(args.output, index=False)
        print(f"[INFO] Saved 0 soft pseudo labels to {args.output}")
        if args.require_agreement:
            print(f"[INFO] Teacher agreement kept {int(agreement_mask.sum())}/{len(agreement_mask)} test rows before thresholds.")
        return

    out = pd.DataFrame(
        {
            "subject": "pseudo_test",
            "classname": [f"c{label}" for label in labels[selected]],
            "img": teacher_df["img"].astype(str).to_numpy()[selected],
            "label_int": labels[selected].astype(int),
            "fold": -1,
            "pseudo_confidence": confidence[selected],
            "pseudo_margin": margin[selected],
            "pseudo_source": str(teacher_path),
            "agreement_sources": ";".join(agreement_sources),
            "teacher_agreement": agreement_mask[selected].astype(bool),
            "hard_pseudo": hard_pseudo.astype(bool),
        }
    )
    for class_idx, column in enumerate(CLASS_COLUMNS):
        out[column] = probs[selected, class_idx]
    out = out[output_columns]

    out.to_csv(args.output, index=False)
    print(f"[INFO] Saved {len(out)} soft pseudo labels to {args.output}")
    if args.require_agreement:
        print(f"[INFO] Teacher agreement kept {int(agreement_mask.sum())}/{len(agreement_mask)} test rows before thresholds.")
    if len(out) > 0:
        print("[INFO] Class counts:")
        print(out["classname"].value_counts().sort_index().to_string())
        print(f"[INFO] Hard pseudo labels: {int(out['hard_pseudo'].sum())}")


if __name__ == "__main__":
    main()
