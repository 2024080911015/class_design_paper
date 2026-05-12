from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pseudo_common import CLASS_COLUMNS, normalize_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import external CNN OOF/test CSVs into strict asset format.")
    parser.add_argument("--asset-dir", default="models/strict_assets")
    parser.add_argument("--asset-name", default="top3cnn")
    parser.add_argument("--cnn-oof-csv", required=True)
    parser.add_argument("--cnn-test-csv", required=True)
    parser.add_argument("--train-index", default="train_index.csv")
    parser.add_argument("--test-index", default="test_index.csv")
    parser.add_argument("--clip-min", type=float, default=1e-7)
    return parser.parse_args()


def load_probability_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in ["img"] + CLASS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if df["img"].duplicated().any():
        duplicated = df.loc[df["img"].duplicated(), "img"].head(5).tolist()
        raise ValueError(f"{path} has duplicate img rows, for example: {duplicated}")
    return df


def align_oof(train_index: pd.DataFrame, oof_df: pd.DataFrame, clip_min: float) -> tuple[pd.DataFrame, np.ndarray]:
    aligned = train_index[["subject", "classname", "img", "label_int", "fold"]].merge(
        oof_df[["img"] + CLASS_COLUMNS + (["label_int"] if "label_int" in oof_df.columns else [])],
        on="img",
        how="left",
        sort=False,
        suffixes=("", "_cnn"),
    )
    if aligned[CLASS_COLUMNS].isna().any().any():
        missing_imgs = aligned.loc[aligned[CLASS_COLUMNS].isna().any(axis=1), "img"].head(5).tolist()
        raise ValueError(f"CNN OOF does not cover every train image, for example: {missing_imgs}")
    if "label_int_cnn" in aligned.columns:
        mismatch = aligned["label_int"].astype(int).to_numpy() != aligned["label_int_cnn"].astype(int).to_numpy()
        if mismatch.any():
            bad = aligned.loc[mismatch, ["img", "label_int", "label_int_cnn"]].head(5).to_dict("records")
            raise ValueError(f"CNN OOF label_int mismatches train index, for example: {bad}")
        aligned = aligned.drop(columns=["label_int_cnn"])

    probs = normalize_probabilities(aligned[CLASS_COLUMNS].to_numpy(dtype=np.float64))
    probs = np.clip(probs, clip_min, 1.0 - clip_min)
    probs = normalize_probabilities(probs)
    aligned.loc[:, CLASS_COLUMNS] = probs
    return aligned, probs


def align_test(test_index: pd.DataFrame, test_df: pd.DataFrame, clip_min: float) -> tuple[pd.DataFrame, np.ndarray]:
    aligned = test_index[["img"]].merge(test_df[["img"] + CLASS_COLUMNS], on="img", how="left", sort=False)
    if aligned[CLASS_COLUMNS].isna().any().any():
        missing_imgs = aligned.loc[aligned[CLASS_COLUMNS].isna().any(axis=1), "img"].head(5).tolist()
        raise ValueError(f"CNN test preds do not cover every test image, for example: {missing_imgs}")
    probs = normalize_probabilities(aligned[CLASS_COLUMNS].to_numpy(dtype=np.float64))
    probs = np.clip(probs, clip_min, 1.0 - clip_min)
    probs = normalize_probabilities(probs)
    aligned.loc[:, CLASS_COLUMNS] = probs
    return aligned, probs


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir)
    train_index = pd.read_csv(asset_dir / args.train_index)
    test_index = pd.read_csv(asset_dir / args.test_index)
    cnn_oof = load_probability_frame(Path(args.cnn_oof_csv))
    cnn_test = load_probability_frame(Path(args.cnn_test_csv))

    oof_aligned, oof_preds = align_oof(train_index, cnn_oof, args.clip_min)
    test_aligned, test_preds = align_test(test_index, cnn_test, args.clip_min)

    asset_dir.mkdir(parents=True, exist_ok=True)
    oof_npy = asset_dir / f"oof_preds_{args.asset_name}.npy"
    test_npy = asset_dir / f"test_preds_{args.asset_name}.npy"
    oof_csv = asset_dir / f"oof_preds_{args.asset_name}.csv"
    test_csv = asset_dir / f"test_preds_{args.asset_name}.csv"
    config_path = asset_dir / f"{args.asset_name}_import_config.json"

    np.save(oof_npy, oof_preds.astype(np.float32))
    np.save(test_npy, test_preds.astype(np.float32))
    oof_aligned.to_csv(oof_csv, index=False)
    test_aligned.to_csv(test_csv, index=False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "asset_name": args.asset_name,
                "cnn_oof_csv": str(Path(args.cnn_oof_csv)),
                "cnn_test_csv": str(Path(args.cnn_test_csv)),
                "num_train": int(len(oof_aligned)),
                "num_test": int(len(test_aligned)),
                "clip_min": args.clip_min,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Imported CNN OOF: {oof_npy}")
    print(f"[INFO] Imported CNN test: {test_npy}")


if __name__ == "__main__":
    main()
