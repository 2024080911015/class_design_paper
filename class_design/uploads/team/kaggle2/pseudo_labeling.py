from __future__ import annotations

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from pseudo_common import (
    CLASS_COLUMNS,
    autocast_context,
    create_timm_model,
    get_preset,
    load_state_dict_cpu,
    normalize_probabilities,
    select_pseudo_labels,
)


class TestImageDataset(Dataset):
    def __init__(
        self,
        sample_submission_path: str | Path,
        test_dir: str | Path,
        fallback_dir: str | Path,
        img_size: int,
        norm_mean: tuple[float, float, float],
        norm_std: tuple[float, float, float],
    ) -> None:
        self.df = pd.read_csv(sample_submission_path)
        self.image_names = self.df["img"].astype(str).tolist()
        self.test_dir = Path(test_dir)
        self.fallback_dir = Path(fallback_dir)
        self.transform = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=list(norm_mean), std=list(norm_std)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        img_path = self.test_dir / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            image = cv2.imread(str(self.fallback_dir / img_name))
        if image is None:
            raise FileNotFoundError(f"Could not read test image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"], img_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hard pseudo labels from fold models or a submission-style probability CSV."
    )
    parser.add_argument(
        "--route",
        default="transformer",
        help="swin/transformer, beit, ensemble, or explicit legacy cnn/effb3.",
    )
    parser.add_argument("--prediction-csv", default=None, help="Use an existing CSV with img,c0..c9 probabilities.")
    parser.add_argument("--sample-submission", default="dataset/sample_submission.csv")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-fallback-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--weights", default=None, help="Fold weight pattern, for example models/x_fold_{fold}.pth.")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--min-margin", type=float, default=0.25)
    parser.add_argument("--per-class-limit", type=int, default=0)
    parser.add_argument("--max-pseudo", type=int, default=0)
    parser.add_argument("--output", default="pseudo_labels.csv")
    parser.add_argument("--prob-output", default=None, help="Optional CSV path for averaged probabilities.")
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_predictions_from_csv(prediction_csv: str | Path):
    df = pd.read_csv(prediction_csv)
    missing = [column for column in ["img"] + CLASS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{prediction_csv} is missing columns: {missing}")

    image_names = df["img"].astype(str).tolist()
    preds = normalize_probabilities(df[CLASS_COLUMNS].to_numpy(dtype=np.float64))
    return image_names, preds


def predict_with_folds(args: argparse.Namespace):
    preset = get_preset(args.route)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    img_size = args.img_size or preset.img_size
    test_dir = args.test_dir or preset.test_dir
    fallback_dir = args.test_fallback_dir or preset.test_fallback_dir
    batch_size = args.batch_size or preset.infer_batch_size
    num_workers = args.num_workers if args.num_workers is not None else preset.num_workers
    model_name = args.model_name or preset.model_name
    weight_pattern = args.weights or preset.initial_weight_pattern
    drop_path_rate = preset.drop_path_rate if args.drop_path_rate is None else args.drop_path_rate
    norm_mean = tuple(preset.norm_mean)
    norm_std = tuple(preset.norm_std)
    amp_enabled = not args.no_amp and device.type == "cuda"

    dataset = TestImageDataset(args.sample_submission, test_dir, fallback_dir, img_size, norm_mean, norm_std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    all_fold_preds: list[np.ndarray] = []
    for fold in args.folds:
        weight_path = Path(weight_pattern.format(fold=fold))
        if not weight_path.exists():
            print(f"[WARN] Missing fold {fold} weights, skipped: {weight_path}")
            continue

        print(f"[INFO] Loading fold {fold}: {weight_path}")
        base_model = create_timm_model(model_name, drop_path_rate)
        state = load_state_dict_cpu(weight_path)
        base_model.load_state_dict(state, strict=True)
        del state

        base_model.to(device)
        base_model.eval()
        model = torch.compile(base_model) if args.compile_model and hasattr(torch, "compile") else base_model

        fold_preds: list[np.ndarray] = []
        with torch.inference_mode():
            for images, _names in tqdm(loader, desc=f"fold {fold}"):
                images = images.to(device, non_blocking=True)
                with autocast_context(device, amp_enabled):
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                fold_preds.append(probs.detach().cpu().numpy())

        all_fold_preds.append(np.concatenate(fold_preds, axis=0))
        del model, base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if not all_fold_preds:
        raise FileNotFoundError(f"No usable weights matched pattern: {weight_pattern}")

    preds = normalize_probabilities(np.mean(np.stack(all_fold_preds, axis=0), axis=0))
    return dataset.image_names, preds


def main() -> None:
    args = parse_args()

    if args.prediction_csv:
        image_names, preds = load_predictions_from_csv(args.prediction_csv)
        source = str(args.prediction_csv)
        print(f"[INFO] Loaded probabilities from {args.prediction_csv}")
    else:
        if args.route.lower() == "ensemble":
            raise ValueError("--route ensemble requires --prediction-csv.")
        image_names, preds = predict_with_folds(args)
        source = args.route

    if args.prob_output:
        prob_df = pd.DataFrame(preds, columns=CLASS_COLUMNS)
        prob_df.insert(0, "img", image_names)
        prob_df.to_csv(args.prob_output, index=False)
        print(f"[INFO] Saved averaged probabilities to {args.prob_output}")

    pseudo_df = select_pseudo_labels(
        image_names=image_names,
        preds=preds,
        source=source,
        threshold=args.threshold,
        min_margin=args.min_margin,
        per_class_limit=args.per_class_limit,
        max_pseudo=args.max_pseudo,
    )
    pseudo_df.to_csv(args.output, index=False)

    print(f"[INFO] Saved {len(pseudo_df)} pseudo labels to {args.output}")
    if len(pseudo_df) > 0:
        counts = pseudo_df["classname"].value_counts().sort_index()
        print(counts.to_string())


if __name__ == "__main__":
    main()
