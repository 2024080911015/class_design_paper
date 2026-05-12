from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from pseudo_common import build_balanced_folds


class PILImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        train_dir: str | Path,
        train_fallback_dir: str | Path,
        test_dir: str | Path,
        test_fallback_dir: str | Path,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.train_dir = Path(train_dir)
        self.train_fallback_dir = Path(train_fallback_dir)
        self.test_dir = Path(test_dir)
        self.test_fallback_dir = Path(test_fallback_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.mode == "test":
            img_name = str(row["img"])
            primary = self.test_dir / img_name
            fallback = self.test_fallback_dir / img_name
        else:
            img_name = str(row["img"])
            class_name = str(row["classname"])
            primary = self.train_dir / class_name / img_name
            fallback = self.train_fallback_dir / class_name / img_name

        path = primary if primary.exists() else fallback
        if not path.exists():
            raise FileNotFoundError(f"Could not read image: {primary}")
        return Image.open(path).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen DINOv2 train/test embeddings.")
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--driver-csv", default="dataset/driver_imgs_list.csv")
    parser.add_argument("--folds-csv", default="train_with_folds.csv")
    parser.add_argument("--sample-submission", default="dataset/sample_submission.csv")
    parser.add_argument("--output-dir", default="models/strict_assets")
    parser.add_argument("--asset-name", default="dinov2")
    parser.add_argument("--train-dir", default="dataset/imgs/train_cropped_v2")
    parser.add_argument("--train-fallback-dir", default="dataset/imgs/train")
    parser.add_argument("--test-dir", default="dataset/imgs/test_cropped_v2")
    parser.add_argument("--test-fallback-dir", default="dataset/imgs/test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Load DINOv2 from local files/cache only. This is the default; no download is attempted.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_false",
        dest="local_files_only",
        help="Explicitly allow transformers to download DINOv2 if it is not available locally.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_training_frame(folds_csv: str | Path, driver_csv: str | Path) -> pd.DataFrame:
    folds_path = Path(folds_csv)
    if folds_path.exists():
        df = pd.read_csv(folds_path).reset_index(drop=True)
    else:
        df = build_balanced_folds(driver_csv)
        df.to_csv(folds_path, index=False)
        print(f"[INFO] Created folds CSV: {folds_path}")
    return df.reset_index(drop=True)


def collate_pil(batch):
    return batch


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def extract(loader, processor, model, device: torch.device, desc: str) -> np.ndarray:
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=desc):
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            features.append(feats.detach().float().cpu().numpy())
    return np.concatenate(features, axis=0)


def main() -> None:
    args = parse_args()
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = AutoImageProcessor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    train_df = load_training_frame(args.folds_csv, args.driver_csv)
    test_df = pd.read_csv(args.sample_submission).reset_index(drop=True)

    train_loader = DataLoader(
        PILImageDataset(
            train_df,
            mode="train",
            train_dir=args.train_dir,
            train_fallback_dir=args.train_fallback_dir,
            test_dir=args.test_dir,
            test_fallback_dir=args.test_fallback_dir,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
    )
    test_loader = DataLoader(
        PILImageDataset(
            test_df,
            mode="test",
            train_dir=args.train_dir,
            train_fallback_dir=args.train_fallback_dir,
            test_dir=args.test_dir,
            test_fallback_dir=args.test_fallback_dir,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
    )

    train_features = extract(train_loader, processor, model, device, f"{args.asset_name} train")
    test_features = extract(test_loader, processor, model, device, f"{args.asset_name} test")
    if args.normalize_features:
        train_features = l2_normalize(train_features)
        test_features = l2_normalize(test_features)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"train_features_{args.asset_name}.npy", train_features.astype(np.float32))
    np.save(output_dir / f"test_features_{args.asset_name}.npy", test_features.astype(np.float32))
    train_df[["subject", "classname", "img", "label_int", "fold"]].to_csv(output_dir / "train_index.csv", index=False)
    test_df[["img"]].to_csv(output_dir / "test_index.csv", index=False)
    print(f"[INFO] Saved DINOv2 features for {args.asset_name} to {output_dir}")


if __name__ == "__main__":
    main()
