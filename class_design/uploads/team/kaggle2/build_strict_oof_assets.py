from __future__ import annotations

import argparse
import gc
import json
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
    build_balanced_folds,
    create_timm_model,
    get_preset,
    load_state_dict_cpu,
    normalize_probabilities,
)


BASE_OOF_NAMES = {
    "effb3": "oof_preds_effb3.npy",
    "swin": "oof_preds_swin.npy",
    "beit": "oof_preds_beit.npy",
}


class ImageFrameDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        mode: str,
        train_dir: str | Path,
        train_fallback_dir: str | Path,
        test_dir: str | Path,
        test_fallback_dir: str | Path,
        tta_mode: str = "base",
        norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.train_dir = Path(train_dir)
        self.train_fallback_dir = Path(train_fallback_dir)
        self.test_dir = Path(test_dir)
        self.test_fallback_dir = Path(test_fallback_dir)
        transform_steps = []
        if tta_mode == "hflip":
            transform_steps.append(A.HorizontalFlip(p=1.0))
        if tta_mode == "zoom":
            zoom_size = int(round(img_size * 1.08))
            transform_steps.extend([A.Resize(zoom_size, zoom_size), A.CenterCrop(img_size, img_size)])
        else:
            transform_steps.append(A.Resize(img_size, img_size))
        transform_steps.extend(
            [
                A.Normalize(mean=list(norm_mean), std=list(norm_std)),
                ToTensorV2(),
            ]
        )
        self.transform = A.Compose(transform_steps)

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

        image = cv2.imread(str(primary))
        if image is None:
            image = cv2.imread(str(fallback))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {primary}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create strict OOF/test prediction and train/test feature assets for one route."
    )
    parser.add_argument("--route", default="swin", help="swin/transformer, beit, cnn/effb3.")
    parser.add_argument("--driver-csv", default="dataset/driver_imgs_list.csv")
    parser.add_argument("--folds-csv", default="train_with_folds.csv")
    parser.add_argument("--sample-submission", default="dataset/sample_submission.csv")
    parser.add_argument("--output-dir", default="models/strict_assets")
    parser.add_argument("--asset-name", default=None, help="Override saved asset suffix, for example pseudo_swin.")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--weights", default=None, help="Fold weight pattern. Default comes from route preset.")
    parser.add_argument("--oof-preds", default=None, help="Existing OOF npy. Default comes from route preset.")
    parser.add_argument("--test-preds-csv", default=None, help="Submission-style CSV for test probabilities.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--train-fallback-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-fallback-dir", default=None)
    parser.add_argument("--norm-mean", nargs=3, type=float, default=None)
    parser.add_argument("--norm-std", nargs=3, type=float, default=None)
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument(
        "--infer-probs-from-weights",
        action="store_true",
        help="Regenerate strict OOF and test probabilities from fold weights instead of loading existing files.",
    )
    parser.add_argument(
        "--prob-tta-modes",
        nargs="+",
        default=["base"],
        choices=["base", "zoom", "hflip"],
        help="TTA modes used for both OOF and test probability regeneration.",
    )
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_training_frame(folds_csv: str | Path, driver_csv: str | Path) -> pd.DataFrame:
    folds_path = Path(folds_csv)
    if folds_path.exists():
        df = pd.read_csv(folds_path).reset_index(drop=True)
    else:
        df = build_balanced_folds(driver_csv)
        df.to_csv(folds_path, index=False)
        print(f"[INFO] Created folds CSV: {folds_path}")
    if "label_int" not in df.columns:
        df["label_int"] = df["classname"].str.extract(r"(\d+)").astype(int)
    return df.reset_index(drop=True)


def choose_alignment_keys(reference: pd.DataFrame, predictions: pd.DataFrame, path: Path) -> list[str]:
    candidates = [["img"], ["classname", "img"], ["subject", "classname", "img"]]
    for keys in candidates:
        if not all(key in reference.columns and key in predictions.columns for key in keys):
            continue
        if reference.duplicated(keys).any() or predictions.duplicated(keys).any():
            continue
        return keys
    raise ValueError(
        f"{path} cannot be safely aligned to the training frame. "
        "Save OOF predictions with unique img/classname metadata."
    )


def load_oof_predictions(oof_path: Path, full_df: pd.DataFrame) -> np.ndarray:
    if oof_path.suffix.lower() == ".csv":
        oof_df = pd.read_csv(oof_path)
        missing = [column for column in CLASS_COLUMNS if column not in oof_df.columns]
        if missing:
            raise ValueError(f"{oof_path} is missing probability columns: {missing}")
        keys = choose_alignment_keys(full_df, oof_df, oof_path)
        aligned = full_df[keys].merge(oof_df[keys + CLASS_COLUMNS], on=keys, how="left", sort=False)
        if aligned[CLASS_COLUMNS].isna().any().any():
            raise ValueError(f"{oof_path} does not cover every row in the training frame.")
        oof_preds = aligned[CLASS_COLUMNS].to_numpy(dtype=np.float64)
    else:
        oof_preds = np.load(oof_path)
        if oof_preds.shape[0] != len(full_df):
            raise ValueError(
                f"{oof_path} has {oof_preds.shape[0]} rows, but the training frame has {len(full_df)} rows."
            )
        print(
            f"[WARN] {oof_path} has no row metadata; assuming it is already aligned to "
            f"{Path('train_with_folds.csv')} order."
        )

    if oof_preds.ndim != 2 or oof_preds.shape[1] != len(CLASS_COLUMNS):
        raise ValueError(f"{oof_path} must have shape (n_train, {len(CLASS_COLUMNS)}), got {oof_preds.shape}.")
    return normalize_probabilities(oof_preds)


def load_test_predictions(test_csv: Path, sample_submission: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    test_df = pd.read_csv(test_csv)
    missing = [column for column in ["img"] + CLASS_COLUMNS if column not in test_df.columns]
    if missing:
        raise ValueError(f"{test_csv} is missing columns: {missing}")
    if test_df["img"].duplicated().any():
        duplicates = test_df.loc[test_df["img"].duplicated(), "img"].head(5).tolist()
        raise ValueError(f"{test_csv} has duplicate img rows, for example: {duplicates}")

    aligned = sample_submission[["img"]].merge(test_df[["img"] + CLASS_COLUMNS], on="img", how="left", sort=False)
    if len(aligned) != len(sample_submission):
        raise ValueError(f"{test_csv} alignment changed row count from {len(sample_submission)} to {len(aligned)}.")
    if aligned[CLASS_COLUMNS].isna().any().any():
        missing_imgs = aligned.loc[aligned[CLASS_COLUMNS].isna().any(axis=1), "img"].head(5).tolist()
        raise ValueError(f"{test_csv} does not cover every test image, for example: {missing_imgs}")

    test_preds = normalize_probabilities(aligned[CLASS_COLUMNS].to_numpy(dtype=np.float64))
    aligned.loc[:, CLASS_COLUMNS] = test_preds
    return test_preds, aligned[["img"] + CLASS_COLUMNS]


def save_standard_predictions(
    args: argparse.Namespace,
    preset,
    asset_name: str,
    output_dir: Path,
    full_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    oof_path = Path(args.oof_preds) if args.oof_preds else Path(preset.save_dir) / BASE_OOF_NAMES[preset.name]
    if not oof_path.exists():
        print(f"[WARN] OOF predictions not found, skipped: {oof_path}")
    else:
        oof_preds = load_oof_predictions(oof_path, full_df)
        np.save(output_dir / f"oof_preds_{asset_name}.npy", oof_preds.astype(np.float32))
        oof_df = full_df[["subject", "classname", "img", "label_int", "fold"]].copy()
        for class_idx, column in enumerate(CLASS_COLUMNS):
            oof_df[column] = oof_preds[:, class_idx]
        oof_df.to_csv(output_dir / f"oof_preds_{asset_name}.csv", index=False)
        print(f"[INFO] Saved {output_dir / f'oof_preds_{asset_name}.npy'}")

    test_csv = Path(args.test_preds_csv) if args.test_preds_csv else Path(preset.save_dir) / preset.submission_name
    if not test_csv.exists():
        print(f"[WARN] Test prediction CSV not found, skipped: {test_csv}")
    else:
        test_preds, aligned_test_df = load_test_predictions(test_csv, sample_submission)
        if test_preds.shape[0] != len(sample_submission):
            raise ValueError(
                f"{test_csv} has {test_preds.shape[0]} rows, but sample submission has {len(sample_submission)} rows."
            )
        np.save(output_dir / f"test_preds_{asset_name}.npy", test_preds.astype(np.float32))
        aligned_test_df.to_csv(output_dir / f"test_preds_{asset_name}.csv", index=False)
        print(f"[INFO] Saved {output_dir / f'test_preds_{asset_name}.npy'}")


def classifier_model_from_fold(
    model_name: str,
    drop_path_rate: float,
    weight_path: Path,
    device: torch.device,
    compile_model: bool,
):
    model = create_timm_model(model_name, drop_path_rate)
    state = load_state_dict_cpu(weight_path)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    model.eval()
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def feature_model_from_fold(
    model_name: str,
    drop_path_rate: float,
    weight_path: Path,
    device: torch.device,
    compile_model: bool,
):
    model = create_timm_model(model_name, drop_path_rate)
    state = load_state_dict_cpu(weight_path)
    model.load_state_dict(state, strict=True)
    del state
    model.reset_classifier(0)
    model.to(device)
    model.eval()
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def swap_hflip_probabilities(probs: np.ndarray) -> np.ndarray:
    swapped = probs.copy()
    swapped[:, 1] = probs[:, 3]
    swapped[:, 3] = probs[:, 1]
    swapped[:, 2] = probs[:, 4]
    swapped[:, 4] = probs[:, 2]
    return swapped


def predict_probabilities(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    desc: str,
) -> np.ndarray:
    probs_list: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.detach().float().cpu().numpy())
    return normalize_probabilities(np.concatenate(probs_list, axis=0))


def predict_tta_probabilities(
    args: argparse.Namespace,
    model,
    frame_df: pd.DataFrame,
    mode: str,
    img_size: int,
    train_dir: str | Path,
    train_fallback_dir: str | Path,
    test_dir: str | Path,
    test_fallback_dir: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    desc: str,
    norm_mean: tuple[float, float, float],
    norm_std: tuple[float, float, float],
) -> np.ndarray:
    tta_preds = []
    for tta_mode in args.prob_tta_modes:
        dataset = ImageFrameDataset(
            frame_df,
            img_size=img_size,
            mode=mode,
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            test_dir=test_dir,
            test_fallback_dir=test_fallback_dir,
            tta_mode=tta_mode,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        preds = predict_probabilities(model, loader, device, amp_enabled, f"{desc} {tta_mode}")
        if tta_mode == "hflip":
            preds = swap_hflip_probabilities(preds)
        tta_preds.append(preds)
    return normalize_probabilities(np.mean(np.stack(tta_preds, axis=0), axis=0))


def infer_strict_predictions(
    args: argparse.Namespace,
    preset,
    asset_name: str,
    output_dir: Path,
    full_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = not args.no_amp and device.type == "cuda"
    model_name = args.model_name or preset.model_name
    weight_pattern = args.weights or preset.initial_weight_pattern
    img_size = args.img_size or preset.img_size
    batch_size = args.batch_size or preset.infer_batch_size
    num_workers = args.num_workers if args.num_workers is not None else preset.num_workers
    drop_path_rate = preset.drop_path_rate if args.drop_path_rate is None else args.drop_path_rate
    train_dir = args.train_dir or preset.train_dir
    train_fallback_dir = args.train_fallback_dir or preset.train_fallback_dir
    test_dir = args.test_dir or preset.test_dir
    test_fallback_dir = args.test_fallback_dir or preset.test_fallback_dir
    norm_mean = tuple(args.norm_mean or preset.norm_mean)
    norm_std = tuple(args.norm_std or preset.norm_std)

    oof_preds = np.zeros((len(full_df), len(CLASS_COLUMNS)), dtype=np.float32)
    test_pred_sum = None
    for fold in args.folds:
        weight_path = Path(weight_pattern.format(fold=fold))
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing fold weights: {weight_path}")

        print(f"[INFO] Regenerating strict probabilities with {weight_path}")
        model = classifier_model_from_fold(
            model_name=model_name,
            drop_path_rate=drop_path_rate,
            weight_path=weight_path,
            device=device,
            compile_model=args.compile_model,
        )

        fold_mask = full_df["fold"].to_numpy() == fold
        fold_df = full_df.loc[fold_mask].reset_index(drop=True)
        fold_preds = predict_tta_probabilities(
            args,
            model,
            frame_df=fold_df,
            mode="train",
            img_size=img_size,
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            test_dir=test_dir,
            test_fallback_dir=test_fallback_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"{asset_name} OOF fold {fold}",
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        oof_preds[np.where(fold_mask)[0]] = fold_preds.astype(np.float32)

        test_preds = predict_tta_probabilities(
            args,
            model,
            frame_df=sample_submission.reset_index(drop=True),
            mode="test",
            img_size=img_size,
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            test_dir=test_dir,
            test_fallback_dir=test_fallback_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"{asset_name} test fold {fold}",
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        if test_pred_sum is None:
            test_pred_sum = np.zeros_like(test_preds, dtype=np.float64)
        test_pred_sum += test_preds.astype(np.float64)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if test_pred_sum is None:
        raise RuntimeError("No test probabilities were generated.")

    test_preds = normalize_probabilities(test_pred_sum / max(1, len(args.folds)))
    oof_preds = normalize_probabilities(oof_preds)

    np.save(output_dir / f"oof_preds_{asset_name}.npy", oof_preds.astype(np.float32))
    np.save(output_dir / f"test_preds_{asset_name}.npy", test_preds.astype(np.float32))

    oof_df = full_df[["subject", "classname", "img", "label_int", "fold"]].copy()
    for class_idx, column in enumerate(CLASS_COLUMNS):
        oof_df[column] = oof_preds[:, class_idx]
    oof_df.to_csv(output_dir / f"oof_preds_{asset_name}.csv", index=False)

    test_df = sample_submission[["img"]].copy()
    for class_idx, column in enumerate(CLASS_COLUMNS):
        test_df[column] = test_preds[:, class_idx]
    test_df.to_csv(output_dir / f"test_preds_{asset_name}.csv", index=False)
    print(f"[INFO] Saved TTA OOF/test probabilities for {asset_name}: {args.prob_tta_modes}")


def predict_features(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    desc: str,
) -> np.ndarray:
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled):
                feats = model(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            features.append(feats.detach().float().cpu().numpy())
    return np.concatenate(features, axis=0)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def extract_features(args: argparse.Namespace, preset, full_df: pd.DataFrame, output_dir: Path, asset_name: str) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = not args.no_amp and device.type == "cuda"
    model_name = args.model_name or preset.model_name
    weight_pattern = args.weights or preset.initial_weight_pattern
    img_size = args.img_size or preset.img_size
    batch_size = args.batch_size or preset.infer_batch_size
    num_workers = args.num_workers if args.num_workers is not None else preset.num_workers
    drop_path_rate = preset.drop_path_rate if args.drop_path_rate is None else args.drop_path_rate
    train_dir = args.train_dir or preset.train_dir
    train_fallback_dir = args.train_fallback_dir or preset.train_fallback_dir
    test_dir = args.test_dir or preset.test_dir
    test_fallback_dir = args.test_fallback_dir or preset.test_fallback_dir
    norm_mean = tuple(args.norm_mean or preset.norm_mean)
    norm_std = tuple(args.norm_std or preset.norm_std)

    test_df = pd.read_csv(args.sample_submission).reset_index(drop=True)
    test_dataset = ImageFrameDataset(
        test_df,
        img_size=img_size,
        mode="test",
        train_dir=train_dir,
        train_fallback_dir=train_fallback_dir,
        test_dir=test_dir,
        test_fallback_dir=test_fallback_dir,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    train_features = None
    test_feature_sum = None

    for fold in args.folds:
        weight_path = Path(weight_pattern.format(fold=fold))
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing fold weights: {weight_path}")

        print(f"[INFO] Extracting strict fold features with {weight_path}")
        model = feature_model_from_fold(
            model_name=model_name,
            drop_path_rate=drop_path_rate,
            weight_path=weight_path,
            device=device,
            compile_model=args.compile_model,
        )

        fold_mask = full_df["fold"].to_numpy() == fold
        fold_df = full_df.loc[fold_mask].reset_index(drop=True)
        fold_dataset = ImageFrameDataset(
            fold_df,
            img_size=img_size,
            mode="train",
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            test_dir=test_dir,
            test_fallback_dir=test_fallback_dir,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        fold_loader = DataLoader(
            fold_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        fold_features = predict_features(model, fold_loader, device, amp_enabled, f"{asset_name} train fold {fold}")
        if train_features is None:
            train_features = np.zeros((len(full_df), fold_features.shape[1]), dtype=np.float32)
        train_features[np.where(fold_mask)[0]] = fold_features.astype(np.float32)

        test_features = predict_features(model, test_loader, device, amp_enabled, f"{asset_name} test fold {fold}")
        if test_feature_sum is None:
            test_feature_sum = np.zeros_like(test_features, dtype=np.float64)
        test_feature_sum += test_features.astype(np.float64)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if train_features is None or test_feature_sum is None:
        raise RuntimeError("No features were extracted.")

    test_features = (test_feature_sum / max(1, len(args.folds))).astype(np.float32)
    if args.normalize_features:
        train_features = l2_normalize(train_features).astype(np.float32)
        test_features = l2_normalize(test_features).astype(np.float32)

    np.save(output_dir / f"train_features_{asset_name}.npy", train_features)
    np.save(output_dir / f"test_features_{asset_name}.npy", test_features)
    print(f"[INFO] Saved {output_dir / f'train_features_{asset_name}.npy'}")
    print(f"[INFO] Saved {output_dir / f'test_features_{asset_name}.npy'}")


def main() -> None:
    args = parse_args()
    preset = get_preset(args.route)
    asset_name = args.asset_name or preset.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_training_frame(args.folds_csv, args.driver_csv)
    sample_submission = pd.read_csv(args.sample_submission).reset_index(drop=True)
    full_df[["subject", "classname", "img", "label_int", "fold"]].to_csv(
        output_dir / "train_index.csv",
        index=False,
    )
    sample_submission[["img"]].to_csv(output_dir / "test_index.csv", index=False)

    if args.infer_probs_from_weights:
        infer_strict_predictions(args, preset, asset_name, output_dir, full_df, sample_submission)
    else:
        save_standard_predictions(args, preset, asset_name, output_dir, full_df, sample_submission)
    if not args.skip_features:
        extract_features(args, preset, full_df, output_dir, asset_name)

    weight_pattern = args.weights or preset.initial_weight_pattern
    oof_path = args.oof_preds or str(Path(preset.save_dir) / BASE_OOF_NAMES[preset.name])
    test_csv = args.test_preds_csv or str(Path(preset.save_dir) / preset.submission_name)
    meta = {
        "asset_name": asset_name,
        "route": args.route,
        "preset_name": preset.name,
        "probability_source": "fold_weights" if args.infer_probs_from_weights else "existing_prediction_files",
        "weight_pattern": weight_pattern,
        "prob_tta_modes": args.prob_tta_modes if args.infer_probs_from_weights else None,
        "existing_oof_preds": None if args.infer_probs_from_weights else oof_path,
        "existing_test_preds_csv": None if args.infer_probs_from_weights else test_csv,
        "feature_source": None if args.skip_features else "fold_weights",
        "norm_mean": args.norm_mean or list(preset.norm_mean),
        "norm_std": args.norm_std or list(preset.norm_std),
    }
    with open(output_dir / f"asset_meta_{asset_name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
