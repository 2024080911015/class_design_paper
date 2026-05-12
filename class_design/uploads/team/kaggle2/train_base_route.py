from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    from sklearn.metrics import log_loss
except ImportError:  # pragma: no cover
    log_loss = None

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
    "swin": "oof_preds_swin.npy",
    "beit": "oof_preds_beit.npy",
}

LOCAL_PRETRAINED = {
    "swin": "swin_base.safetensors",
    "beit": "beit_large.safetensors",
}


class DriverDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        train_dir: str | Path,
        train_fallback_dir: str | Path,
        img_size: int,
        norm_mean: tuple[float, float, float],
        norm_std: tuple[float, float, float],
        is_train: bool,
        coarse_dropout: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.train_dir = Path(train_dir)
        self.train_fallback_dir = Path(train_fallback_dir)

        transforms = [A.Resize(img_size, img_size)]
        if is_train:
            transforms.extend(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.6),
                    A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
                    A.GaussNoise(p=0.3),
                ]
            )
            if coarse_dropout:
                transforms.append(
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(1, 16),
                        hole_width_range=(1, 16),
                        fill=0,
                        p=0.35,
                    )
                )
        transforms.extend([A.Normalize(mean=list(norm_mean), std=list(norm_std)), ToTensorV2()])
        self.transform = A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        class_name = str(row["classname"])
        img_name = str(row["img"])
        primary = self.train_dir / class_name / img_name
        fallback = self.train_fallback_dir / class_name / img_name

        image = cv2.imread(str(primary))
        if image is None:
            image = cv2.imread(str(fallback))
        if image is None:
            raise FileNotFoundError(f"Could not read train image: {primary}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"], int(row["label_int"])


class TestDataset(Dataset):
    def __init__(
        self,
        sample_submission: pd.DataFrame,
        test_dir: str | Path,
        test_fallback_dir: str | Path,
        img_size: int,
        norm_mean: tuple[float, float, float],
        norm_std: tuple[float, float, float],
    ) -> None:
        self.df = sample_submission.reset_index(drop=True)
        self.image_names = self.df["img"].astype(str).tolist()
        self.test_dir = Path(test_dir)
        self.test_fallback_dir = Path(test_fallback_dir)
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
        primary = self.test_dir / img_name
        fallback = self.test_fallback_dir / img_name
        image = cv2.imread(str(primary))
        if image is None:
            image = cv2.imread(str(fallback))
        if image is None:
            raise FileNotFoundError(f"Could not read test image: {primary}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised base Swin/BEiT fold weights without notebooks.")
    parser.add_argument("--route", default="swin", choices=["swin", "transformer", "beit"])
    parser.add_argument("--driver-csv", default="dataset/driver_imgs_list.csv")
    parser.add_argument("--folds-csv", default="train_with_folds.csv")
    parser.add_argument("--sample-submission", default="dataset/sample_submission.csv")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pretrained-weights", default=None, help="Local timm safetensors/pth file. No download is attempted.")
    parser.add_argument("--run-root", default="models/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-canonical", action="store_true", help="Do not copy outputs to the canonical paths used by the pipeline.")
    parser.add_argument("--strict-asset-dir", default="models/strict_assets")
    parser.add_argument("--skip-strict-assets", action="store_true")
    parser.add_argument("--skip-test-preds", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--infer-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--head-lr-mult", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--class-weight-c9", type=float, default=None)
    parser.add_argument("--coarse-dropout", action="store_true", default=None)
    parser.add_argument("--no-coarse-dropout", action="store_false", dest="coarse_dropout")

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--train-fallback-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-fallback-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-tf32", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing local pretrained weights: {path}. The script will not download models; place the file first."
        )

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Loading .safetensors requires safetensors.") from exc

        state = load_file(str(path))
    else:
        state = load_state_dict_cpu(path)

    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    classifier_prefixes = ("head.", "fc.", "classifier.")
    return {
        key: value
        for key, value in state.items()
        if not any(str(key).startswith(prefix) for prefix in classifier_prefixes)
    }


def make_scheduler(optimizer, total_updates: int, warmup_ratio: float):
    warmup_steps = int(total_updates * warmup_ratio)
    try:
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
    except ImportError:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates))


def make_optimizer(model, route_name: str, lr: float, weight_decay: float, head_lr_mult: float | None):
    if head_lr_mult is None:
        head_lr_mult = 10.0 if route_name == "beit" else 1.0

    if head_lr_mult == 1.0:
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("head.", "fc.", "classifier.")):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    if not head_params:
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return AdamW(
        [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": lr * head_lr_mult},
        ],
        weight_decay=weight_decay,
    )


def load_training_frame(folds_csv: str | Path, driver_csv: str | Path) -> pd.DataFrame:
    folds_path = Path(folds_csv)
    if folds_path.exists():
        df = pd.read_csv(folds_path).reset_index(drop=True)
    else:
        df = build_balanced_folds(driver_csv)
        df.to_csv(folds_path, index=False)
        print(f"[INFO] Created folds file: {folds_path}")

    if "label_int" not in df.columns:
        df["label_int"] = df["classname"].str.extract(r"(\d+)").astype(int)
    return df.reset_index(drop=True)


def predict_probabilities(model, loader: DataLoader, device: torch.device, amp_enabled: bool, desc: str) -> np.ndarray:
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
            preds.append(probs.detach().float().cpu().numpy())
    return normalize_probabilities(np.concatenate(preds, axis=0))


def save_oof_csv(path: Path, full_df: pd.DataFrame, oof_preds: np.ndarray) -> None:
    out = full_df[["subject", "classname", "img", "label_int", "fold"]].copy()
    for class_idx, column in enumerate(CLASS_COLUMNS):
        out[column] = oof_preds[:, class_idx]
    out.to_csv(path, index=False)


def copy_if_needed(src: Path, dst: Path, enabled: bool) -> None:
    if not enabled:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_command(command: list[str]) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    preset = get_preset(args.route)
    if preset.name not in {"swin", "beit"}:
        raise ValueError("train_base_route.py is intentionally limited to Transformer routes: swin and beit.")

    set_seed(args.seed)
    if not args.no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    route_name = preset.name
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"base_{route_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = not args.no_amp and device.type == "cuda"
    model_name = args.model_name or preset.model_name
    img_size = args.img_size or preset.img_size
    batch_size = args.batch_size or max(16, preset.batch_size)
    infer_batch_size = args.infer_batch_size or preset.infer_batch_size
    num_workers = args.num_workers if args.num_workers is not None else preset.num_workers
    lr = args.lr if args.lr is not None else (5e-5 if route_name == "swin" else 2e-5)
    weight_decay = args.weight_decay if args.weight_decay is not None else (0.05 if route_name == "swin" else 1e-2)
    drop_path_rate = preset.drop_path_rate if args.drop_path_rate is None else args.drop_path_rate
    train_dir = args.train_dir or preset.train_dir
    train_fallback_dir = args.train_fallback_dir or preset.train_fallback_dir
    test_dir = args.test_dir or preset.test_dir
    test_fallback_dir = args.test_fallback_dir or preset.test_fallback_dir
    class_weight_c9 = args.class_weight_c9 if args.class_weight_c9 is not None else (2.0 if route_name == "swin" else 1.5)
    coarse_dropout = args.coarse_dropout
    if coarse_dropout is None:
        coarse_dropout = route_name == "beit"

    pretrained_path = Path(args.pretrained_weights or LOCAL_PRETRAINED[route_name])
    pretrained_state = load_pretrained_state(pretrained_path)

    full_df = load_training_frame(args.folds_csv, args.driver_csv)
    sample_submission = pd.read_csv(args.sample_submission).reset_index(drop=True)
    oof_preds = np.zeros((len(full_df), len(CLASS_COLUMNS)), dtype=np.float32)
    fold_logs: list[dict] = []
    canonical_enabled = not args.no_canonical

    config = {
        "route": route_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_name": model_name,
        "pretrained_weights": str(pretrained_path),
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "infer_batch_size": infer_batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "drop_path_rate": drop_path_rate,
        "class_weight_c9": class_weight_c9,
        "coarse_dropout": coarse_dropout,
        "norm_mean": list(preset.norm_mean),
        "norm_std": list(preset.norm_std),
        "canonical_enabled": canonical_enabled,
    }
    with open(run_dir / "base_training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for fold in args.folds:
        print(f"\n[INFO] Training {route_name} fold {fold}")
        train_df = full_df[full_df["fold"] != fold].copy()
        val_df = full_df[full_df["fold"] == fold].copy()
        val_indices = val_df.index.to_numpy()

        train_dataset = DriverDataset(
            train_df,
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            img_size=img_size,
            norm_mean=preset.norm_mean,
            norm_std=preset.norm_std,
            is_train=True,
            coarse_dropout=coarse_dropout,
        )
        val_dataset = DriverDataset(
            val_df,
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            img_size=img_size,
            norm_mean=preset.norm_mean,
            norm_std=preset.norm_std,
            is_train=False,
            coarse_dropout=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        base_model = create_timm_model(model_name, drop_path_rate)
        missing, unexpected = base_model.load_state_dict(pretrained_state, strict=False)
        print(f"[INFO] Pretrained load: missing={len(missing)}, unexpected={len(unexpected)}")
        base_model.to(device)
        model = torch.compile(base_model) if args.compile_model and hasattr(torch, "compile") else base_model

        class_weights = torch.ones(len(CLASS_COLUMNS), dtype=torch.float32, device=device)
        class_weights[9] = float(class_weight_c9)
        train_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        valid_criterion = nn.CrossEntropyLoss()
        optimizer = make_optimizer(model, route_name, lr, weight_decay, args.head_lr_mult)
        optimizer.zero_grad(set_to_none=True)
        total_updates = max(1, math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs)
        scheduler = make_scheduler(optimizer, total_updates, args.warmup_ratio)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_val_loss = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        run_weight_path = weights_dir / f"best_model_{route_name}_fold_{fold}.pth"

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            running_count = 0
            pbar = tqdm(train_loader, desc=f"{route_name} fold {fold} epoch {epoch + 1}/{args.epochs} train")
            for step, (images, labels) in enumerate(pbar, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast_context(device, amp_enabled):
                    logits = model(images)
                    loss = train_criterion(logits, labels) / args.accumulation_steps

                scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * args.accumulation_steps * images.size(0)
                running_count += images.size(0)

                if step % args.accumulation_steps == 0 or step == len(train_loader):
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                pbar.set_postfix(loss=f"{running_loss / max(1, running_count):.4f}")

            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            fold_pred_batches: list[np.ndarray] = []
            with torch.inference_mode():
                for images, labels in tqdm(val_loader, desc=f"{route_name} fold {fold} valid"):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast_context(device, amp_enabled):
                        logits = model(images)
                        loss = valid_criterion(logits, labels)
                        probs = torch.softmax(logits, dim=1)
                    val_loss_sum += float(loss.detach().cpu()) * images.size(0)
                    val_count += images.size(0)
                    fold_pred_batches.append(probs.detach().float().cpu().numpy())

            avg_val_loss = val_loss_sum / max(1, val_count)
            current_preds = normalize_probabilities(np.concatenate(fold_pred_batches, axis=0))
            epoch_log = {
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": running_loss / max(1, running_count),
                "val_loss": avg_val_loss,
            }
            fold_logs.append(epoch_log)
            print(f"[INFO] Fold {fold} epoch {epoch + 1}: val_loss={avg_val_loss:.5f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(base_model.state_dict(), run_weight_path)
                oof_preds[val_indices] = current_preds.astype(np.float32)
                print(f"[INFO] Saved improved fold {fold}: {run_weight_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    print(f"[INFO] Early stopping fold {fold}")
                    break

        canonical_weight = Path(preset.initial_weight_pattern.format(fold=fold))
        copy_if_needed(run_weight_path, canonical_weight, canonical_enabled)
        fold_logs.append({"fold": fold, "best_epoch": best_epoch, "best_val_loss": best_val_loss})

        del model, base_model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_preds = normalize_probabilities(oof_preds)
    run_oof_npy = run_dir / BASE_OOF_NAMES[route_name]
    run_oof_csv = run_dir / f"oof_preds_{route_name}.csv"
    np.save(run_oof_npy, oof_preds.astype(np.float32))
    save_oof_csv(run_oof_csv, full_df, oof_preds)

    canonical_oof = Path(preset.save_dir) / BASE_OOF_NAMES[route_name]
    copy_if_needed(run_oof_npy, canonical_oof, canonical_enabled)
    if canonical_enabled:
        save_oof_csv(Path(preset.save_dir) / f"oof_preds_{route_name}.csv", full_df, oof_preds)

    oof_loss = None
    if log_loss is not None and np.all(oof_preds.sum(axis=1) > 0):
        oof_loss = float(log_loss(full_df["label_int"].to_numpy(), np.clip(oof_preds, 1e-7, 1.0 - 1e-7)))
        print(f"[INFO] Final base OOF log loss: {oof_loss:.5f}")

    if not args.skip_test_preds:
        test_dataset = TestDataset(
            sample_submission,
            test_dir=test_dir,
            test_fallback_dir=test_fallback_dir,
            img_size=img_size,
            norm_mean=preset.norm_mean,
            norm_std=preset.norm_std,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        test_sum = None
        for fold in args.folds:
            weight_path = weights_dir / f"best_model_{route_name}_fold_{fold}.pth"
            base_model = create_timm_model(model_name, drop_path_rate)
            state = load_state_dict_cpu(weight_path)
            base_model.load_state_dict(state, strict=True)
            del state
            base_model.to(device)
            base_model.eval()
            model = torch.compile(base_model) if args.compile_model and hasattr(torch, "compile") else base_model
            fold_test = predict_probabilities(model, test_loader, device, amp_enabled, f"{route_name} fold {fold} test")
            if test_sum is None:
                test_sum = np.zeros_like(fold_test, dtype=np.float64)
            test_sum += fold_test.astype(np.float64)
            del model, base_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        test_preds = normalize_probabilities(test_sum / max(1, len(args.folds)))
        test_df = sample_submission[["img"]].copy()
        for class_idx, column in enumerate(CLASS_COLUMNS):
            test_df[column] = test_preds[:, class_idx]
        run_test_csv = run_dir / preset.submission_name
        run_test_npy = run_dir / f"test_preds_{route_name}.npy"
        test_df.to_csv(run_test_csv, index=False)
        np.save(run_test_npy, test_preds.astype(np.float32))
        canonical_test = Path(preset.save_dir) / preset.submission_name
        copy_if_needed(run_test_csv, canonical_test, canonical_enabled)

    pd.DataFrame(fold_logs).to_csv(run_dir / "training_log.csv", index=False)
    summary = {
        "route": route_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "oof_loss": oof_loss,
        "canonical_enabled": canonical_enabled,
        "canonical_oof": str(canonical_oof) if canonical_enabled else None,
    }
    with open(run_dir / "base_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.skip_strict_assets:
        weight_pattern = str(weights_dir / f"best_model_{route_name}_fold_{{fold}}.pth")
        command = [
            sys.executable,
            script_path("build_strict_oof_assets.py"),
            "--route",
            args.route,
            "--driver-csv",
            args.driver_csv,
            "--folds-csv",
            args.folds_csv,
            "--sample-submission",
            args.sample_submission,
            "--weights",
            weight_pattern,
            "--output-dir",
            args.strict_asset_dir,
            "--model-name",
            model_name,
            "--img-size",
            str(img_size),
            "--drop-path-rate",
            str(drop_path_rate),
            "--train-dir",
            str(train_dir),
            "--train-fallback-dir",
            str(train_fallback_dir),
            "--test-dir",
            str(test_dir),
            "--test-fallback-dir",
            str(test_fallback_dir),
            "--infer-probs-from-weights",
            "--prob-tta-modes",
            "base",
            "--normalize-features",
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.no_amp:
            command.append("--no-amp")
        if args.compile_model:
            command.append("--compile")
        command.extend(["--folds", *[str(fold) for fold in args.folds]])
        command.extend(["--batch-size", str(infer_batch_size)])
        command.extend(["--num-workers", str(num_workers)])
        run_command(command)

        if not args.skip_manifest:
            manifest_command = [
                sys.executable,
                script_path("validate_strict_assets.py"),
                "--asset-dir",
                args.strict_asset_dir,
                "--fail-on-issues",
            ]
            run_command(manifest_command)


if __name__ == "__main__":
    main()
