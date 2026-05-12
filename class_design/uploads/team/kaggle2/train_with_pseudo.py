from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
)


class DriverPseudoDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        train_dir: str | Path,
        train_fallback_dir: str | Path,
        test_dir: str | Path,
        test_fallback_dir: str | Path,
        img_size: int,
        is_train: bool,
        pseudo_weight: float,
        confidence_weighting: bool,
        norm_mean: tuple[float, float, float],
        norm_std: tuple[float, float, float],
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.train_dir = Path(train_dir)
        self.train_fallback_dir = Path(train_fallback_dir)
        self.test_dir = Path(test_dir)
        self.test_fallback_dir = Path(test_fallback_dir)
        self.pseudo_weight = float(pseudo_weight)
        self.confidence_weighting = bool(confidence_weighting)

        transforms = [A.Resize(img_size, img_size)]
        if is_train:
            transforms.extend(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.6),
                    A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
                    A.GaussNoise(p=0.3),
                ]
            )
        transforms.extend(
            [
                A.Normalize(mean=list(norm_mean), std=list(norm_std)),
                ToTensorV2(),
            ]
        )
        self.transform = A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _is_pseudo(row: pd.Series) -> bool:
        if bool(row.get("is_pseudo", False)):
            return True
        if str(row.get("subject", "")) == "pseudo_test":
            return True
        try:
            return int(row.get("fold", 0)) < 0
        except (TypeError, ValueError):
            return False

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        is_pseudo = self._is_pseudo(row)
        img_name = str(row["img"])

        if is_pseudo:
            primary = self.test_dir / img_name
            fallback = self.test_fallback_dir / img_name
        else:
            class_name = str(row["classname"])
            primary = self.train_dir / class_name / img_name
            fallback = self.train_fallback_dir / class_name / img_name

        image = cv2.imread(str(primary))
        if image is None:
            image = cv2.imread(str(fallback))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {primary}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        label = int(row["label_int"])
        weight = 1.0
        if is_pseudo:
            confidence = float(row.get("pseudo_confidence", 1.0))
            weight = self.pseudo_weight
            if self.confidence_weighting:
                weight *= max(0.0, min(1.0, confidence))

        teacher_probs = np.zeros(len(CLASS_COLUMNS), dtype=np.float32)
        has_soft = all(column in row.index for column in CLASS_COLUMNS)
        if has_soft:
            raw_probs = row[CLASS_COLUMNS].to_numpy(dtype=np.float32)
            if np.isfinite(raw_probs).all() and raw_probs.sum() > 0:
                teacher_probs = raw_probs / max(float(raw_probs.sum()), 1e-12)
            else:
                teacher_probs[label] = 1.0
        else:
            teacher_probs[label] = 1.0

        hard_pseudo = bool(row.get("hard_pseudo", True))

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(teacher_probs, dtype=torch.float32),
            torch.tensor(is_pseudo, dtype=torch.bool),
            torch.tensor(hard_pseudo, dtype=torch.bool),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune timm models with hard or soft pseudo labels.")
    parser.add_argument(
        "--route",
        default="transformer",
        help="swin/transformer or beit. Legacy cnn/effb3 is only used when explicitly requested.",
    )
    parser.add_argument("--driver-csv", default="dataset/driver_imgs_list.csv")
    parser.add_argument("--folds-csv", default="train_with_folds.csv")
    parser.add_argument("--pseudo-csv", default="pseudo_labels.csv")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--pseudo-weight", type=float, default=0.3)
    parser.add_argument(
        "--pseudo-sample-ratio",
        type=float,
        default=1.0,
        help="Expected pseudo examples per real example in each epoch. Set below 0 to use plain concat shuffle.",
    )
    parser.add_argument("--confidence-weighting", action="store_true")
    parser.add_argument("--pseudo-threshold", type=float, default=0.0)
    parser.add_argument("--soft-pseudo", action="store_true", help="Use c0..c9 columns as teacher soft labels.")
    parser.add_argument("--soft-kl-weight", type=float, default=0.7)
    parser.add_argument("--hard-pseudo-weight", type=float, default=0.3)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-c9", type=float, default=1.0)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--train-fallback-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-fallback-dir", default=None)
    parser.add_argument("--initial-weights", default=None)
    parser.add_argument("--save-weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-random-start", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    try:
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    except ImportError:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))


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


def load_pseudo_frame(pseudo_csv: str | Path, threshold: float) -> pd.DataFrame:
    pseudo_df = pd.read_csv(pseudo_csv)
    required = {"img", "classname", "label_int"}
    missing = sorted(required - set(pseudo_df.columns))
    if missing:
        raise ValueError(f"{pseudo_csv} is missing required columns: {missing}")

    if "pseudo_confidence" in pseudo_df.columns and threshold > 0:
        pseudo_df = pseudo_df[pseudo_df["pseudo_confidence"] >= threshold].copy()

    pseudo_df["subject"] = "pseudo_test"
    pseudo_df["fold"] = -1
    pseudo_df["is_pseudo"] = True
    if "pseudo_confidence" not in pseudo_df.columns:
        pseudo_df["pseudo_confidence"] = 1.0
    if "hard_pseudo" not in pseudo_df.columns:
        pseudo_df["hard_pseudo"] = True

    return pseudo_df.reset_index(drop=True)


def compute_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    teacher_probs: torch.Tensor,
    is_pseudo: torch.Tensor,
    hard_pseudo: torch.Tensor,
    criterion,
    args: argparse.Namespace,
) -> torch.Tensor:
    if not args.soft_pseudo:
        loss_vec = criterion(logits, labels)
        return (loss_vec * weights).mean()

    loss = logits.sum() * 0.0
    real_mask = ~is_pseudo
    pseudo_mask = is_pseudo
    hard_mask = is_pseudo & hard_pseudo

    if real_mask.any():
        real_loss = criterion(logits[real_mask], labels[real_mask]).mean()
        loss = loss + real_loss

    if pseudo_mask.any() and args.soft_kl_weight > 0:
        temperature = float(args.kl_temperature)
        teacher = teacher_probs[pseudo_mask]
        teacher = torch.pow(torch.clamp(teacher, min=1e-12), 1.0 / temperature)
        teacher = teacher / torch.clamp(teacher.sum(dim=1, keepdim=True), min=1e-12)
        student_log_probs = F.log_softmax(logits[pseudo_mask] / temperature, dim=1)
        kl_vec = F.kl_div(student_log_probs, teacher, reduction="none").sum(dim=1) * (temperature ** 2)
        kl_loss = (kl_vec * weights[pseudo_mask]).mean()
        loss = loss + args.soft_kl_weight * kl_loss

    if hard_mask.any() and args.hard_pseudo_weight > 0:
        hard_loss_vec = criterion(logits[hard_mask], labels[hard_mask])
        hard_loss = (hard_loss_vec * weights[hard_mask]).mean()
        loss = loss + args.hard_pseudo_weight * hard_loss

    return loss


def make_pseudo_balanced_sampler(train_df: pd.DataFrame, pseudo_sample_ratio: float):
    if pseudo_sample_ratio < 0:
        return None

    is_pseudo = train_df["is_pseudo"].astype(bool).to_numpy()
    real_count = int((~is_pseudo).sum())
    pseudo_count = int(is_pseudo.sum())
    if real_count == 0 or pseudo_count == 0:
        return None

    sample_weights = np.zeros(len(train_df), dtype=np.float64)
    sample_weights[~is_pseudo] = 1.0 / real_count
    sample_weights[is_pseudo] = float(pseudo_sample_ratio) / pseudo_count
    num_samples = max(1, int(math.ceil(real_count * (1.0 + float(pseudo_sample_ratio)))))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def main() -> None:
    args = parse_args()
    preset = get_preset(args.route)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = not args.no_amp and device.type == "cuda"
    img_size = args.img_size or preset.img_size
    batch_size = args.batch_size or preset.batch_size
    num_workers = args.num_workers if args.num_workers is not None else preset.num_workers
    lr = args.lr if args.lr is not None else preset.lr
    weight_decay = args.weight_decay if args.weight_decay is not None else preset.weight_decay
    model_name = args.model_name or preset.model_name
    drop_path_rate = preset.drop_path_rate if args.drop_path_rate is None else args.drop_path_rate
    train_dir = args.train_dir or preset.train_dir
    train_fallback_dir = args.train_fallback_dir or preset.train_fallback_dir
    test_dir = args.test_dir or preset.test_dir
    test_fallback_dir = args.test_fallback_dir or preset.test_fallback_dir
    norm_mean = tuple(preset.norm_mean)
    norm_std = tuple(preset.norm_std)
    output_dir = Path(args.output_dir or preset.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_pattern = args.initial_weights or preset.initial_weight_pattern
    if args.save_weights:
        save_pattern = args.save_weights
    elif args.output_dir:
        save_pattern = str(output_dir / Path(preset.pseudo_weight_pattern).name)
    else:
        save_pattern = preset.pseudo_weight_pattern

    full_df = load_training_frame(args.folds_csv, args.driver_csv)
    pseudo_df = load_pseudo_frame(args.pseudo_csv, args.pseudo_threshold)
    if len(pseudo_df) == 0:
        raise ValueError("No pseudo labels left after filtering.")

    print(f"[INFO] Route: {preset.name}")
    print(f"[INFO] Pseudo labels: {len(pseudo_df)}")
    print(pseudo_df["classname"].value_counts().sort_index().to_string())

    oof_preds = np.zeros((len(full_df), len(CLASS_COLUMNS)), dtype=np.float32)

    for fold in args.folds:
        print(f"\n[INFO] Fold {fold}")
        original_train = full_df[full_df["fold"] != fold].copy()
        val_df = full_df[full_df["fold"] == fold].copy()
        val_indices = val_df.index.to_numpy()

        original_train["is_pseudo"] = False
        val_df["is_pseudo"] = False
        train_df = pd.concat([original_train, pseudo_df], axis=0, ignore_index=True)

        print(f"[INFO] Train rows: {len(train_df)} ({len(pseudo_df)} pseudo)")
        print(f"[INFO] Valid rows: {len(val_df)}")

        train_dataset = DriverPseudoDataset(
            train_df,
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            test_dir=test_dir,
            test_fallback_dir=test_fallback_dir,
            img_size=img_size,
            is_train=True,
            pseudo_weight=args.pseudo_weight,
            confidence_weighting=args.confidence_weighting,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        val_dataset = DriverPseudoDataset(
            val_df,
            train_dir=train_dir,
            train_fallback_dir=train_fallback_dir,
            test_dir=test_dir,
            test_fallback_dir=test_fallback_dir,
            img_size=img_size,
            is_train=False,
            pseudo_weight=args.pseudo_weight,
            confidence_weighting=False,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        train_sampler = make_pseudo_balanced_sampler(train_df, args.pseudo_sample_ratio)
        if train_sampler is not None:
            print(
                "[INFO] Balanced pseudo sampler: "
                f"real:pseudo ~= 1:{args.pseudo_sample_ratio}, samples/epoch={len(train_sampler)}"
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
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
        init_path = Path(initial_pattern.format(fold=fold))
        if init_path.exists():
            print(f"[INFO] Loading initial weights: {init_path}")
            state = load_state_dict_cpu(init_path)
            base_model.load_state_dict(state, strict=True)
            del state
        elif not args.allow_random_start:
            raise FileNotFoundError(f"Missing initial weights for fold {fold}: {init_path}")
        else:
            print("[WARN] Initial weights missing; starting from random weights.")

        base_model.to(device)
        model = torch.compile(base_model) if args.compile_model and hasattr(torch, "compile") else base_model

        class_weights = torch.ones(len(CLASS_COLUMNS), dtype=torch.float32, device=device)
        class_weights[9] = float(args.class_weight_c9)
        train_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
            reduction="none",
        )
        valid_criterion = nn.CrossEntropyLoss(reduction="mean")

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad(set_to_none=True)
        total_updates = max(1, math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs)
        scheduler = make_scheduler(optimizer, total_updates, args.warmup_ratio)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        save_path = Path(save_pattern.format(fold=fold))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            running_count = 0

            pbar = tqdm(train_loader, desc=f"fold {fold} epoch {epoch + 1}/{args.epochs} train")
            for step, (images, labels, weights, teacher_probs, is_pseudo, hard_pseudo) in enumerate(pbar, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                teacher_probs = teacher_probs.to(device, non_blocking=True)
                is_pseudo = is_pseudo.to(device, non_blocking=True)
                hard_pseudo = hard_pseudo.to(device, non_blocking=True)

                with autocast_context(device, amp_enabled):
                    logits = model(images)
                    loss = compute_training_loss(
                        logits=logits,
                        labels=labels,
                        weights=weights,
                        teacher_probs=teacher_probs,
                        is_pseudo=is_pseudo,
                        hard_pseudo=hard_pseudo,
                        criterion=train_criterion,
                        args=args,
                    )
                    loss = loss / args.accumulation_steps

                scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * args.accumulation_steps * images.size(0)
                running_count += images.size(0)

                if step % args.accumulation_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                pbar.set_postfix(loss=f"{running_loss / max(1, running_count):.4f}")

            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            fold_preds: list[np.ndarray] = []
            with torch.inference_mode():
                for images, labels, _weights, _teacher_probs, _is_pseudo, _hard_pseudo in tqdm(
                    val_loader,
                    desc=f"fold {fold} valid",
                ):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast_context(device, amp_enabled):
                        logits = model(images)
                        loss = valid_criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)
                    val_loss_sum += float(loss.detach().cpu()) * images.size(0)
                    val_count += images.size(0)
                    fold_preds.append(probs.detach().cpu().numpy())

            avg_val_loss = val_loss_sum / max(1, val_count)
            current_preds = np.concatenate(fold_preds, axis=0)
            print(f"[INFO] Fold {fold} epoch {epoch + 1}: val_loss={avg_val_loss:.5f}")

            if avg_val_loss < best_val_loss:
                print(f"[INFO] Saving improved fold {fold}: {best_val_loss:.5f} -> {avg_val_loss:.5f}")
                best_val_loss = avg_val_loss
                torch.save(base_model.state_dict(), save_path)
                oof_preds[val_indices] = current_preds
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    print(f"[INFO] Early stopping fold {fold}")
                    break

        del model, base_model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_path = output_dir / preset.oof_name
    np.save(oof_path, oof_preds)
    print(f"[INFO] Saved OOF predictions: {oof_path}")

    completed = np.where(oof_preds.sum(axis=1) > 0)[0]
    if len(completed) == len(full_df) and log_loss is not None:
        y_true = full_df["label_int"].to_numpy()
        clipped = np.clip(oof_preds, 1e-7, 1.0 - 1e-7)
        clipped = clipped / clipped.sum(axis=1, keepdims=True)
        print(f"[INFO] Final OOF log loss: {log_loss(y_true, clipped):.5f}")
    elif len(completed) != len(full_df):
        print("[WARN] OOF log loss skipped because not all folds were trained.")
    else:
        print("[WARN] OOF log loss skipped because scikit-learn is not installed.")


if __name__ == "__main__":
    main()
