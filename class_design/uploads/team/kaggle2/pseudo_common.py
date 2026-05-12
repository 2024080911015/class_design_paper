from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


CLASS_COLUMNS = [f"c{i}" for i in range(10)]


@dataclass(frozen=True)
class RoutePreset:
    name: str
    model_name: str
    img_size: int
    save_dir: str
    initial_weight_pattern: str
    pseudo_weight_pattern: str
    submission_name: str
    oof_name: str
    drop_path_rate: float
    batch_size: int
    infer_batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    train_dir: str = "dataset/imgs/train_cropped_v2"
    train_fallback_dir: str = "dataset/imgs/train"
    test_dir: str = "dataset/imgs/test_cropped_v2"
    test_fallback_dir: str = "dataset/imgs/test"
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


PRESETS = {
    "cnn": RoutePreset(
        name="effb3",
        model_name="tf_efficientnet_b3.ns_jft_in1k",
        img_size=300,
        save_dir="models/effb3",
        initial_weight_pattern="models/effb3/best_model_effb3_fold_{fold}.pth",
        pseudo_weight_pattern="models/effb3/pseudo_best_model_effb3_fold_{fold}.pth",
        submission_name="submission_effb3_5fold.csv",
        oof_name="pseudo_oof_preds_effb3.npy",
        drop_path_rate=0.2,
        batch_size=32,
        infer_batch_size=128,
        num_workers=8,
        lr=1e-5,
        weight_decay=1e-4,
    ),
    "swin": RoutePreset(
        name="swin",
        model_name="swin_base_patch4_window12_384.ms_in22k",
        img_size=384,
        save_dir="models",
        initial_weight_pattern="models/best_model_swin_fold_{fold}.pth",
        pseudo_weight_pattern="models/pseudo_best_model_swin_fold_{fold}.pth",
        submission_name="submission_swin_5fold_fixed.csv",
        oof_name="pseudo_oof_preds_swin.npy",
        drop_path_rate=0.3,
        batch_size=16,
        infer_batch_size=128,
        num_workers=4,
        lr=8e-6,
        weight_decay=1e-4,
    ),
    "beit": RoutePreset(
        name="beit",
        model_name="beit_large_patch16_224.in22k_ft_in22k_in1k",
        img_size=224,
        save_dir="models/beit",
        initial_weight_pattern="models/beit/best_model_beit_fold_{fold}.pth",
        pseudo_weight_pattern="models/beit/pseudo_best_model_beit_fold_{fold}.pth",
        submission_name="submission_beit_5fold.csv",
        oof_name="pseudo_oof_preds_beit.npy",
        drop_path_rate=0.1,
        batch_size=16,
        infer_batch_size=128,
        num_workers=8,
        lr=6e-6,
        weight_decay=1e-4,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5),
    ),
}

PRESETS["effb3"] = PRESETS["cnn"]
PRESETS["transformer"] = PRESETS["swin"]


def get_preset(route: str) -> RoutePreset:
    try:
        return PRESETS[route.lower()]
    except KeyError as exc:
        names = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown route '{route}'. Available routes: {names}") from exc


def build_balanced_folds(driver_csv: str | Path, n_splits: int = 5) -> pd.DataFrame:
    df = pd.read_csv(driver_csv).reset_index(drop=True)
    if "label_int" not in df.columns:
        df["label_int"] = df["classname"].str.extract(r"(\d+)").astype(int)

    driver_counts = df.groupby("subject").size().sort_values(ascending=False)
    fold_totals = np.zeros(n_splits, dtype=np.int64)
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]

    for subject, count in driver_counts.items():
        fold_idx = int(np.argmin(fold_totals))
        fold_groups[fold_idx].append(subject)
        fold_totals[fold_idx] += int(count)

    df["fold"] = -1
    for fold_idx, subjects in enumerate(fold_groups):
        df.loc[df["subject"].isin(subjects), "fold"] = fold_idx

    return df


def normalize_probabilities(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    preds = np.clip(preds, 0.0, None)
    row_sums = preds.sum(axis=1, keepdims=True)
    return preds / np.clip(row_sums, 1e-12, None)


def select_pseudo_labels(
    image_names: Sequence[str],
    preds: np.ndarray,
    source: str,
    threshold: float,
    min_margin: float,
    per_class_limit: int,
    max_pseudo: int,
) -> pd.DataFrame:
    preds = normalize_probabilities(preds)
    top_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    top2 = np.partition(preds, -2, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]

    mask = (confidences >= threshold) & (margins >= min_margin)
    selected = np.where(mask)[0]

    if per_class_limit > 0 and selected.size > 0:
        limited: list[int] = []
        for class_idx in range(len(CLASS_COLUMNS)):
            class_indices = selected[top_labels[selected] == class_idx]
            order = np.argsort(-confidences[class_indices])
            limited.extend(class_indices[order[:per_class_limit]].tolist())
        selected = np.asarray(limited, dtype=np.int64)

    if max_pseudo > 0 and selected.size > max_pseudo:
        order = np.argsort(-confidences[selected])
        selected = selected[order[:max_pseudo]]

    if selected.size == 0:
        columns = [
            "subject",
            "classname",
            "img",
            "label_int",
            "fold",
            "pseudo_confidence",
            "pseudo_margin",
            "pseudo_source",
        ] + CLASS_COLUMNS
        return pd.DataFrame(columns=columns)

    order = np.lexsort((np.asarray(image_names, dtype=object)[selected], -confidences[selected]))
    selected = selected[order]

    out = pd.DataFrame(
        {
            "subject": "pseudo_test",
            "classname": [f"c{label}" for label in top_labels[selected]],
            "img": np.asarray(image_names, dtype=object)[selected],
            "label_int": top_labels[selected].astype(int),
            "fold": -1,
            "pseudo_confidence": confidences[selected],
            "pseudo_margin": margins[selected],
            "pseudo_source": source,
        }
    )
    for class_idx, column in enumerate(CLASS_COLUMNS):
        out[column] = preds[selected, class_idx]
    return out


def load_state_dict_cpu(weight_path: str | Path):
    import torch

    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if isinstance(state, dict):
        for prefix in ("module.", "_orig_mod."):
            if any(str(key).startswith(prefix) for key in state):
                state = {
                    (key[len(prefix) :] if str(key).startswith(prefix) else key): value
                    for key, value in state.items()
                }

    return state


def create_timm_model(model_name: str, drop_path_rate: float):
    import timm

    kwargs = {"pretrained": False, "num_classes": len(CLASS_COLUMNS)}
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    return timm.create_model(model_name, **kwargs)


def autocast_context(device, enabled: bool):
    if not enabled or getattr(device, "type", None) != "cuda":
        return nullcontext()

    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()
