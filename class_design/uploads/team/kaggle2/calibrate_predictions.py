from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from pseudo_common import CLASS_COLUMNS, normalize_probabilities


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.replace(",", " ").split() if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF temperature and clipping calibration.")
    parser.add_argument("--asset-dir", default="models/strict_assets")
    parser.add_argument("--oof-preds", default="graph_knn_oof_preds.npy")
    parser.add_argument("--test-preds", default="graph_knn_test_preds.npy")
    parser.add_argument("--train-index", default="train_index.csv")
    parser.add_argument("--test-index", default="test_index.csv")
    parser.add_argument("--output-prefix", default="graph_knn_calibrated")
    parser.add_argument("--temperature-grid", default="0.65,0.75,0.85,0.95,1.0,1.05,1.15,1.3,1.5")
    parser.add_argument("--clip-grid", default="1e-7,1e-6,1e-5,1e-4")
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--class-temperature-grid", default="0.75,0.9,1.0,1.1,1.25")
    parser.add_argument("--class-passes", type=int, default=2)
    return parser.parse_args()


def apply_temperature(probs: np.ndarray, temperature: float, class_temps: np.ndarray | None, clip_min: float) -> np.ndarray:
    logits = np.log(np.clip(probs, clip_min, 1.0))
    logits = logits / temperature
    if class_temps is not None:
        logits = logits / class_temps.reshape(1, -1)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return normalize_probabilities(exp_logits)


def evaluate(y_true, probs, temperature, class_temps, clip_min):
    calibrated = apply_temperature(probs, temperature, class_temps, clip_min)
    calibrated = np.clip(calibrated, clip_min, 1.0 - clip_min)
    calibrated = normalize_probabilities(calibrated)
    return log_loss(y_true, calibrated, labels=range(10)), calibrated


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir)
    train_index = pd.read_csv(asset_dir / args.train_index)
    test_index = pd.read_csv(asset_dir / args.test_index)
    y_true = train_index["label_int"].to_numpy()
    oof = normalize_probabilities(np.load(asset_dir / args.oof_preds))
    test = normalize_probabilities(np.load(asset_dir / args.test_preds))
    expected_oof_shape = (len(train_index), len(CLASS_COLUMNS))
    expected_test_shape = (len(test_index), len(CLASS_COLUMNS))
    if oof.shape != expected_oof_shape:
        raise ValueError(f"{args.oof_preds} has shape {oof.shape}, expected {expected_oof_shape}.")
    if test.shape != expected_test_shape:
        raise ValueError(f"{args.test_preds} has shape {test.shape}, expected {expected_test_shape}.")

    best = {"loss": float("inf"), "temperature": 1.0, "clip_min": 1e-7, "class_temps": None}
    best_oof = None
    for temperature in parse_float_list(args.temperature_grid):
        for clip_min in parse_float_list(args.clip_grid):
            loss, calibrated = evaluate(y_true, oof, temperature, None, clip_min)
            if loss < best["loss"]:
                best = {"loss": float(loss), "temperature": float(temperature), "clip_min": float(clip_min), "class_temps": None}
                best_oof = calibrated

    class_temps = None
    if args.per_class:
        class_temps = np.ones(len(CLASS_COLUMNS), dtype=np.float64)
        grid = parse_float_list(args.class_temperature_grid)
        for _pass in range(args.class_passes):
            for class_idx in range(len(CLASS_COLUMNS)):
                local_best = (best["loss"], class_temps[class_idx], best_oof)
                for candidate in grid:
                    trial_temps = class_temps.copy()
                    trial_temps[class_idx] = candidate
                    loss, calibrated = evaluate(y_true, oof, best["temperature"], trial_temps, best["clip_min"])
                    if loss < local_best[0]:
                        local_best = (float(loss), float(candidate), calibrated)
                class_temps[class_idx] = local_best[1]
                if local_best[0] < best["loss"]:
                    best["loss"] = local_best[0]
                    best_oof = local_best[2]
        best["class_temps"] = class_temps.tolist()

    assert best_oof is not None
    best_test = apply_temperature(
        test,
        temperature=best["temperature"],
        class_temps=np.asarray(best["class_temps"]) if best["class_temps"] is not None else None,
        clip_min=best["clip_min"],
    )
    best_test = np.clip(best_test, best["clip_min"], 1.0 - best["clip_min"])
    best_test = normalize_probabilities(best_test)

    np.save(asset_dir / f"{args.output_prefix}_oof_preds.npy", best_oof.astype(np.float32))
    np.save(asset_dir / f"{args.output_prefix}_test_preds.npy", best_test.astype(np.float32))

    oof_df = pd.DataFrame(best_oof, columns=CLASS_COLUMNS)
    oof_df.insert(0, "img", train_index["img"].values)
    oof_df["label_int"] = y_true
    oof_df.to_csv(asset_dir / f"{args.output_prefix}_oof_preds.csv", index=False)

    test_df = pd.DataFrame(best_test, columns=CLASS_COLUMNS)
    test_df.insert(0, "img", test_index["img"].values)
    test_df.to_csv(asset_dir / f"{args.output_prefix}_test_preds.csv", index=False)

    with open(asset_dir / f"{args.output_prefix}_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"[INFO] Best calibration: {best}")


if __name__ == "__main__":
    main()
