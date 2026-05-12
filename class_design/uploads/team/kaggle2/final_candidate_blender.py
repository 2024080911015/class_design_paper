from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pseudo_common import CLASS_COLUMNS, normalize_probabilities


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.replace(",", " ").split() if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF-searched log-space blender for final transformer candidates.")
    parser.add_argument("--asset-dir", default="models/strict_assets")
    parser.add_argument(
        "--candidate-prefixes",
        nargs="+",
        default=[
            "graph_knn_calibrated",
            "graph_knn_conservative_calibrated",
            "final_graph_knn_calibrated",
            "final_graph_knn_conservative_calibrated",
        ],
    )
    parser.add_argument("--train-index", default="train_index.csv")
    parser.add_argument("--test-index", default="test_index.csv")
    parser.add_argument("--weight-grid", default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    parser.add_argument("--clip-grid", default="1e-7,1e-6,1e-5,1e-4")
    parser.add_argument("--temperature-grid", default="0.70,0.80,0.90,1.0,1.10,1.25,1.50")
    parser.add_argument("--class-temperature-grid", default="0.80,0.90,1.0,1.10,1.25")
    parser.add_argument("--class-passes", type=int, default=2)
    parser.add_argument("--no-per-class", action="store_true")
    parser.add_argument("--output-prefix", default="transformer_candidate_blend")
    parser.add_argument("--submission", default="submission_transformer_system.csv")
    return parser.parse_args()


def log_loss_score(y_true: np.ndarray, probs: np.ndarray, clip_min: float) -> float:
    probs = np.clip(probs, clip_min, 1.0 - clip_min)
    probs = normalize_probabilities(probs)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true.astype(int)])))


def log_blend(sources: list[np.ndarray], weights: list[float], clip_min: float) -> np.ndarray:
    logits = np.zeros_like(sources[0], dtype=np.float64)
    for source, weight in zip(sources, weights):
        if weight <= 0:
            continue
        logits += float(weight) * np.log(np.clip(source, clip_min, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probabilities(np.exp(logits))


def apply_temperature(
    probs: np.ndarray,
    temperature: float,
    class_temps: np.ndarray | None,
    clip_min: float,
) -> np.ndarray:
    logits = np.log(np.clip(probs, clip_min, 1.0)) / temperature
    if class_temps is not None:
        logits = logits / class_temps.reshape(1, -1)
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probabilities(np.exp(logits))


def load_candidate(asset_dir: Path, prefix: str, n_train: int, n_test: int):
    oof_path = asset_dir / f"{prefix}_oof_preds.npy"
    test_path = asset_dir / f"{prefix}_test_preds.npy"
    if not oof_path.exists() or not test_path.exists():
        print(f"[WARN] Missing candidate skipped: {prefix}")
        return None
    oof = np.load(oof_path)
    test = np.load(test_path)
    expected_oof = (n_train, len(CLASS_COLUMNS))
    expected_test = (n_test, len(CLASS_COLUMNS))
    if oof.shape != expected_oof:
        raise ValueError(f"{oof_path} has shape {oof.shape}, expected {expected_oof}.")
    if test.shape != expected_test:
        raise ValueError(f"{test_path} has shape {test.shape}, expected {expected_test}.")
    return normalize_probabilities(oof), normalize_probabilities(test), oof_path, test_path


def search_weights(
    y_true: np.ndarray,
    oof_sources: list[np.ndarray],
    weight_grid: list[float],
    clip_grid: list[float],
):
    mesh = np.array(np.meshgrid(*([weight_grid] * len(oof_sources)))).T.reshape(-1, len(oof_sources))
    best = {"loss": float("inf"), "weights": None, "clip_min": None}
    best_oof = None
    for raw_weights in mesh:
        weights = raw_weights.astype(np.float64).tolist()
        if all(weight <= 0 for weight in weights):
            continue
        for clip_min in clip_grid:
            preds = log_blend(oof_sources, weights, clip_min)
            loss = log_loss_score(y_true, preds, clip_min)
            if loss < best["loss"]:
                best = {"loss": float(loss), "weights": weights, "clip_min": float(clip_min)}
                best_oof = preds
    if best_oof is None:
        raise RuntimeError("Candidate weight search failed.")
    return best, best_oof


def calibrate(
    y_true: np.ndarray,
    oof: np.ndarray,
    test: np.ndarray,
    temperature_grid: list[float],
    clip_grid: list[float],
    class_temperature_grid: list[float],
    class_passes: int,
    per_class: bool,
):
    best = {"loss": float("inf"), "temperature": 1.0, "clip_min": 1e-7, "class_temps": None}
    best_oof = None
    for temperature in temperature_grid:
        for clip_min in clip_grid:
            preds = apply_temperature(oof, temperature, None, clip_min)
            loss = log_loss_score(y_true, preds, clip_min)
            if loss < best["loss"]:
                best = {
                    "loss": float(loss),
                    "temperature": float(temperature),
                    "clip_min": float(clip_min),
                    "class_temps": None,
                }
                best_oof = preds

    if per_class:
        class_temps = np.ones(len(CLASS_COLUMNS), dtype=np.float64)
        for _ in range(class_passes):
            for class_idx in range(len(CLASS_COLUMNS)):
                local_loss = best["loss"]
                local_temp = class_temps[class_idx]
                local_oof = best_oof
                for candidate in class_temperature_grid:
                    trial_temps = class_temps.copy()
                    trial_temps[class_idx] = candidate
                    preds = apply_temperature(oof, best["temperature"], trial_temps, best["clip_min"])
                    loss = log_loss_score(y_true, preds, best["clip_min"])
                    if loss < local_loss:
                        local_loss = float(loss)
                        local_temp = float(candidate)
                        local_oof = preds
                class_temps[class_idx] = local_temp
                if local_loss < best["loss"]:
                    best["loss"] = local_loss
                    best_oof = local_oof
        best["class_temps"] = class_temps.tolist()

    assert best_oof is not None
    best_test = apply_temperature(
        test,
        best["temperature"],
        np.asarray(best["class_temps"]) if best["class_temps"] is not None else None,
        best["clip_min"],
    )
    best_oof = np.clip(best_oof, best["clip_min"], 1.0 - best["clip_min"])
    best_test = np.clip(best_test, best["clip_min"], 1.0 - best["clip_min"])
    return normalize_probabilities(best_oof), normalize_probabilities(best_test), best


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir)
    train_index = pd.read_csv(asset_dir / args.train_index)
    test_index = pd.read_csv(asset_dir / args.test_index)
    y_true = train_index["label_int"].to_numpy(dtype=int)

    candidate_names: list[str] = []
    oof_sources: list[np.ndarray] = []
    test_sources: list[np.ndarray] = []
    paths = {}
    for prefix in args.candidate_prefixes:
        loaded = load_candidate(asset_dir, prefix, len(train_index), len(test_index))
        if loaded is None:
            continue
        oof, test, oof_path, test_path = loaded
        candidate_names.append(prefix)
        oof_sources.append(oof)
        test_sources.append(test)
        paths[prefix] = {"oof": str(oof_path), "test": str(test_path)}

    if len(oof_sources) < 1:
        raise FileNotFoundError("Need at least one available candidate for final blending.")

    weight_grid = parse_float_list(args.weight_grid)
    clip_grid = parse_float_list(args.clip_grid)
    if len(oof_sources) == 1:
        weight_best = {
            "loss": log_loss_score(y_true, oof_sources[0], min(clip_grid)),
            "weights": [1.0],
            "clip_min": min(clip_grid),
        }
        blended_oof = oof_sources[0]
        blended_test = test_sources[0]
    else:
        weight_best, blended_oof = search_weights(y_true, oof_sources, weight_grid, clip_grid)
        blended_test = log_blend(test_sources, weight_best["weights"], weight_best["clip_min"])
    calibrated_oof, calibrated_test, calibration = calibrate(
        y_true=y_true,
        oof=blended_oof,
        test=blended_test,
        temperature_grid=parse_float_list(args.temperature_grid),
        clip_grid=clip_grid,
        class_temperature_grid=parse_float_list(args.class_temperature_grid),
        class_passes=args.class_passes,
        per_class=not args.no_per_class,
    )

    oof_path = asset_dir / f"{args.output_prefix}_oof_preds.npy"
    test_path = asset_dir / f"{args.output_prefix}_test_preds.npy"
    oof_csv_path = asset_dir / f"{args.output_prefix}_oof_preds.csv"
    test_csv_path = asset_dir / f"{args.output_prefix}_test_preds.csv"
    config_path = asset_dir / f"{args.output_prefix}_config.json"

    np.save(oof_path, calibrated_oof.astype(np.float32))
    np.save(test_path, calibrated_test.astype(np.float32))

    metadata_columns = [column for column in ["subject", "classname", "img", "label_int", "fold"] if column in train_index.columns]
    oof_df = train_index[metadata_columns].copy()
    for class_idx, column in enumerate(CLASS_COLUMNS):
        oof_df[column] = calibrated_oof[:, class_idx]
    oof_df.to_csv(oof_csv_path, index=False)

    test_df = pd.DataFrame(calibrated_test, columns=CLASS_COLUMNS)
    test_df.insert(0, "img", test_index["img"].values)
    test_df.to_csv(test_csv_path, index=False)
    test_df.to_csv(args.submission, index=False)

    config = {
        "candidates": candidate_names,
        "candidate_paths": paths,
        "weight_search": {
            "loss": weight_best["loss"],
            "clip_min": weight_best["clip_min"],
            "weights": dict(zip(candidate_names, weight_best["weights"])),
        },
        "calibration": calibration,
        "final_oof_log_loss": log_loss_score(y_true, calibrated_oof, calibration["clip_min"]),
        "submission": args.submission,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] Candidate weights: {config['weight_search']['weights']}")
    print(f"[INFO] Final candidate blend OOF log loss: {config['final_oof_log_loss']:.6f}")
    print(f"[INFO] Saved submission: {args.submission}")
    print(f"[INFO] Saved config: {config_path}")


if __name__ == "__main__":
    main()
