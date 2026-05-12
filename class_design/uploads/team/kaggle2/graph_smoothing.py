from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import log_loss
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover
    raise ImportError("graph_smoothing.py requires scikit-learn.") from exc

from pseudo_common import CLASS_COLUMNS, normalize_probabilities


@dataclass
class SearchResult:
    loss: float
    feature_preset: str
    feature_weights: dict[str, float]
    oof_neighbor_mode: str
    k: int
    alpha: float
    clip_min: float
    w_self: float
    w_test: float
    w_train: float


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.replace(",", " ").split() if x]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.replace(",", " ").split() if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF-searched KNN graph smoothing for State Farm predictions.")
    parser.add_argument("--asset-dir", default="models/strict_assets")
    parser.add_argument("--prob-models", nargs="+", default=["swin", "beit"])
    parser.add_argument("--feature-models", nargs="+", default=["swin", "beit", "dinov2_crop", "dinov2_full"])
    parser.add_argument("--output-prefix", default="graph_knn")
    parser.add_argument("--k-grid", default="5,10,20,40")
    parser.add_argument("--alpha-grid", default="0,5,15,30")
    parser.add_argument("--clip-grid", default="1e-7,1e-6,1e-5")
    parser.add_argument("--weight-grid", default="0,0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--feature-weights", default=None, help="Fixed comma/space weights aligned to --feature-models.")
    parser.add_argument("--no-feature-preset-search", action="store_true")
    parser.add_argument("--feature-dirichlet-count", type=int, default=5)
    parser.add_argument("--feature-dirichlet-seed", type=int, default=42)
    parser.add_argument("--feature-dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--prob-weights", default=None, help="Fixed comma/space weights aligned to --prob-models.")
    parser.add_argument("--prob-weight-grid", default="0,0.5,1.0,1.5,2.0")
    parser.add_argument("--prob-blend-clip", type=float, default=1e-7)
    parser.add_argument("--prob-weight-search", action="store_true", default=True)
    parser.add_argument("--no-prob-weight-search", action="store_false", dest="prob_weight_search")
    parser.add_argument(
        "--oof-neighbor-mode",
        choices=["transductive", "train_only"],
        default="transductive",
        help="transductive uses train->test neighbors in OOF; train_only is the conservative ablation.",
    )
    parser.add_argument("--max-k", type=int, default=None)
    return parser.parse_args()


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def validate_probability_shape(path: Path, preds: np.ndarray, expected_rows: int) -> None:
    expected = (expected_rows, len(CLASS_COLUMNS))
    if preds.shape != expected:
        raise ValueError(f"{path} has shape {preds.shape}, expected {expected}.")


def load_probability_arrays(asset_dir: Path, model_names: list[str], expected_train: int, expected_test: int):
    oof_list = []
    test_list = []
    used_models = []
    for name in model_names:
        oof_path = asset_dir / f"oof_preds_{name}.npy"
        test_path = asset_dir / f"test_preds_{name}.npy"
        if not oof_path.exists() or not test_path.exists():
            print(f"[WARN] Missing prob assets for {name}, skipped.")
            continue
        oof_preds = np.load(oof_path)
        test_preds = np.load(test_path)
        validate_probability_shape(oof_path, oof_preds, expected_train)
        validate_probability_shape(test_path, test_preds, expected_test)
        oof_list.append(normalize_probabilities(oof_preds))
        test_list.append(normalize_probabilities(test_preds))
        used_models.append(name)

    if not oof_list:
        raise FileNotFoundError("No probability assets found.")
    return oof_list, test_list, used_models


def search_probability_blend(
    y_true: np.ndarray,
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    used_models: list[str],
    model_names: list[str],
    args: argparse.Namespace,
):
    def loss_score(preds: np.ndarray, clip_min: float) -> float:
        clipped = normalize_probabilities(np.clip(preds, clip_min, 1.0 - clip_min))
        return float(log_loss(y_true, clipped, labels=range(10)))

    if args.prob_weights:
        raw_weights = parse_float_list(args.prob_weights)
        if len(raw_weights) != len(model_names):
            raise ValueError("--prob-weights must align to the original --prob-models list.")
        weights = np.asarray([raw_weights[model_names.index(name)] for name in used_models], dtype=np.float64)
        if np.all(weights <= 0):
            raise ValueError("--prob-weights cannot all be zero.")
        oof = log_blend(oof_list, weights.tolist(), args.prob_blend_clip)
        test = log_blend(test_list, weights.tolist(), args.prob_blend_clip)
        loss = loss_score(oof, args.prob_blend_clip)
        return oof, test, used_models, weights, float(loss)

    if not args.prob_weight_search:
        weights = np.ones(len(used_models), dtype=np.float64)
        oof = log_blend(oof_list, weights.tolist(), args.prob_blend_clip)
        test = log_blend(test_list, weights.tolist(), args.prob_blend_clip)
        loss = loss_score(oof, args.prob_blend_clip)
        return oof, test, used_models, weights, float(loss)

    grid = parse_float_list(args.prob_weight_grid)
    mesh = np.array(np.meshgrid(*([grid] * len(used_models)))).T.reshape(-1, len(used_models))
    best_loss = float("inf")
    best_weights = None
    best_oof = None
    for raw_weights in mesh:
        weights = raw_weights.astype(np.float64)
        if np.all(weights <= 0):
            continue
        preds = log_blend(oof_list, weights.tolist(), args.prob_blend_clip)
        loss = loss_score(preds, args.prob_blend_clip)
        if loss < best_loss:
            best_loss = float(loss)
            best_weights = weights
            best_oof = preds
    assert best_oof is not None and best_weights is not None
    best_test = log_blend(test_list, best_weights.tolist(), args.prob_blend_clip)
    return best_oof, best_test, used_models, best_weights, best_loss


def load_probabilities(
    asset_dir: Path,
    model_names: list[str],
    expected_train: int,
    expected_test: int,
    y_true: np.ndarray,
    args: argparse.Namespace,
):
    oof_list, test_list, used_models = load_probability_arrays(asset_dir, model_names, expected_train, expected_test)
    return search_probability_blend(y_true, oof_list, test_list, used_models, model_names, args)


def validate_feature_shape(path: Path, features: np.ndarray, expected_rows: int) -> None:
    if features.ndim != 2 or features.shape[0] != expected_rows:
        raise ValueError(f"{path} has shape {features.shape}, expected first dimension {expected_rows}.")


def load_feature_arrays(asset_dir: Path, model_names: list[str], expected_train: int, expected_test: int):
    arrays = {}
    for name in model_names:
        train_path = asset_dir / f"train_features_{name}.npy"
        test_path = asset_dir / f"test_features_{name}.npy"
        if not train_path.exists() or not test_path.exists():
            print(f"[WARN] Missing feature assets for {name}, skipped.")
            continue
        train_features = np.load(train_path)
        test_features = np.load(test_path)
        validate_feature_shape(train_path, train_features, expected_train)
        validate_feature_shape(test_path, test_features, expected_test)
        arrays[name] = (l2_normalize(train_features), l2_normalize(test_features))
    if not arrays:
        raise FileNotFoundError("No feature assets found.")
    return arrays


def feature_presets(
    feature_names: list[str],
    fixed_weights: str | None,
    search_presets: bool,
    dirichlet_count: int,
    dirichlet_seed: int,
    dirichlet_alpha: float,
):
    if fixed_weights:
        weights = parse_float_list(fixed_weights)
        if len(weights) != len(feature_names):
            raise ValueError("--feature-weights must align to loaded feature names.")
        weight_map = {name: float(weight) for name, weight in zip(feature_names, weights)}
        return [("fixed", weight_map)]

    uniform = {name: 1.0 / len(feature_names) for name in feature_names}
    presets = [("uniform", uniform)]
    if search_presets:
        for name in feature_names:
            presets.append((f"only_{name}", {feature_name: float(feature_name == name) for feature_name in feature_names}))
        if dirichlet_count > 0:
            rng = np.random.default_rng(dirichlet_seed)
            alpha = np.full(len(feature_names), max(float(dirichlet_alpha), 1e-6), dtype=np.float64)
            for preset_idx in range(dirichlet_count):
                weights = rng.dirichlet(alpha)
                presets.append(
                    (
                        f"dirichlet_{preset_idx:03d}",
                        {name: float(weight) for name, weight in zip(feature_names, weights)},
                    )
                )
    return presets


def combine_features(feature_arrays: dict[str, tuple[np.ndarray, np.ndarray]], weights: dict[str, float]):
    train_parts = []
    test_parts = []
    for name, weight in weights.items():
        if weight <= 0 or name not in feature_arrays:
            continue
        scale = np.sqrt(weight)
        train_part, test_part = feature_arrays[name]
        train_parts.append(train_part * scale)
        test_parts.append(test_part * scale)
    if not train_parts:
        raise ValueError("Feature weights selected no available features.")
    return l2_normalize(np.concatenate(train_parts, axis=1)), l2_normalize(np.concatenate(test_parts, axis=1))


def remove_self_neighbors(indices: np.ndarray, distances: np.ndarray, max_k: int):
    cleaned_idx = np.empty((indices.shape[0], max_k), dtype=np.int64)
    cleaned_dist = np.empty((distances.shape[0], max_k), dtype=np.float32)
    row_ids = np.arange(indices.shape[0])
    for row in range(indices.shape[0]):
        keep = indices[row] != row_ids[row]
        cleaned_idx[row] = indices[row][keep][:max_k]
        cleaned_dist[row] = distances[row][keep][:max_k]
    return cleaned_idx, cleaned_dist


def knn_indices(train_features: np.ndarray, test_features: np.ndarray, max_k: int):
    train_nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    train_nn.fit(train_features)
    train_train_dist, train_train_idx = train_nn.kneighbors(train_features)
    train_train_idx, train_train_dist = remove_self_neighbors(train_train_idx, train_train_dist, max_k)

    test_nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    test_nn.fit(test_features)
    test_test_dist, test_test_idx = test_nn.kneighbors(test_features)
    test_test_idx, test_test_dist = remove_self_neighbors(test_test_idx, test_test_dist, max_k)

    train_test_dist, train_test_idx = test_nn.kneighbors(train_features, n_neighbors=max_k)
    test_train_dist, test_train_idx = train_nn.kneighbors(test_features, n_neighbors=max_k)
    return {
        "train_train": (train_train_idx, 1.0 - train_train_dist),
        "test_test": (test_test_idx, 1.0 - test_test_dist),
        "train_test": (train_test_idx, 1.0 - train_test_dist),
        "test_train": (test_train_idx, 1.0 - test_train_dist),
    }


def aggregate_neighbors(base_probs: np.ndarray, indices: np.ndarray, sims: np.ndarray, k: int, alpha: float) -> np.ndarray:
    idx = indices[:, :k]
    sim = sims[:, :k]
    if alpha == 0:
        weights = np.ones_like(sim, dtype=np.float64)
    else:
        weights = np.exp(alpha * (sim - sim.max(axis=1, keepdims=True)))
    gathered = base_probs[idx]
    weighted = (gathered * weights[:, :, None]).sum(axis=1)
    return normalize_probabilities(weighted / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None))


def log_blend(sources: list[np.ndarray], weights: list[float], clip_min: float) -> np.ndarray:
    logs = np.zeros_like(sources[0], dtype=np.float64)
    for source, weight in zip(sources, weights):
        if weight <= 0:
            continue
        logs += weight * np.log(np.clip(source, clip_min, 1.0))
    logs -= logs.max(axis=1, keepdims=True)
    probs = np.exp(logs)
    return normalize_probabilities(probs)


def evaluate_grid(
    y_true: np.ndarray,
    oof_self: np.ndarray,
    test_self: np.ndarray,
    neighbor_data,
    k_values: list[int],
    alpha_values: list[float],
    clip_values: list[float],
    weight_values: list[float],
    feature_preset_name: str,
    feature_weights: dict[str, float],
    oof_neighbor_mode: str,
):
    def loss_score(preds: np.ndarray, clip_min: float) -> float:
        clipped = normalize_probabilities(np.clip(preds, clip_min, 1.0 - clip_min))
        return float(log_loss(y_true, clipped, labels=range(10)))

    best = SearchResult(
        loss=float("inf"),
        feature_preset=feature_preset_name,
        feature_weights=feature_weights,
        oof_neighbor_mode=oof_neighbor_mode,
        k=-1,
        alpha=-1,
        clip_min=-1,
        w_self=-1,
        w_test=-1,
        w_train=-1,
    )
    best_oof = None
    best_sources = None

    train_train_idx, train_train_sim = neighbor_data["train_train"]
    train_test_idx, train_test_sim = neighbor_data["train_test"]

    for k in k_values:
        for alpha in alpha_values:
            p_test_neighbors = None
            if oof_neighbor_mode == "transductive":
                p_test_neighbors = aggregate_neighbors(test_self, train_test_idx, train_test_sim, k, alpha)
            p_train_neighbors = aggregate_neighbors(oof_self, train_train_idx, train_train_sim, k, alpha)

            for clip_min in clip_values:
                for w_self in weight_values:
                    test_weight_values = [0.0] if oof_neighbor_mode == "train_only" else weight_values
                    for w_test in test_weight_values:
                        for w_train in weight_values:
                            if w_self == 0 and w_test == 0 and w_train == 0:
                                continue
                            sources = [oof_self, p_train_neighbors]
                            weights = [w_self, w_train]
                            if p_test_neighbors is not None:
                                sources.insert(1, p_test_neighbors)
                                weights.insert(1, w_test)
                            preds = log_blend(sources, weights, clip_min)
                            loss = loss_score(preds, clip_min)
                            if loss < best.loss:
                                best = SearchResult(
                                    loss=float(loss),
                                    feature_preset=feature_preset_name,
                                    feature_weights=feature_weights,
                                    oof_neighbor_mode=oof_neighbor_mode,
                                    k=int(k),
                                    alpha=float(alpha),
                                    clip_min=float(clip_min),
                                    w_self=float(w_self),
                                    w_test=float(w_test),
                                    w_train=float(w_train),
                                )
                                best_oof = preds.astype(np.float32)
                                best_sources = (p_test_neighbors, p_train_neighbors)
    return best, best_oof, best_sources


def apply_best_to_test(test_self, oof_self, neighbor_data, result: SearchResult):
    test_test_idx, test_test_sim = neighbor_data["test_test"]
    test_train_idx, test_train_sim = neighbor_data["test_train"]
    p_train_neighbors = aggregate_neighbors(oof_self, test_train_idx, test_train_sim, result.k, result.alpha)
    if result.oof_neighbor_mode == "train_only":
        return log_blend(
            [test_self, p_train_neighbors],
            [result.w_self, result.w_train],
            result.clip_min,
        )

    p_test_neighbors = aggregate_neighbors(test_self, test_test_idx, test_test_sim, result.k, result.alpha)
    return log_blend(
        [test_self, p_test_neighbors, p_train_neighbors],
        [result.w_self, result.w_test, result.w_train],
        result.clip_min,
    )


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir)
    train_index = pd.read_csv(asset_dir / "train_index.csv")
    test_index = pd.read_csv(asset_dir / "test_index.csv")
    y_true = train_index["label_int"].to_numpy()

    oof_self, test_self, used_prob_models, prob_weights, prob_loss = load_probabilities(
        asset_dir,
        args.prob_models,
        expected_train=len(train_index),
        expected_test=len(test_index),
        y_true=y_true,
        args=args,
    )
    print("[INFO] Probability models:", dict(zip(used_prob_models, prob_weights.tolist())))
    print(f"[INFO] Base probability OOF log loss: {prob_loss:.6f}")

    feature_arrays = load_feature_arrays(
        asset_dir,
        args.feature_models,
        expected_train=len(train_index),
        expected_test=len(test_index),
    )
    loaded_feature_names = list(feature_arrays)
    max_k = args.max_k or max(parse_int_list(args.k_grid))
    max_k = min(max_k, len(train_index) - 1, len(test_index) - 1)
    if max_k < 1:
        raise ValueError("Need at least two train and two test samples for graph smoothing.")
    k_values = [k for k in parse_int_list(args.k_grid) if k <= max_k]
    if not k_values:
        k_values = [max_k]
    alpha_values = parse_float_list(args.alpha_grid)
    clip_values = parse_float_list(args.clip_grid)
    weight_values = parse_float_list(args.weight_grid)

    global_best = None
    global_best_oof = None
    global_best_neighbor_data = None

    presets = feature_presets(
        loaded_feature_names,
        args.feature_weights,
        search_presets=not args.no_feature_preset_search,
        dirichlet_count=args.feature_dirichlet_count,
        dirichlet_seed=args.feature_dirichlet_seed,
        dirichlet_alpha=args.feature_dirichlet_alpha,
    )
    for preset_name, weights in presets:
        print(f"[INFO] Searching feature preset: {preset_name} {weights}")
        train_features, test_features = combine_features(feature_arrays, weights)
        neighbor_data = knn_indices(train_features, test_features, max_k=max_k)
        result, best_oof, _sources = evaluate_grid(
            y_true=y_true,
            oof_self=oof_self,
            test_self=test_self,
            neighbor_data=neighbor_data,
            k_values=k_values,
            alpha_values=alpha_values,
            clip_values=clip_values,
            weight_values=weight_values,
            feature_preset_name=preset_name,
            feature_weights=weights,
            oof_neighbor_mode=args.oof_neighbor_mode,
        )
        print(f"[INFO] Best for {preset_name}: {asdict(result)}")
        if global_best is None or result.loss < global_best.loss:
            global_best = result
            global_best_oof = best_oof
            global_best_neighbor_data = neighbor_data

    assert global_best is not None and global_best_oof is not None and global_best_neighbor_data is not None
    test_preds = apply_best_to_test(test_self, oof_self, global_best_neighbor_data, global_best)

    oof_path = asset_dir / f"{args.output_prefix}_oof_preds.npy"
    test_path = asset_dir / f"{args.output_prefix}_test_preds.npy"
    np.save(oof_path, global_best_oof.astype(np.float32))
    np.save(test_path, test_preds.astype(np.float32))

    oof_csv = pd.DataFrame(global_best_oof, columns=CLASS_COLUMNS)
    oof_csv.insert(0, "img", train_index["img"].values)
    oof_csv["label_int"] = y_true
    oof_csv.to_csv(asset_dir / f"{args.output_prefix}_oof_preds.csv", index=False)

    test_csv = pd.DataFrame(test_preds, columns=CLASS_COLUMNS)
    test_csv.insert(0, "img", test_index["img"].values)
    test_csv.to_csv(asset_dir / f"{args.output_prefix}_test_preds.csv", index=False)

    config = asdict(global_best)
    config["prob_models"] = dict(zip(used_prob_models, prob_weights.tolist()))
    config["prob_oof_loss"] = prob_loss
    with open(asset_dir / f"{args.output_prefix}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Global best: {config}")
    print(f"[INFO] Saved {test_csv.shape[0]} test predictions to {asset_dir / f'{args.output_prefix}_test_preds.csv'}")


if __name__ == "__main__":
    main()
