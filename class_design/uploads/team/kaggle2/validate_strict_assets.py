from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pseudo_common import CLASS_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate strict asset shapes and write assets_manifest.json.")
    parser.add_argument("--asset-dir", default="models/strict_assets")
    parser.add_argument("--train-index", default="train_index.csv")
    parser.add_argument("--test-index", default="test_index.csv")
    parser.add_argument("--output", default="assets_manifest.json")
    parser.add_argument("--fail-on-issues", action="store_true")
    return parser.parse_args()


def asset_name_from(path: Path, prefix: str) -> str:
    name = path.stem
    return name[len(prefix) :]


def safe_shape(path: Path) -> list[int] | None:
    try:
        arr = np.load(path, mmap_mode="r")
        return list(arr.shape)
    except Exception:
        return None


def row_sum_stats(path: Path) -> dict | None:
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != len(CLASS_COLUMNS):
            return None
        row_sums = np.asarray(arr.sum(axis=1))
        return {
            "min": float(np.min(row_sums)),
            "max": float(np.max(row_sums)),
            "mean": float(np.mean(row_sums)),
        }
    except Exception:
        return None


def load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir)
    train_index_path = asset_dir / args.train_index
    test_index_path = asset_dir / args.test_index
    issues: list[str] = []

    if not train_index_path.exists():
        issues.append(f"Missing train index: {train_index_path}")
        n_train = None
    else:
        n_train = len(pd.read_csv(train_index_path))

    if not test_index_path.exists():
        issues.append(f"Missing test index: {test_index_path}")
        n_test = None
    else:
        n_test = len(pd.read_csv(test_index_path))

    names: set[str] = set()
    for pattern, prefix in [
        ("oof_preds_*.npy", "oof_preds_"),
        ("test_preds_*.npy", "test_preds_"),
        ("train_features_*.npy", "train_features_"),
        ("test_features_*.npy", "test_features_"),
        ("asset_meta_*.json", "asset_meta_"),
    ]:
        for path in asset_dir.glob(pattern):
            names.add(asset_name_from(path, prefix))

    assets: dict[str, dict] = {}
    for name in sorted(names):
        entry = {
            "name": name,
            "files": {},
            "has_probabilities": False,
            "has_features": False,
            "issues": [],
        }
        paths = {
            "oof_preds": asset_dir / f"oof_preds_{name}.npy",
            "test_preds": asset_dir / f"test_preds_{name}.npy",
            "train_features": asset_dir / f"train_features_{name}.npy",
            "test_features": asset_dir / f"test_features_{name}.npy",
            "meta": asset_dir / f"asset_meta_{name}.json",
        }
        for key, path in paths.items():
            if not path.exists():
                continue
            if key == "meta":
                entry["files"][key] = {"path": str(path), "meta": load_meta(path)}
                continue
            shape = safe_shape(path)
            entry["files"][key] = {"path": str(path), "shape": shape}

        if "oof_preds" in entry["files"] or "test_preds" in entry["files"]:
            entry["has_probabilities"] = True
        if "train_features" in entry["files"] or "test_features" in entry["files"]:
            entry["has_features"] = True

        if "oof_preds" in entry["files"]:
            shape = entry["files"]["oof_preds"]["shape"]
            if shape != [n_train, len(CLASS_COLUMNS)]:
                entry["issues"].append(f"oof_preds shape {shape} != [{n_train}, {len(CLASS_COLUMNS)}]")
            entry["files"]["oof_preds"]["row_sum"] = row_sum_stats(paths["oof_preds"])
        if "test_preds" in entry["files"]:
            shape = entry["files"]["test_preds"]["shape"]
            if shape != [n_test, len(CLASS_COLUMNS)]:
                entry["issues"].append(f"test_preds shape {shape} != [{n_test}, {len(CLASS_COLUMNS)}]")
            entry["files"]["test_preds"]["row_sum"] = row_sum_stats(paths["test_preds"])
        if "train_features" in entry["files"]:
            shape = entry["files"]["train_features"]["shape"]
            if not shape or shape[0] != n_train:
                entry["issues"].append(f"train_features first dim {shape} != {n_train}")
        if "test_features" in entry["files"]:
            shape = entry["files"]["test_features"]["shape"]
            if not shape or shape[0] != n_test:
                entry["issues"].append(f"test_features first dim {shape} != {n_test}")

        if entry["issues"]:
            issues.extend([f"{name}: {issue}" for issue in entry["issues"]])
        assets[name] = entry

    manifest = {
        "asset_dir": str(asset_dir),
        "n_train": n_train,
        "n_test": n_test,
        "class_columns": CLASS_COLUMNS,
        "assets": assets,
        "issues": issues,
    }

    output_path = asset_dir / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Wrote manifest: {output_path}")
    print(f"[INFO] Assets: {len(assets)} | issues: {len(issues)}")
    if issues:
        for issue in issues[:20]:
            print(f"[ISSUE] {issue}")
        if len(issues) > 20:
            print(f"[ISSUE] ... {len(issues) - 20} more")
        if args.fail_on_issues:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
