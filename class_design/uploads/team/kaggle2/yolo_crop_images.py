from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO person crops and crop quality flags for train/test images.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--allow-yolo-download", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-size", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--driver-csv", default="dataset/driver_imgs_list.csv")
    parser.add_argument("--sample-submission", default="dataset/sample_submission.csv")
    parser.add_argument("--train-source-dir", default="dataset/imgs/train")
    parser.add_argument("--test-source-dir", default="dataset/imgs/test")
    parser.add_argument("--train-target-dir", default="dataset/imgs/train_cropped_v2")
    parser.add_argument("--test-target-dir", default="dataset/imgs/test_cropped_v2")
    parser.add_argument("--train-flags", default="crop_flags_train.csv")
    parser.add_argument("--test-flags", default="crop_flags_test.csv")

    parser.add_argument("--pad-x", type=float, default=0.15)
    parser.add_argument("--pad-y-top", type=float, default=0.05)
    parser.add_argument("--person-height-ratio", type=float, default=0.65)
    parser.add_argument("--weak-min-area-ratio", type=float, default=0.06)
    parser.add_argument("--weak-max-area-ratio", type=float, default=0.95)
    parser.add_argument("--weak-min-aspect", type=float, default=0.35)
    parser.add_argument("--weak-max-aspect", type=float, default=3.00)
    return parser.parse_args()


def load_yolo_model(model_path: str, allow_download: bool):
    path = Path(model_path)
    if path.suffix and not path.exists() and not allow_download:
        raise FileNotFoundError(
            f"Missing YOLO weights: {model_path}. Put the file in place or pass --allow-yolo-download explicitly."
        )
    from ultralytics import YOLO

    return YOLO(model_path)


def train_rows(driver_csv: str | Path, source_root: Path) -> pd.DataFrame:
    df = pd.read_csv(driver_csv).reset_index(drop=True)
    if "label_int" not in df.columns:
        df["label_int"] = df["classname"].str.extract(r"(\d+)").astype(int)
    df["source_path"] = df.apply(lambda row: str(source_root / str(row["classname"]) / str(row["img"])), axis=1)
    df["target_rel"] = df.apply(lambda row: str(Path(str(row["classname"])) / str(row["img"])), axis=1)
    return df


def test_rows(sample_submission: str | Path, source_root: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_submission).reset_index(drop=True)
    df["classname"] = ""
    df["label_int"] = -1
    df["source_path"] = df["img"].astype(str).map(lambda img: str(source_root / img))
    df["target_rel"] = df["img"].astype(str)
    return df


def choose_person_box(result) -> tuple[np.ndarray | None, float]:
    if len(result.boxes) == 0:
        return None, 0.0
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    best_idx = int(np.argmax(areas))
    return boxes[best_idx], float(confs[best_idx])


def crop_from_box(
    image: np.ndarray,
    box: np.ndarray,
    pad_x_ratio: float,
    pad_y_top_ratio: float,
    person_height_ratio: float,
) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    person_h = max(1, y2 - y1)
    pad_x = int(max(0, x2 - x1) * pad_x_ratio)
    pad_y_top = int(person_h * pad_y_top_ratio)
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1 - pad_y_top)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y1 + int(person_h * person_height_ratio))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        crop_x1, crop_y1, crop_x2, crop_y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return image, {
            "crop_failed": True,
            "crop_touch_edge": True,
            "crop_area_ratio": 1.0,
            "crop_aspect_ratio": w / max(1.0, float(h)),
        }

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    area_ratio = ((crop_x2 - crop_x1) * (crop_y2 - crop_y1)) / max(1.0, float(w * h))
    aspect_ratio = (crop_x2 - crop_x1) / max(1.0, float(crop_y2 - crop_y1))
    touch_edge = crop_x1 <= 0 or crop_y1 <= 0 or crop_x2 >= w or crop_y2 >= h
    return crop, {
        "crop_failed": False,
        "crop_touch_edge": bool(touch_edge),
        "crop_area_ratio": float(area_ratio),
        "crop_aspect_ratio": float(aspect_ratio),
    }


def process_split(
    model,
    rows: pd.DataFrame,
    target_root: Path,
    flags_path: Path,
    args: argparse.Namespace,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"crop {flags_path.stem}"):
        img_name = str(row["img"])
        class_name = str(row.get("classname", ""))
        source_path = Path(str(row["source_path"]))
        target_path = target_root / str(row["target_rel"])
        target_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(source_path))
        if image is None:
            records.append(
                {
                    "img": img_name,
                    "classname": class_name,
                    "source_path": str(source_path),
                    "crop_path": str(target_path),
                    "detected": 0,
                    "bbox_conf": 0.0,
                    "bbox_area_ratio": 0.0,
                    "crop_area_ratio": 0.0,
                    "crop_aspect_ratio": 0.0,
                    "crop_touch_edge": False,
                    "crop_ok": False,
                    "yolo_fail": True,
                    "crop_failed": True,
                    "crop_weak": True,
                    "fallback_full_image": False,
                }
            )
            continue

        h, w = image.shape[:2]
        detected = 0
        bbox_conf = 0.0
        bbox_area_ratio = 0.0
        fallback_full = False
        crop_ok = False
        result = model.predict(
            image,
            classes=[0],
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            device=args.device,
        )[0]
        box, bbox_conf = choose_person_box(result)
        if box is None:
            crop = image
            fallback_full = True
            crop_meta = {
                "crop_failed": False,
                "crop_touch_edge": True,
                "crop_area_ratio": 1.0,
                "crop_aspect_ratio": w / max(1.0, float(h)),
            }
        else:
            detected = int(len(result.boxes))
            bbox_area_ratio = (
                max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])) / max(1.0, float(w * h))
            )
            crop, crop_meta = crop_from_box(
                image,
                box,
                pad_x_ratio=args.pad_x,
                pad_y_top_ratio=args.pad_y_top,
                person_height_ratio=args.person_height_ratio,
            )
            crop_ok = not crop_meta["crop_failed"]

        if args.overwrite or not target_path.exists():
            final = cv2.resize(crop, (args.output_size, args.output_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(target_path), final)

        crop_weak = (
            crop_meta["crop_area_ratio"] < args.weak_min_area_ratio
            or crop_meta["crop_area_ratio"] > args.weak_max_area_ratio
            or crop_meta["crop_aspect_ratio"] < args.weak_min_aspect
            or crop_meta["crop_aspect_ratio"] > args.weak_max_aspect
            or bool(crop_meta["crop_touch_edge"])
        )
        yolo_fail = detected <= 0
        records.append(
            {
                "img": img_name,
                "classname": class_name,
                "source_path": str(source_path),
                "crop_path": str(target_path),
                "detected": detected,
                "bbox_conf": float(bbox_conf),
                "bbox_area_ratio": float(bbox_area_ratio),
                "crop_area_ratio": float(crop_meta["crop_area_ratio"]),
                "crop_aspect_ratio": float(crop_meta["crop_aspect_ratio"]),
                "crop_touch_edge": bool(crop_meta["crop_touch_edge"]),
                "crop_ok": bool(crop_ok and not yolo_fail),
                "yolo_fail": bool(yolo_fail),
                "crop_failed": bool(crop_meta["crop_failed"]),
                "crop_weak": bool(crop_weak),
                "fallback_full_image": bool(fallback_full),
            }
        )

    flags_df = pd.DataFrame(records)
    flags_df.to_csv(flags_path, index=False)
    print(f"[INFO] Saved crop flags: {flags_path}")
    if len(flags_df):
        print(flags_df[["yolo_fail", "crop_failed", "crop_weak"]].mean().to_string())


def main() -> None:
    args = parse_args()
    model = load_yolo_model(args.model, args.allow_yolo_download)

    if args.split in {"train", "both"}:
        source = Path(args.train_source_dir)
        target = Path(args.train_target_dir)
        rows = train_rows(args.driver_csv, source)
        process_split(model, rows, target, Path(args.train_flags), args)

    if args.split in {"test", "both"}:
        source = Path(args.test_source_dir)
        target = Path(args.test_target_dir)
        rows = test_rows(args.sample_submission, source)
        process_split(model, rows, target, Path(args.test_flags), args)


if __name__ == "__main__":
    main()
