#!/usr/bin/env python3
"""Convert TT100K detection annotations into cropped classification splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def _load_annotations(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected TT100K annotations.json to be a JSON object.")
    return payload


def _resolve_images_root(root: Path) -> Path:
    for candidate in [root / "data", root]:
        if (candidate / "train").exists() or (candidate / "test").exists():
            return candidate
    raise FileNotFoundError(f"Could not find TT100K image root under {root}")


def _iter_objects(img_info: dict) -> list[dict]:
    for key in ["objects", "objs"]:
        objs = img_info.get(key)
        if isinstance(objs, list):
            return objs
    return []


def _obj_label(obj: dict) -> str | None:
    for key in ["category", "type", "name"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _obj_bbox(obj: dict) -> tuple[int, int, int, int] | None:
    bbox = obj.get("bbox") or obj.get("BBox") or obj.get("bndbox")
    if isinstance(bbox, dict):
        keys = {k.lower(): v for k, v in bbox.items()}
        xmin = keys.get("xmin", keys.get("x1", keys.get("left")))
        ymin = keys.get("ymin", keys.get("y1", keys.get("top")))
        xmax = keys.get("xmax", keys.get("x2", keys.get("right")))
        ymax = keys.get("ymax", keys.get("y2", keys.get("bottom")))
        if None not in (xmin, ymin, xmax, ymax):
            return int(xmin), int(ymin), int(xmax), int(ymax)
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2), int(y2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/tt100k_raw")
    parser.add_argument("--output-root", default="data/tt100k_signs_clf")
    parser.add_argument("--min-size", type=int, default=12)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images_root = _resolve_images_root(raw_root)
    annotations_path = images_root / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"TT100K annotations not found: {annotations_path}")

    annotations = _load_annotations(annotations_path)
    imgs = annotations.get("imgs", annotations)
    if not isinstance(imgs, dict):
        raise ValueError("TT100K annotations missing `imgs` mapping.")

    split_counter: dict[str, Counter] = {"train": Counter(), "test": Counter()}
    total_crops = 0

    for image_id, img_info in imgs.items():
        if not isinstance(img_info, dict):
            continue
        path_value = img_info.get("path") or img_info.get("img_path") or f"{image_id}.jpg"
        rel_path = Path(path_value)
        split = "train" if rel_path.parts and rel_path.parts[0] == "train" else "test"
        image_path = images_root / rel_path
        if not image_path.exists():
            continue
        objects = _iter_objects(img_info)
        if not objects:
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        stem = image_path.stem
        for index, obj in enumerate(objects):
            label = _obj_label(obj)
            bbox = _obj_bbox(obj)
            if label is None or bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                continue

            class_dir = out_root / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            crop = image.crop((x1, y1, x2, y2))
            crop.save(class_dir / f"{stem}_{index}.png")
            split_counter[split][label] += 1
            total_crops += 1

    summary = {
        "status": "completed",
        "raw_root": str(raw_root),
        "images_root": str(images_root),
        "output_root": str(out_root),
        "total_crops": total_crops,
        "train_counts": dict(sorted(split_counter["train"].items())),
        "test_counts": dict(sorted(split_counter["test"].items())),
        "num_classes": len(set(split_counter["train"]) | set(split_counter["test"])),
    }

    # Keep train/test class_to_idx stable for ImageFolder by materializing
    # the union of class directories under both splits, even when one split
    # has zero samples for a class.
    all_classes = sorted(set(split_counter["train"]) | set(split_counter["test"]))
    for split in ["train", "test"]:
        for label in all_classes:
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
