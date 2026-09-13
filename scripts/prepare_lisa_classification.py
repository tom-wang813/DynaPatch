#!/usr/bin/env python3
"""Convert LISA traffic-sign annotations into cropped classification splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image


def _slugify(label: str) -> str:
    text = label.strip()
    # Normalize common camelCase labels from augmented YOLO variants.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _discover_annotation_csvs(raw_root: Path) -> list[Path]:
    return sorted(
        p for p in raw_root.rglob("*.csv")
        if "annotation" in p.name.lower() or "annotations" in p.name.lower()
    )


def _find_image(raw_root: Path, filename: str) -> Path | None:
    matches = list(raw_root.rglob(filename))
    if matches:
        return matches[0]
    stem = Path(filename).stem
    candidates = list(raw_root.rglob(f"{stem}.*"))
    return candidates[0] if candidates else None


def _parse_row(row: dict) -> tuple[str, tuple[int, int, int, int]] | None:
    keys = {k.lower().strip(): v for k, v in row.items()}
    label = (
        keys.get("annotation tag")
        or keys.get("annotation_tag")
        or keys.get("label")
        or keys.get("sign type")
        or keys.get("signtype")
    )
    if not label:
        return None

    def _get(*names: str) -> int | None:
        for name in names:
            value = keys.get(name)
            if value not in (None, ""):
                try:
                    return int(float(value))
                except ValueError:
                    return None
        return None

    x1 = _get("upper left corner x", "ulx", "xmin", "x1")
    y1 = _get("upper left corner y", "uly", "ymin", "y1")
    x2 = _get("lower right corner x", "lrx", "xmax", "x2")
    y2 = _get("lower right corner y", "lry", "ymax", "y2")
    if None in (x1, y1, x2, y2):
        return None
    return _slugify(str(label)), (x1, y1, x2, y2)


def _split_of(path: Path) -> str:
    lowered = str(path).lower()
    if "train" in lowered:
        return "train"
    if "test" in lowered or "valid" in lowered or "val" in lowered:
        return "test"
    # Fallback: treat unknown folders conservatively as train.
    return "train"


def _discover_yolo_dataset_root(raw_root: Path) -> Path | None:
    candidates = [raw_root]
    candidates.extend(p for p in raw_root.rglob("*") if p.is_dir())
    for candidate in candidates:
        if (candidate / "data.yaml").exists():
            return candidate
    return None


def _load_yolo_names(dataset_root: Path) -> list[str]:
    data_yaml = (dataset_root / "data.yaml").read_text(encoding="utf-8")
    match = re.search(r"names:\s*\[(.*)\]", data_yaml, flags=re.S)
    if not match:
        raise ValueError(f"Unable to parse names list from {dataset_root / 'data.yaml'}")
    content = match.group(1)
    names = re.findall(r"'([^']+)'|\"([^\"]+)\"", content)
    flattened = [a or b for a, b in names]
    if not flattened:
        raise ValueError(f"No class names found in {dataset_root / 'data.yaml'}")
    return [_slugify(name) for name in flattened]


def _infer_label_from_image_name(image_path: Path, known_labels: set[str]) -> str | None:
    stem = image_path.stem
    stem = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem)
    match = re.match(r"^(.*?)(?:_[0-9].*|_png.*|_jpg.*)$", stem)
    candidate = match.group(1) if match else stem
    label = _slugify(candidate)
    return label if label in known_labels else None


def _prepare_from_yolo(raw_root: Path, out_root: Path, min_size: int) -> dict:
    dataset_root = _discover_yolo_dataset_root(raw_root)
    if dataset_root is None:
        raise FileNotFoundError(f"No YOLO-style LISA dataset found under {raw_root}")

    names = _load_yolo_names(dataset_root)
    known_labels = set(names)
    split_counter: dict[str, Counter] = {"train": Counter(), "test": Counter()}
    total_crops = 0
    missing_labels = 0
    missing_images = 0

    for source_split in ["train", "valid", "test"]:
        images_dir = dataset_root / source_split / "images"
        labels_dir = dataset_root / source_split / "labels"
        if not images_dir.exists():
            continue
        out_split = "train" if source_split == "train" else "test"
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            if label_path.exists():
                lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not lines:
                    continue
                for idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    if cls_id < 0 or cls_id >= len(names):
                        continue
                    xc, yc, w, h = map(float, parts[1:])
                    box_w = max(1, int(round(w * width)))
                    box_h = max(1, int(round(h * height)))
                    center_x = xc * width
                    center_y = yc * height
                    x1 = max(0, int(round(center_x - box_w / 2)))
                    y1 = max(0, int(round(center_y - box_h / 2)))
                    x2 = min(width, max(x1 + 1, int(round(center_x + box_w / 2))))
                    y2 = min(height, max(y1 + 1, int(round(center_y + box_h / 2))))
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        continue
                    label = names[cls_id]
                    class_dir = out_root / out_split / label
                    class_dir.mkdir(parents=True, exist_ok=True)
                    crop = image.crop((x1, y1, x2, y2))
                    crop.save(class_dir / f"{image_path.stem}_{idx}.png")
                    split_counter[out_split][label] += 1
                    total_crops += 1
            else:
                inferred_label = _infer_label_from_image_name(image_path, known_labels)
                if inferred_label is None:
                    missing_labels += 1
                    continue
                class_dir = out_root / out_split / inferred_label
                class_dir.mkdir(parents=True, exist_ok=True)
                image.save(class_dir / f"{image_path.stem}.png")
                split_counter[out_split][inferred_label] += 1
                total_crops += 1

    all_classes = sorted(set(split_counter["train"]) | set(split_counter["test"]) | set(names))
    for split in ["train", "test"]:
        for label in all_classes:
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    return {
        "status": "completed",
        "raw_root": str(raw_root),
        "dataset_format": "yolo_detection",
        "output_root": str(out_root),
        "total_crops": total_crops,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "train_counts": dict(sorted(split_counter["train"].items())),
        "test_counts": dict(sorted(split_counter["test"].items())),
        "num_classes": len(all_classes),
        "classes": all_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/lisa_raw")
    parser.add_argument("--output-root", default="data/lisa_signs_clf")
    parser.add_argument("--min-size", type=int, default=8)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    yolo_root = _discover_yolo_dataset_root(raw_root)
    if yolo_root is not None:
        summary = _prepare_from_yolo(raw_root, out_root, args.min_size)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    annotation_files = _discover_annotation_csvs(raw_root)
    if not annotation_files:
        raise FileNotFoundError(
            f"No LISA annotation CSV files found under {raw_root}. "
            "Place the official dataset archive contents under data/lisa_raw first."
        )

    split_counter: dict[str, Counter] = {"train": Counter(), "test": Counter()}
    total_crops = 0
    missing_images = 0

    for csv_path in annotation_files:
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                image_name = (
                    row.get("Filename")
                    or row.get("filename")
                    or row.get("file")
                    or row.get("image")
                )
                if not image_name:
                    continue
                parsed = _parse_row(row)
                if parsed is None:
                    continue
                label, (x1, y1, x2, y2) = parsed
                image_path = _find_image(raw_root, image_name)
                if image_path is None or not image_path.exists():
                    missing_images += 1
                    continue

                split = _split_of(image_path)
                image = Image.open(image_path).convert("RGB")
                width, height = image.size

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                    continue

                class_dir = out_root / split / label
                class_dir.mkdir(parents=True, exist_ok=True)
                crop = image.crop((x1, y1, x2, y2))
                crop.save(class_dir / f"{image_path.stem}_{idx}.png")
                split_counter[split][label] += 1
                total_crops += 1

    all_classes = sorted(set(split_counter["train"]) | set(split_counter["test"]))
    for split in ["train", "test"]:
        for label in all_classes:
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "completed",
        "raw_root": str(raw_root),
        "output_root": str(out_root),
        "annotation_files": [str(p) for p in annotation_files],
        "total_crops": total_crops,
        "missing_images": missing_images,
        "train_counts": dict(sorted(split_counter["train"].items())),
        "test_counts": dict(sorted(split_counter["test"].items())),
        "num_classes": len(all_classes),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
