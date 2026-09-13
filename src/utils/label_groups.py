"""Utilities for dataset-specific label-group mappings used by safety-aware models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _expand_rule_names(classes: list[str], rule: dict | None) -> list[str]:
    if not rule:
        return []
    matched: set[str] = set()
    for name in rule.get("exact", []):
        if name in classes:
            matched.add(name)
    for prefix in rule.get("prefixes", []):
        matched.update(name for name in classes if name.startswith(prefix))
    return sorted(matched)


def _tt100k_classes(summary_path: Path) -> list[str]:
    payload = _load_json(summary_path)
    train = set(payload.get("train_counts", {}))
    test = set(payload.get("test_counts", {}))
    classes = sorted(train | test)
    if not classes:
        raise ValueError(f"No TT100K classes found in {summary_path}")
    return classes


def _lisa_classes(summary_path: Path) -> list[str]:
    payload = _load_json(summary_path)
    classes = list(payload.get("classes", []))
    if not classes:
        raise ValueError(f"No LISA classes found in {summary_path}")
    return classes


def _gtsrb_groups(cfg: dict, num_classes: int) -> tuple[list[str], list[int]]:
    group_names = list(cfg["groups"].keys()) + ["other"]
    other_idx = len(group_names) - 1
    class_to_group = [other_idx for _ in range(num_classes)]
    for group_idx, group_name in enumerate(cfg["groups"].keys()):
        for class_idx in cfg["groups"][group_name]:
            class_to_group[int(class_idx)] = group_idx
    return group_names, class_to_group


def _rule_based_groups(classes: list[str], cfg_groups: dict[str, dict]) -> tuple[list[str], list[int]]:
    index_of = {name: idx for idx, name in enumerate(classes)}
    group_names = list(cfg_groups.keys()) + ["other"]
    other_idx = len(group_names) - 1
    class_to_group = [other_idx for _ in classes]
    for group_idx, group_name in enumerate(cfg_groups.keys()):
        matched = _expand_rule_names(classes, cfg_groups[group_name])
        for name in matched:
            class_to_group[index_of[name]] = group_idx
    return group_names, class_to_group


def build_class_group_matrix_for_dataset(dataset_name: str) -> tuple[torch.Tensor, list[str], list[int]]:
    """Return a one-hot class->group matrix for a traffic-sign dataset.

    The returned tensor has shape `[num_classes, num_groups]`.
    """
    dataset = str(dataset_name)
    if dataset == "gtsrb":
        cfg = _load_yaml(ROOT / "configs" / "risk" / "gtsrb_safety.yaml")
        num_classes = int(cfg["num_classes"])
        group_names, class_to_group = _gtsrb_groups(cfg, num_classes)
    elif dataset == "tt100k_signs":
        cfg = _load_yaml(ROOT / "configs" / "risk" / "tt100k_signs_safety.yaml")
        classes = _tt100k_classes(ROOT / cfg["summary_path"])
        group_names, class_to_group = _rule_based_groups(classes, cfg["groups"])
        num_classes = len(classes)
    elif dataset == "lisa_signs":
        cfg = _load_yaml(ROOT / "configs" / "risk" / "lisa_signs_safety.yaml")
        classes = _lisa_classes(ROOT / cfg["summary_path"])
        group_names, class_to_group = _rule_based_groups(classes, cfg["groups"])
        num_classes = len(classes)
    else:
        raise ValueError(f"Unsupported dataset for group-aware ViT LoRA: {dataset_name}")

    num_groups = len(group_names)
    matrix = torch.zeros(num_classes, num_groups, dtype=torch.float32)  # [classes, groups]
    for class_idx, group_idx in enumerate(class_to_group):
        matrix[class_idx, int(group_idx)] = 1.0
    return matrix, group_names, class_to_group
