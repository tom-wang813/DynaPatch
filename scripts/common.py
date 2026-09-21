#!/usr/bin/env python3
"""Shared helpers for package-facing reproduction entrypoints."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT
MANIFEST_PATH = PACKAGE_ROOT / "artifacts" / "checkpoints" / "manifest.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_backbone_path(dataset: str, backbone: str) -> str:
    payload = load_manifest()
    for entry in payload["frozen_backbones"]:
        if entry["dataset"] == dataset and entry["backbone"] == backbone:
            return str(entry["checkpoint_path"])
    raise ValueError(f"Missing frozen backbone manifest entry for {dataset}/{backbone}")


def repair_checkpoint_candidates(rel_path: str) -> list[str]:
    path = Path(rel_path)
    candidates = [str(path)]
    parts = path.parts
    if "checkpoints" in parts and "train" not in parts:
        checkpoint_index = parts.index("checkpoints")
        train_variant = Path(*parts[:checkpoint_index], "train", *parts[checkpoint_index:])
        candidates.append(str(train_variant))
    return candidates


def resolve_existing_path(*rel_paths: str) -> str | None:
    for rel_path in rel_paths:
        for candidate in repair_checkpoint_candidates(rel_path):
            if (ROOT / candidate).exists():
                return candidate
    return None


def print_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))


def run_command(cmd: list[str], *, dry_run: bool = False) -> int:
    print_command(cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def python_entrypoint(script_relative_path: str, *args: str) -> list[str]:
    return [sys.executable, str(PACKAGE_ROOT / "scripts" / script_relative_path), *args]
