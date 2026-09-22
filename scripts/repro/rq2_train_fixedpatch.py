#!/usr/bin/env python3
"""Thin CLI driver for the FixedPatch (FPa) ablation training -- the actual config resolution
lives in src/experiment/stage3.py::build_fixed_patch_config (same convention every other
stage3_repair entrypoint in this repo follows: src/ holds the logic, scripts/ just wires a CLI to
it). See that function's docstring for why the split/artifact paths need overriding at all, and
scripts/checkpoint_eval/fixedpatch.py for scoring the resulting checkpoint.

Usage:
  uv run python scripts/repro/rq2_train_fixedpatch.py --dataset gtsrb --backbone resnet50
  uv run python scripts/repro/rq2_train_fixedpatch.py --dataset gtsrb --backbone resnet50 \
      --override train_loop.epochs=40 --override train_loop.early_stop_patience=12
  uv run python scripts/repro/rq2_train_fixedpatch.py --all
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiment.stage3 import build_fixed_patch_config  # noqa: E402
from src.experiment.train_stage3 import run_stage3_experiment  # noqa: E402

SEED = 101
ALL_SETTINGS = [
    (ds, bb)
    for ds in ("gtsrb", "tt100k_signs", "lisa_signs")
    for bb in ("resnet50", "convnext_tiny", "densenet121", "vgg16")
]


def _parse_value(raw: str) -> object:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def parse_overrides(pairs: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for p in pairs:
        key, _, value = p.partition("=")
        out[key] = _parse_value(value)
    return out


def train_one(dataset: str, backbone: str, overrides: dict[str, object]) -> Path:
    cfg = build_fixed_patch_config(dataset, backbone, SEED, overrides=overrides)
    art_root = Path(str(cfg.artifacts.root))
    t0 = time.time()
    run_stage3_experiment(cfg)
    elapsed = time.time() - t0
    print(f"[FixedPatch] {dataset}/{backbone} s{SEED}: trained in {elapsed:.1f}s")

    src_ckpt = art_root / "checkpoints" / "repair_best.pt"
    if not src_ckpt.exists():
        raise SystemExit(f"[FixedPatch] expected checkpoint not found: {src_ckpt}")

    dst_dir = ROOT / f"artifacts/checkpoints/baselines/FixedPatch/{dataset}_{backbone}_s{SEED}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_ckpt = dst_dir / "repair_best.pt"
    shutil.copy2(src_ckpt, dst_ckpt)
    print(f"[FixedPatch] saved -> {dst_ckpt}")
    return dst_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--backbone")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--override", action="append", default=[],
                     help="dot.path=value, repeatable, e.g. --override train_loop.epochs=40")
    a = ap.parse_args()

    if a.all:
        settings = ALL_SETTINGS
    elif a.dataset and a.backbone:
        settings = [(a.dataset, a.backbone)]
    else:
        raise SystemExit("pass --dataset/--backbone for one setting, or --all for all 12")

    overrides = parse_overrides(a.override)
    times = {}
    for dataset, backbone in settings:
        t0 = time.time()
        train_one(dataset, backbone, overrides)
        times[f"{dataset}/{backbone}"] = time.time() - t0
    print("\n[FixedPatch] timing summary:")
    for k, v in times.items():
        print(f"  {k}: {v:.1f}s")


if __name__ == "__main__":
    main()
