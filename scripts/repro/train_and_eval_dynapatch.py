#!/usr/bin/env python3
"""End-to-end "reviewer starts from nothing but data + frozen backbone" demo: train DynaPatch's
hypernetwork from scratch (src/experiment/train_stage3.py::run_stage3_experiment, same as
scripts/run_resolved_experiment.py's stage3_repair path -- method.name is left at train.yaml's
own default, hypernet_only, unlike scripts/repro/rq2_train_fixedpatch.py which overrides it to
fixed_patch), then dump gate-evidence, fit a 9-feature DPGate on it (reusing
scripts/repro/rq3_fit_eval_gate.py's fitting/evaluation code, not a second implementation), and
report ungated vs gated RR/Reg/CReg plus the gate's own classification performance.

Only seeds 101/202/303 are supported -- those are the only ones with full split manifests
(artifacts/bug_sets/shuffled_split_seed{101,202,303}/, including bug_val and clean_calib, which
gate fitting needs). Seed 101 already has shipped checkpoints/gates (see
artifacts/checkpoints/manifest.json); 202/303 currently ship no repair checkpoint or gate at all
in this checkout, so running this at --seed 202 or 303 is a genuine from-scratch reproduction, not
a re-score of something already on disk.

The trained checkpoint IS copied to the canonical artifacts/checkpoints/dynapatch/<ds>_<bb>_s<seed>/
repair_best.pt location (not a scratch path) -- that's the real, correct place for a DynaPatch
repair checkpoint at a supported seed, and scripts/checkpoint_eval/dynapatch.py and
scripts/repro/rq3_fit_eval_gate.py both already expect to find it there.

Usage:
  uv run python scripts/repro/train_and_eval_dynapatch.py --dataset gtsrb --backbone resnet50 --seed 202
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/checkpoint_eval"))
sys.path.insert(0, str(ROOT / "scripts/repro"))

from omegaconf import OmegaConf  # noqa: E402
from src.experiment.train_stage3 import run_stage3_experiment  # noqa: E402
from src.experiment.deploy_eval import run_deploy_eval  # noqa: E402
import common  # noqa: E402
import dynapatch as dynapatch_ckpt_eval  # noqa: E402
import rq3_fit_eval_gate as gate_eval  # noqa: E402

SUPPORTED_SEEDS = (101, 202, 303)


def train_cfg(dataset: str, backbone: str, seed: int, art_root: Path,
              overrides: dict[str, object] | None = None) -> OmegaConf:
    cfg = OmegaConf.load(ROOT / f"configs/shuffled_split_source/{dataset}/{backbone}/train.yaml")
    sdir = common.split_dir(dataset, backbone, seed)
    OmegaConf.update(cfg, "artifacts.root", str(art_root), merge=True)
    OmegaConf.update(cfg, "experiment.id", f"dynapatch_fromscratch_{dataset}_{backbone}_s{seed}", merge=True)
    OmegaConf.update(cfg, "experiment.name", f"dynapatch_fromscratch_{dataset}_{backbone}_s{seed}", merge=True)
    OmegaConf.update(cfg, "data.bug_indices_path", str(sdir / f"{dataset}_bug_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.bug_train_indices_path", str(sdir / f"{dataset}_bug_train_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.bug_val_indices_path", str(sdir / f"{dataset}_bug_val_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.bug_eval_indices_path", str(sdir / f"{dataset}_bug_eval_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.clean_eval_indices_path", str(sdir / f"{dataset}_clean_eval_indices.json"), merge=True)
    OmegaConf.update(cfg, "seed", seed, merge=True)
    for dotpath, value in (overrides or {}).items():
        OmegaConf.update(cfg, dotpath, value, merge=True)
    return cfg


def train(dataset: str, backbone: str, seed: int, overrides: dict[str, object] | None = None,
          force: bool = False) -> Path:
    art_root = ROOT / f"outputs/repro/dynapatch_fromscratch_train/{dataset}_{backbone}_s{seed}"
    dst = ROOT / f"artifacts/checkpoints/dynapatch/{dataset}_{backbone}_s{seed}/repair_best.pt"
    if dst.exists() and not force:
        print(f"[train] checkpoint already at {dst}, skipping training")
        return dst

    cfg = train_cfg(dataset, backbone, seed, art_root, overrides)
    t0 = time.time()
    run_stage3_experiment(cfg)
    print(f"[train] {dataset}/{backbone} s{seed}: trained in {time.time() - t0:.1f}s")

    src_ckpt = art_root / "checkpoints" / "repair_best.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dst)
    print(f"[train] saved -> {dst}")
    return dst


def dump_gate_evidence(dataset: str, backbone: str, seed: int, ckpt: Path) -> None:
    """Same deploy_direct / deploy_direct_calib dump pass scripts/checkpoint_eval/dynapatch.py's
    --gate flow runs, reused directly rather than reimplemented."""
    gate_root = ROOT / f"outputs/effect_dump_v8_s{seed}/{dataset}/{backbone}"
    for tag, held_suffix in (("deploy_direct", "bug_eval_indices"), ("deploy_direct_calib", "bug_val_indices")):
        out = gate_root / tag
        if (out / "predictions" / "clean_eval_predictions.csv").exists():
            print(f"[dump] skip done ({tag}) -> {out}")
            continue
        print(f"[dump] gate-evidence dump ({tag}) -> {out}")
        cfg = dynapatch_ckpt_eval.resolved_cfg(dataset, backbone, seed, out, ckpt, "cuda:0",
                                                held_suffix=held_suffix, save_route_features=True)
        run_deploy_eval(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gtsrb", "tt100k_signs", "lisa_signs"])
    ap.add_argument("--backbone", required=True, choices=["resnet50", "convnext_tiny", "densenet121", "vgg16"])
    ap.add_argument("--seed", type=int, required=True, choices=SUPPORTED_SEEDS)
    ap.add_argument("--override", action="append", default=[],
                     help="dot.path=value, repeatable, e.g. --override train_loop.epochs=40")
    ap.add_argument("--force-retrain", action="store_true",
                     help="retrain even if a checkpoint already exists at the canonical path")
    a = ap.parse_args()

    def _cast(raw: str) -> object:
        for c in (int, float):
            try:
                return c(raw)
            except ValueError:
                continue
        return raw
    overrides = {k: _cast(v) for k, v in (o.split("=", 1) for o in a.override)}

    ckpt = train(a.dataset, a.backbone, a.seed, overrides, force=a.force_retrain)
    dump_gate_evidence(a.dataset, a.backbone, a.seed, ckpt)

    ungated = common.summarize(
        (ROOT / f"outputs/effect_dump_v8_s{a.seed}/{a.dataset}/{a.backbone}/deploy_direct/predictions"),
        a.dataset,
    )
    print(f"\n[ungated] {a.dataset}/{a.backbone} s{a.seed}: {ungated}")

    gate_result = gate_eval.run_one(a.dataset, a.backbone, a.seed)
    print(f"[gate] preonly (DPInput, 3-feature, fit fresh): {gate_result['preonly']}")
    print(f"[gate] prepost (shipped 9-feature gate, if any): {gate_result['prepost']}")
    print(f"[gate] prepost_refit (9-feature, fit fresh -- THIS is the from-scratch 'gated' number "
          f"for seeds with no shipped gate): {gate_result['prepost_refit']}")

    out_path = ROOT / f"outputs/repro/dynapatch_fromscratch_{a.dataset}_{a.backbone}_s{a.seed}.json"
    out_path.write_text(json.dumps({"ungated": ungated, "gate": gate_result}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
