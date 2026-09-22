#!/usr/bin/env python3
"""RQ4: sweep the shipped DPGate's regression-cost lambda in {0.5, 1, 2, 4} (paper.tex's Table
rq4_summary sweep, confirmed from its own text) using the SAME base/patched logits
scripts/checkpoint_eval/dynapatch.py's --gate pass already dumped under
outputs/effect_dump_v8_s101/<ds>/<bb>/deploy_direct/ -- FeatureGate.decide() is re-evaluated at
each lambda on cached features, no re-running the network per lambda point. Reuses
scripts/repro/rq3_fit_eval_gate.py's load_split/crit_mask and the shipped
artifacts/checkpoints/gates/<ds>_<bb>_s101.json (same gate as RQ1/RQ3's Pre+Post column, no
refitting).

Usage:
  uv run python scripts/repro/rq4_sweep_gate_lambda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/checkpoint_eval"))
sys.path.insert(0, str(ROOT / "scripts/repro"))

from src.models.dynapatch.gate import FeatureGate, compute_gate_features  # noqa: E402
from rq3_fit_eval_gate import load_split, crit_mask, dump_root, SHIPPED_GATE_ROOT, DEFAULT_SEED as SEED  # noqa: E402

DUMP_ROOT = dump_root(SEED)

LAMBDAS = (0.5, 1.0, 2.0, 4.0)
ALL_SETTINGS = [
    (ds, bb)
    for ds in ("gtsrb", "tt100k_signs", "lisa_signs")
    for bb in ("resnet50", "convnext_tiny", "densenet121", "vgg16")
]


def metrics_at(bug_eval: dict, clean_eval: dict, dataset: str,
                gate: FeatureGate | None, lam: float | None) -> dict:
    """lam=None -> ungated (DynaPatch-NoGate, i.e. always apply the patch)."""
    if gate is None:
        held_final = bug_eval["patched_correct"]
        clean_final = clean_eval["patched_correct"]
    else:
        held_accept = gate.decide(compute_gate_features(bug_eval["base_logits"], bug_eval["patched_logits"]), lam=lam)
        held_final = torch.where(held_accept, bug_eval["patched_correct"], bug_eval["base_correct"])
        clean_accept = gate.decide(compute_gate_features(clean_eval["base_logits"], clean_eval["patched_logits"]), lam=lam)
        clean_final = torch.where(clean_accept, clean_eval["patched_correct"], clean_eval["base_correct"])

    rr_held = float(held_final.float().mean())
    reg_mask = clean_eval["base_correct"] & (~clean_final)
    reg = float(reg_mask.float().mean())
    cmask = crit_mask(dataset, clean_eval["label"])
    creg = float(reg_mask[cmask].float().mean()) if cmask.any() else 0.0
    return {"rr_held": rr_held, "reg": reg, "creg": creg}


def run_one(dataset: str, backbone: str) -> dict:
    direct_dir = DUMP_ROOT / dataset / backbone / "deploy_direct"
    bug_eval = load_split(direct_dir, "repair_holdout_unseen")
    clean_eval = load_split(direct_dir, "clean_eval")

    out = {"nogate": metrics_at(bug_eval, clean_eval, dataset, None, None)}
    gate_path = SHIPPED_GATE_ROOT / f"{dataset}_{backbone}_s{SEED}.json"
    if not gate_path.exists():
        for lam in LAMBDAS:
            out[f"lambda={lam}"] = None
        return out
    gate = FeatureGate.load(gate_path)
    for lam in LAMBDAS:
        out[f"lambda={lam}"] = metrics_at(bug_eval, clean_eval, dataset, gate, lam)
    return out


def main() -> None:
    results = {}
    for dataset, backbone in ALL_SETTINGS:
        print(f"=== {dataset}/{backbone} ===")
        r = run_one(dataset, backbone)
        results[f"{dataset}/{backbone}"] = r
        for k, v in r.items():
            print(f"  {k}: {v}")

    out_path = ROOT / "outputs/repro/rq4_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
