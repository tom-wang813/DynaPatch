#!/usr/bin/env python3
"""RQ3: evaluate DPGate (pre-only "DPInput" vs pre+post "DPGate") from the base/patched logits
already dumped by scripts/checkpoint_eval/dynapatch.py's --gate pass (run once per setting by
scripts/repro/rq1_aggregate.py) under outputs/effect_dump_v8_s101/<ds>/<bb>/deploy_direct{,_calib}/.
No new gate math -- everything here is src/models/dynapatch/gate.py's
compute_gate_features/gain_labels/FeatureGate, the exact code path src/experiment/deploy_eval.py
uses at deploy time.

**Pre+Post (DPGate)**: loaded straight from the shipped `artifacts/checkpoints/gates/<ds>_<bb>_s101.json`
-- the paper's own already-fit 9-feature gate. No refitting, no fitting-pool decisions to get
wrong.

**Pre-only (DPInput)**: no shipped checkpoint exists for this ablation (see
artifacts/checkpoints/manifest.json), so it's fit here on bug_train + bug_val + **clean_calib**
(a dedicated calibration split, disjoint from clean_eval) via one extra deploy-eval pass
(`_clean_calib_pass`, output under outputs/repro/gate_calib_pool/). Earlier draft of this script
fit on clean_eval instead of clean_calib -- i.e. the same clean population later used to report
Reg/CReg, a selection-on-the-evaluation-set leak (see this repo's own Evidence Hygiene rule #3).
Fixed here by fitting only on clean_calib and reporting Reg/CReg purely from the held-out
`deploy_direct` dump's clean_eval, which the fit never sees.

Evaluation pool (the "direct" dump, i.e. the real test-time split, for BOTH gate variants):
repair_holdout_unseen there is the actual bug_eval -- gate classification metrics
(accuracy/precision/recall/f1, matching paper.tex's Table rq3_gate_clf) are computed there (bug
population only, so gain in {0,+1}, per paper.tex's "binary evaluation" paragraph), and
RR_held/Reg/CReg (Table rq3_gate_effect) from the same dump's repair_holdout_unseen + clean_eval,
decide()'d at lambda=1 (the natural point, no threshold search).

Usage:
  uv run python scripts/repro/rq3_fit_eval_gate.py --settings gtsrb/resnet50   # one setting
  uv run python scripts/repro/rq3_fit_eval_gate.py                            # all 12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/checkpoint_eval"))

from src.models.dynapatch.gate import FEATURE_NAMES, FeatureGate, compute_gate_features, gain_labels  # noqa: E402
from src.experiment.deploy_eval import run_deploy_eval  # noqa: E402
import common  # noqa: E402
import dynapatch as dynapatch_ckpt_eval  # noqa: E402

DEFAULT_SEED = 101  # RQ1-4's own numbers always use seed 101, matching the 6 baselines (which
                     # only ever ship seed-101 checkpoints) -- --seed 202/303 is for reviewers
                     # checking a from-scratch DynaPatch+gate run at one of the other two seeds
                     # that DO have full split manifests (artifacts/bug_sets/shuffled_split_seed{202,303}/).


def dump_root(seed: int) -> Path:
    return ROOT / f"outputs/effect_dump_v8_s{seed}"


GATE_OUT = ROOT / "outputs/repro/gates"
CALIB_POOL_ROOT = ROOT / "outputs/repro/gate_calib_pool"
SHIPPED_GATE_ROOT = ROOT / "artifacts/checkpoints/gates"
PRE_ONLY_IDX = [FEATURE_NAMES.index(n) for n in ("pB_max", "pB_margin", "H_base")]

ALL_SETTINGS = [
    (ds, bb)
    for ds in ("gtsrb", "tt100k_signs", "lisa_signs")
    for bb in ("resnet50", "convnext_tiny", "densenet121", "vgg16")
]


def load_split(dump_dir: Path, split: str) -> dict:
    """dataset_index-ordered rows: base/patched logits + label/base_correct/patched_correct from
    the predictions CSV (written in the same iteration order as the .npy dumps -- verified by
    matching dataset_index between the two)."""
    pred_dir = dump_dir / "predictions"
    rows = list(csv.DictReader((pred_dir / f"{split}_predictions.csv").open()))
    idx_npy = np.load(pred_dir / f"dataset_indices_{split}.npy")
    assert [int(r["dataset_index"]) for r in rows] == idx_npy.tolist(), \
        f"{dump_dir}/{split}: CSV row order does not match .npy dataset_indices order"
    base_logits = torch.from_numpy(np.load(pred_dir / f"base_logits_{split}.npy"))
    patched_logits = torch.from_numpy(np.load(pred_dir / f"patched_logits_{split}.npy"))
    base_correct = torch.tensor([r["base_correct"] == "True" for r in rows])
    patched_correct = torch.tensor([r["patched_correct"] == "True" for r in rows])
    label = torch.tensor([int(r["label"]) for r in rows])
    return {"base_logits": base_logits, "patched_logits": patched_logits,
            "base_correct": base_correct, "patched_correct": patched_correct, "label": label}


def crit_mask(dataset: str, label: torch.Tensor) -> torch.Tensor:
    crit = common.crit_classes(dataset)
    return torch.tensor([int(y) in crit for y in label.tolist()])


def clean_calib_pass(dataset: str, backbone: str, seed: int) -> Path:
    """One extra deploy-eval pass scoring the DynaPatch repair checkpoint against clean_calib
    (not clean_eval) with save_route_features=True, so pre-only gate fitting has a clean
    population that is NOT the one Reg/CReg gets reported on later. Cached across runs."""
    out_root = CALIB_POOL_ROOT / f"{dataset}_{backbone}_s{seed}"
    pred_dir = out_root / "predictions"
    if (pred_dir / "clean_eval_predictions.csv").exists():
        return out_root
    ckpt = dynapatch_ckpt_eval.repair_checkpoint_path(dataset, backbone, seed)
    sdir = common.split_dir(dataset, backbone, seed)
    cfg = dynapatch_ckpt_eval.resolved_cfg(
        dataset, backbone, seed, out_root, ckpt, "cuda:0",
        held_suffix="bug_val_indices", save_route_features=True,
    )
    # clean_calib_indices.json instead of resolved_cfg's default clean_eval_indices.json --
    # the whole point of this pass.
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, "data.clean_eval_indices_path", str(sdir / f"{dataset}_clean_calib_indices.json"), merge=True)
    run_deploy_eval(cfg)
    return out_root


def _fitting_pool(dataset: str, backbone: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    calib_dir = dump_root(seed) / dataset / backbone / "deploy_direct_calib"
    bug_parts = [load_split(calib_dir, s) for s in ("repair_support_seen", "repair_holdout_unseen")]
    clean_calib_dir = clean_calib_pass(dataset, backbone, seed)
    clean_part = load_split(clean_calib_dir, "clean_eval")  # "clean_eval" split-name here holds clean_calib rows
    parts = bug_parts + [clean_part]
    return (torch.cat([p["base_logits"] for p in parts]), torch.cat([p["patched_logits"] for p in parts]),
            torch.cat([p["base_correct"] for p in parts]), torch.cat([p["patched_correct"] for p in parts]))


def fit_preonly_gate(dataset: str, backbone: str, seed: int) -> FeatureGate | None:
    base_logits, patched_logits, base_correct, patched_correct = _fitting_pool(dataset, backbone, seed)
    feats = compute_gate_features(base_logits, patched_logits)[:, PRE_ONLY_IDX]
    gain = gain_labels(base_correct, patched_correct)
    if len(set(gain.tolist())) < 2:
        return None  # not estimable (e.g. too few positive/negative rows at this seed)
    gate = FeatureGate.fit(feats, gain)
    GATE_OUT.mkdir(parents=True, exist_ok=True)
    gate.save(GATE_OUT / f"{dataset}_{backbone}_preonly_s{seed}.json")
    return gate


def load_shipped_prepost_gate(dataset: str, backbone: str, seed: int) -> FeatureGate | None:
    path = SHIPPED_GATE_ROOT / f"{dataset}_{backbone}_s{seed}.json"
    if not path.exists():
        return None
    return FeatureGate.load(path)


def evaluate(dataset: str, backbone: str, seed: int, gate: FeatureGate, feature_idx: list[int] | None) -> dict:
    direct_dir = dump_root(seed) / dataset / backbone / "deploy_direct"
    bug_eval = load_split(direct_dir, "repair_holdout_unseen")
    clean_eval = load_split(direct_dir, "clean_eval")

    def decide(split: dict) -> torch.Tensor:
        feats = compute_gate_features(split["base_logits"], split["patched_logits"])
        if feature_idx is not None:
            feats = feats[:, feature_idx]
        return gate.decide(feats, lam=1.0)

    # --- classification metrics on the bug (failure) population only: gain in {0,+1} there ---
    accept = decide(bug_eval)
    gain = gain_labels(bug_eval["base_correct"], bug_eval["patched_correct"])
    y_true = (gain == 1)
    y_pred = accept
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # --- deployment effect: RR_held / Reg / CReg with the gate's accept/reject applied ---
    held_final_correct = torch.where(accept, bug_eval["patched_correct"], bug_eval["base_correct"])
    rr_held = float(held_final_correct.float().mean())

    clean_accept = decide(clean_eval)
    clean_final_correct = torch.where(clean_accept, clean_eval["patched_correct"], clean_eval["base_correct"])
    reg_mask = clean_eval["base_correct"] & (~clean_final_correct)
    reg = float(reg_mask.float().mean())
    cmask = crit_mask(dataset, clean_eval["label"])
    creg = float(reg_mask[cmask].float().mean()) if cmask.any() else 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "rr_held": rr_held, "reg": reg, "creg": creg,
            "n_bug_eval": len(y_true), "n_clean_eval": len(clean_eval["label"])}


def run_one(dataset: str, backbone: str, seed: int = DEFAULT_SEED) -> dict:
    out = {}

    gate = fit_preonly_gate(dataset, backbone, seed)
    out["preonly"] = evaluate(dataset, backbone, seed, gate, PRE_ONLY_IDX) if gate is not None else None

    gate = load_shipped_prepost_gate(dataset, backbone, seed)
    out["prepost"] = evaluate(dataset, backbone, seed, gate, None) if gate is not None else None

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    settings = [tuple(s.split("/")) for s in a.settings] if a.settings else ALL_SETTINGS

    results = {}
    for dataset, backbone in settings:
        print(f"=== {dataset}/{backbone} (seed {a.seed}) ===")
        r = run_one(dataset, backbone, a.seed)
        results[f"{dataset}/{backbone}"] = r
        for tag in ("preonly", "prepost"):
            print(f"  {tag}: {r[tag]}")

    out_path = ROOT / f"outputs/repro/rq3_raw_s{a.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # Tables rq3_gate_clf / rq3_gate_effect: per setting, Pre (DPInput) vs Pre+Post (DPGate),
    # plus the mean over settings where both variants are available
    tables = [("rq3_gate_clf", ("accuracy", "precision", "recall", "f1"), 3),
              ("rq3_gate_effect", ("rr_held", "reg", "creg"), 4)]
    both = {k: r for k, r in results.items() if r["preonly"] and r["prepost"]}
    for name, metrics, digits in tables:
        header = "".join(f"{m + ' Pre':>16}{m + ' Pre+Post':>18}" for m in metrics)
        print(f"\nTable {name} (seed {a.seed})")
        print(f"{'Setting':<26}{header}")
        lines = ["setting," + ",".join(f"{m}_pre,{m}_prepost" for m in metrics)]
        rows = list(both.items())
        rows.append(("Mean", {tag: {m: sum(r[tag][m] for r in both.values()) / len(both)
                                     for m in metrics} for tag in ("preonly", "prepost")}))
        for setting, r in rows:
            vals = [r[tag][m] for m in metrics for tag in ("preonly", "prepost")]
            cells = "".join(f"{v:>16.{digits}f}" if i % 2 == 0 else f"{v:>18.{digits}f}"
                            for i, v in enumerate(vals))
            print(f"{setting:<26}{cells}")
            lines.append(f"{setting}," + ",".join(str(v) for v in vals))
        (out_path.parent / f"{name}.csv").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}\nwrote {out_path.parent}/rq3_gate_clf.csv\n"
          f"wrote {out_path.parent}/rq3_gate_effect.csv")


if __name__ == "__main__":
    main()
