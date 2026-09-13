#!/usr/bin/env python3
"""RQ1 main table (tab:rq1): RR_held / Reg / CReg for the four RQ1 methods, all at the SAME
RAW, ungated operating point (always apply the produced patch, no gate/routing of any kind --
not even the baseline's own estimator).

Written to add a CReg column tab:rq1 never had. First attempt (2026-09-07) wrongly rebuilt
NN-Patching/PatchNAS's RR/Reg from `outputs/prior_patch_persample/` (a single, non-seeded patch
generation run) and got numbers close to but not matching the existing table (0.653 vs 0.660 for
gtsrb/resnet50 NN-Patching) -- close enough to look like a bug, but actually a real
nondeterminism artefact: NN-Patching/PatchNAS's patch generation has no fixed torch seed (see
`note/RESEARCH_STATE.md`/memory `dynapatch-prior-patch-nondeterministic`, RR spans up to 0.105
across repeats). The actually-canonical, ALREADY-VERIFIED source is
`outputs/baseline_prior_patches_mlp_r5_ungated.json` (`scripts/baseline_prior_patches.py
--estimator mlp --torch-seed 0 --repeats 5 --tau -1.0`), the exact source
`scripts/paper_latex_tables.py:rq1()`'s `_ungated_baseline_rr_reg()` already reads to build the
existing table (confirmed 2026-09-07: recomputing from it reproduces 0.660/0.646 and 0.748/0.889
exactly, and its CReg values match the already-published RQ1_DATA.md/RQ3_DATA.md RQ3.13
"Fully ungated" row exactly). This script now reads NN-Patching/PatchNAS from THAT json (which
already carries a precomputed CReg per seed) instead of recomputing from raw predictions.

FixedPatch/DynaPatch-NoGate still come from `outputs/sample_frame.csv.gz` (methods "FixedPatch"
and "DynaPatch (ungated)" -- both rows already unconditional-apply, no `routed`/`gate_applied`
values) using the same RR_held/Reg/CReg definitions as
`scripts/analysis_gate_transplant.py:metrics_at`/`_creg` (same `critical_classes()` source as
every RQ2/RQ4 table); cross-checked against `outputs/ALL_RQ_DATA.csv` (rq=RQ1, per-seed rows,
averaged) -- matches to the precision both sources carry.

Seeds are averaged within each setting (project convention, `RQ2_BASELINE_BEHAVIOR.md`'s data
contract); all 12 settings have full 3-seed coverage for all four methods (verified 2026-09-07).

    .venv/bin/python scripts/table_rq1_raw_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analysis_gate_transplant import critical_classes  # noqa: E402

SEEDS = (101, 202, 303)
DATASETS = ("gtsrb", "tt100k_signs", "lisa_signs")
BACKBONES = ("resnet50", "convnext_tiny", "vgg16", "densenet121")
SETTINGS = [(d, b) for d in DATASETS for b in BACKBONES]
LABEL = {"gtsrb": "G", "tt100k_signs": "T", "lisa_signs": "L"}
BB_LABEL = {"resnet50": "RN", "convnext_tiny": "CN", "vgg16": "VG", "densenet121": "DN"}


def cell_from_arrays(held_base_correct, held_correct_after,
                      clean_base_correct, clean_correct_after, clean_label, critical) -> dict:
    h = ~held_base_correct
    rr_held = float(held_correct_after[h].mean()) if h.any() else float("nan")

    c = clean_base_correct
    reg = float((c & ~clean_correct_after).sum() / c.sum()) if c.any() else float("nan")

    crit_mask = c & np.isin(clean_label, list(critical)) if critical else np.zeros_like(c)
    creg = float((crit_mask & ~clean_correct_after).sum() / crit_mask.sum()) \
        if crit_mask.any() else float("nan")
    return {"RR_held": rr_held, "Reg": reg, "CReg": creg}


def sample_frame_cells(sf: pd.DataFrame, method: str, display: str, critical_by_ds: dict) -> list[dict]:
    recs = []
    sub = sf[sf.method == method]
    for (ds, bb), seed in [(s, sd) for s in SETTINGS for sd in SEEDS]:
        setting = f"{ds}/{bb}"
        crit = critical_by_ds[ds]
        cell = sub[(sub.setting == setting) & (sub.seed == seed)]
        held = cell[cell.split == "held"]
        clean = cell[cell.split == "clean"]
        if not len(held) or not len(clean):
            continue
        m = cell_from_arrays(
            held.base_correct.to_numpy(), held.patched_correct.to_numpy(),
            clean.base_correct.to_numpy(), clean.patched_correct.to_numpy(),
            clean.true_class.to_numpy(), crit)
        recs.append({"method": display, "setting": setting, "seed": seed, **m})
    return recs


def ungated_baseline_cells(method: str) -> list[dict]:
    """NN-Patching/PatchNAS at their RAW ungated (route_rate=1.0) operating point --
    `outputs/baseline_prior_patches_mlp_r5_ungated.json`, the seeded (`--torch-seed 0
    --repeats 5`) protocol `scripts/paper_latex_tables.py:_ungated_baseline_rr_reg` already
    reads for `tab:rq1`'s RR/Reg columns. NOT `outputs/prior_patch_persample/` -- that dump has
    no fixed torch seed and does not reproduce this table (see module docstring). CReg is
    already precomputed per seed in this json; not rederived here."""
    j = json.loads((ROOT / "outputs/baseline_prior_patches_mlp_r5_ungated.json").read_text())
    recs = []
    for ds, bb in SETTINGS:
        setting = f"{ds}/{bb}"
        by_seed = j[method][setting]
        for seed_str, m in by_seed.items():
            recs.append({"method": method, "setting": setting, "seed": int(seed_str),
                         "RR_held": m["RR_held"], "Reg": m["Reg"], "CReg": m["CReg"]})
    return recs


def main() -> None:
    critical_by_ds = {d: critical_classes(d) for d in DATASETS}
    for d, crit in critical_by_ds.items():
        assert crit, f"empty critical-class set for {d} -- artifacts/risk/{d}_safety_risk_matrix.json missing or empty"

    sf = pd.read_csv(ROOT / "outputs/sample_frame.csv.gz", low_memory=False)

    recs = []
    recs += sample_frame_cells(sf, "FixedPatch", "FixedPatch", critical_by_ds)
    recs += sample_frame_cells(sf, "DynaPatch (ungated)", "DynaPatch-NoGate", critical_by_ds)
    recs += ungated_baseline_cells("NN-Patching")
    recs += ungated_baseline_cells("PatchNAS")

    cell = pd.DataFrame(recs)
    out = ROOT / "outputs" / "rq1_raw_metrics"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_cell.csv", index=False)

    per_setting = cell.groupby(["method", "setting"], as_index=False)[
        ["RR_held", "Reg", "CReg"]].mean()
    per_setting.to_csv(out / "per_setting.csv", index=False)

    n_expected = len(SETTINGS) * len(SEEDS)
    for method, grp in cell.groupby("method"):
        assert len(grp) == n_expected, f"{method}: {len(grp)}/{n_expected} cells"
        assert grp[["RR_held", "Reg", "CReg"]].isna().to_numpy().sum() == 0, \
            f"{method}: NaN in RR_held/Reg/CReg -- check critical-class overlap or empty populations"

    methods = ["FixedPatch", "DynaPatch-NoGate", "NN-Patching", "PatchNAS"]
    print(f"{'Setting':14}" + "".join(f"{m + ' RR':>18}" for m in methods)
          + "".join(f"{m + ' Reg':>18}" for m in methods)
          + "".join(f"{m + ' CReg':>18}" for m in methods))
    for ds, bb in SETTINGS:
        setting = f"{ds}/{bb}"
        label = f"{LABEL[ds]}-{BB_LABEL[bb]}"
        row = per_setting[per_setting.setting == setting].set_index("method")
        vals = [row.loc[m, c] for c in ("RR_held", "Reg", "CReg") for m in methods]
        print(f"{label:14}" + "".join(f"{v:18.4f}" for v in vals))
    overall = per_setting.groupby("method")[["RR_held", "Reg", "CReg"]].mean()
    print("\nMean over 12 settings:")
    print(overall.loc[methods])


if __name__ == "__main__":
    main()
