#!/usr/bin/env python3
"""Ablation: does the shipped pre+post gate need its 3 within-split rank features?

`m_base_r`, `dm_r`, `dnorm_r` are computed by ranking each evaluation split against ITSELF
(`analyze_response_gate_lobo.py`'s `rank()`), not against a mapping fixed on repair data. That
makes a flagged input's own gate feature depend on which other inputs happen to be in the same
evaluation batch -- not per-input, contrary to the deployment story ("a gate fitted on one
architecture can be evaluated on another", one input at a time). This script refits the IDENTICAL
shipped pipeline (`gate_lambda_sweep.py`'s `dp_curve()`, lambda=1.0 = natural threshold) with
those 3 features removed, and reports RR_held/Reg/CReg per setting against the shipped 12-feature
number, so the price of being honestly per-input is a measured number, not a guess.

Fixed 2026-09-07: `GLS.FEATS` (== `LAYERS["4 pre+post"]`) now IS the 9-feature honest set by
default (the fix this script argued for was adopted as the shipped gate) -- so the two arms below
are read explicitly off `LAYERS["4-legacy pre+post"]` (pre-fix, 12 feat) and
`LAYERS["4 pre+post"]` (post-fix, 9 feat) instead of deriving FEATS_9 from GLS.FEATS, which would
now silently produce a 6-feature arm instead of 9.

    .venv/bin/python scripts/analysis_gate_norank_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gate_lambda_sweep as GLS  # noqa: E402
from probe_gonogo_pre_vs_prepost import LAYERS  # noqa: E402

RANK_FEATS = {"m_base_r", "dm_r", "dnorm_r"}
FEATS_12 = list(LAYERS["4-legacy pre+post"])
FEATS_9 = list(LAYERS["4 pre+post"])
assert set(FEATS_12) - set(FEATS_9) == RANK_FEATS, "expected exactly the 3 rank feats to drop"


def run(feats: list[str]) -> pd.DataFrame:
    GLS.FEATS = feats
    try:
        df = GLS.dp_curve()
    finally:
        GLS.FEATS = FEATS_12
    return df[df["lambda"] == 1.0].drop(columns="lambda")


def main() -> None:
    r12 = run(FEATS_12).groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    r9 = run(FEATS_9).groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    m = r12.merge(r9, on="setting", suffixes=("_12feat", "_9feat_norank"))
    m = m[["setting", "RR_held_12feat", "RR_held_9feat_norank",
           "Reg_12feat", "Reg_9feat_norank", "CReg_12feat", "CReg_9feat_norank"]]
    print(m.round(4).to_string(index=False))
    print()
    print("pooled (mean over settings, uniform weight):")
    print(m.drop(columns="setting").mean(numeric_only=True).round(4))

    out = ROOT / "outputs" / "gate_norank_ablation"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "per_setting.csv", index=False)
    print("\n[written] outputs/gate_norank_ablation/per_setting.csv")


if __name__ == "__main__":
    main()
