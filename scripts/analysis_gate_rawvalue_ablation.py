#!/usr/bin/env python3
"""Third rank-feature variant, also per-input-feasible: use the RAW scalar (m_base, dm, dnorm)
directly instead of any rank/percentile transform of it.

`analysis_gate_fixedrank_ablation.py` showed that replacing the transductive rank with a properly
per-input one (fixed empirical-CDF reference fit on repair data) recovers essentially nothing over
just deleting the three features -- RR 0.4518 vs 0.4505, Reg 0.0040 vs 0.0041. That rules out "the
percentile position of this scalar among repair data" as informative, but percentile-by
construction throws away the scalar's magnitude (0.89 and 0.51 give different ranks only if they
land on different sides of the repair distribution's mass -- the number itself is discarded).

This variant keeps the magnitude: it swaps the `*_r` columns for the raw scalar and lets the
already-existing `fit_gain()` pipeline (`StandardScaler` fit on the SAME repair rows used to fit
the gate, `.transform` applied to report rows without looking at any other report row) do the
per-input-legal standardization. No new fitting step is added here; this only changes which three
columns enter a pipeline that was already train-fit/test-apply.

    .venv/bin/python scripts/analysis_gate_rawvalue_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gate_lambda_sweep as GLS  # noqa: E402
import probe_gonogo_pre_vs_prepost as G  # noqa: E402
from gate_protocol_b import train_cells  # noqa: E402
from probe_gonogo_pre_vs_prepost import fit_gain  # noqa: E402

RANK_TO_RAW = {"m_base_r": "m_base_raw", "dm_r": "dm_raw", "dnorm_r": "dnorm_raw"}
FEATS = list(GLS.FEATS)  # column NAMES are unchanged; we swap what each rank name maps to


def use_raw(f: dict) -> dict:
    """Same dict, but `f[name]` for name in RANK_TO_RAW now returns the raw (unranked) scalar."""
    out = dict(f)
    for rank_name, raw_name in RANK_TO_RAW.items():
        out[rank_name] = f[raw_name]
    return out


def dp_curve_raw() -> pd.DataFrame:
    """`gate_lambda_sweep.dp_curve()`, with the 3 rank columns replaced by raw scalars."""
    report = GLS._load_split(lambda: GLS.Z.attach_criticality(
        GLS.Z.attach_idx(GLS.Z.add_derived(G.build_cells()))))
    train = GLS._load_split(lambda: train_cells("both", "calib"))
    rows = []
    for k in sorted(set(train) & set(report)):
        seed, ds, bb = k
        c, h = train[k]["clean"], train[k]["held"]
        npos = int((h["gain"][h["flip"]] == 1).sum())
        nneg = int((c["gain"][c["flip"]] == -1).sum())
        if npos < GLS.MIN_POS or nneg < GLS.MIN_POS:
            continue
        c_f, h_f = use_raw(c["f"]), use_raw(h["f"])
        X = np.vstack([G.mat(cf, FEATS, d["flip"]) for cf, d in ((c_f, c), (h_f, h))])
        g = np.concatenate([d["gain"][d["flip"]] for d in (c, h)])
        mdl = fit_gain(X, g)
        if mdl is None:
            continue
        rc, rh = report[k]["clean"], report[k]["held"]
        rc_f, rh_f = use_raw(rc["f"]), use_raw(rh["f"])
        pos_c, neg_c = GLS.posneg(mdl, G.mat(rc_f, FEATS))
        pos_h, neg_h = GLS.posneg(mdl, G.mat(rh_f, FEATS))
        cd = int(rc["crit"].sum())
        for lam in GLS.LAMBDAS:
            keep_h = ~rh["flip"] | (pos_h > lam * neg_h)
            keep_c = ~rc["flip"] | (pos_c > lam * neg_c)
            rr_held = float(rh["y"][keep_h].sum() / rh["n"])
            reg = float(rc["y"][keep_c].sum() / rc["n"])
            creg = float((rc["y"] & keep_c & rc["crit"]).sum() / max(cd, 1))
            rows.append({"setting": f"{ds}/{bb}", "seed": seed, "lambda": lam,
                         "RR_held": rr_held, "Reg": reg, "CReg": creg})
    return pd.DataFrame(rows)


def main() -> None:
    shipped = GLS.dp_curve()
    shipped = shipped[shipped["lambda"] == 1.0].drop(columns="lambda")
    raw = dp_curve_raw()
    raw = raw[raw["lambda"] == 1.0].drop(columns="lambda")

    a = shipped.groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    b = raw.groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    m = a.merge(b, on="setting", suffixes=("_shipped_transductive", "_rawvalue_perinput"))
    m = m[["setting",
           "RR_held_shipped_transductive", "RR_held_rawvalue_perinput",
           "Reg_shipped_transductive", "Reg_rawvalue_perinput",
           "CReg_shipped_transductive", "CReg_rawvalue_perinput"]]
    print(m.round(4).to_string(index=False))
    print()
    print("pooled (mean over settings, uniform weight):")
    print(m.drop(columns="setting").mean(numeric_only=True).round(4))

    out = ROOT / "outputs" / "gate_rawvalue_ablation"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "per_setting.csv", index=False)
    print("\n[written] outputs/gate_rawvalue_ablation/per_setting.csv")


if __name__ == "__main__":
    main()
