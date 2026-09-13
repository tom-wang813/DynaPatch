#!/usr/bin/env python3
"""Ablation: replace the transductive `*_r` rank features with a FIXED reference fit once on
repair data, applied unchanged to a single incoming row -- the only thing that is physically
possible at per-input inference time (see `analysis_gate_norank_ablation.py`'s docstring and
`gate_protocol_b.py`'s own "Second-order caveat" about ranking calibration and reporting rows on
different grids).

`m_base_r`, `dm_r`, `dnorm_r` in the shipped pipeline are `argsort(argsort(v))/(n-1)` computed
fresh on whatever population is passed in -- the repair (train) split ranks against itself, and
the test (report) split ranks against ITSELF too, never against a fixed repair-side reference.
This script builds that reference once from the SAME repair rows used to fit the gate
(`gate_protocol_b.train_cells("both", "calib")`'s clean+held sides, matching `gate_lambda_sweep.
dp_curve()` exactly), as an empirical CDF, and maps every row -- train and report alike -- through
that one fixed step function. A held-out row's feature value then depends only on itself and on
data collected before it existed, matching the per-input deployment story.

Requires the additive `*_raw` keys added to `analyze_response_gate_lobo.build_features()`
(m_base_raw, dm_raw, dnorm_raw) -- delete `outputs/_response_gate_cache*` once before the first
run of this script so stale caches (predating those keys) are not reused.

    .venv/bin/python scripts/analysis_gate_fixedrank_ablation.py
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
FEATS = list(GLS.FEATS)  # the shipped 12-feature "4 pre+post" set, names unchanged


def fixed_ref(*sides: dict) -> dict[str, np.ndarray]:
    """One empirical CDF per raw scalar, built by concatenating the given repair-side dicts."""
    return {rank_name: np.sort(np.concatenate([s["f"][raw_name] for s in sides]))
            for rank_name, raw_name in RANK_TO_RAW.items()}


def apply_fixed_rank(f: dict, ref: dict[str, np.ndarray]) -> dict:
    """Map every row of `f` through the fixed reference -- no dependence on any other row of `f`."""
    out = dict(f)
    for rank_name, raw_name in RANK_TO_RAW.items():
        sorted_ref = ref[rank_name]
        n = len(sorted_ref)
        out[rank_name] = np.searchsorted(sorted_ref, f[raw_name], side="right") / max(n, 1)
    return out


def dp_curve_fixed_rank() -> pd.DataFrame:
    """`gate_lambda_sweep.dp_curve()`, byte-identical except the 3 rank features are looked up
    through a fixed repair-fit reference instead of `analyze_response_gate_lobo.rank()`."""
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
        ref = fixed_ref(c, h)
        c_f, h_f = apply_fixed_rank(c["f"], ref), apply_fixed_rank(h["f"], ref)
        X = np.vstack([G.mat(cf, FEATS, d["flip"]) for cf, d in ((c_f, c), (h_f, h))])
        g = np.concatenate([d["gain"][d["flip"]] for d in (c, h)])
        mdl = fit_gain(X, g)
        if mdl is None:
            continue
        rc, rh = report[k]["clean"], report[k]["held"]
        rc_f, rh_f = apply_fixed_rank(rc["f"], ref), apply_fixed_rank(rh["f"], ref)
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
    fixed = dp_curve_fixed_rank()
    fixed = fixed[fixed["lambda"] == 1.0].drop(columns="lambda")

    a = shipped.groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    b = fixed.groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    m = a.merge(b, on="setting", suffixes=("_shipped_transductive", "_fixedref_perinput"))
    m = m[["setting",
           "RR_held_shipped_transductive", "RR_held_fixedref_perinput",
           "Reg_shipped_transductive", "Reg_fixedref_perinput",
           "CReg_shipped_transductive", "CReg_fixedref_perinput"]]
    print(m.round(4).to_string(index=False))
    print()
    print("pooled (mean over settings, uniform weight):")
    print(m.drop(columns="setting").mean(numeric_only=True).round(4))

    out = ROOT / "outputs" / "gate_fixedrank_ablation"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "per_setting.csv", index=False)
    print("\n[written] outputs/gate_fixedrank_ablation/per_setting.csv")


if __name__ == "__main__":
    main()
