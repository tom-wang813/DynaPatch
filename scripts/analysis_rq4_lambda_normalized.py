#!/usr/bin/env python3
"""RQ4 redesign: does the gate remove regression-causing patch applications faster than
successful repairs as the regression cost lambda increases?

Uses only the paper's existing metrics (RR, Reg, CReg) -- no new metric is defined. This script
adds two things `scripts/gate_lambda_sweep.py` does not report:

  1. The No-gate reference point (RR_nogate, Reg_nogate, CReg_nogate) per (setting, seed): every
     proposed patch applied unconditionally, no gate fit needed or used. Uses every (setting,
     seed) in the cached report cells -- NOT restricted to the seeds `gate_lambda_sweep.py`'s
     MIN_POS check would accept a gate fit on (an earlier version of this function applied that
     restriction to "match" the lambda rows' population; that was a bug, not a safety margin --
     gate-fittability is irrelevant to a quantity that involves no gate. Fixed 2026-09-07, see
     `no_gate_rows()`'s own docstring). This reproduces RQ1/RQ2/RQ4.1-4.4's published DPNoGate
     mean RR of 0.5231 exactly; the lambda rows below still correctly use only the gate-fittable
     seeds, since gating does require a fit.

  2. Two purely visualization-side derived quantities -- NOT new metrics, just percentage views
     of RR/Reg/CReg already defined in the paper:
       (a) per-setting percentage decrease in RR and in Reg from No-gate to lambda=1 (panel a data)
       (b) 12-setting-mean RR/Reg/CReg at each lambda, expressed as a percentage of the
           12-setting-mean No-gate value (panel b data, ratio-of-means, matching the paper's
           existing "gate retains X% of the ungated mean repair rate" convention)

    .venv/bin/python scripts/analysis_rq4_lambda_normalized.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402
import gate_zoo as Z  # noqa: E402
import probe_gonogo_pre_vs_prepost as G  # noqa: E402
from gate_lambda_sweep import _load_split  # noqa: E402
import names as N  # noqa: E402

SETTING_LABEL = {f"{d}/{b}": lab for d, b, lab in N.SETTING_ORDER}


def no_gate_rows() -> pd.DataFrame:
    """Per (setting, seed): RR_nogate, Reg_nogate, CReg_nogate -- every proposed patch applied
    unconditionally, no gate involved at all.

    CORRECTED (2026-09-07, caught by user review): this function used to apply `dp_curve()`'s
    MIN_POS cell-inclusion filter (skip a seed if the gate could not be fit on it) before
    computing the no-gate rate. That filter belongs to GATE FITTING; "no gate" does not fit a
    gate and does not need one to exist, so tying its population to gate-fittability was wrong,
    not merely a stricter-than-necessary choice. It silently dropped 1 of 3 seeds for
    lisa_signs/vgg16 and 1 of 3 for lisa_signs/convnext_tiny, understating the true no-gate mean
    RR by half a point (0.518 instead of 0.523) relative to RQ1/RQ2/RQ4.1-4.4's own published
    DPNoGate figure, which correctly uses every seed. Verified directly against the raw report
    cells with no filter: mean RR/Reg/CReg over all (setting, seed) rows reproduces 0.5231/
    0.0066/0.0043 exactly, matching RQ4.1-4.4. This function now reports every (setting, seed)
    in `report`, regardless of whether a gate could be fit on the corresponding training cell --
    the lambda-sweep rows this joins against still correctly use only the gate-fittable seeds
    (gating DOES require a fit), so a 12-setting mean joining this column against the lambda
    columns will average over slightly different per-setting seed counts for those two settings;
    that is expected and correct, not a mismatch to paper over -- the two quantities have
    genuinely different eligible populations."""
    report = _load_split(lambda: Z.attach_criticality(Z.attach_idx(Z.add_derived(G.build_cells()))))
    rows = []
    for k in sorted(report):
        seed, ds, bb = k
        rc, rh = report[k]["clean"], report[k]["held"]
        cd = int(rc["crit"].sum())
        rows.append({
            "setting": f"{ds}/{bb}", "seed": seed,
            "RR_held": float(rh["y"].sum() / rh["n"]),
            "Reg": float(rc["y"].sum() / rc["n"]),
            "CReg": float((rc["y"] & rc["crit"]).sum() / max(cd, 1)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    nogate = no_gate_rows()
    sweep = pd.read_csv(ROOT / "outputs/gate_lambda_sweep/per_cell.csv")
    sweep = sweep[sweep.method == "DynaPatch"].drop(columns="method")

    # per-setting mean across seeds, matching this project's setting-balanced convention
    nogate_setting = nogate.groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    nogate_setting["lambda"] = "No gate"
    sweep_setting = sweep.groupby(["setting", "lambda"], as_index=False)[
        ["RR_held", "Reg", "CReg"]].mean()
    both = pd.concat([nogate_setting, sweep_setting], ignore_index=True)

    # ---------------------------------------------------------------- summary table
    summary = both.groupby("lambda")[["RR_held", "Reg", "CReg"]].mean()
    order = ["No gate", 0.5, 1.0, 2.0, 4.0]
    summary = summary.loc[order]

    # ---------------------------------------------------------------- panel (a): per-setting,
    # lambda=1 vs No gate, percentage decrease in RR and in Reg
    piv = both.pivot(index="setting", columns="lambda", values=["RR_held", "Reg"])
    panel_a = pd.DataFrame({
        "setting": piv.index,
        "label": [SETTING_LABEL[s] for s in piv.index],
        "RR_nogate": piv[("RR_held", "No gate")],
        "RR_lambda1": piv[("RR_held", 1.0)],
        "Reg_nogate": piv[("Reg", "No gate")],
        "Reg_lambda1": piv[("Reg", 1.0)],
    }).reset_index(drop=True)
    panel_a["pct_decrease_RR"] = (panel_a.RR_nogate - panel_a.RR_lambda1) / panel_a.RR_nogate * 100
    panel_a["pct_decrease_Reg"] = (panel_a.Reg_nogate - panel_a.Reg_lambda1) / panel_a.Reg_nogate * 100
    panel_a["above_diagonal"] = panel_a.pct_decrease_Reg > panel_a.pct_decrease_RR

    # ---------------------------------------------------------------- panel (b): 12-setting-mean
    # ratio to 12-setting-mean No-gate, i.e. ratio-of-means (matches the paper's existing "gate
    # retains X% of the ungated mean repair rate" convention, not a mean-of-per-setting-ratios)
    base = summary.loc["No gate"]
    panel_b = pd.DataFrame({
        "lambda": [0.5, 1.0, 2.0, 4.0],
        "RR_pct_of_nogate": [float(summary.loc[l, "RR_held"] / base.RR_held * 100) for l in (0.5, 1.0, 2.0, 4.0)],
        "Reg_pct_of_nogate": [float(summary.loc[l, "Reg"] / base.Reg * 100) for l in (0.5, 1.0, 2.0, 4.0)],
        "CReg_pct_of_nogate": [float(summary.loc[l, "CReg"] / base.CReg * 100) for l in (0.5, 1.0, 2.0, 4.0)],
    })

    out = ROOT / "outputs" / "rq4_lambda_normalized"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary_table.csv")
    panel_a.to_csv(out / "panel_a_per_setting.csv", index=False)
    panel_b.to_csv(out / "panel_b_relative.csv", index=False)

    RM.write_section(
        "RQ4LambdaNormalized",
        "RQ4 redesign: No-gate reference, per-setting symmetry (panel a), relative-to-no-gate "
        "trend (panel b) -- RR/Reg/CReg only, no new metric",
        f"""
No-gate row uses every (setting, seed) in the cached report cells, no gate-fittability
restriction (fixed 2026-09-07 -- see `no_gate_rows()`'s docstring); it reproduces
RQ1/RQ2/RQ4.1-4.4's published DPNoGate mean RR (0.5231) exactly. The lambda rows still come from
`outputs/gate_lambda_sweep/per_cell.csv`, which correctly restricts to gate-fittable seeds.

**Summary table** (12-setting mean; only RR/Reg/CReg, the paper's existing metrics):
""",
        [("summary_table", summary.reset_index()),
         ("panel_a_per_setting", panel_a),
         ("panel_b_relative_to_nogate", panel_b)],
    )
    print(f"[written] {out.relative_to(ROOT)}/*.csv")
    print("\nSummary table (12-setting mean):")
    print(summary.round(4).to_string())
    print("\nPanel (a) per-setting (lambda=1 vs No gate):")
    print(panel_a[["label", "pct_decrease_RR", "pct_decrease_Reg", "above_diagonal"]]
          .round(1).to_string(index=False))
    print("\nPanel (b) relative to No-gate (%):")
    print(panel_b.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
