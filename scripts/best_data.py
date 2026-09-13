#!/usr/bin/env python3
"""One flat table: every method at its BEST configuration, on all 12 settings.

No RQ grouping, no epoch/variant/audit columns, no shipped-default rows. One row per
(method, setting, seed) with the four metrics. What "best" means per method:

  FullFT / HeadOnly / LastDelta / WeightedRetrain   40 epochs (the matched, longer budget)
  Arachne        per-setting best bound_scale over {2,4,8,16,32,64,128}
  TopKSearch     per-setting best step_scale  over {8,32,128}
  DistRep        the paper's full PSO budget 5x40x40 / clean-cap 2048   (1 seed, the rest are 3)
  NN-Patching / PatchNAS    swept error estimator (MLP)
  FixedPatch     per-setting best lr over {1e-2,1e-1}, 60 epochs  (a lr=1e-3 arm
                 exists but only for 3 settings x 1 seed, so the 3-seed requirement
                 in all_rq_data.py drops it -- it never enters the sweep)
  DynaPatch      40 epochs, no early stop; one gated row at theta=0 (commit iff the fitted
                 gate's own score favours beneficial over harmful -- no target r, no
                 threshold search, no calibration split). The gate is fitted WITHIN the
                 setting on bug_train + bug_val + clean_calib (scripts/gate_protocol_b.py).
                 RR_seen is blank on that row because the gate saw bug_train.

Metrics:  RR_test = repair rate on held-out failures (the paper's RR)
          RR_seen = repair rate on the evidence failures the repair was fitted on
          Reg / CReg = regression on clean_test, all classes / critical classes

Gated DynaPatch rows are seed-pooled by construction (the gate curve pools the three seeds),
so their seed column reads `pooled`.

Usage:  .venv/bin/python scripts/best_data.py
"""
from __future__ import annotations
import csv, json, statistics as st, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names as N  # noqa: E402  -- the ONLY place display labels may come from

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV, OUT_TXT = ROOT / "outputs/BEST_DATA.csv", ROOT / "outputs/BEST_DATA.txt"
ORDER = N.SETTING_ORDER
LBL = N.SETTING_LABEL
rows = []


def add(method, setting, seed, rr_seen, rr_test, reg, creg):
    rows.append(dict(method=method, setting=setting, label=LBL.get(setting, setting), seed=seed,
                     RR_seen=rr_seen, RR_test=rr_test, Reg=reg, CReg=creg))


# --- weight-editing / fine-tuning / search baselines -------------------------------------
# Which rq4_final.csv row is the reported configuration of which method is decided in
# names.py, not here. Rows that are misnamed or under-budget are rejected there by name.
for r in csv.DictReader((ROOT / "outputs/rq4_final.csv").open()):
    key = N.key_for_rq4_row(r["method"])
    if key is not None:
        add(N.display(key), r["setting"], r["seed"],
            r["RR_repair"], r["RR_held"], r["Reg"], r["CReg"])

# --- prior patch-based methods ------------------------------------------------------------
for name, key in ((N.display("NNPatch"), "NN-Patching"), (N.display("PatchNAS"), "PatchNAS")):
    # Prefer the 5-draw file. A single draw of these two methods is not reportable: with no
    # torch seed they moved 200 of 360 values between identical runs, and even seeded, the
    # spread across 5 draws reaches .105 in RR_held -- larger than several per-setting gaps.
    # The _r5 file stores the median plus __min/__max; the single-draw file is the fallback.
    _pp = next(p for p in (ROOT / "outputs/baseline_prior_patches_mlp_r5.json",
                           ROOT / "outputs/baseline_prior_patches_mlp.json") if p.is_file())
    for stg, per_seed in json.load(_pp.open())[key].items():
        for s, v in per_seed.items():
            add(name, stg, s, v["RR_seen"], v["RR_held"], v["Reg"], v["CReg"])

# --- fixed patch + ungated DynaPatch (both 40ep, from the ablation trees) -----------------
a = pd.read_csv(ROOT / "outputs/ALL_RQ_DATA.csv")
a = a[a.rq == "RQ1"].pivot_table(index=["method", "setting", "seed"], columns="metric",
                                 values="value").reset_index()
for _, r in a.iterrows():
    name = N.display("FP") if r["method"].startswith("FixedPatch") else N.display("DPNoGate")
    add(name, r["setting"], r["seed"], r["RR_seen"], r["RR_held"], r["Reg"], r["CReg"])

# --- gated DynaPatch, at the no-calibration natural threshold -----------------------------
# The shipped gate protocol lives in names.py, not here. It is protocol C: fitted within the
# setting on bug_train + bug_val + clean_calib. No leave-one-backbone-out anywhere.
# FIXED 2026-09-04 (two rounds): round 1 scanned N.GATE_CURVES (theta chosen by searching
# S_held + S_clean^test, an eval-set selection bug affecting every "DynaPatch @ r=X" row in
# the paper). Round 2 (this one) dropped r-targeting/calibration entirely: bug_val/clean_calib,
# the population round 1 used to pick theta, is ALSO part of what fits the gate under protocol
# C's shipped "both" mode, so it was never fully disjoint calibration either. Now reads
# N.GATE_NATURAL_POINT: theta=0, the fitted model's own class decision, no target r, no
# threshold search, no calibration split of any kind. See note/RESEARCH_STATE.md.
GATE = N.GATE_CURVE_NAME
nat = pd.read_csv(ROOT / N.GATE_NATURAL_POINT)
nat_pooled = nat.groupby("setting", as_index=False)[["RR_held", "Reg", "CReg"]].mean()
for _, b in nat_pooled.iterrows():
    add(N.gate_label_natural(), b.setting, "pooled", "", b.RR_held, b.Reg, b.CReg)

with OUT_CSV.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["method", "setting", "label", "seed",
                                       "RR_seen", "RR_test", "Reg", "CReg"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# Paper order first (the \begin{itemize} of "Compared Methods"), internal ablations last and
# visibly marked. Nothing here is a free-text string any more.
METHODS = ([N.gate_label_natural()]
           + [N.display(k) for k in N.PAPER_ORDER if k != "DP"]
           + [N.display(k) for k in N.INTERNAL_ORDER])
agg = {}
for r in rows:
    for m in ("RR_seen", "RR_test", "Reg", "CReg"):
        if r[m] not in ("", None):
            agg.setdefault((r["method"], m), {}).setdefault(r["setting"], []).append(float(r[m]))
lines = [__doc__.strip().split("\n\n")[0], ""]
for metric in ("RR_test", "RR_seen", "Reg", "CReg"):
    dec = 4 if metric in ("Reg", "CReg") else 3
    hdr = f"{'method':22s}" + "".join(f"{l:>9s}" for _, _, l in ORDER) + f"{'MEAN':>9s}"
    lines += ["", f"### {metric}", hdr, "-" * len(hdr)]
    for m in METHODS:
        per = agg.get((m, metric))
        if not per:
            continue
        vals = [st.mean(per[f"{d}/{b}"]) if f"{d}/{b}" in per else None for d, b, _ in ORDER]
        got = [v for v in vals if v is not None]
        lines.append(f"{m:22s}"
                     + "".join(f"{v:9.{dec}f}" if v is not None else f"{'--':>9s}" for v in vals)
                     + (f"{st.mean(got):9.{dec}f}" if got else f"{'--':>9s}"))
OUT_TXT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT_CSV} ({len(rows)} rows) and {OUT_TXT}")
