#!/usr/bin/env python3
"""P2 -- what input-specific patch generation buys, split by seen vs unseen failures.

The contrast is FixedPatch (one shared delta, trained on the same evidence) against DynaPatch
without the gate (a delta generated per input). Neither uses the gate, so the difference between
them is the generator.

Why the seen/unseen split is not optional
-----------------------------------------
The earlier reading of this contrast was that a shared patch "cannot express" the corrections.
That was retracted: once the fixed delta is trained properly it reaches RR_seen ~0.74, so it can
express them -- on the failures it was fitted on. The gap lives entirely in generalisation.
Reporting the two splits pooled would re-measure that gap and mislabel it as expressiveness.

    seen    repair_support_seen  -- the failures the patch was fitted on
    held    bug_eval             -- failures neither patch has seen

Per-class rates use the TRUE CLASS, for the reason given in note/AUDIT.md: the ordered pair
(y -> y_hat) leaves 1-6 estimable types per LISA cell.

Also emitted: the per-class scatter that the "which failures does a shared patch handle badly"
question needs -- for every (setting, seed, class) with >=5 held-out failures, both methods'
per-class repair rate side by side, plus the stratum the FIXED patch falls in (low/medium/high).
That stratification is the direct test of "input-specific generation particularly helps the
failure types a single shared patch handles badly", and it must be defined by the FIXED arm
alone, never by the difference, or it selects on the outcome.

    .venv/bin/python scripts/analysis_p2_inputspecific.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import names as N  # noqa: E402
import raw_md as RM  # noqa: E402

FIXED = N.display("FP")
GEN = N.display("DPNoGate")
MIN_N = 5


def main() -> None:
    df = pd.read_csv(ROOT / "outputs/sample_frame.csv.gz", low_memory=False)
    df = df[df.method.isin([FIXED, GEN])]
    if df.empty:
        raise SystemExit("FixedPatch / DynaPatch-NoGate not in the frame")

    # ---------------------------------------------------------------- cell level, per split
    rows = []
    for (m, stg, sd, sp), d in df.groupby(["method", "setting", "seed", "split"]):
        if sp == "clean":
            c = d[d.base_correct]
            rows.append({"method": m, "setting": stg, "seed": sd, "split": sp,
                         "n": len(c), "n_event": int(c.regressed.sum()),
                         "rate": float(c.regressed.mean()) if len(c) else np.nan,
                         "n_classes": int(c.true_class.nunique())})
            continue
        f = d[~d.base_correct]
        if not len(f):
            continue
        g = f.groupby("true_class").repaired.agg(["sum", "size"])
        g5 = g[g["size"] >= MIN_N]
        r5 = (g5["sum"] / g5["size"]) if len(g5) else pd.Series(dtype=float)
        rows.append({"method": m, "setting": stg, "seed": sd, "split": sp,
                     "n": len(f), "n_event": int(f.repaired.sum()),
                     "rate": float(f.repaired.mean()),
                     "n_classes": int(len(g)), "n_classes_est": int(len(g5)),
                     "RR_median_class": float(r5.median()) if len(r5) else np.nan,
                     "RR_p10_class": float(r5.quantile(.10)) if len(r5) else np.nan,
                     "cov0.5": float((r5 >= .5).mean()) if len(r5) else np.nan})
    cell = pd.DataFrame(rows)

    # ---------------------------------------------------------------- paired, per split
    piv = cell.pivot_table(index=["setting", "seed", "split"], columns="method",
                           values=["rate", "n_event", "n", "RR_median_class", "cov0.5"])
    piv.columns = [f"{a}__{'FIX' if b == FIXED else 'GEN'}" for a, b in piv.columns]
    paired = piv.reset_index()
    for c in ("rate", "RR_median_class", "cov0.5"):
        if f"{c}__GEN" in paired and f"{c}__FIX" in paired:
            paired[f"d_{c}"] = paired[f"{c}__GEN"] - paired[f"{c}__FIX"]

    # ---------------------------------------------------------------- per CLASS scatter
    sc = []
    for (stg, sd, sp), d in df[df.split.isin(("seen", "held"))].groupby(
            ["setting", "seed", "split"]):
        f = d[~d.base_correct]
        w = {}
        for m in (FIXED, GEN):
            x = f[f.method == m]
            if not len(x):
                continue
            w[m] = x.groupby("true_class").repaired.agg(["sum", "size"])
        if len(w) < 2:
            continue
        common = w[FIXED].index.intersection(w[GEN].index)
        for c in common:
            nf, ng = w[FIXED].loc[c, "size"], w[GEN].loc[c, "size"]
            if min(nf, ng) < MIN_N:
                continue
            rf = w[FIXED].loc[c, "sum"] / nf
            rg = w[GEN].loc[c, "sum"] / ng
            sc.append({"setting": stg, "seed": sd, "split": sp, "true_class": int(c),
                       "n_fail_FIX": int(nf), "n_fail_GEN": int(ng),
                       "RR_FIX": float(rf), "RR_GEN": float(rg), "delta": float(rg - rf),
                       # stratum defined by the FIXED arm alone: defining it on the difference
                       # would select on the very outcome being tested
                       "fixed_stratum": "low" if rf < .3 else ("medium" if rf < .7 else "high")})
    per_class = pd.DataFrame(sc)

    RM.write_section("P2", "P2 — input-specific vs shared patch, split by seen/unseen (raw)", f"""
`{FIXED}` is one delta shared by every input; `{GEN}` generates a delta per input. **Neither
uses the gate**, so the difference between them is the generator alone.

`split` is never collapsed:

| split | source | meaning |
|---|---|---|
| `seen` | `repair_support_seen` | the failures the patch was FITTED on |
| `held` | `bug_eval` | failures neither patch has seen |
| `clean` | `clean_test` | where regression is measured (`n_event` = regressions) |

On `seen`/`held`, `n` is the deployed model's failures and `n_event` is how many the method
fixed. Per-class columns use the true class and only classes with >= {MIN_N} failures.

**Sign warning for Block B.** `d_rate = rate(GEN) - rate(FIX)` in every split, but `rate` is a
REPAIR rate on `seen`/`held` and a REGRESSION rate on `clean`. So a positive `d_rate` is the
generator doing better on the first two and doing WORSE on the third. Do not read the three
splits with one sign convention.

The third block is per (setting, seed, class): both arms' repair rate for the same class, with
`fixed_stratum` set by the FIXED arm's own rate (low <.3, medium .3-.7, high >=.7). The stratum
is deliberately NOT defined on the difference — that would select on the outcome.
""", [("Block A — per cell, per method, per split", cell.sort_values(
            ["split", "setting", "seed", "method"])),
      ("Block B — the same cells paired (GEN minus FIX)", paired.sort_values(
            ["split", "setting", "seed"])),
      ("Block C — per class, both arms side by side", per_class.sort_values(
            ["split", "setting", "seed", "true_class"]))])

    out = ROOT / "outputs" / "p2_inputspecific"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_cell.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    per_class.to_csv(out / "per_class.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/  ({len(cell)} cells, {len(per_class)} classes)")


if __name__ == "__main__":
    main()
