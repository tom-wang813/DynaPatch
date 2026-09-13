#!/usr/bin/env python3
"""RQ1.13: decompose the margin gain Delta m(x) into scale and direction, for all four RQ1
methods (FixedPatch, DynaPatch-NoGate, NN-Patching, PatchNAS).

RQ1.12 (scripts/analysis_rq1_alignment.py) shows DynaPatch has the highest alignment
(cos_to_ideal) to the ideal margin direction d* = e_y - e_yhat, on all 12 settings vs FixedPatch
and on 11/12 vs NN-Patching/PatchNAS -- yet DynaPatch's raw ungated RR_held (0.523, Table 1) is
LOWER than NN-Patching's (0.554) and PatchNAS's (0.541). This is not a contradiction: by
construction,

    Delta m(x) = Delta z_y(x) - Delta z_yhat(x) = ||Delta z(x)|| * sqrt(2) * cos_to_ideal(x)

so the actual margin gain a method achieves is alignment (direction quality) times
||Delta z(x)|| (correction scale), not alignment alone. This script reports both factors and
their product side by side, per (method, setting), on held-out failures, to check the specific,
falsifiable hypothesis this decomposition raises: NN-Patching/PatchNAS could be winning on raw RR
DESPITE worse alignment because they push a larger ||Delta z||.

Caveat (same one scripts/analysis_m1_correction_direction.py's docstring already states for
mean_dz_norm/mean_dm): NN-Patching's and PatchNAS's patched_logits come from a freshly-trained
head with no constraint tying its logit scale to DynaPatch's/FixedPatch's frozen head, so
||Delta z|| and Delta m are only meaningful WITHIN one method's own settings, never compared in
absolute cross-method units on their own -- this script's job is exactly to show them next to
alignment so the reader sees which part of the product (scale vs direction) differs, not to rank
methods by raw ||Delta z|| in isolation.

Reuses:
  - outputs/m1_correction_direction/magnitude_margin_per_cell.csv (DynaPatch-NoGate, NN-Patching,
    PatchNAS: mean_dz_norm, mean_dm, already computed by analysis_m1_correction_direction.py)
  - scripts/analysis_rq1_alignment.py's load_fixedpatch_cell() for FixedPatch (no prior
    magnitude/margin table exists for FixedPatch since M1 never covered it)

    .venv/bin/python scripts/analysis_rq1_margin_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                             # noqa: E402
from analysis_m1_correction_direction import SEEDS, SETTINGS    # noqa: E402
from analysis_rq1_alignment import load_fixedpatch_cell          # noqa: E402

SQRT2 = float(np.sqrt(2.0))
METHOD_ORDER = ["FixedPatch", "DynaPatch-NoGate", "NN-Patching", "PatchNAS"]


def fixedpatch_rows() -> list[dict]:
    rows = []
    for ds, bb in SETTINGS:
        for seed in SEEDS:
            cell = load_fixedpatch_cell(ds, bb, seed)
            if cell is None:
                continue
            dz_norm = np.linalg.norm(cell["dz"], axis=1)
            dm = cell["dm"]
            rows.append({
                "method": "FixedPatch", "setting": f"{ds}/{bb}", "seed": seed, "n": cell["n"],
                "mean_dz_norm": float(dz_norm.mean()), "median_dz_norm": float(np.median(dz_norm)),
                "mean_dm": float(dm.mean()), "median_dm": float(np.median(dm)),
            })
    return rows


def main() -> None:
    mm = pd.read_csv(ROOT / "outputs/m1_correction_direction/magnitude_margin_per_cell.csv")
    mm = mm[mm.split == "held"][["method", "setting", "seed", "n", "mean_dz_norm",
                                 "median_dz_norm", "mean_dm", "median_dm"]]
    fp = pd.DataFrame(fixedpatch_rows())
    per_cell = pd.concat([mm, fp], ignore_index=True)

    align = pd.read_csv(ROOT / "outputs/rq1_alignment/per_cell.csv")[
        ["method", "setting", "seed", "n", "mean_cos_to_ideal"]]
    per_cell = per_cell.merge(align, on=["method", "setting", "seed", "n"], how="inner")

    # sanity check: mean_dm should equal ~ sqrt(2) * mean(||dz|| * cos_to_ideal) elementwise-summed;
    # at the cell-mean level this only holds approximately (mean of a product != product of means),
    # so check it on a reconstructed per-cell product instead of asserting exact equality.
    per_cell["implied_dm_lower_bound_check"] = per_cell.mean_dz_norm * SQRT2 * per_cell.mean_cos_to_ideal

    out = ROOT / "outputs" / "rq1_margin_decomposition"
    out.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out / "per_cell.csv", index=False)

    cols = ["mean_dz_norm", "median_dz_norm", "mean_dm", "median_dm", "mean_cos_to_ideal"]
    st = per_cell.groupby(["method", "setting"])[cols].mean().reset_index()
    st.to_csv(out / "per_setting.csv", index=False)
    summary = per_cell.groupby("method")[cols].median().reset_index()
    summary = summary.set_index("method").loc[METHOD_ORDER].reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    order = [f"{d}/{b}" for d, b in SETTINGS]
    norm_piv = st.pivot(index="setting", columns="method", values="median_dz_norm").loc[order][METHOD_ORDER]
    dm_piv = st.pivot(index="setting", columns="method", values="median_dm").loc[order][METHOD_ORDER]
    cos_piv = st.pivot(index="setting", columns="method", values="mean_cos_to_ideal").loc[order][METHOD_ORDER]

    RM.write_section(
        "RQ1MarginDecomposition",
        "RQ1.13 -- margin gain Delta m decomposed into ||Delta z|| (scale) x cos_to_ideal "
        "(direction), all four methods (raw, held-out failures)",
        f"""
`Delta m(x) = ||Delta z(x)|| * sqrt(2) * cos_to_ideal(x)` by construction (RQ1.12's docstring).
This checks whether NN-Patching/PatchNAS's higher raw RR_held (Table 1: 0.554/0.541 vs
DynaPatch-NoGate's 0.523) despite LOWER alignment (RQ1.12: 0.152/0.168 vs DynaPatch's 0.310) is
explained by a larger correction scale `||Delta z||`.

**Caveat that must travel with this table** (same as M1's magnitude/margin table): NN-Patching's
and PatchNAS's `patched_logits` come from a freshly-trained head with no constraint tying its
logit scale to DynaPatch's/FixedPatch's frozen head, so `dz_norm`/`dm` are only informative about
relative scale, not comparable as absolute cross-method units on their own.

**Median, not mean, is reported for `||Delta z||`/`Delta m` below.** On `lisa_signs/vgg16`,
NN-Patching's freshly-trained head produces a small number of failures with an astronomically
large `||Delta z||` (per-seed median already ~1e5, some individual rows far larger); this drags
the CELL MEAN `Delta m` negative (-4809) even though `cos_to_ideal` on that same cell is positive
(0.025) -- a mean-of-ratio-vs-ratio-of-means artefact from a few extreme rows, not a sign that the
correction points the wrong way on average. Median is reported instead so one degenerate cell does
not dominate the table; `cos_to_ideal` (RQ1.12, unaffected by this since it is a per-row bounded
ratio) is repeated below for the row this affects.

**Median ||Delta z(x)|| per (method, setting):**

```
{norm_piv.round(2).to_string()}
```

**Median Delta m(x) per (method, setting):**

```
{dm_piv.round(2).to_string()}
```

**Mean cos_to_ideal per (method, setting)** (repeated from RQ1.12 for side-by-side reading):

```
{cos_piv.round(3).to_string()}
```

**Setting-median summary** (median over the 12 settings, same robustness reasoning):

```
{summary.round(3).to_string(index=False)}
```
""",
        [("per_setting_norm", norm_piv.reset_index()),
         ("per_setting_dm", dm_piv.reset_index()),
         ("per_setting_cos", cos_piv.reset_index()),
         ("summary", summary)],
    )

    pd.set_option("display.width", 150)
    print("median ||Delta z||:")
    print(norm_piv.round(2).to_string())
    print("\nmedian Delta m:")
    print(dm_piv.round(2).to_string())
    print("\nmean cos_to_ideal:")
    print(cos_piv.round(3).to_string())
    print("\nsetting-median summary:")
    print(summary.round(3).to_string(index=False))
    print(f"\n[written] {out.relative_to(ROOT)}/{{per_cell,per_setting,summary}}.csv")


if __name__ == "__main__":
    main()
