#!/usr/bin/env python3
"""RQ1 evidence #2 -- is FixedPatch's repair concentrated in a few classes?

Scope note: this is deliberately NOT `analysis_p1_regression.py`. That script's 14-method
comparison (Arachne, FullFT, DistrRep, DynaPatch-gated, ...) backs the separate "why does our
method beat the baseline zoo" mechanism analysis in note/RESULTS_MECHANISM.md. RQ1's claim is
narrower: FixedPatch (one shared patch) vs DynaPatch-NoGate (per-input patch, same architecture,
same training, gate off -- see scripts/names.py's `DPNoGate`). Mixing in the other baselines here
answers a different question and must not happen again -- see note/PITFALLS.md.

Two numbers per (method, setting, seed), both about the SAME held-out failure pool RQ1.1 reports
repair rate on:

  top20_share   the literal, student-readable number: share of this method's OWN repaired
                failures contributed by its own top 20% of classes (ranked by that method's
                repair count). This is what was asked for -- report it as the headline.
  repair_spread the count-robust null check from analysis_p1_regression.py (imported, not
                reimplemented): distinct classes reached, divided by the expected count under a
                random draw of the same size from the failure pool. Only needed because top20_share
                alone conflates "concentrated" with "few total repairs"; here FixedPatch and
                DynaPatch-NoGate have comparable n per setting (see RQ1.1's Mean n column) so that
                confound is small, but report both rather than assume it away.

Source: outputs/sample_frame.csv.gz, method column filtered to {FixedPatch, DynaPatch (ungated)}
-- the same file RQ1_DATA.md's RQ1.1-1.6 tables read, already rebuilt for the 2026-09-05
last_affine change on vgg16/convnext_tiny (verified: this file's FixedPatch/DynaPatch (ungated)
gtsrb/vgg16 held RR reproduce RQ1_DATA.md's RQ1.1 row, 0.6088/0.6000).

    .venv/bin/python scripts/analysis_rq1_concentration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analysis_p1_regression import spread_vs_null  # noqa: E402 -- reuse, do not reimplement

OUT = ROOT / "outputs" / "rq1_concentration"
METHODS = ["FixedPatch", "DynaPatch (ungated)"]


def top_k_share(counts: np.ndarray, frac: float = 0.2) -> float:
    """Share of this method's OWN total contributed by its own top `frac` of classes.

    Valid as a headline number only when comparing arms with comparable n (true here -- see
    module docstring); analysis_p1_regression.py's `conc()` rejects it for the 14-method
    comparison specifically because n varies by orders of magnitude there.
    """
    if counts.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * len(counts))))
    return float(np.sort(counts)[::-1][:k].sum() / counts.sum())


def per_cell(d: pd.DataFrame) -> dict | None:
    h = d[(d.split == "held") & (~d.base_correct)]
    if not len(h):
        return None
    g = h.groupby("true_class").repaired.sum()
    rp = h[h.repaired]
    spread, z = spread_vs_null(rp.true_class.to_numpy(), h.true_class.to_numpy())
    return {"n_classes_failed": int((h.groupby("true_class").size() > 0).sum()),
            "n_repaired": int(len(rp)),
            "top20_share": top_k_share(g.to_numpy(), 0.2),
            "repair_spread": spread, "repair_spread_z": z}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "outputs/sample_frame.csv.gz")
    df = df[df.method.isin(METHODS)]
    recs = []
    for (m, stg, sd), d in df.groupby(["method", "setting", "seed"]):
        r = per_cell(d)
        if r:
            recs.append({"method": m, "setting": stg, "seed": sd, **r})
    cell = pd.DataFrame(recs)
    cell.to_csv(OUT / "per_cell.csv", index=False)
    st = cell.groupby(["method", "setting"]).median(numeric_only=True).reset_index()
    st.to_csv(OUT / "per_setting.csv", index=False)

    import raw_md as RM
    RM.write_section("RQ1_CONC", "RQ1 concentration -- FixedPatch vs DynaPatch-NoGate (raw, per cell)", """
Unit is the TRUE CLASS, same population as RQ1.1's held-split repair rate. One row per (method,
setting, seed); FixedPatch and DynaPatch (ungated) = DynaPatch-NoGate only -- see
scripts/analysis_rq1_concentration.py's module docstring for why the other baselines are
deliberately absent here.

| column | meaning |
|---|---|
| `top20_share` | share of this method's own repairs contributed by its own top-20%-of-classes |
| `repair_spread` / `_z` | count-robust null check (imported from analysis_p1_regression.py) |
""", [("", cell[["method", "setting", "seed", "n_classes_failed", "n_repaired",
                 "top20_share", "repair_spread", "repair_spread_z"]]
       .sort_values(["method", "setting", "seed"]))])

    lines = ["| Setting | FP n_rep | FP top20 share | FP spread (z) | "
             "DPNoGate n_rep | DPNoGate top20 share | DPNoGate spread (z) |",
             "|---|---|---|---|---|---|---|"]
    for stg in sorted(st.setting.unique()):
        row = st[st.setting == stg].set_index("method")
        if not {"FixedPatch", "DynaPatch (ungated)"}.issubset(row.index):
            continue
        fp, dp = row.loc["FixedPatch"], row.loc["DynaPatch (ungated)"]
        lines.append(
            f"| {stg} | {fp.n_repaired:.0f} | {fp.top20_share:.3f} | "
            f"{fp.repair_spread:.3f} ({fp.repair_spread_z:.1f}) | "
            f"{dp.n_repaired:.0f} | {dp.top20_share:.3f} | "
            f"{dp.repair_spread:.3f} ({dp.repair_spread_z:.1f}) |")
    print("\n".join(lines))
    print(f"\n[written] {OUT.relative_to(ROOT)}/per_cell.csv, per_setting.csv; "
          "note/ANALYSIS_RAW.md#RQ1_CONC")


if __name__ == "__main__":
    main()
