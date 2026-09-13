#!/usr/bin/env python3
"""RQ2 (gate): extract DynaPatch's own pre vs pre+post classification quality and deployment
RR/Reg/CReg, from the ALREADY-PUBLISHED, last_affine-corrected markdown tables in
note/RQ3_DATA.md (RQ3.2 and RQ3.4), rather than re-reading the raw
outputs/gate_ablation_natural/natural_points_{pre,pre_post}.csv files directly.

Also extracts RQ3.3 (confidence/entropy response by patch outcome) and RQ3.9a (pre+post gate
feature importance) for the mechanism question T1/T2/T3 raise but do not answer: post-info
changes gate DECISIONS (T1/T2/T3), but what SIGNAL in post-info is the gate actually reading?
RQ3.3 shows successful repairs raise patched-response confidence and lower entropy while
regressions do the reverse; RQ3.9a shows the pre+post gate's post-only features (pP_max, dm_r,
dH, kl, dnorm_r, dp_c -- named by a `p`/`d` prefix over the pre-only features m_base_r, rho_worst,
pB_max, H_base, rho, pB_margin) are weighted accordingly. Both were already computed; this only
assembles them into the same paper table shape as T1/T2/T3.

Why not read the raw CSVs directly (2026-09-06): those files are dated 2026-09-04 23:37-49,
predating the 2026-09-05 last_affine re-fit that RQ3_DATA.md's own header note says RQ3.2/RQ3.4
specifically incorporate (vgg16/convnext_tiny re-fit and re-evaluated for all three evidence
arms). Re-parsing the markdown means this script reads the exact numbers already vetted and
published in the note, not a stale pre-last_affine snapshot.

This is a data-shape convenience script (parses two known table blocks with a fixed schema), not
a new experiment or a new computation -- every number here already exists in note/RQ3_DATA.md.

    .venv/bin/python scripts/extract_rq2_gate_dynapatch.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "note" / "RQ3_DATA.md"


def _extract_table(text: str, header_marker: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(header_marker))
    rows = []
    header = None
    for line in lines[start:]:
        if not line.strip().startswith("|"):
            if header is not None:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(cells)
    return pd.DataFrame(rows, columns=header)


def main() -> None:
    text = NOTE.read_text()

    clf = _extract_table(text, "### RQ3.2")
    for c in ("Accuracy", "Precision", "Recall", "F1"):
        clf[c] = clf[c].astype(float)
    clf = clf[clf.Evidence.isin(["pre", "pre+post"])].copy()
    clf.columns = [c.lower() for c in clf.columns]
    clf.to_csv(ROOT / "outputs" / "rq2_gate_classification_dynapatch.csv", index=False)

    dep = _extract_table(text, "### RQ3.4")
    for c in ("RR_held", "Reg", "CReg"):
        dep[c] = dep[c].astype(float)
    dep = dep[dep.Evidence.isin(["pre", "pre+post"])].copy()
    dep.columns = [c.lower() for c in dep.columns]
    dep.to_csv(ROOT / "outputs" / "rq2_gate_deployment_dynapatch.csv", index=False)

    print(f"classification: {len(clf)} rows -> outputs/rq2_gate_classification_dynapatch.csv")
    print(clf.groupby("evidence")[["accuracy", "precision", "recall", "f1"]].mean().round(4))
    print(f"\ndeployment: {len(dep)} rows -> outputs/rq2_gate_deployment_dynapatch.csv")
    print(dep.groupby("evidence")[["rr_held", "reg", "creg"]].mean().round(4))
    assert clf.setting.nunique() == 12 and dep.setting.nunique() == 12

    resp = _extract_table(text, "### RQ3.3")
    for c in ("Confidence change", "Entropy change"):
        resp[c] = resp[c].astype(float)
    resp.columns = [c.lower().replace(" ", "_") for c in resp.columns]
    resp_st = resp.groupby("outcome", as_index=False)[["confidence_change", "entropy_change"]].mean()
    resp_st.to_csv(ROOT / "outputs" / "rq2_gate_mechanism_response.csv", index=False)
    print(f"\nresponse signature (setting-balanced mean over {resp.setting.nunique()} settings) "
         f"-> outputs/rq2_gate_mechanism_response.csv")
    print(resp_st.round(4).to_string(index=False))
    assert resp.setting.nunique() == 12

    # RQ3.9a's header cell is literally "Mean |coefficient|" (the bars are the metric name, not
    # markdown column separators), which breaks a naive split("|") on the header row only --
    # every DATA row has exactly 5 plain cells, so skip the header/separator by content and
    # parse data rows directly instead of reusing `_extract_table`.
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("### RQ3.9a"))
    imp_rows = []
    for line in lines[start:]:
        if not line.strip().startswith("|"):
            if imp_rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] in ("Method",) or set(cells[0]) <= {"-", ":"}:
            continue
        imp_rows.append(cells)
    imp = pd.DataFrame(imp_rows, columns=["method", "feature", "mean_coefficient",
                                          "mean_abs_coefficient", "n_settings"])
    for c in ("mean_coefficient", "mean_abs_coefficient"):
        imp[c] = imp[c].astype(float)
    POST_FEATURES = {"pP_max", "dm_r", "dH", "kl", "dnorm_r", "dp_c"}
    dp = imp[(imp.method == "DynaPatch") & (imp.feature.isin(POST_FEATURES))].copy()
    dp = dp.sort_values("mean_coefficient", key=lambda s: s.abs(), ascending=False)
    dp.to_csv(ROOT / "outputs" / "rq2_gate_mechanism_features.csv", index=False)
    print(f"\nDynaPatch's pre+post gate, post-only feature coefficients "
         f"-> outputs/rq2_gate_mechanism_features.csv")
    print(dp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
