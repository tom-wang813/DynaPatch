#!/usr/bin/env python3
"""Emit the paper's own 9 result tables (RQ1-RQ4), either as paste-ready LaTeX (the paper's own
house style) or as plain Markdown for quick terminal reading.

House style (LaTeX):
  * settings named by the D-A scheme (G-RN ...), three dataset blocks separated by
    \\specialrule{\\lightrulewidth}{2pt}{2pt}
  * the best cell of each row in \\textbf{}
  * a Median (G/T) row and a Mean (all) row at the foot of every per-setting table
  * method columns use the \\HeadFT / \\FullFT / \\Arachne / ... macros from names.py

Everything is read from small CSVs/JSONs -- no feature tensors are loaded, so this is safe to
run while a sweep is using the machine.

2026-09-21: cut down from an earlier, much larger version that also rendered a long tail of
mechanism/ablation tables (type/ideal alignment, clean-margin decomposition, gate-mechanism
coefficients, postinfo flip counts, lambda-mechanism removal tables, the RQ4 frontier figure...)
that trace to RQ1.x/RQ2.x/RQ3.x/RQ4.x exploratory sub-questions, not to a table in the paper
itself. This file now emits exactly the 9 tables (+1 figure) the paper's Results section has;
each `_rq<N>_...()` private helper is named for the RQ NUMBER IT FEEDS, and the four public entry
points -- rq1()/rq2()/rq3()/rq4() -- are exactly the paper's own RQ1-4. See git history for the
longer version and its extra tables if a reviewer question needs one of them reconstructed.

2026-09-21: default output is now Markdown, printed to the terminal, nothing written to disk --
for a quick "does this number still look right" read. `--format latex` (or `--outdir`, which
implies it) switches to the paper's exact LaTeX, still the source of truth for what actually
gets pasted into the paper.

  RQ1  comparison with existing methods (FullFT/HeadFT/Arachne/DistrRep/NNPatch/PatchNAS/DP):
       tab:rq1_rr (repair rate per setting), tab:rq1_summary (mean RR/Reg/CReg per method)
  RQ2  input-specific patch (FixedPatch vs DynaPatch-NoGate, causal reassignment, alignment):
       tab:rq2_ungated_persetting, tab:rq2_ungated_summary, tab:rq2_direction, tab:rq2_norm
  RQ3  post-information (DynaPatch's own gate, pre-only vs pre+post):
       tab:rq3_gate_clf (classification quality), tab:rq3_gate_effect (RR/Reg/CReg), fig:features
  RQ4  regression-cost controllability (DynaPatch's own gate, cost-weighted lambda sweep):
       tab:rq4_summary

Usage:
  .venv/bin/python scripts/paper_tables.py                       # markdown, prints, no files
  .venv/bin/python scripts/paper_tables.py --rq 4                 # just RQ4
  .venv/bin/python scripts/paper_tables.py --format latex         # paper LaTeX, prints only
  .venv/bin/python scripts/paper_tables.py --outdir note/tables   # paper LaTeX, also writes .tex
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names as N  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Set once by main() before any _rq*() call; read by table()/method_table() and the three
# hand-built (non-table()/method_table()) renderers below. A module global, not a parameter
# threaded through every helper, because this is a CLI script with one output format per run,
# not a library -- threading `fmt` through all ~15 helpers would just be noise.
FMT = "latex"


def blocks() -> list[list[tuple[str, str, str]]]:
    """SETTING_ORDER split into the three dataset blocks."""
    out, cur, ds = [], [], None
    for d, b, lab in N.SETTING_ORDER:
        if ds is not None and d != ds:
            out.append(cur); cur = []
        cur.append((d, b, lab)); ds = d
    out.append(cur)
    return out


def delatex(s: str) -> str:
    """Best-effort LaTeX -> plain text, for the markdown terminal view ONLY -- the LaTeX
    strings themselves (captions/headers/notes) stay the single source of truth; this never
    writes back to them. Not meant to be exact -- good enough to read at a glance."""
    s = s.replace(r"$\uparrow$", "\u2191").replace(r"$\downarrow$", "\u2193")
    s = re.sub(r"\\makecell\{([^{}]*)\}", lambda m: m.group(1).replace("\\\\", " "), s)
    s = re.sub(r"\\(?:mathrm|text|textnormal|textbf)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)\{\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)\{([^{}]*)\}", r"\1(\2)", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def fmt(v, dec: int, bold: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{dec}f}"
    if not bold:
        return s
    return f"**{s}**" if FMT == "markdown" else f"\\textbf{{{s}}}"


def table(caption: str, label: str, headers: list[str], rows: dict[str, list],
          dec: int, bold_max: bool | None, foot: list[tuple[str, list]],
          colspec: str | None = None, note: str = "",
          bold_groups: list[list[int]] | None = None) -> str:
    """`bold_groups` lists the column index sets that COMPETE with each other; the best cell in
    each group is bolded. Pass None/[] where the columns are different metrics (Acc vs Prec vs
    Rec vs F1) or different operating points -- bolding the row maximum there says nothing, it
    just marks whichever metric happens to run highest.
    """
    def row_marks(vals: list) -> list[bool]:
        mark = [False] * len(vals)
        for grp in (bold_groups or []):
            good = [(i, vals[i]) for i in grp
                    if i < len(vals) and vals[i] is not None and not np.isnan(vals[i])]
            if not good:
                continue
            star = (max if bold_max else min)(v for _, v in good)
            for i, v in good:
                if abs(v - star) < 1e-9:
                    mark[i] = True
        return mark

    if FMT == "markdown":
        hdr = [delatex(h) for h in headers]
        L = [f"**{delatex(caption)}**  ({label})", "",
             "| Setting | " + " | ".join(hdr) + " |",
             "|" + "---|" * (len(hdr) + 1)]
        for blk in blocks():
            for _, _, lab in blk:
                vals = rows.get(lab, [None] * len(headers))
                mark = row_marks(vals)
                L.append(f"| {lab} | " + " | ".join(
                    fmt(v, dec, mark[i]) for i, v in enumerate(vals)) + " |")
        for name, vals in foot:
            L.append(f"| **{name}** | " + " | ".join(fmt(v, dec) for v in vals) + " |")
        if note:
            L += ["", delatex(note)]
        return "\n".join(L) + "\n"

    spec = colspec or ("l" + "c" * len(headers))
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{\textnormal{{{caption}}}}}", rf"\label{{{label}}}",
         r"\small", r"\setlength{\tabcolsep}{4.0pt}",
         rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         "Setting & " + " & ".join(headers) + r" \\", r"\midrule"]
    for bi, blk in enumerate(blocks()):
        if bi:
            L.append(r"\specialrule{\lightrulewidth}{2pt}{2pt}")
        for _, _, lab in blk:
            vals = rows.get(lab, [None] * len(headers))
            mark = row_marks(vals)
            L.append(f"{lab} & " + " & ".join(
                fmt(v, dec, mark[i]) for i, v in enumerate(vals)) + r" \\")
    L.append(r"\midrule")
    for name, vals in foot:
        L.append(f"{name} & " + " & ".join(fmt(v, dec) for v in vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(rf"\vspace{{2pt}}{{\footnotesize {note}}}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def agg_rows(rows: dict[str, list], ncol: int) -> list[tuple[str, list]]:
    gt = [lab for _, _, lab in N.SETTING_ORDER if not lab.startswith("L-")]
    allx = [lab for _, _, lab in N.SETTING_ORDER]
    def col(keys, i, f):
        v = [rows[k][i] for k in keys if k in rows and rows[k][i] is not None
             and not np.isnan(rows[k][i])]
        return f(v) if v else float("nan")
    return [("Median (G/T)", [col(gt, i, st.median) for i in range(ncol)]),
            ("Mean (all)", [col(allx, i, st.mean) for i in range(ncol)])]


def method_table(caption: str, label: str, headers: list[str], row_labels: list[str],
                 rows: list[list], dec: int | list[int], colspec: str | None = None,
                 note: str = "", row_header: str = "Method") -> str:
    """Small booktabs table indexed by METHOD (or another row concept, via `row_header`), not by
    setting. `dec` may be one int (all columns) or a list (one per column)."""
    decs = dec if isinstance(dec, list) else [dec] * len(headers)

    if FMT == "markdown":
        hdr = [delatex(h) for h in headers]
        L = [f"**{delatex(caption)}**  ({label})", "",
             f"| {row_header} | " + " | ".join(hdr) + " |",
             "|" + "---|" * (len(hdr) + 1)]
        for lab, vals in zip(row_labels, rows):
            L.append(f"| {delatex(lab)} | " + " | ".join(
                fmt(v, d) for v, d in zip(vals, decs)) + " |")
        if note:
            L += ["", delatex(note)]
        return "\n".join(L) + "\n"

    spec = colspec or ("l" + "c" * len(headers))
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{\textnormal{{{caption}}}}}", rf"\label{{{label}}}",
         r"\small", r"\setlength{\tabcolsep}{4.0pt}",
         rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         f"{row_header} & " + " & ".join(headers) + r" \\", r"\midrule"]
    for lab, vals in zip(row_labels, rows):
        L.append(f"{lab} & " + " & ".join(fmt(v, d) for v, d in zip(vals, decs)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(rf"\vspace{{2pt}}{{\footnotesize {note}}}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ RQ1

def _natural_points() -> pd.DataFrame:
    """One row per (dataset, backbone), pooled over seeds, at theta=0 (commit iff the fitted
    gate's own score favours beneficial over harmful) -- no target r, no threshold search, no
    calibration split of any kind. The ONLY admissible source for a reported gated-DynaPatch
    number (names.py:GATE_NATURAL_POINT)."""
    d = pd.read_csv(ROOT / N.GATE_NATURAL_POINT)
    d[["dataset", "backbone"]] = d.setting.str.split("/", n=1, expand=True)
    return d.groupby(["dataset", "backbone"], as_index=False)[
        ["RR_held", "Reg", "CReg"]].mean()


def _rq1_baseline_rows() -> dict[str, dict[str, dict[str, list]]]:
    """RR_held/Reg/CReg per (method, setting), the 6 comparison baselines --
    outputs/rq1/comparison_baselines.csv (FullFT/HeadFT/Arachne/DistrRep) and
    outputs/rq1/baseline_prior_patches_mlp.json (NN-Patching/PatchNAS, their own best/routed
    operating point -- this is the RQ1 comparison-table number, a DIFFERENT operating point
    from RQ2's raw-ungated one; see _rq2_ungated_summary())."""
    per: dict[str, dict[str, dict[str, list]]] = {}
    for row in csv.DictReader((ROOT / "outputs/rq1/comparison_baselines.csv").open()):
        key = N.key_for_rq4_row(row["method"])
        if key is None:
            continue
        d = per.setdefault(key, {}).setdefault(row["setting"], {})
        for m in ("RR_held", "Reg", "CReg"):
            if row[m] not in ("", None):
                d.setdefault(m, []).append(float(row[m]))
    j = json.loads((ROOT / "outputs/rq1/baseline_prior_patches_mlp.json").read_text())
    for key, jk in (("NNPatch", "NN-Patching"), ("PatchNAS", "PatchNAS")):
        for stg, seeds in j.get(jk, {}).items():
            d = per.setdefault(key, {}).setdefault(stg, {})
            for v in seeds.values():
                for m in ("RR_held", "Reg", "CReg"):
                    d.setdefault(m, []).append(float(v[m]))
    return per


def _rq1_rr() -> str:
    per, nat = _rq1_baseline_rows(), _natural_points()
    ORDER = ["FullFT", "HeadFT", "Arachne", "DistrRep", "NNPatch", "PatchNAS"]
    hdr = [N.latex(k) for k in ORDER] + [r"\DP"]
    rr = {}
    for ds, bb, lab in N.SETTING_ORDER:
        stg = f"{ds}/{bb}"
        def g(key):
            v = per.get(key, {}).get(stg, {}).get("RR_held")
            return st.mean(v) if v else float("nan")
        p = nat[(nat.dataset == ds) & (nat.backbone == bb)]
        pt_rr = float(p.RR_held.iloc[0]) if len(p) else float("nan")
        rr[lab] = [g(k) for k in ORDER] + [pt_rr]
    return table(
        r"\textbf{RQ1:} Repair rate across the 12 settings.",
        "tab:rq1_rr", hdr, rr, 3, True, agg_rows(rr, len(hdr)),
        bold_groups=[list(range(len(hdr)))])


def _rq1_summary() -> str:
    per, nat = _rq1_baseline_rows(), _natural_points()
    ORDER = [("FullFT", r"\FullFT"), ("HeadFT", r"\HeadFT"), ("Arachne", r"\Arachne"),
            ("DistrRep", r"\DistrRep"), ("NNPatch", r"\NNPatch"), ("PatchNAS", r"\PatchNAS")]
    def method_mean(key: str, metric: str) -> float:
        vals = [st.mean(d[metric]) for d in per.get(key, {}).values() if metric in d]
        return st.mean(vals) if vals else float("nan")
    row_labels = [lab for _, lab in ORDER] + [r"\DP"]
    rows = [[method_mean(k, "RR_held"), method_mean(k, "Reg"), method_mean(k, "CReg")]
            for k, _ in ORDER]
    rows.append([nat.RR_held.mean(), nat.Reg.mean(), nat.CReg.mean()])
    return method_table(
        r"\textbf{RQ1:} Mean repair and regression rates across the 12 settings.",
        "tab:rq1_summary", [r"\RR{} $\uparrow$", r"\Reg{} $\downarrow$", r"\CReg{} $\downarrow$"],
        row_labels, rows, [3, 4, 4])


def rq1() -> str:
    """Paper RQ1: comparison with existing methods."""
    return "\n".join([_rq1_rr(), _rq1_summary()])


# ------------------------------------------------------------------ RQ2

def _rq2_ungated_persetting() -> str:
    """T1: FixedPatch vs DynaPatch-NoGate, per setting, both at their raw ungated (always-apply)
    operating point -- outputs/rq2/ungated_fixedpatch_dynapatch.csv. Custom 2-level header
    (method spans 3 metric columns each), so built directly rather than via `table()`."""
    a = pd.read_csv(ROOT / "outputs/rq2/ungated_fixedpatch_dynapatch.csv")
    a = a.pivot_table(index=["method", "setting"], columns="metric",
                      values="value").reset_index()
    def g(method: str, stg: str, metric: str) -> float:
        r = a[(a.method.str.startswith(method)) & (a.setting == stg)]
        return float(r[metric].iloc[0]) if len(r) else float("nan")

    fp_rr, dp_rr = {}, {}
    for blk in blocks():
        for ds, bb, lab in blk:
            stg = f"{ds}/{bb}"
            fp_rr[lab] = [g("FixedPatch", stg, m) for m in ("RR_held", "Reg", "CReg")]
            dp_rr[lab] = [g("DynaPatch-NoGate", stg, m) for m in ("RR_held", "Reg", "CReg")]

    def mean_of(d: dict, i: int) -> float:
        return st.mean(v[i] for v in d.values())
    mean_fp = [mean_of(fp_rr, i) for i in range(3)]
    mean_dp = [mean_of(dp_rr, i) for i in range(3)]

    def cells(fp: list, dp: list, decs=(3, 4, 4)) -> tuple[list, list]:
        # higher-is-better for RR (index 0), lower-is-better for Reg/CReg -- a tie bolds BOTH
        # sides (independent comparisons), not "whichever side isn't excluded by the other".
        fp_wins = [fp[0] >= dp[0]] + [fp[i] <= dp[i] for i in (1, 2)]
        dp_wins = [dp[0] >= fp[0]] + [dp[i] <= fp[i] for i in (1, 2)]
        return ([fmt(fp[i], decs[i], fp_wins[i]) for i in range(3)],
                [fmt(dp[i], decs[i], dp_wins[i]) for i in range(3)])

    caption = r"\textbf{RQ2:} Per-setting repair, regression, and critical-class regression rates without the gate."
    label = "tab:rq2_ungated_persetting"

    if FMT == "markdown":
        metric_hdr = ["RR\u2191", "Reg\u2193", "CReg\u2193"]
        L = [f"**{delatex(caption)}**  ({label})", "",
             "| Setting | " + " | ".join(f"FPa {m}" for m in metric_hdr)
             + " | " + " | ".join(f"DPNoGate {m}" for m in metric_hdr) + " |",
             "|" + "---|" * 7]
        for blk in blocks():
            for _, _, lab in blk:
                fpc, dpc = cells(fp_rr[lab], dp_rr[lab])
                L.append(f"| {lab} | " + " | ".join(fpc + dpc) + " |")
        fpc, dpc = cells(mean_fp, mean_dp)
        L.append(f"| **Mean** | " + " | ".join(fpc + dpc) + " |")
        return "\n".join(L) + "\n"

    L = [r"\begin{table}[t]", "", r"\centering",
         rf"\caption{{\textnormal{{{caption}}}}}",
         rf"\label{{{label}}}", r"\footnotesize",
         r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}} l ccc ccc @{}}", r"\toprule",
         r"& \multicolumn{3}{c}{\FPa{}} & \multicolumn{3}{c}{\DPNoGate{}} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
         r"Setting & \RR{} $\uparrow$ & \Reg{} $\downarrow$ & \CReg{} $\downarrow$ & "
         r"\RR{} $\uparrow$ & \Reg{} $\downarrow$ & \CReg{} $\downarrow$ \\", r"\midrule"]
    for bi, blk in enumerate(blocks()):
        if bi:
            L.append(r"\specialrule{\lightrulewidth}{2pt}{2pt}")
        for _, _, lab in blk:
            fpc, dpc = cells(fp_rr[lab], dp_rr[lab])
            L.append(f"{lab} & " + " & ".join(fpc + dpc) + r" \\")
    L.append(r"\midrule")
    fpc, dpc = cells(mean_fp, mean_dp)
    L.append("Mean & " + " & ".join(fpc + dpc) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L) + "\n"


def _rq2_ungated_summary() -> str:
    """T2: FixedPatch/DynaPatch-NoGate (raw ungated, as T1) plus NN-Patching/PatchNAS at THEIR
    raw ungated (always-apply, route_rate=1.0) operating point --
    outputs/rq2/baseline_prior_patches_mlp_r5_ungated.json. NOT their own gate's routed
    prediction (that is RQ3's transplant question, a different table)."""
    a = pd.read_csv(ROOT / "outputs/rq2/ungated_fixedpatch_dynapatch.csv")
    a = a.pivot_table(index="method", columns="metric", values="value",
                      aggfunc="mean").reset_index()
    def own(method: str, metric: str) -> float:
        r = a[a.method.str.startswith(method)]
        return float(r[metric].iloc[0]) if len(r) else float("nan")
    j = json.loads((ROOT / "outputs/rq2/baseline_prior_patches_mlp_r5_ungated.json").read_text())
    def ungated(method: str, metric: str) -> float:
        vals = [st.mean(m[metric] for m in by_seed.values())
                for by_seed in j.get(method, {}).values()]
        return st.mean(vals) if vals else float("nan")
    ORDER = [("FixedPatch", r"\FPa{}", own), ("DynaPatch-NoGate", r"\DPNoGate{}", own),
             ("NN-Patching", r"\NNPatch{}", ungated), ("PatchNAS", r"\PatchNAS{}", ungated)]
    row_labels = [lab for _, lab, _ in ORDER]
    rows = [[fn(m, "RR_held"), fn(m, "Reg"), fn(m, "CReg")] for m, _, fn in ORDER]
    return method_table(
        r"\textbf{RQ2:} Mean repair, regression, and critical-class regression rates without "
        r"the gate.", "tab:rq2_ungated_summary",
        [r"\RR{} $\uparrow$", r"\Reg{} $\downarrow$", r"\CReg{} $\downarrow$"],
        row_labels, rows, [4, 4, 4])


def _rq2_direction() -> str:
    """T3: causal reassignment (RQ1.8/1.9) -- destroying which failure gets which correction
    (direction shuffle) collapses RR_held for every per-input method; destroying only magnitude
    does not. outputs/rq2/patch_reassignment_{,nnpatching_,patchnas_}v1/summary.json."""
    files = {r"\DPNoGate{}": "outputs/rq2/patch_reassignment_v1/summary.json",
            r"\NNPatch{}": "outputs/rq2/patch_reassignment_nnpatching_v1/summary.json",
            r"\PatchNAS{}": "outputs/rq2/patch_reassignment_patchnas_v1/summary.json"}
    row_labels, rows = [], []
    for lab, path in files.items():
        j = json.loads((ROOT / path).read_text())
        agg = j["aggregate"]
        dshuf = agg["global_direction_shuffle"]
        mshuf = agg["global_magnitude_shuffle"]
        row_labels.append(lab)
        rows.append([-dshuf["median_original_minus_mean_RR"],
                     -mshuf["median_original_minus_mean_RR"]])
    return method_table(
        r"\textbf{RQ2:} Mean repair-rate decrease after direction and magnitude reassignment "
        r"across all settings.", "tab:rq2_direction", ["Direction", "Magnitude"],
        row_labels, rows, 3, colspec="lcc")


def _rq2_norm() -> str:
    """T4: directional alignment, logit-change magnitude, and repair margin gain --
    outputs/rq2/alignment/summary.csv (mean cos_to_ideal) and
    outputs/rq2/margin_decomposition/summary.csv (median ||dz||, median dm)."""
    dec = pd.read_csv(ROOT / "outputs/rq2/margin_decomposition/summary.csv").set_index("method")
    align = pd.read_csv(ROOT / "outputs/rq2/alignment/summary.csv").set_index("method")
    order = [("FixedPatch", r"\FPa{}"), ("DynaPatch-NoGate", r"\DPNoGate{}"),
            ("NN-Patching", r"\NNPatch{}"), ("PatchNAS", r"\PatchNAS{}")]
    row_labels = [lab for _, lab in order]
    rows = [[align.loc[m, "mean"], dec.loc[m, "median_dz_norm"], dec.loc[m, "median_dm"]]
            for m, _ in order]
    return method_table(
        r"\textbf{RQ2:} Directional alignment, logit-change magnitude, and repair margin gain.",
        "tab:rq2_norm",
        [r"\makecell{Alignment \\ $\alpha_M(x) \uparrow$}", r"$\|\Delta \ell_M\|_2$",
         r"$\Delta m_M^{\mathrm{rep}}$"],
        row_labels, rows, [3, 1, 1], colspec="l ccc")


def rq2() -> str:
    """Paper RQ2: input-specific patch."""
    return "\n".join([_rq2_ungated_persetting(), _rq2_ungated_summary(),
                       _rq2_direction(), _rq2_norm()])


# ------------------------------------------------------------------ RQ3

def _rq3_gate_clf() -> str:
    """T1: DynaPatch's own gate, classification quality, pre-only vs pre+post (theta=0) --
    outputs/rq3/gate_classification.csv (scripts/extract_rq3_gate_evidence.py)."""
    d = pd.read_csv(ROOT / "outputs/rq3/gate_classification.csv")
    rows = {}
    for ds, bb, lab in N.SETTING_ORDER:
        stg = f"{ds}/{bb}"
        r = d[d.setting == stg].set_index("evidence")
        def g(arm, met):
            return float(r.loc[arm, met]) if arm in r.index else float("nan")
        rows[lab] = [g("pre", "accuracy"), g("pre+post", "accuracy"),
                     g("pre", "precision"), g("pre+post", "precision"),
                     g("pre", "recall"), g("pre+post", "recall"),
                     g("pre", "f1"), g("pre+post", "f1")]
    hdr = []
    for h in (r"\Acc", r"\Prec", r"\Rec", r"\FOne"):
        hdr += [rf"{h} (p)", rf"{h} (p+p)"]
    return table(
        r"\textbf{RQ3:} Gate classification performance with pre-information and "
        r"pre+post-information.", "tab:rq3_gate_clf", hdr, rows, 3, True,
        agg_rows(rows, 8), colspec="l" + "cc@{\\hspace{4pt}}" * 3 + "cc",
        bold_groups=[[0, 1], [2, 3], [4, 5], [6, 7]])


def _rq3_gate_effect() -> str:
    """T2: DynaPatch's own deployed RR_held/Reg/CReg, pre-only vs pre+post (theta=0) --
    outputs/rq3/gate_deployment.csv (scripts/extract_rq3_gate_evidence.py)."""
    d = pd.read_csv(ROOT / "outputs/rq3/gate_deployment.csv")
    rows = {}
    for ds, bb, lab in N.SETTING_ORDER:
        stg = f"{ds}/{bb}"
        r = d[d.setting == stg].set_index("evidence")
        def g(arm, met):
            return float(r.loc[arm, met]) if arm in r.index else float("nan")
        rows[lab] = [g("pre", "rr_held"), g("pre+post", "rr_held"),
                     g("pre", "reg"), g("pre+post", "reg"),
                     g("pre", "creg"), g("pre+post", "creg")]
    hdr = []
    for h in (r"\RR{}", r"\Reg{}", r"\CReg{}"):
        hdr += [rf"{h} (p)", rf"{h} (p+p)"]
    return table(
        r"\textbf{RQ3:} Repair and regression with pre-information and pre+post-information.",
        "tab:rq3_gate_effect", hdr, rows, 4, True, agg_rows(rows, 6),
        colspec="l" + "cc@{\\hspace{4pt}}" * 3, bold_groups=[[0, 1]])


def _rq3_features_figure() -> str:
    """F1: distributions of gate features for beneficial and non-beneficial patch outcomes --
    plotted by a separate figure script; this just emits the matching figure environment (or,
    for the markdown terminal view, a one-line pointer -- there is no figure to render there)."""
    if FMT == "markdown":
        return "**fig:features** -- see images/features (not renderable in a terminal)\n"
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=\linewidth]{images/features}",
        r"\caption{Distributions of gate features for beneficial and non-beneficial patch "
        r"outcomes.}",
        r"\label{fig:features}", r"\end{figure}",
    ]) + "\n"


def rq3() -> str:
    """Paper RQ3: post-information."""
    return "\n".join([_rq3_gate_clf(), _rq3_gate_effect(), _rq3_features_figure()])


# ------------------------------------------------------------------ RQ4

def _rq4_summary() -> str:
    """DynaPatch-NoGate versus the cost-weighted gate at lambda in {0.5, 1, 2, 4} (lambda=1
    recovers the shipped natural threshold) -- outputs/rq2/ungated_fixedpatch_dynapatch.csv
    (NoGate reference) and outputs/rq4/gate_lambda_sweep/per_cell.csv
    (scripts/gate_lambda_sweep.py). Percentages are relative to the NoGate row."""
    a = pd.read_csv(ROOT / "outputs/rq2/ungated_fixedpatch_dynapatch.csv")
    a = a.pivot_table(index="method", columns="metric", values="value",
                      aggfunc="mean").reset_index()
    nogate_row = a[a.method.str.startswith("DynaPatch-NoGate")].iloc[0]
    base = {m: float(nogate_row[m]) for m in ("RR_held", "Reg", "CReg")}

    d = pd.read_csv(ROOT / "outputs/rq4/gate_lambda_sweep/per_cell.csv")
    dp = d[d.method == "DynaPatch"]
    per_setting = dp.groupby(["setting", "lambda"])[["RR_held", "Reg", "CReg"]].mean()
    pooled = per_setting.groupby("lambda").mean()
    metrics = ("RR_held", "Reg", "CReg")
    decs = {"RR_held": 3, "Reg": 4, "CReg": 4}

    caption = (r"\textbf{RQ4:} Mean repair and regression rates under different regression "
               r"costs. Parentheses show relative changes from \DPNoGate{}.")
    label = "tab:rq4_summary"

    if FMT == "markdown":
        L = [f"**{delatex(caption)}**  ({label})", "",
             "| Gate setting | RR\u2191 | Reg\u2193 | CReg\u2193 |", "|---|---|---|---|",
             "| DPNoGate | " + " | ".join(fmt(base[m], decs[m]) for m in metrics) + " |"]
        for lam in sorted(pooled.index):
            row = pooled.loc[lam]
            vals = " | ".join(
                f"{fmt(row[m], decs[m])} ({(row[m] - base[m]) / base[m] * 100:+.1f}%)"
                for m in metrics)
            L.append(f"| lambda={lam:g} | {vals} |")
        return "\n".join(L) + "\n"

    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{\textnormal{{{caption}}}}}",
         rf"\label{{{label}}}", r"\footnotesize", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} lccc @{}}", r"\toprule",
         r"Gate setting & \RR{} $\uparrow$ & \Reg{} $\downarrow$ & \CReg{} $\downarrow$ \\",
         r"\midrule",
         r"\DPNoGate{} & " + " & ".join(fmt(base[m], decs[m]) for m in metrics) + r" \\",
         r"\midrule"]
    for lam in sorted(pooled.index):
        row = pooled.loc[lam]
        vals = " & ".join(fmt(row[m], decs[m]) for m in metrics)
        pct = " & ".join(f"{{\\scriptsize(${(row[m] - base[m]) / base[m] * 100:+.1f}\\%$)}}"
                         for m in metrics)
        L.append(f"$\\lambda={lam:g}$ & {vals} \\\\")
        L.append(f"& {pct} \\\\")
        L.append(r"\addlinespace[2pt]")
        L.append(r"\midrule")
    if L[-1] == r"\midrule":
        L.pop(); L.pop()
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L) + "\n"


def rq4() -> str:
    """Paper RQ4: regression-cost controllability."""
    return _rq4_summary()


# ------------------------------------------------------------------ public RQ entry points

RQ_TABLES = {"1": rq1, "2": rq2, "3": rq3, "4": rq4}


def main() -> None:
    global FMT
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="Also write paper-ready LaTeX .tex files here (implies --format "
                         "latex for the files; nothing is written unless this is given).")
    ap.add_argument("--format", default=None, choices=["markdown", "latex"],
                    help="Terminal output format. Default: markdown, unless --outdir is given "
                         "(then latex, to match what gets written).")
    ap.add_argument("--rq", default="all", choices=["1", "2", "3", "4", "all"],
                    help="Print/write just one RQ's tables, or all four (default).")
    a = ap.parse_args()
    FMT = a.format or ("latex" if a.outdir else "markdown")
    names_ = list(RQ_TABLES) if a.rq == "all" else [a.rq]
    for name in names_:
        print(f"===== RQ{name} " + "=" * 60)
        print(RQ_TABLES[name]())
    if a.outdir:
        out = ROOT / a.outdir
        out.mkdir(parents=True, exist_ok=True)
        FMT = "latex"
        for name in names_:
            (out / f"rq{name}.tex").write_text(RQ_TABLES[name]())
        print(f"wrote {len(names_)} .tex file(s) to {out}")


if __name__ == "__main__":
    main()
