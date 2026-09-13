#!/usr/bin/env python3
"""Emit RQ1-RQ4 as paste-ready LaTeX, in the paper's own house style.

House style, taken from the drafts already in the paper:
  * settings named by the D-A scheme (G-RN ...), three dataset blocks separated by
    \\specialrule{\\lightrulewidth}{2pt}{2pt}
  * the best cell of each row in \\textbf{}
  * a Median (G/T) row and a Mean (all) row at the foot
  * method columns use the \\HeadFT / \\FullFT / \\Arachne / ... macros from names.py

Everything is read from small CSVs -- no feature tensors are loaded, so this is safe to run
while a sweep is using the machine.

Sources
  RQ1  outputs/ALL_RQ_DATA.csv (rq=RQ1)            FixedPatch vs the conditioned patch
  RQ2  outputs/rq2_gate_{classification,deployment}_dynapatch.csv (run
       scripts/extract_rq2_gate_dynapatch.py first -- parses the already-published,
       last_affine-corrected RQ3.2/RQ3.4 tables in note/RQ3_DATA.md), plus
       outputs/baseline_prior_patches_mlp_r5{,_ungated}.json and
       outputs/gate_transplant/per_cell_{2_prestrong,4_pre+post}.csv for the transplant table
  RQ3  outputs/gate_natural_point_protocolC_ep40ns.csv  the natural (theta=0) operating point
  RQ4  outputs/rq4_final.csv + baseline_prior_patches_mlp.json + the RQ3 natural point

Usage:  .venv/bin/python scripts/paper_latex_tables.py [--outdir note/tables]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names as N  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GT = [s for s in ("gtsrb", "tt100k_signs") for _ in range(4)]


def blocks() -> list[list[tuple[str, str, str]]]:
    """SETTING_ORDER split into the three dataset blocks."""
    out, cur, ds = [], [], None
    for d, b, lab in N.SETTING_ORDER:
        if ds is not None and d != ds:
            out.append(cur); cur = []
        cur.append((d, b, lab)); ds = d
    out.append(cur)
    return out


def fmt(v, dec: int, bold: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{dec}f}"
    return f"\\textbf{{{s}}}" if bold else s


def table(caption: str, label: str, headers: list[str], rows: dict[str, list],
          dec: int, bold_max: bool | None, foot: list[tuple[str, list]],
          colspec: str | None = None, note: str = "",
          bold_groups: list[list[int]] | None = None) -> str:
    """`bold_groups` lists the column index sets that COMPETE with each other; the best cell in
    each group is bolded. Pass None/[] where the columns are different metrics (Acc vs Prec vs
    Rec vs F1) or different operating points -- bolding the row maximum there says nothing, it
    just marks whichever metric happens to run highest.
    """
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
            L.append(f"{lab} & " + " & ".join(
                fmt(v, dec, mark[i]) for i, v in enumerate(vals)) + r" \\")
    L.append(r"\midrule")
    for name, vals in foot:
        L.append(f"{name} & " + " & ".join(
            fmt(v, dec)
            for v in vals) + r" \\")
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


# ------------------------------------------------------------------ RQ1

def _ungated_baseline_rr_reg() -> dict[str, dict[str, tuple[float, float]]]:
    """NN-Patching/PatchNAS at their RAW ungated (always-apply, route_rate=1.0) operating
    point -- outputs/baseline_prior_patches_mlp_r5_ungated.json (--tau -1.0, same
    estimator/seed/repeats protocol as the paper's own-gate numbers). NOT their own gate's
    routed prediction -- see note/RQ1_INPUT_SPECIFICITY.md §5 for why RQ1 needs baselines at
    this operating point specifically."""
    j = json.loads((ROOT / "outputs/baseline_prior_patches_mlp_r5_ungated.json").read_text())
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for method, by_setting in j.items():
        out[method] = {}
        for stg, by_seed in by_setting.items():
            rr = st.mean(m["RR_held"] for m in by_seed.values())
            reg = st.mean(m["Reg"] for m in by_seed.values())
            out[method][stg] = (rr, reg)
    return out


def rq1() -> str:
    """T1: RR_held (bolded, best of 4) and Reg (context, not bolded) for all four ungated
    methods -- FixedPatch/DynaPatch-NoGate from ALL_RQ_DATA.csv, NN-Patching/PatchNAS at their
    raw ungated operating point (see `_ungated_baseline_rr_reg`)."""
    a = pd.read_csv(ROOT / "outputs/ALL_RQ_DATA.csv")
    a = a[a.rq == "RQ1"].pivot_table(index=["method", "setting"], columns="metric",
                                     values="value").reset_index()
    ungated = _ungated_baseline_rr_reg()
    rows = {}
    for _, _, lab in N.SETTING_ORDER:
        stg = [f"{d}/{b}" for d, b, l in N.SETTING_ORDER if l == lab][0]
        def g(m, met):
            r = a[(a.method.str.startswith(m)) & (a.setting == stg)]
            return float(r[met].iloc[0]) if len(r) else float("nan")
        nnp_rr, nnp_reg = ungated["NN-Patching"].get(stg, (float("nan"), float("nan")))
        pns_rr, pns_reg = ungated["PatchNAS"].get(stg, (float("nan"), float("nan")))
        rows[lab] = [g("FixedPatch", "RR_held"), g("DynaPatch-NoGate", "RR_held"),
                     nnp_rr, pns_rr,
                     g("FixedPatch", "Reg"), g("DynaPatch-NoGate", "Reg"), nnp_reg, pns_reg]
    hdr = [r"\FPa", r"\DPNoGate", r"\NNPatch", r"\PatchNAS"]
    return table(
        r"\textbf{RQ1:} A single shared correction (\FPa) versus an input-conditioned "
        r"correction (\DPNoGate), alongside \NNPatch/\PatchNAS at their own RAW, ungated "
        r"operating point (route\_rate=1.0 -- their own gate's behaviour is a separate, RQ2 "
        r"question). None of the four rows is gated. $\mathrm{RR}$ (cols 1--4, bolded, best "
        r"of four) is on held-out failures $\mathcal{D}^f_{\mathrm{test}}$; \Reg\ (cols 5--8, "
        r"not bolded, shown for context only) is on $\mathcal{D}^c_{\mathrm{test}}$.",
        "tab:rq1", hdr + hdr, rows, 3, True, agg_rows(rows, 8),
        colspec="l" + "c" * 4 + "@{\\hspace{10pt}}" + "c" * 4,
        bold_groups=[[0, 1, 2, 3]],
        note=r"\NNPatch/\PatchNAS's ungated \Reg\ is catastrophic (mean 0.84--0.86, see "
             r"RQ1.10/RQ2.12) -- their patch head has no clean-replay objective and no "
             r"``leave it unchanged'' default; shown here for the same reason FixedPatch's "
             r"\Reg\ is shown, not because it is a competitive operating point.")


# ------------------------------------------------------------------ RQ2
# 2026-09-06: replaced the old single r=0.80 table (`outputs/gate_protocol_b.csv`, the r-grid
# threshold-search protocol `scripts/build_results_draft.py:_gate_points()`'s docstring flags as
# an eval-set selection bug, confirmed 2026-09-04 -- fixed project-wide by moving to the theta=0
# natural-point protocol RQ3/RQ4 already use) with three tables built on that theta=0 protocol:
# does post-info make the CLASSIFIER better (T1, DynaPatch only), does it make the DEPLOYED
# system better (T2, DynaPatch only), and does the same post-info benefit show up when our gate
# architecture is transplanted onto NN-Patching/PatchNAS's own raw patches, compared against
# their own native (pre-only) estimator and no gate at all (T3).

def rq2_classification() -> str:
    """T1: DynaPatch's own gate, classification quality, pre-only vs pre+post (RQ3.2, theta=0).
    Single method -- no need to spread this across NN-Patching/PatchNAS too; T3 covers the
    cross-method post-info question at the deployment level instead."""
    d = pd.read_csv(ROOT / "outputs/rq2_gate_classification_dynapatch.csv")
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
        r"\textbf{RQ2:} \DPGate's own classification quality, pre-only evidence versus the "
        r"shipped pre+post evidence, at the natural threshold (theta=0, no target r or "
        r"calibration split). ``p'' = pre-only, ``p+p'' = pre+post. Each metric pair is "
        r"(pre, pre+post); bolded = better of the two.",
        "tab:rq2a", hdr,
        rows, 3, True,
        agg_rows(rows, 8),
        colspec="l" + "cc@{\\hspace{6pt}}" * 3 + "cc",
        bold_groups=[[0, 1], [2, 3], [4, 5], [6, 7]],
        note=r"Every column pair is (pre, pre+post) for the SAME metric, not different metrics "
             r"-- bolding compares the two evidence arms, not \Acc\ against \Prec. Accuracy/"
             r"Precision/F1 improve with post-info in most settings; Recall drops on average "
             r"(mean 0.986 $\to$ 0.950) -- the classifier becomes pickier, not just better "
             r"(RQ3.12).")


def rq2_deployment() -> str:
    """T2: DynaPatch's own deployed RR_held/Reg/CReg, pre-only vs pre+post (RQ3.4, theta=0)."""
    d = pd.read_csv(ROOT / "outputs/rq2_gate_deployment_dynapatch.csv")
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
    for h in (r"$\mathrm{RR}_{\mathrm{held}}$", r"\Reg", r"\CReg"):
        hdr += [rf"{h} (p)", rf"{h} (p+p)"]
    return table(
        r"\textbf{RQ2:} \DPGate's deployed repair rate and regression, pre-only evidence "
        r"versus the shipped pre+post evidence, at the natural threshold (theta=0). ``p'' = "
        r"pre-only, ``p+p'' = pre+post. Only the $\mathrm{RR}$ pair is bolded (higher = "
        r"better); \Reg/\CReg\ are lower-is-better and left unbolded so bolding never mixes "
        r"two opposite directions in one table.",
        "tab:rq2b", hdr,
        rows, 4, True,
        agg_rows(rows, 6),
        colspec="l" + "cc@{\\hspace{6pt}}" * 3,
        bold_groups=[[0, 1]],
        note=r"Post-info trades a small amount of $\mathrm{RR}_{\mathrm{held}}$ (mean 0.511 "
             r"$\to$ 0.494) for roughly half the \Reg/\CReg\ (0.0014/0.0016 $\to$ 0.0009/0.0010) "
             r"-- the same conservatism shift as T1's recall drop, read at the deployment level "
             r"(RQ3.11/3.12).")


def rq2_transplant() -> str:
    """T3: NN-Patching/PatchNAS across four gating regimes -- fully ungated (no gate at all),
    their own native estimator (pre-only by construction -- these methods have no post-info
    concept, so there is no "their own gate, pre+post" arm to report), our gate architecture
    transplanted with pre-only features, and our gate architecture transplanted with pre+post
    features. Isolates two different questions in one table: does GATING at all help (ungated
    vs either gate), and does POST-INFO specifically help once the gate architecture is held
    fixed (our-gate pre-only vs our-gate pre+post) -- the same question T1/T2 ask for DynaPatch's
    own gate, asked here for a patch mechanism DynaPatch had no part in generating."""
    ungated = json.loads((ROOT / "outputs/baseline_prior_patches_mlp_r5_ungated.json").read_text())
    own = json.loads((ROOT / "outputs/baseline_prior_patches_mlp_r5.json").read_text())
    pre = pd.read_csv(ROOT / "outputs/gate_transplant/per_cell_2_prestrong.csv")
    prepost = pd.read_csv(ROOT / "outputs/gate_transplant/per_cell_4_pre+post.csv")

    def json_mean(j, method) -> tuple[float, float, float]:
        rr, reg, creg = [], [], []
        for by_seed in j.get(method, {}).values():
            for m in by_seed.values():
                rr.append(m["RR_held"]); reg.append(m["Reg"]); creg.append(m["CReg"])
        return (sum(rr) / len(rr), sum(reg) / len(reg), sum(creg) / len(creg))

    def csv_mean(d, method) -> tuple[float, float, float]:
        r = d[d.method == method]
        return (float(r.RR_held.mean()), float(r.Reg.mean()), float(r.CReg.mean()))

    row_labels = ["Fully ungated", "Own estimator", "Our gate, pre-only", "Our gate, pre+post"]
    rows = []
    for regime_fn in (lambda m: json_mean(ungated, m), lambda m: json_mean(own, m),
                      lambda m: csv_mean(pre, m), lambda m: csv_mean(prepost, m)):
        nn = regime_fn("NN-Patching")
        pn = regime_fn("PatchNAS")
        rows.append([*nn, *pn])
    return method_table(
        r"\textbf{RQ2:} \NNPatch/\PatchNAS's raw patch output across four gating regimes: no "
        r"gate at all, their own native (pre-only) estimator, and our gate architecture "
        r"transplanted with pre-only versus pre+post features. \NNPatch/\PatchNAS have no "
        r"post-info concept of their own, so ``their own gate, pre+post'' is not a reportable "
        r"arm.",
        "tab:rq2c",
        [r"\NNPatch\ $\mathrm{RR}$", r"\NNPatch\ \Reg", r"\NNPatch\ \CReg",
         r"\PatchNAS\ $\mathrm{RR}$", r"\PatchNAS\ \Reg", r"\PatchNAS\ \CReg"],
        row_labels, rows, 3,
        colspec="l" + "ccc@{\\hspace{10pt}}ccc",
        note=r"Ungated $\to$ own estimator isolates what gating at all buys (\Reg\ collapses "
             r"from $\sim$0.82--0.86 to $\sim$0.01--0.02). Own estimator $\to$ our-gate "
             r"pre-only is NOT an improvement on $\mathrm{RR}$ (0.374/0.317 $\to$ 0.358/0.345) "
             r"-- swapping in our gate architecture alone does not help. Our-gate pre-only "
             r"$\to$ pre+post recovers $\mathrm{RR}$ to near \DP's own gated point (0.494/0.477 "
             r"vs \DP's 0.494) while \Reg/\CReg\ drop further still -- the same post-info "
             r"benefit T1/T2 show for \DP's own gate also holds when the same gate architecture "
             r"is applied to a patch mechanism \DP had no part in generating (RQ3.13).")


# ------------------------------------------------------------------ RQ3

def _natural_points() -> pd.DataFrame:
    """One row per (dataset, backbone), pooled over seeds, at theta=0 (commit iff the fitted
    gate's own score favours beneficial over harmful) -- no target r, no threshold search, no
    calibration split of any kind. FIXED 2026-09-04, second round: `_curve_point()`'s original
    r-grid version picked "the q whose frac_Reg_removed crosses r, then the highest RR_held
    among those" off outputs/gate_curves_protocolC_ep40ns.csv -- an eval-set selection bug.
    The first fix (this function's predecessor, `_deploy_points()`) instead targeted r on
    bug_val/clean_calib alone, but that population is ALSO part of what fits the gate under
    the shipped protocol, so it was still in-sample for the model. theta=0 needs no such split
    at all. See note/RESEARCH_STATE.md.
    """
    d = pd.read_csv(ROOT / N.GATE_NATURAL_POINT)
    d[["dataset", "backbone"]] = d.setting.str.split("/", n=1, expand=True)
    return d.groupby(["dataset", "backbone"], as_index=False)[
        ["RR_held", "Reg", "CReg"]].mean()


# ------------------------------------------------------------------ RQ4
# 2026-09-06 renumbering (user confirmed): this content -- DynaPatch's own NoGate-vs-gated
# repair/regression trade-off -- was captioned "RQ3" but is conceptually the paper's RQ4
# ("how effectively does DynaPatch control the repair-regression trade-off", per
# DRAFT_RESULTS_NEW_RQS.md's original four-RQ index and this function's own pre-existing
# comment below about "RQ4 controllability" owning this path). Renamed rq3()->
# rq4_controllability(), tab:rq3{a,b}->tab:rq4{a,b}, caption RQ3->RQ4. The comparison-with-
# existing-methods content that used to live in tab:rq4{a,b,c,d} is renamed to RQ3 below
# (rq3_comparison/rq3_coverage/rq3_regression_profile), and the old rq3c() (post-info net-flip)
# moved into the RQ2 (gate/post-info) file as rq2_postinfo_flip() -- same topic, not a separate RQ.

def rq4_controllability() -> tuple[str, str]:
    c = pd.read_csv(ROOT / N.GATE_CURVES)   # q=0 (ungated) baseline reference only -- no
    nat = _natural_points()                 # threshold search involved, safe to keep
    rr, reg = {}, {}
    for ds, bb, lab in N.SETTING_ORDER:
        s = c[(c.dataset == ds) & (c.backbone == bb)].iloc[0]
        p = nat[(nat.dataset == ds) & (nat.backbone == bb)]
        pt = {"RR": float(p.RR_held.iloc[0]), "Reg": float(p.Reg.iloc[0]),
              "CReg": float(p.CReg.iloc[0])} if len(p) else None
        rr[lab] = [float(s.RR_base)] + [pt["RR"] if pt else float("nan")]
        reg[lab] = [float(s.Reg_base)] + [pt["Reg"] if pt else float("nan")] \
                   + [float(s.CReg_base)] + [pt["CReg"] if pt else float("nan")]
    hdr = [r"\DPNoGate", r"\DP\ (gated)"]
    t1 = table(
        r"\textbf{RQ4:} Repair rate on held-out failures, no gate vs the natural-threshold "
        r"gated point (theta=0, no target r or calibration split).",
        "tab:rq4a", hdr, rr, 3, None, agg_rows(rr, 2))
    t2 = table(
        r"\textbf{RQ4:} The regression that gate buys. \Reg\ over "
        r"$\mathcal{D}^c_{\mathrm{test}}$ (left) and \CReg\ over its safety-critical rows "
        r"(right); lower is better.",
        "tab:rq4b", hdr + hdr, reg, 4, None, agg_rows(reg, 4),
        colspec="l" + "c" * 2 + "@{\\hspace{10pt}}" + "c" * 2)
    return t1, t2


def rq4_lambda() -> str:
    """T3 (tab:rq4_lambda): cost-weighted controllability, ALL THREE methods (DynaPatch,
    NN-Patching, PatchNAS) side by side -- scripts/gate_lambda_sweep.py, commit iff
    P(gain=+1) > lambda * P(gain=-1), lambda=1 recovers the shipped theta=0 rule. Unlike the
    quantile-q sweep (rq4_frontier_figure(), DynaPatch only, user-scoped that way 2026-09-06),
    lambda touches the report population's score distribution not at all -- a pure modelling
    choice applied identically everywhere -- so this was already the natural place a cross-
    method comparison belonged, and it was already computed. **Recovery note (2026-09-06)**:
    this table previously existed in note/tables/rq4.tex as a manually-appended block (not
    generated by this script), and was silently DESTROYED when this script's full-overwrite
    rq4.tex regeneration ran without first checking the file's existing content. Recovered by
    verifying outputs/gate_lambda_sweep/per_cell.csv (the underlying computation was never
    touched, only the manual tex block was lost) reproduces the exact same numbers under
    setting-balanced averaging (mean within setting across seeds, then mean across the 12
    settings), and promoting it to a proper generator function so this cannot happen silently
    again."""
    d = pd.read_csv(ROOT / "outputs/gate_lambda_sweep/per_cell.csv")
    per_setting = d.groupby(["method", "lambda", "setting"])[["RR_held", "Reg", "CReg"]].mean()
    st = per_setting.groupby(["method", "lambda"]).mean()
    lambdas = sorted(d["lambda"].unique())
    methods = ["DynaPatch", "NN-Patching", "PatchNAS"]
    method_hdr = [r"\DP", "NN-Patch", "PatchNAS"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\textnormal{\textbf{RQ4:} Repair rate, regression, and critical regression "
         r"under different regression costs. All nine cells are the 12-setting mean from the "
         r"same cost-weighted gate sweep (`scripts/gate_lambda_sweep.py`): commit iff "
         r"$P(\text{gain}=+1) > \lambda \cdot P(\text{gain}=-1)$, $\lambda=1$ recovers the "
         r"natural threshold.}}",
         r"\label{tab:rq4_lambda}",
         r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{c ccc ccc ccc}", r"\toprule",
         r"\multirow{2}{*}{$\lambda$} & \multicolumn{3}{c}{RR $\uparrow$} & "
         r"\multicolumn{3}{c}{Reg $\downarrow$} & \multicolumn{3}{c}{CReg $\downarrow$} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
         "& " + " & ".join(method_hdr * 3) + r" \\", r"\midrule"]
    for lam in lambdas:
        vals = []
        for metric in ("RR_held", "Reg", "CReg"):
            for m in methods:
                vals.append(float(st.loc[(m, lam), metric]))
        dec = [3] * 3 + [4] * 6
        L.append(f"{lam:.1f} & " + " & ".join(fmt(v, d) for v, d in zip(vals, dec)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(L) + "\n"


def rq4_mechanism() -> str:
    """T4/T5 (tab:rq4_score/tab:rq4_removal): why does tightening lambda (T3) remove regression
    faster than repair, rather than reporting that fall as if it were itself a finding -- raising
    lambda can only shrink the accepted set (RQ4.6's nested-set monotonicity), so "RR/Reg both
    fall as lambda rises" is close to guaranteed by the decision rule, not evidence on its own.
    scripts/analysis_gate_lambda_mechanism.py answers the question that actually matters: does
    the fitted score rank harmful patch applications below beneficial ones?

    T4 (score_by_outcome): mean gate log-odds score log P(gain=+1) - log P(gain=-1), by the true
    eventual outcome of the flipped row -- regression / ineffective change / successful repair
    (same three-way partition as RQ3.3/tab:rq2d). Setting-balanced; 11/12 settings individually
    show the full order regression < ineffective change < successful repair (the one exception,
    lisa_signs/convnext_tiny, has n=1 in the ineffective-change group -- a single sample, not a
    contradicting trend).

    T5 (nested_removal): lambda's accepted sets are nested per setting/seed (RQ4.6's argument
    applies identically here), so raising lambda only ever moves rows from kept to rejected.
    Restricted to the HELD (failure) side, where a flip can genuinely go either way -- the clean
    side cannot be asked the same question, since a proposed change on an already-correct input
    is a regression BY CONSTRUCTION, so clean-side removal is a magnitude fact (it already IS
    T3's falling Reg) not a mix-shift one. At every lambda step, the newly-rejected held rows are
    enriched for ineffective changes relative to the ambient rate in the population still
    eligible to be rejected just before that step -- 11/12 settings show this enrichment when
    pooled over all four steps (the exception, gtsrb/convnext_tiny, has only 3 total held rows
    removed across all four steps against an already near-zero 1.8% ambient ineffective rate --
    underpowered, not contradicting)."""
    score = pd.read_csv(ROOT / "outputs/gate_lambda_mechanism/score_by_outcome_overall.csv")
    order = ["regression", "ineffective change", "successful repair"]
    score = score.set_index("outcome").loc[order]
    t4 = method_table(
        r"\textbf{RQ4:} \DPGate's fitted score (log $P(\mathrm{gain}=+1)$ $-$ log "
        r"$P(\mathrm{gain}=-1)$, the same quantity $\lambda$ thresholds against), by the true "
        r"eventual outcome of the flipped row, setting-balanced mean over 12 settings. Higher "
        r"is more committed toward applying the patch.",
        "tab:rq4_score", ["$n$", "Mean score"],
        [o.capitalize() for o in order],
        [[int(score.loc[o, "n"]), float(score.loc[o, "mean_score"])] for o in order],
        [0, 3], row_header="True outcome",
        note=r"11/12 settings individually show this exact order (regression $<$ ineffective "
             r"$<$ successful repair); the one exception has a single-sample ineffective-change "
             r"group.")

    rem = pd.read_csv(ROOT / "outputs/gate_lambda_mechanism/nested_removal_pooled.csv")
    rem["n_removed_held"] = rem.removed_successful_repair + rem.removed_ineffective_change
    rem["frac_removed_ineffective"] = rem.removed_ineffective_change / rem.n_removed_held.replace(0, np.nan)
    t5 = method_table(
        r"\textbf{RQ4:} Held-side rows newly rejected at each $\lambda$ step, pooled over 12 "
        r"settings: how many (of those still eligible to be rejected just before this step) are "
        r"actually ineffective changes versus how many were successful repairs, against the "
        r"ambient ineffective-change rate in that pre-step population. No clean-side column: a "
        r"clean-side rejection is a regression by construction, so it is already the falling Reg "
        r"column of Table~\ref{tab:rq4_lambda}, not "
        r"separate evidence.",
        "tab:rq4_removal", ["$n$ removed (held)", "\\% ineffective (removed)",
                             "\\% ineffective (ambient)"],
        [t.replace("→", r"$\to$") for t in rem.transition],
        [[int(r.n_removed_held), float(r.frac_removed_ineffective) * 100,
          float(r.pre_held_ineffective_rate) * 100] for _, r in rem.iterrows()],
        [0, 1, 1], row_header="$\\lambda$ step",
        note=r"Ambient rate is the ineffective-change share of the held population still "
             r"eligible to be rejected immediately before that step; the removed share sits "
             r"well above it at every step (2.8--3.2$\times$), i.e.\ tightening $\lambda$ does "
             r"not reject held-side rows in proportion to their ambient mix -- it preferentially "
             r"rejects the ones that were not going to repair anything.")
    return t4 + "\n" + t5


def rq4_frontier_figure() -> str:
    """F1 (fig:rq4): the repair-regression frontier -- sweeping the gate's veto threshold q
    across its full range (59 steps/setting, scripts/analysis_rq4_frontier.py) traces a smooth,
    monotonic trade-off in 12/12 settings (verified: RR_held never increases as q increases,
    zero violations in any setting/step), with the Ungated and deployed theta=0 points from
    tab:rq4a/b marked on the same curve. Answers "is the threshold actually controllable" with
    the full curve, not just the two endpoints tab:rq4a/b report."""
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=\linewidth]{figures/rq4_frontier/frontier_grid.pdf}",
        r"\caption{\textnormal{\textbf{RQ4:} Repair-regression frontier, one panel per "
        r"setting, tracing the gate's veto threshold $q$ across its full range "
        r"(59 steps/setting). $\mathrm{RR}_{\mathrm{held}}$ is non-increasing in $q$ in "
        r"12/12 settings with zero violations across all 708 (setting, step) points -- the "
        r"threshold is a well-behaved, monotone knob, not a pair of disconnected operating "
        r"points. The deployed $\theta=0$ point sits on (or very near, for the small LISA "
        r"settings where seed-averaging can shift it slightly off the seed-pooled curve) the "
        r"frontier its own gate traces.}}",
        r"\label{fig:rq4}", r"\end{figure}",
    ]) + "\n"


# ------------------------------------------------------------------ RQ3 (comparison with
# existing methods -- was captioned "RQ4"/tab:rq4{a,b,c,d}, renamed 2026-09-06, see note above)

def rq3_comparison() -> tuple[str, str]:
    per: dict[str, dict[str, dict[str, list]]] = {}
    for row in csv.DictReader((ROOT / "outputs/rq4_final.csv").open()):
        key = N.key_for_rq4_row(row["method"])
        if key is None:
            continue
        d = per.setdefault(key, {}).setdefault(row["setting"], {})
        for m in ("RR_held", "Reg", "CReg"):
            if row[m] not in ("", None):
                d.setdefault(m, []).append(float(row[m]))
    j = json.loads((ROOT / "outputs/baseline_prior_patches_mlp.json").read_text())
    for key, jk in (("NNPatch", "NN-Patching"), ("PatchNAS", "PatchNAS")):
        for stg, seeds in j.get(jk, {}).items():
            d = per.setdefault(key, {}).setdefault(stg, {})
            for v in seeds.values():
                for m in ("RR_held", "Reg", "CReg"):
                    d.setdefault(m, []).append(float(v[m]))
    nat = _natural_points()

    ORDER = ["FullFT", "HeadFT", "Arachne", "DistrRep", "NNPatch", "PatchNAS"]
    hdr = [N.latex(k) for k in ORDER] + [r"\DP"]
    rr, reg = {}, {}
    for ds, bb, lab in N.SETTING_ORDER:
        stg = f"{ds}/{bb}"
        def g(key, m):
            v = per.get(key, {}).get(stg, {}).get(m)
            return st.mean(v) if v else float("nan")
        p = nat[(nat.dataset == ds) & (nat.backbone == bb)]
        pt = {"RR": float(p.RR_held.iloc[0]), "Reg": float(p.Reg.iloc[0])} if len(p) else None
        rr[lab] = [g(k, "RR_held") for k in ORDER] + [pt["RR"] if pt else float("nan")]
        reg[lab] = [g(k, "Reg") for k in ORDER] + [pt["Reg"] if pt else float("nan")]
    t1 = table(
        r"\textbf{RQ3:} Repair rate on held-out failures, \DP\ at the natural threshold "
        r"(theta=0, no target r or calibration split).",
        "tab:rq3a", hdr, rr, 3, True, agg_rows(rr, len(hdr)),
        bold_groups=[list(range(len(hdr)))])
    t2 = table(
        r"\textbf{RQ3:} Regression on $\mathcal{D}^c_{\mathrm{test}}$; lower is better. "
        r"\DP\ at the natural threshold.",
        "tab:rq3b", hdr, reg, 4, False, agg_rows(reg, len(hdr)),
        bold_groups=[list(range(len(hdr)))])
    return t1, t2


def rq3_coverage() -> str:
    """T3 (tab:rq3c): why do baselines win/lose -- failure-type coverage and repair strength.
    Extracted from note/RQ2_BASELINE_BEHAVIOR.md section 3 (already recomputed end-to-end after
    last_affine; see scripts/extract_rq3_baseline_tables.py). Explains FullFT's strength (high
    coverage, mostly fully-repaired types) and Arachne/DistrRep's weakness (low coverage, mostly
    zero-repair types) in the same table as DynaPatch-NoGate."""
    d = pd.read_csv(ROOT / "outputs/rq3_baseline_failure_type.csv").set_index("method")
    ORDER = [("FullFT", r"\FullFT"), ("HeadFT", r"\HeadFT"), ("Arachne", r"\Arachne"),
            ("DistrRep", r"\DistrRep"), ("NN-Patching", r"\NNPatch"),
            ("PatchNAS", r"\PatchNAS"), ("DynaPatch-NoGate", r"\DPNoGate")]
    row_labels = [lab for _, lab in ORDER]
    rows = [[d.loc[m, "coverage_at_0.5"], d.loc[m, "type-rr_p10"], d.loc[m, "type-rr_median"],
            d.loc[m, "zero-repair_types"], d.loc[m, "fully_repaired_types"]]
            for m, _ in ORDER]
    return method_table(
        r"\textbf{RQ3:} Failure-type coverage and repair strength, ungated methods. "
        r"$\mathrm{cov}@0.5$ = fraction of "
        r"estimable failure types with per-type $\mathrm{RR}\ge 0.5$; the last two columns "
        r"are the fraction of estimable types with ZERO repair and FULL (100\%) repair.",
        "tab:rq3c",
        [r"cov@0.5", r"Type-RR P10", r"Type-RR median", r"Zero-repair share",
         r"Full-repair share"],
        row_labels, rows, 3,
        note=r"FullFT's strength is coverage BREADTH, not just per-type strength once reached: "
             r"74\% of its estimable types are fully repaired, vs 11\%/5\% for Arachne/"
             r"DistrRep. Arachne/DistrRep's low aggregate $\mathrm{RR}$ (Table~\ref{tab:rq3a}) "
             r"is not explained by partial-but-nonzero repair everywhere -- 70\%/43\% of their "
             r"estimable types get ZERO repair. NN-Patching/PatchNAS sit in between: moderate "
             r"coverage (0.60/0.53) with a genuine mix of zero, partial, and full repair, not a "
             r"narrow-coverage failure mode.")


def rq3_regression_profile() -> str:
    """T4 (tab:rq3d): why do some methods "look safe" on mean Reg alone but are not, and why is
    DynaPatch's gate different from just having low mean Reg -- worst-class severity and breadth,
    from note/RQ2_BASELINE_BEHAVIOR.md section 4."""
    d = pd.read_csv(ROOT / "outputs/rq3_baseline_regression_profile.csv").set_index("method")
    ORDER = [("FullFT", r"\FullFT"), ("HeadFT", r"\HeadFT"), ("Arachne", r"\Arachne"),
            ("DistrRep", r"\DistrRep"), ("NN-Patching", r"\NNPatch"),
            ("PatchNAS", r"\PatchNAS"), ("DynaPatch-NoGate", r"\DPNoGate"),
            ("DynaPatch (gated, theta=0)", r"\DP\ (gated)")]
    row_labels = [lab for _, lab in ORDER]
    rows = [[d.loc[m, "mean_reg"], d.loc[m, "mean_worst-class_reg"],
            d.loc[m, "fraction_of_clean_classes_hit"]] for m, _ in ORDER]
    return method_table(
        r"\textbf{RQ3:} Regression distribution -- breadth and worst-case severity, not just "
        r"the mean. Lower is better on all three columns.",
        "tab:rq3d",
        [r"Mean \Reg", r"Mean worst-class \Reg", r"Frac.\ clean classes hit"],
        row_labels, rows, 4,
        note=r"Mean \Reg\ alone is misleading for Arachne: 0.0078 is the second-LOWEST mean, "
             r"but its worst-class \Reg\ (0.5146) is the second-HIGHEST -- a low mean hides "
             r"localized spikes concentrated in a few classes, not diffuse damage. DistrRep is "
             r"worst on all three simultaneously (broad, diffuse damage). DynaPatch-NoGate is "
             r"NOT safe by itself (mean \Reg\ 0.0066, comparable to the other baselines); the "
             r"gate is what moves it to the lowest value on all three columns "
             r"(0.0009/0.1642/0.0489) -- synthesis supplies repair, the gate supplies "
             r"preservation.")


def rq2_postinfo_flip() -> str:
    """T6 (tab:rq2f) for the RQ2 (gate/post-info) file -- moved here 2026-09-06 (was rq3c(),
    tab:rq3c, captioned "RQ3") since this is post-info content, same topic as T1-T5, not the
    controllability question RQ4 now owns. Postinfo gate flip (RQ3.11 in note/RQ3_DATA.md):
    per-sample decision flips, pre-only vs shipped pre+post, at theta=0. Counts, not rates --
    so the foot is a pooled Total, not a Median/Mean, and bold_groups is None (the two columns
    are different axes, not a competition)."""
    cell = pd.read_csv(ROOT / "outputs/postinfo_gate_flip/per_setting.csv")
    held = cell[cell.side == "held"]
    clean = cell[cell.side == "clean"]

    def net_held(stg: str) -> float:
        h = held[held.setting == stg]
        g = lambda t: int(h[(h.transition == t) & (h.gain_label == "beneficial (+1)")]
                          ["size"].sum())
        return g("veto -> commit") - g("commit -> veto")

    def net_clean(stg: str) -> float:
        c = clean[clean.setting == stg]
        g = lambda t: int(c[(c.transition == t) & (c.gain_label == "harmful (-1)")]
                          ["size"].sum())
        return g("commit -> veto") - g("veto -> commit")

    rows = {}
    for ds, bb, lab in N.SETTING_ORDER:
        stg = f"{ds}/{bb}"
        rows[lab] = [float(net_held(stg)), float(net_clean(stg))]
    tot = [sum(v[i] for v in rows.values()) for i in range(2)]
    return table(
        r"\textbf{RQ2:} Adding post-repair evidence to \DP's own gate: net repairs gained "
        r"on held-out failures versus net regressions newly prevented on clean inputs, at the "
        r"natural threshold (theta=0). Positive is a gain from adding post-info; negative is a "
        r"loss.",
        "tab:rq2f",
        ["Net repairs (held)", "Net Reg.\\ prevented (clean)"],
        rows, 0, None, [("Total (pooled)", tot)],
        note=r"Counts, not rates, pooled over 12 settings $\times$ 3 seeds. Held: 3 newly "
             r"committed vs.\ 36 newly vetoed beneficial repairs (net $-33$). Clean: 146 newly "
             r"vetoed vs.\ 28 newly committed harmful changes (net $+118$). 11/12 settings have "
             r"net repairs $\le 0$; all 12 have net regressions prevented $\ge 0$.")


def method_table(caption: str, label: str, headers: list[str], row_labels: list[str],
                 rows: list[list], dec: int | list[int], colspec: str | None = None,
                 note: str = "", row_header: str = "Method") -> str:
    """Small booktabs table indexed by METHOD (or another row concept, via `row_header`), not by
    setting -- T2/T3 have 3-4 rows total, `table()`'s per-setting dataset-block structure
    (blocks()/agg_rows()) does not apply. `dec` may be one int (all columns) or a list (one per
    column, e.g. integer cell counts alongside 4-decimal deltas)."""
    spec = colspec or ("l" + "c" * len(headers))
    decs = dec if isinstance(dec, list) else [dec] * len(headers)
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


def rq1b() -> str:
    """T2: RQ1.9's causal reassignment summary (direction vs magnitude shuffle), all three
    per-input methods side by side -- outputs/patch_reassignment_{,nnpatching_,patchnas_}v1/
    summary.json."""
    files = {r"\DPNoGate": "outputs/patch_reassignment_v1/summary.json",
            r"\NNPatch": "outputs/patch_reassignment_nnpatching_v1/summary.json",
            r"\PatchNAS": "outputs/patch_reassignment_patchnas_v1/summary.json"}
    row_labels, rows = [], []
    for lab, path in files.items():
        j = json.loads((ROOT / path).read_text())
        agg = j["aggregate"]
        orig = agg["original"]["median_RR_held"]
        dshuf = agg["global_direction_shuffle"]
        mshuf = agg["global_magnitude_shuffle"]
        row_labels.append(lab)
        rows.append([orig,
                    -dshuf["median_original_minus_mean_RR"], dshuf["cells_original_above_null_p975"],
                    -mshuf["median_original_minus_mean_RR"], mshuf["cells_original_above_null_p975"]])
    return method_table(
        r"\textbf{RQ1:} Causal reassignment test (RQ1.8/1.9): destroying which failure gets "
        r"which correction (direction shuffle) collapses $\mathrm{RR}_{\mathrm{held}}$ for "
        r"every per-input method; destroying only magnitude does not. $\Delta$ = shuffled "
        r"minus original (negative = RR dropped); ``cells'' = of 36 (setting $\times$ seed) "
        r"cells exceeding the null's 97.5th percentile.",
        "tab:rq1b",
        ["Original RR", r"$\Delta$ direction", "cells", r"$\Delta$ magnitude", "cells"],
        row_labels, rows, [4, 4, 0, 4, 0],
        colspec="lc@{\\hspace{6pt}}cc@{\\hspace{10pt}}cc",
        note=r"NN-Patching/PatchNAS's pre-registered auto-decision reads ``INCONCLUSIVE'' "
             r"rather than ``SUPPORT'' only because it keys off a different condition "
             r"(global\_effect\_shuffle, 29/36 cells) than the direction shuffle shown here "
             r"(30/36); see note/RQ1\_DATA.md RQ1.9.")


def rq1c() -> str:
    """T3: per-class coverage/regression, all four methods at the SAME ungated operating point
    -- outputs/p1_regression/per_cell.csv (FixedPatch/DynaPatch-NoGate) and
    outputs/p1_regression_ungated_baselines/per_cell.csv (NN-Patching/PatchNAS, ungated)."""
    def cov_table(path: str, methods: list[str]) -> pd.DataFrame:
        pc = pd.read_csv(ROOT / path.replace("per_cell.csv", "per_class.csv"))
        pc = pc[(pc.kind == "repair") & (pc.method.isin(methods))]
        pc = pc[pc.n_pool >= 5].copy()
        pc["rr_c"] = pc.n_events / pc.n_pool
        g = pc.groupby(["method", "setting", "seed"])
        return g["rr_c"].agg(cov0_5=lambda s: (s >= 0.5).mean(),
                             cov1_0=lambda s: (s >= 1.0).mean()).reset_index()

    a_cov = cov_table("outputs/p1_regression/per_cell.csv",
                      ["FixedPatch", "DynaPatch (ungated)"])
    b_cov = cov_table("outputs/p1_regression_ungated_baselines/per_cell.csv",
                      ["NN-Patching [ungated]", "PatchNAS [ungated]"])
    cov = pd.concat([a_cov, b_cov])
    cov_st = cov.groupby(["method", "setting"]).median(numeric_only=True).reset_index()

    a_reg = pd.read_csv(ROOT / "outputs/p1_regression/per_cell.csv")
    b_reg = pd.read_csv(ROOT / "outputs/p1_regression_ungated_baselines/per_cell.csv")
    reg = pd.concat([a_reg[["method", "setting", "seed", "Reg_overall"]],
                    b_reg[["method", "setting", "seed", "Reg_overall"]]])
    reg_st = reg.groupby(["method", "setting"]).median(numeric_only=True).reset_index()
    merged = cov_st.merge(reg_st, on=["method", "setting"])
    summary = merged.groupby("method")[["cov0_5", "cov1_0", "Reg_overall"]].median()

    order = [("FixedPatch", r"\FPa"), ("DynaPatch (ungated)", r"\DPNoGate"),
            ("NN-Patching [ungated]", r"\NNPatch"), ("PatchNAS [ungated]", r"\PatchNAS")]
    row_labels = [lab for _, lab in order]
    rows = [[summary.loc[m, "cov0_5"], summary.loc[m, "cov1_0"], summary.loc[m, "Reg_overall"]]
            for m, _ in order]
    return method_table(
        r"\textbf{RQ1:} Per-class repair coverage at the ungated operating point, all four "
        r"methods (RQ1.10). $\mathrm{cov}@t$ = fraction of estimable classes ($\ge 5$ held-out "
        r"failures) with per-class $\mathrm{RR} \ge t$.",
        "tab:rq1c", [r"cov@0.5", r"cov@1.0", r"\Reg\ overall"],
        row_labels, rows, 3,
        note=r"On repair breadth alone the ungated baselines are not behind FixedPatch/"
             r"DynaPatch-NoGate -- the entire gap is in \Reg\ (0.84--0.86 vs 0.005--0.010), "
             r"consistent with RQ2.12's clean-replay ablation.")


def rq1d() -> str:
    """T4 (still filed under RQ1, per user's naming convention): the RQ1.6-style within/between
    failure-type cosine-distance test, extended to NN-Patching/PatchNAS -- unlike RQ1.6 itself
    (DynaPatch's generator-internal `patch_vec`, which the baselines have no analogue of), this
    reads the M1 series' logit-space `Delta z(x)` (`outputs/m1_correction_direction/
    direction_per_cell.csv`, `split=held`, `grouping=failure_type`), which all three per-input
    methods have."""
    df = pd.read_csv(ROOT / "outputs/m1_correction_direction/direction_per_cell.csv")
    d = df[(df.split == "held") & (df.grouping == "failure_type")]
    summary = d.groupby("method")[["cos_within", "cos_between", "cos_gap"]].mean()
    order = [("DynaPatch-NoGate", r"\DPNoGate"), ("NN-Patching", r"\NNPatch"),
            ("PatchNAS", r"\PatchNAS")]
    row_labels = [lab for _, lab in order]
    rows = [[summary.loc[m, "cos_within"], summary.loc[m, "cos_between"],
            summary.loc[m, "cos_gap"]] for m, _ in order]
    return method_table(
        r"\textbf{RQ1:} Within- vs between-failure-type cosine distance of each method's own "
        r"correction $\Delta z(x)$ (logit space; mean over 11 estimable settings x 3 seeds, "
        r"held-out failures -- $\mathtt{lisa\_signs/densenet121}$ has too few estimable "
        r"failure types). Lower $\mathrm{cos}_{\mathrm{within}}$ / higher $\mathrm{cos}_{"
        r"\mathrm{between}}$ = more type-structured directions; $\mathrm{gap} = \mathrm{cos}_{"
        r"\mathrm{between}} - \mathrm{cos}_{\mathrm{within}}$.",
        "tab:rq1d",
        [r"$\mathrm{cos}_{\mathrm{within}}$", r"$\mathrm{cos}_{\mathrm{between}}$", "gap"],
        row_labels, rows, 3,
        note=r"89/96 (method, setting, seed) cells' gap clears a 200-draw label-shuffle null "
             r"at $p<0.05$; all 7 that do not are a VGG16 backbone (mostly \PatchNAS, e.g.\ "
             r"$\mathtt{lisa\_signs/vgg16}$ seed 202: gap $-0.24$, $p=0.29$) -- reported, not "
             r"hidden. All three methods show a positive gap on average; \DP's is largest, "
             r"narrowing through \NNPatch\ to \PatchNAS. Source: "
             r"\texttt{outputs/m1\_correction\_direction/direction\_per\_cell.csv}.")


def rq1e() -> str:
    """T5: mean alignment of each method's own correction Delta z(x) to the IDEAL MARGIN
    direction d* = e_y - e_yhat (scale-free: cos_to_ideal(x) = Delta m(x) / (||Delta z(x)|| *
    sqrt(2)) is a true cosine, so -- unlike raw Delta z norm / Delta m, not cross-method
    comparable since NN-Patching/PatchNAS/FixedPatch's freshly-trained heads have no constraint
    tying their logit scale to DynaPatch's frozen one -- this one IS comparable across methods).
    Includes FixedPatch (2026-09-06 addition -- the mechanistically most important comparison
    point, since it is the zero-input-conditioning floor): required a one-off redeploy, see
    `scripts/analysis_rq1_alignment.py`'s docstring for the checkpoint gotcha. Reads that
    script's promoted output (`outputs/rq1_alignment/`) rather than recomputing inline."""
    piv = pd.read_csv(ROOT / "outputs/rq1_alignment/per_setting.csv").pivot(
        index="setting", columns="method", values="mean_cos_to_ideal")
    rows = {}
    for ds, bb, lab in N.SETTING_ORDER:
        stg = f"{ds}/{bb}"
        r = piv.loc[stg] if stg in piv.index else None
        rows[lab] = [float(r.get(m, float("nan"))) if r is not None else float("nan")
                    for m in ("FixedPatch", "DynaPatch-NoGate", "NN-Patching", "PatchNAS")]
    return table(
        r"\textbf{RQ1:} Mean alignment of each method's own correction $\Delta z(x)$ to the "
        r"ideal margin direction $d^\star = e_y - e_{\hat y}$ on held-out failures "
        r"($\mathrm{cos\_to\_ideal}(x) = \Delta m(x) / (\|\Delta z(x)\| \sqrt{2})$, a true "
        r"cosine -- scale-free, so comparable across methods despite their differently-scaled "
        r"logit heads). Bolded = best of the four per setting.",
        "tab:rq1e", [r"\FPa", r"\DPNoGate", r"\NNPatch", r"\PatchNAS"], rows, 3, True,
        agg_rows(rows, 4), bold_groups=[[0, 1, 2, 3]],
        note=r"\DP\ has the higher mean alignment than ALL THREE other methods in 11/12 "
             r"settings (the one exception: $\mathtt{tt100k\_signs/convnext\_tiny}$, \NNPatch\ "
             r"0.273 vs \DP\ 0.206), and beats \FPa\ specifically in 12/12. Mean over settings: "
             r"\DP\ 0.310, \PatchNAS\ 0.168, \NNPatch\ 0.152, \FPa\ 0.128 -- the first measured "
             r"(not asserted) sense in which \DP's corrections are better-aimed, and the "
             r"closing link to \FPa: a single shared correction cannot align as well with "
             r"heterogeneous per-failure targets as an input-conditioned one does.")


def rq1f() -> str:
    """T6: margin-gain decomposition (RQ1.13). Delta m(x) = ||Delta z(x)|| * sqrt(2) *
    cos_to_ideal(x) by construction, so T5's alignment gap (DynaPatch highest) does not by
    itself say which method wins on raw margin gain -- that also depends on correction scale.
    Median (not mean) ||Delta z||/Delta m over the 12 settings: one NN-Patching cell
    (lisa_signs/vgg16) has a handful of failures with ||Delta z|| ~1e5 from its freshly-trained
    head, which drags the MEAN Delta m of that cell negative despite a positive mean
    cos_to_ideal -- a mean-of-ratio-vs-ratio-of-means artefact, not a sign of misdirection; median
    keeps that one cell from dominating the whole-method summary. cos_to_ideal repeats T5's
    already-published mean (outputs/rq1_alignment/summary.csv), not a re-aggregation, so the two
    tables read consistently."""
    dec = pd.read_csv(ROOT / "outputs/rq1_margin_decomposition/summary.csv").set_index("method")
    align = pd.read_csv(ROOT / "outputs/rq1_alignment/summary.csv").set_index("method")
    order = [("FixedPatch", r"\FPa"), ("DynaPatch-NoGate", r"\DPNoGate"),
            ("NN-Patching", r"\NNPatch"), ("PatchNAS", r"\PatchNAS")]
    row_labels = [lab for _, lab in order]
    rows = [[dec.loc[m, "median_dz_norm"], dec.loc[m, "median_dm"], align.loc[m, "mean"]]
            for m, _ in order]
    return method_table(
        r"\textbf{RQ1:} Margin gain decomposed into scale and direction (RQ1.13): "
        r"$\Delta m(x) = \|\Delta z(x)\| \sqrt{2}\, \mathrm{cos\_to\_ideal}(x)$. Median (not "
        r"mean) $\|\Delta z\|$/$\Delta m$ over the 12 settings; $\mathrm{cos\_to\_ideal}$ "
        r"repeats \Cref{tab:rq1e}'s mean.",
        "tab:rq1f",
        [r"Median $\|\Delta z\|$", r"Median $\Delta m$", r"Mean $\mathrm{cos\_to\_ideal}$"],
        row_labels, rows, [1, 1, 3],
        note=r"\NNPatch/\PatchNAS's median $\|\Delta z\|$ runs roughly 2--10$\times$ \DP's on "
             r"10/12 settings despite lower alignment, giving them a larger median $\Delta m$ "
             r"(11.9/13.1 vs \DP's 4.1) -- a bigger, less precisely aimed correction can still "
             r"produce a larger raw margin gain, consistent with their much worse \Reg\ (0.84--"
             r"0.86, \Cref{tab:rq1c}). One \NNPatch\ cell "
             r"($\mathtt{lisa\_signs/vgg16}$) is a scale outlier (median $\|\Delta z\|\approx"
             r"8.7\!\times\!10^{4}$) from its freshly-trained head; median rather than mean is "
             r"used so this one cell does not dominate the table. Not a controlled ablation -- "
             r"no experiment here holds scale fixed while varying direction -- and not offered "
             r"to explain \DP's raw ungated $\mathrm{RR}$ relative to \NNPatch/\PatchNAS "
             r"(\Cref{tab:rq1}); source: \texttt{outputs/rq1\_margin\_decomposition/}.")


def rq2_mechanism() -> str:
    """T4/T5 (tab:rq2d/tab:rq2e): the mechanism question T1-T3 raise but do not answer -- post-
    info changes gate DECISIONS, but what SIGNAL in post-info is the gate reading? T4 (RQ3.3):
    mean confidence/entropy change by patch outcome, setting-balanced over 12 settings --
    successful repairs raise confidence and lower entropy, regressions do the reverse. T5
    (RQ3.9a): DynaPatch's own pre+post gate's post-only feature coefficients (named by a `p`/`d`
    prefix over the pre-only features), sorted by |coefficient| -- pP_max (patched confidence)
    and dm_r (delta margin ratio) carry the largest, sign-STABLE positive weight (mean
    coefficient == mean |coefficient|, i.e. same sign in all 12 settings); dH (entropy change)
    carries a negative weight -- exactly the RQ3.3 signature the gate would need to read to
    separate successful repairs from regressions."""
    resp = pd.read_csv(ROOT / "outputs/rq2_gate_mechanism_response.csv").set_index("outcome")
    order = ["successful repair", "ineffective change", "regression"]
    t4 = method_table(
        r"\textbf{RQ2:} Prediction response by patch outcome (RQ3.3), setting-balanced mean "
        r"over all 12 settings: successful repairs raise patched-response confidence and lower "
        r"entropy; regressions do the reverse.",
        "tab:rq2d", [r"$\Delta$Confidence", r"$\Delta$Entropy"],
        [o.capitalize() for o in order],
        [[float(resp.loc[o, "confidence_change"]), float(resp.loc[o, "entropy_change"])]
         for o in order],
        4, row_header="Outcome")

    feat = pd.read_csv(ROOT / "outputs/rq2_gate_mechanism_features.csv")
    FEATURE_NAME = {"pP_max": r"$p^P_{\max}$ (patched confidence)",
                    "dm_r": r"$\Delta m_r$ (delta margin ratio)",
                    "dH": r"$\Delta H$ (delta entropy)",
                    "dnorm_r": r"$\Delta\|\cdot\|_r$ (delta norm ratio)",
                    "dp_c": r"$\Delta p_c$ (delta prob.\ at base's top-1 class)",
                    "kl": r"$\mathrm{KL}$(base $\|$ patched)"}
    t5 = method_table(
        r"\textbf{RQ2:} DynaPatch's own pre+post gate (RQ3.9a): mean logistic-regression "
        r"coefficient for gain$=+1$ (beneficial), POST-ONLY features, sorted by "
        r"$|\mathrm{coefficient}|$, averaged over 12 settings. Positive = pushes the gate "
        r"toward committing the patch.",
        "tab:rq2e", ["Mean coefficient", r"Mean $|\mathrm{coefficient}|$"],
        [FEATURE_NAME[f] for f in feat.feature],
        [[float(r.mean_coefficient), float(r.mean_abs_coefficient)] for _, r in feat.iterrows()],
        4,
        note=r"Mean coefficient $=$ mean $|\mathrm{coefficient}|$ for $p^P_{\max}$/$\Delta m_r$ "
             r"means the sign is the SAME in all 12 settings (stable); $\Delta H$'s mean "
             r"($-0.29$) is smaller in magnitude than its mean $|\cdot|$ ($0.42$), i.e.\ its "
             r"sign is not always negative, but negative on average -- consistent with T4: "
             r"successful repairs lower entropy, regressions raise it, so a negative weight on "
             r"$\Delta H$ pushes the gate away from committing exactly the response pattern "
             r"regressions show.", row_header="Feature")
    return t4 + "\n" + t5


def rq1g() -> str:
    """T7 (tab:rq1g): closes the gap between failure-side mechanism (T5/T6) and clean-side
    damage (RQ1.10's Reg/CReg) by computing the exact parallel scale/margin decomposition on
    ORIGINALLY-CORRECT inputs instead of held-out failures (RQ1.14, 2026-09-06). Median over
    12 settings, regressed vs stayed-correct rows, all three per-input methods."""
    st = pd.read_csv(ROOT / "outputs/rq1_clean_margin_decomposition/per_setting.csv")
    med = st.groupby(["method", "outcome"])[["median_dz_norm", "median_dm"]].median()
    order = [("FixedPatch", r"\FPa"), ("DynaPatch-NoGate", r"\DPNoGate"),
            ("NN-Patching", r"\NNPatch"), ("PatchNAS", r"\PatchNAS")]
    row_labels = [lab for _, lab in order]
    rows = [[med.loc[(m, "regressed"), "median_dm"], med.loc[(m, "stayed_correct"), "median_dm"],
            med.loc[(m, "regressed"), "median_dz_norm"],
            med.loc[(m, "stayed_correct"), "median_dz_norm"]] for m, _ in order]
    return method_table(
        r"\textbf{RQ1:} Scale/margin decomposition on ORIGINALLY-CORRECT (clean) inputs, "
        r"regressed vs stayed-correct rows (RQ1.14) -- the exact parallel to T5/T6 but "
        r"anchored on preserving $y$ against its strongest base-model competitor $c^\star = "
        r"\arg\max_{k\neq y} z_{\mathrm{base}}(x)[k]$ rather than correcting a wrong "
        r"prediction. Median over 12 settings.",
        "tab:rq1g",
        [r"Median $\Delta m'$ (regressed)", r"Median $\Delta m'$ (stayed correct)",
         r"Median $\|\Delta z\|$ (regressed)", r"Median $\|\Delta z\|$ (stayed correct)"],
        row_labels, rows, [3, 3, 1, 1],
        note=r"All four methods show a clearly NEGATIVE median $\Delta m'$ on regressed rows "
             r"(margin eroded toward the competitor) vs.\ near-zero/positive on stayed-correct "
             r"rows -- the same mechanism family as T5/T6, now measured on the population Reg "
             r"is actually computed on. \FPa\ isolates direction from scale mechanically: since "
             r"it applies the SAME correction to every input, its $\|\Delta z\|$ is essentially "
             r"identical between regressed/stayed-correct rows (17.8 vs 17.7), yet $\Delta m'$ "
             r"still separates cleanly -- for a truly fixed correction, preserving correctness "
             r"is a pure direction question. \NNPatch/\PatchNAS/\DPNoGate's $\|\Delta z\|$ IS "
             r"larger on regressed rows, but the WITHIN-method gap ($\sim$10--15\%) is far "
             r"smaller than the CROSS-method gap T6 already reports -- the two must not be "
             r"conflated. Correlational, not a controlled ablation: consistent with, does not "
             r"prove, ``larger/less-aimed corrections cause more collateral damage.''")


def rq1_figure() -> str:
    """F1: the shuffle-distribution figure (scripts/plot_rq1_shuffle_distribution.py renders
    the png/pdf; this just emits the matching figure environment so it lives in the SAME merged
    rq1.tex as the four tables, not a separate file)."""
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=\linewidth]{figures/rq1/shuffle_distribution.pdf}",
        r"\caption{\textnormal{\textbf{RQ1:} Destroying which failure gets which correction "
        r"(direction shuffle) collapses $\mathrm{RR}_{\mathrm{held}}$ for every per-input "
        r"method (\DPNoGate/\NNPatch/\PatchNAS), not just \DP's own; destroying only "
        r"magnitude does not. Each box is the spread of that condition's per-cell mean "
        r"$\mathrm{RR}_{\mathrm{held}}$ across the 36 (setting $\times$ seed) cells "
        r"(RQ1.8/1.9); see \Cref{tab:rq1b} for the summary statistics and significance counts "
        r"this figure visualises, including the within-class/within-type conditions dropped "
        r"from this figure for space.}}",
        r"\label{fig:rq1}", r"\end{figure}",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="note/tables")
    a = ap.parse_args()
    out = ROOT / a.outdir
    out.mkdir(parents=True, exist_ok=True)
    # RQ1's four-method argument (T1-T5 + F1) merged into ONE file, per explicit request
    # (2026-09-06: consolidate RQ1 into one file instead of many small .tex files) -- previously five separate
    # files (rq1/rq1b/rq1c/rq1d.tex + fig_rq1.tex), now one, in argument order: headline
    # RR+Reg (T1) -> causal shuffle summary (T2) -> its figure (F1) -> per-class coverage (T3)
    # -> failure-type cosine geometry (T4) -> alignment-to-ideal-direction (T5) -> margin-gain
    # scale x direction decomposition (T6, 2026-09-06).
    rq1_merged = "\n".join([rq1(), rq1b(), rq1_figure(), rq1c(), rq1d(), rq1e(), rq1f(), rq1g()])
    # RQ2's post-info tables, same one-file-per-RQ convention as RQ1: T1 classifier quality ->
    # T2 deployed RR/Reg/CReg (both DynaPatch's own gate) -> T3 the same post-info question for
    # a patch mechanism DynaPatch had no part in generating -> T4/T5 the mechanism question
    # T1-T3 raise but do not answer (what SIGNAL in post-info is the gate reading) -> T6
    # (2026-09-06 renumbering: moved here from the old tab:rq3c -- net repairs-vs-regressions-
    # prevented from adding post-info is the same topic as T1-T5, not a separate RQ).
    rq2_merged = "\n".join([rq2_classification(), rq2_deployment(), rq2_transplant(),
                           rq2_mechanism(), rq2_postinfo_flip()])
    # RQ3 = comparison with existing methods (2026-09-06 renumbering, user confirmed: this
    # content was captioned "RQ4"/tab:rq4{a,b,c,d} -- old-RQ2's content under the pre-session
    # numbering, promoted to the paper's RQ3 the same way old-RQ3 was promoted to RQ2). Headline
    # RR/Reg (T1/T2, content unchanged) -> failure-type coverage/strength (T3) -> regression
    # breadth/severity (T4), the two tables explaining WHY the T1/T2 comparison looks this way.
    t3a, t3b = rq3_comparison()
    rq3_merged = "\n".join([t3a, t3b, rq3_coverage(), rq3_regression_profile()])
    # RQ4 = controllability (2026-09-06 renumbering: was captioned "RQ3"/tab:rq3{a,b} --
    # rq4_controllability()'s own pre-existing docstring/comment already said this belongs to
    # "RQ4 controllability", i.e. the mislabeling predates this session's renumbering work).
    # F1 (2026-09-06, added same session): the swept-threshold frontier figure -- the single
    # theta=0 point in T1/T2 does not by itself show the threshold is a controllable knob. T3
    # (recovered 2026-09-06, see rq4_lambda()'s own docstring): the cost-weighted cross-method
    # comparison (DynaPatch/NN-Patching/PatchNAS) that used to be a manually-appended block in
    # this file and was silently lost on a prior full-overwrite regeneration.
    t4a, t4b = rq4_controllability()
    rq4_merged = "\n".join([t4a, t4b, rq4_frontier_figure(), rq4_lambda(), rq4_mechanism()])
    tables = (("rq1", rq1_merged), ("rq2", rq2_merged), ("rq3", rq3_merged),
              ("rq4", rq4_merged))
    for name, tex in tables:
        (out / f"{name}.tex").write_text(tex)
        print(f"===== {name}.tex " + "=" * 60)
        print(tex)
    print(f"wrote {len(tables)} .tex files to {out}")


if __name__ == "__main__":
    main()
