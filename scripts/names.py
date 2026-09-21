#!/usr/bin/env python3
"""THE single source of truth for method, gate and setting names.

Why this file exists
--------------------
Names in this project have been wrong twice, in the same way both times: an output DIRECTORY
was named after the method someone intended to run, the method that actually ran was something
else, and every downstream table inherited the directory's name.

  outputs/.../distrrep/      ran plain full fine-tuning, NOT Li Calsi et al.'s DistrRep
  outputs/.../arachne_style/ ran a greedy top-k coordinate search, NOT Sohn et al.'s Arachne

Both were printed under the prior work's name in tables that were read and acted on. So:

  PATHS ARE NOT NAMES. Nothing downstream may derive a display label from a directory.
  Every table script imports from here, and here only.

The display names and LaTeX macros below are exactly the ones declared in the paper's
\\subsection{Compared Methods}. A method that is NOT in the paper's list carries
`in_paper=False`, and table scripts must either drop it or mark it as an unpublished
internal ablation. `check_paper_coverage()` fails loudly if the two drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- settings

# The paper's D-A scheme: D in {G,T,L}, A in {RN,CN,VG,DN}. Order is the table's column order.
SETTING_ORDER: list[tuple[str, str, str]] = [
    ("gtsrb", "resnet50", "G-RN"),
    ("gtsrb", "convnext_tiny", "G-CN"),
    ("gtsrb", "vgg16", "G-VG"),
    ("gtsrb", "densenet121", "G-DN"),
    ("tt100k_signs", "resnet50", "T-RN"),
    ("tt100k_signs", "convnext_tiny", "T-CN"),
    ("tt100k_signs", "vgg16", "T-VG"),
    ("tt100k_signs", "densenet121", "T-DN"),
    ("lisa_signs", "resnet50", "L-RN"),
    ("lisa_signs", "convnext_tiny", "L-CN"),
    ("lisa_signs", "vgg16", "L-VG"),
    ("lisa_signs", "densenet121", "L-DN"),
]
SETTING_LABEL = {f"{d}/{b}": lab for d, b, lab in SETTING_ORDER}

DATASET_LATEX = {"gtsrb": r"\GTSRB", "tt100k_signs": r"\TTK", "lisa_signs": r"\LISA"}
BACKBONE_LATEX = {"resnet50": r"\ResNet", "convnext_tiny": r"\ConvNeXt",
                  "vgg16": r"\VGG", "densenet121": r"\DenseNet"}

# --------------------------------------------------------------------------- methods


@dataclass(frozen=True)
class Method:
    key: str            # the stable identifier used everywhere in code
    display: str        # plain-text name for terminal tables
    latex: str          # the macro declared in the paper preamble
    in_paper: bool      # False => an internal ablation the paper does not claim
    note: str = ""      # what it actually is, when the name alone could mislead


METHODS: dict[str, Method] = {m.key: m for m in [
    # --- fine-tuning baselines (paper section "Fine-Tuning Baselines") ---
    Method("HeadFT", "HeadFT", r"\HeadFT", True,
           "backbone frozen, classification head updated on the repair data"),
    Method("FullFT", "FullFT", r"\FullFT", True,
           "all parameters updated. NOTE: this is what outputs/*/distrrep/ actually ran"),

    # --- weight-modification repair (paper section "Weight-Modification Repair") ---
    Method("Arachne", "Arachne", r"\Arachne", True,
           "Sohn et al. TOSEM'22, our PyTorch re-implementation: bidirectional Pareto "
           "localisation + differential evolution. Reads outputs/arachne_bs*_v8_*"),
    Method("DistrRep", "DistrRep", r"\DistrRep", True,
           "Li Calsi et al. ICST'23, our re-implementation: 3-phase distributed PSO. "
           "Reads outputs/distrepPSO_full_v8_* -- NEVER outputs/*/distrrep/"),

    # --- patch-based repair (paper section "Patch-Based Repair Baselines") ---
    Method("NNPatch", "NN-Patching", r"\NNPatch", True,
           "Kauschke & Fuernkranz: patch on the internal representation + error estimator"),
    Method("PatchNAS", "PatchNAS", r"\PatchNAS", True,
           "Fang et al. AAAI'23: frozen model, searched lightweight patch head"),

    # --- proposed method and variants (paper section "Proposed Method and Variants") ---
    Method("FP", "FixedPatch", r"\FPa", True,
           "one fixed patch parameter shared by every input"),
    Method("DPNoGate", "DynaPatch (ungated)", r"\DPNoGate", True,
           "input-conditioned patch, applied unconditionally"),
    Method("DP", "DynaPatch", r"\DP", True,
           "the full framework: DPGen proposes, DPGate authorises"),
]}

# The paper's \\begin{itemize} order in "Compared Methods".
PAPER_ORDER = ["HeadFT", "FullFT", "Arachne", "DistrRep", "NNPatch", "PatchNAS",
               "FP", "DPNoGate", "DP"]
INTERNAL_ORDER: list[str] = []

# --------------------------------------------------------------------------- row -> key

# The `method` column of outputs/rq1/comparison_baselines.csv -> the canonical key, for the ONE configuration
# each method is reported at. Configurations that exist on disk but are not the reported one
# (shorter epoch budgets, reduced search budgets, single sweep points) are deliberately absent:
# leaving them out here is what stops a stale variant from reaching a table.
RQ4_METHOD_TO_KEY: dict[str, str] = {
    "Head-Only Fine-Tuning (40ep)":            "HeadFT",
    "Full fine-tuning (40ep)":                 "FullFT",
    "Arachne(DE) (swept bound_scale)":         "Arachne",
    "DistRep(PSO) real, FULL budget (1 seed)": "DistrRep",
}

# Rows that must NEVER be read as a method: they are earlier, misnamed or lower-budget builds.
RQ4_METHOD_REJECT: dict[str, str] = {
    "Full fine-tuning [shipped as 'DistrRep'] (12ep)":
        "misnamed AND under-trained; use 'Full fine-tuning (40ep)' as FullFT",
    "DistRep(PSO) real, reduced budget":
        "3x15x15 / clean-cap 512, a lower bound; use the FULL budget row",
    "Head-Only Fine-Tuning (12ep, shipped)": "under-trained; use the 40ep row",
    "Weighted Retraining (12ep, shipped)":   "under-trained; use the 40ep row",
    "Always Patch (ungated)":
        "the 12-epoch SHIPPED patch. The gated rows and DPNoGate are both 40-epoch "
        "no-early-stop (ep40ns); mixing the two puts two training budgets in one column. "
        "DPNoGate is sourced from ungated_fixedpatch_dynapatch.csv's 'DynaPatch-NoGate (40ep, no early stop)'.",
    "DynaPatch [2nd round] @ CReg=0":        "an operating point, not a method; use gate_label()",
    # r=0.60/0.80/0.90 rows dropped entirely 2026-09-04 (not just renamed) -- both the CReg=0
    # and r-indexed operating points defined a reported number by searching or targeting
    # something on the split it is reported on. See gate_label_natural()'s docstring.
}


def gate_label(r: float) -> str:
    """Display name for the gated method at a target regression-removal fraction.

    SUPERSEDED 2026-09-04: no reported row is built at a target r any more (see
    GATE_NATURAL_POINT's docstring). Kept only so RQ4_METHOD_REJECT's old entries below still
    resolve; use gate_label_natural() for the one gated DynaPatch row that is reported now.
    """
    return f"{METHODS['DP'].display} (r={r:.2f})"


def gate_label_natural() -> str:
    """Display name for the gated method at the no-calibration natural threshold (theta=0)."""
    return f"{METHODS['DP'].display} (gated)"


def display(key: str) -> str:
    return METHODS[key].display


def latex(key: str) -> str:
    m = METHODS[key]
    if not m.in_paper:
        raise KeyError(f"{key} has no paper macro: {m.note}")
    return m.latex


def key_for_rq4_row(method_str: str) -> str | None:
    """Canonical key for an outputs/rq1/comparison_baselines.csv row, or None if the row is not a method."""
    if method_str in RQ4_METHOD_REJECT:
        return None
    return RQ4_METHOD_TO_KEY.get(method_str)


# --------------------------------------------------------------------------- gates


@dataclass(frozen=True)
class Gate:
    key: str            # the id used inside outputs/gate_zoo_curves*.csv
    display: str        # plain-text name for terminal tables
    protocol: str       # how it is fitted -- the thing that decides whether it is admissible
    admissible: bool    # False => must not be reported as a result
    note: str = ""


# `protocol` is not decoration. Two of these are fitted on rows that overlap the reporting rows
# at the IMAGE level, and one reads the ground-truth label of the input it is judging.
GATES: dict[str, Gate] = {g.key: g for g in [
    Gate("L4 pre+post", "DPGate", "leave-one-backbone-out", True,
         "OURS. 12 features (pre-repair + post-repair), 3-class logistic over "
         "gain in {-1,0,+1}, score = P(+1) - P(-1). Fitted on the other three "
         "backbones, so no row it scores took part in its own fit."),
    Gate("L2 pre-strong", "Pre-repair evidence", "leave-one-backbone-out", True,
         "base confidence, margin, entropy, margin rank, ||delta||. No patch output."),
    Gate("L3 post-only", "Post-repair evidence", "leave-one-backbone-out", True,
         "patch response only. No base-side evidence."),
    Gate("L1 magnitude", "Patch magnitude", "leave-one-backbone-out", True,
         "||delta|| alone -- the curve-sliding null, not a rival"),
    Gate("L5 pre label-free", "Label-free pre-repair", "leave-one-backbone-out", True,
         "entropy + PCS: the runtime-computable half of Ishimoto et al.'s features"),
    Gate("L6 GatedFusion", "Gated Fusion (stand-in)", "leave-one-backbone-out", True,
         "EACL'23 decision input, not their architecture. Label it a stand-in."),
    Gate("L7 PredictionUpdate", "Prediction update", "leave-one-backbone-out", True,
         "the positive-congruent / negative-flip-suppression family"),
    Gate("R1 post-confidence", "Patched confidence", "no fitting", True,
         "threshold on p^P_max. Cannot see the base model at all."),
    Gate("R2 post-uncertainty", "Patched entropy", "no fitting", True, ""),
    Gate("R3 prediction-update", "Confidence gain", "no fitting", True,
         "p^P_max - p^B_max: adopt the new answer only when it is more confident"),
    Gate("R4 pre-confidence", "Base confidence", "no fitting", True,
         "do not touch inputs the base was sure about"),
    Gate("Z0 label oracle (upper bound)", "Label oracle (upper bound)", "label-privileged", False,
         "reads p^B[y] of the input being judged. A ceiling, never a rival."),
    # W1 / W2 / P1 (within setting, cross-seed) were REMOVED on 2026-09-01, generator and
    # curve rows both. The three bug-split seeds share one frozen backbone and their clean_test
    # sets overlap ~80% at the image level, so those gates were fitted on the rows they scored.
]}

SHIPPED_GATE = "L4 pre+post"      # the FEATURE SET key inside gate_zoo.LEARNED

# ---- the shipped gate PROTOCOL -------------------------------------------------------------
# 2026-09-01: the gate is fitted WITHIN the setting, on bug_train + bug_val + clean_calib of
# the same setting and seed (scripts/gate_protocol_b.py --train-on both). Leave-one-backbone-out
# is no longer the shipped protocol and must not appear in the paper.
#
# Measured cost of the switch (scripts/protocol_ab_tables.py prints both side by side):
#   RQ3 mean RR_held   r=.60 .5703 -> .5621   r=.80 .5552 -> .5556   r=.90 .5410 -> .5307
#   Reg / CReg         identical to 4 decimals, marginally LOWER under C at every r
#   RQ4 ranking        unchanged
# The one thing that weakens is the pre+post vs pre-only ablation (11/0/1 -> 6/1/5 at r=.95),
# because C has far fewer negative examples to estimate the 6 post-repair dimensions from.
# REMOVED 2026-09-04: GATE_CURVES (the frontier over the full QGRID) is a fitting-side and
# frontier-shape diagnostic ONLY. Picking a single row from it -- "the q whose frac_Reg_removed
# on S_held+S_clean^test crosses r, then take the highest RR_held among those" -- selects that
# operating point using the SAME population RR_held/Reg/CReg are then reported on
# (scripts/gate_zoo.py:operating_points() computes and evaluates on the same (c, h) argument).
# `outputs/rq4/gate_curves_protocolC_ep40ns.csv` was confirmed byte-identical to
# `outputs/gate_ablation_raw/curves_pre_post.csv` (2026-09-04), and this is exactly the pattern
# that file's consumers (`scripts/table_rq4_final.py:dynapatch_rows()`,
# `scripts/best_data.py`'s gated-DynaPatch block) used to pick every reported r=0.60/0.80/0.90
# row. Fixed 2026-09-04, user decision: no paper table may pick a reported operating point off
# a curve fitted AND evaluated on the report population, ever again.
#
# SUPERSEDED 2026-09-04 (same day): GATE_DEPLOY_POINTS was the first fix -- theta chosen by
# targeting r on clean_calib/bug_val's OWN frontier instead of the report population. User
# caught a remaining problem: under the shipped "both" fit protocol, clean_calib/bug_val is
# ALSO part of what fits the gate (see GATE_PROTOCOL above), so a threshold picked from that
# same population is in-sample for the MODEL even though it never touches the report
# population -- not fully "clean" calibration, and the natural suspect for why realised r
# systematically undershot the target (the model's own score distribution looks more
# separated on rows it was fit on than on genuinely unseen rows). Rather than carve out yet
# another disjoint split (shrinking an already tiny D_gate further, worst for LISA), the user's
# decision: drop r-targeting and calibration entirely.
#
# GATE_NATURAL_POINT is the replacement and the ONLY admissible source for a gated DynaPatch
# number now: theta=0 -- commit iff the fitted model's own score favours beneficial over
# harmful (P(gain=+1) > P(gain=-1)). No target r, no threshold search of any kind, no
# calibration split, nothing that could be in-sample OR eval-set-selected -- the model's raw
# class decision, evaluated once on S_held + S_clean^test. One row per (setting, seed).
GATE_NATURAL_POINT = "outputs/rq4/gate_natural_point_protocolC_ep40ns.csv"
GATE_CURVE_NAME = "DPGate"          # the `gate` column inside GATE_NATURAL_POINT / GATE_CURVES
GATE_PROTOCOL = ("fitted within setting on bug_train + bug_val + clean_calib; theta=0 (no "
                 "target r, no calibration split); reported on S_held + S_clean^test")

# Under this protocol the gate is fitted on bug_train, so a repair rate measured on the
# evidence set is IN-SAMPLE for the gate. RR_seen is therefore not reportable for gated rows.
GATED_RR_SEEN_REPORTABLE = False

# Kept for full-frontier PLOTS ONLY (e.g. a Reg-vs-RR_held curve figure showing every q) --
# never for picking the row a table reports as "DynaPatch @ r=X". See the removal note above.
GATE_CURVES = "outputs/rq4/gate_curves_protocolC_ep40ns.csv"

def gate_display(key: str) -> str:
    return GATES[key].display


def assert_gate_admissible(key: str) -> str:
    g = GATES[key]
    if not g.admissible:
        raise ValueError(f"gate {key!r} ({g.display}) is not reportable: {g.note}")
    return key


# --------------------------------------------------------------------------- guards


def check_paper_coverage() -> list[str]:
    """Return the discrepancies between this file and the paper's method list."""
    problems = []
    for k in PAPER_ORDER:
        if k not in METHODS:
            problems.append(f"paper lists {k} but METHODS has no entry")
        elif not METHODS[k].in_paper:
            problems.append(f"{k} is in PAPER_ORDER but marked in_paper=False")
    for k, m in METHODS.items():
        if m.in_paper and k not in PAPER_ORDER:
            problems.append(f"{k} is marked in_paper but is not in PAPER_ORDER")
        if m.in_paper and not m.latex:
            problems.append(f"{k} is marked in_paper but declares no LaTeX macro")
    covered = set(RQ4_METHOD_TO_KEY.values())
    for k in PAPER_ORDER:
        if k in ("DP", "DPNoGate", "FP", "NNPatch", "PatchNAS"):
            continue  # sourced from the gate curves / ALL_RQ_DATA / prior-patch json
        if k not in covered:
            problems.append(f"{k} has no row mapping in RQ4_METHOD_TO_KEY")
    return problems


if __name__ == "__main__":
    probs = check_paper_coverage()
    print("=== paper method list ===")
    for k in PAPER_ORDER:
        m = METHODS[k]
        print(f"  {m.latex:12s} {m.display:22s} {m.note}")
    print("\n=== internal only (must be marked, or dropped) ===")
    for k in INTERNAL_ORDER:
        m = METHODS[k]
        print(f"  {'--':12s} {m.display:22s} {m.note}")
    print("\n=== gates NOT reportable ===")
    for g in GATES.values():
        if not g.admissible:
            print(f"  {g.key:32s} ({g.protocol}) {g.note.splitlines()[0]}")
    print("\n" + ("OK: names.py agrees with the paper" if not probs else "PROBLEMS:"))
    for p in probs:
        print("  " + p)
