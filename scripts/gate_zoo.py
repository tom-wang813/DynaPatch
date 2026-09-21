#!/usr/bin/env python3
"""RQ1: do EXISTING deployable runtime gates all collapse under stringent regression control?

Why this script exists
----------------------
Section 12/13 compared our pre+post gate against our own pre-only gate and found a crossover at
r* ~ 0.75 of the baseline regression removed. That is a comparison against ourselves. It cannot
support the claim "existing runtime gates fail in the high-assurance regime", because the only
rival on that frontier was a baseline we built. This script assembles the actual gate families
from the literature and puts every one of them on the SAME frontier, so the failure -- if it is
real -- is visible as a property of the regime rather than of one weak baseline.

The zoo (all deployable: nothing reads the ground-truth label of the input being judged)
-----------------------------------------------------------------------------------------
Learned, leave-one-backbone-out, identical 3-class expected-gain target (probe_gonogo):
  L1 magnitude          ||delta|| only -- the curve-sliding null, not a rival
  L2 pre-strong         learned pre-repair gate: p^B_max, margin, entropy, rank-margin, ||delta||
  L3 post-only          post-repair response only
  L4 pre+post           OURS: the union
  L5 pre label-free     Entropy + PCS, i.e. the runtime-computable half of Ishimoto et al.'s
                        feature set (LPS and Loss need y and are drawn as a ceiling elsewhere)
  L6 GatedFusion        p^P_max + H(p^P): a stand-in for EACL'23 Gated Fusion, which mixes old
                        and new model predictions from a learned gate. Their gate reads a single
                        model's latent and co-trains the new model; neither is available for a
                        frozen third-party repair, so this is their DECISION INPUT, not their
                        architecture. Marked as a stand-in wherever it is reported.
  L7 PredictionUpdate   dp_max + dm_r + flip: the prediction-update family (Yan et al.'s
                        positive-congruent line), which decides per input whether to adopt the
                        new model's answer in order to suppress negative flips.

Fixed rules, NO fitting at all (a deployed system can run these on day zero):
  R1 post-confidence    score = p^P_max
  R2 post-uncertainty   score = -H(p^P)
  R3 prediction-update  score = p^P_max - p^B_max  -- adopt the new answer only when the patched
                        model is more confident than the base. This is the textbook update rule.
  R4 pre-confidence     score = -p^B_max           -- do not touch inputs the base was sure about

REMOVED 2026-09-01 -- the within-setting protocol is leaky (see WITHIN below), so P1 and the
W* protocol controls no longer run and no longer appear in any curve file:
  P1 PatchNAS-style     logistic probe on the router's latent features. PatchNAS (AAAI'23) gates
                        a frozen-backbone patch at deployment time from intermediate-feature
                        evidence. A latent coordinate has no meaning across architectures, so
                        this gate CANNOT be run leave-one-backbone-out; it is therefore given the
                        EASIER within-setting protocol. If it still fails, that is not a protocol
                        artefact -- it had the advantage.

The frontier
------------
Identical to scripts/curve_risk_coverage_crossover.py: sweep the veto fraction q over flipped
rows, and read RR_held against the fraction of baseline regression removed. Gates are compared at
matched regression removed, never at matched q, because the deployment question is "for the safety
I bought, what did I pay".

CReg is carried alongside Reg on every point of every curve so that
scripts/criticality_diagnosis.py can re-read the same frontier on the critical subpopulation
without refitting anything.

Usage:
  .venv/bin/python scripts/gate_zoo.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

import probe_gonogo_pre_vs_prepost as G  # noqa: E402  (same directory)
import analyze_response_gate_lobo as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DSS, BBS, SEEDS = L.DSS, L.BBS, L.SEEDS

# ---- the zoo -------------------------------------------------------------------------------

# RANK_FEATS = {"m_base_r", "dm_r", "dnorm_r"} are computed by ranking each evaluation split
# against ITSELF (analyze_response_gate_lobo.py's rank()), not against a mapping fixed on repair
# data -- a flagged input's own gate feature then depends on which other inputs happen to be in
# the same evaluation batch, contrary to the "per-input, one at a time" deployment story this
# whole zoo is supposed to satisfy (see scripts/analysis_gate_norank_ablation.py). Fixed 2026-09-07:
# `L4 pre+post` (== names.SHIPPED_GATE, the gate every RQ1-4 headline number is fitted with) had
# all 3 removed -- 12 features down to 9 -- confirmed by rerunning
# scripts/analysis_gate_norank_ablation.py (12feat RR_held=0.4936/Reg=0.0009/CReg=0.0010 vs the
# 9feat honest refit RR_held=0.4505/Reg=0.0041/CReg=0.0028, natural threshold, 12-setting mean).
# The pre-fix 12-feature list is kept below under an explicit legacy name so
# analysis_gate_norank_ablation.py/analysis_gate_norank_full.py can still compute that comparison
# without silently picking up whatever L4 pre+post happens to mean at import time.
# NOT YET FIXED, flagged rather than silently left inconsistent: `L1 magnitude` (pure dnorm_r) and
# `L2 pre-strong` (contains m_base_r, dnorm_r) share the identical rank-feature legitimacy issue
# and have not been re-derived -- neither is the paper's shipped/headline gate, so this was out of
# scope for the 2026-09-07 fix, but a rerun of either should not be trusted as deployable without
# the same treatment.
LEARNED = {
    "L0 pre-only (DPInput)": ["pB_max", "pB_margin", "H_base"],
    "L1 magnitude":        ["dnorm_r"],
    "L2 pre-strong":       ["pB_max", "pB_margin", "H_base", "m_base_r", "dnorm_r"],
    "L3 post-only":        ["pP_max", "kl", "dH", "dp_c", "rho", "rho_worst", "dm_r"],
    "L4-legacy pre+post (12feat, non-deployable rank feats, pre-2026-09-07)":
                           ["pB_max", "pB_margin", "H_base", "m_base_r", "dnorm_r",
                            "pP_max", "kl", "dH", "dp_c", "rho", "rho_worst", "dm_r"],
    "L4 pre+post":         ["pB_max", "pB_margin", "H_base",
                            "pP_max", "kl", "dH", "dp_c", "rho", "rho_worst"],
    "L5 pre label-free":   ["H_base", "pB_margin"],
    "L6 GatedFusion":      ["pP_max", "H_patched"],
    # `flip` is deliberately absent: the threshold only ever ranks rows the patch already
    # flips, so the indicator is constant on the actionable set and contributes nothing.
    "L7 PredictionUpdate": ["dp_max", "dm_r", "dp_c"],
}
# NOT a rival and NOT deployable: LPS and Loss read p^B[y], the ground-truth label OF THE INPUT
# BEING JUDGED. A runtime gate does not have it -- if it did, it would already know whether the
# base was right and would not need a gate at all. Carried only to draw the ceiling, so that
# "how much of the headroom a label would buy did we capture" is a number and not a gesture.
ORACLE = "Z0 label oracle (upper bound)"
LEARNED[ORACLE] = ["H_base", "pB_margin", "lps", "nll"]
# name -> (feature name, sign). score = sign * f[name]; higher score = authorise the flip.
FIXED = {
    "R1 post-confidence":   ("pP_max", +1.0),
    "R2 post-uncertainty":  ("H_patched", -1.0),
    "R3 prediction-update": ("dp_max", +1.0),
    "R4 pre-confidence":    ("pB_max", -1.0),
}
LATENT = "P1 PatchNAS-style"
# Protocol-matched controls. P1 cannot be run leave-one-backbone-out (a latent coordinate does
# not survive a change of architecture), so it is fit within setting -- the easier protocol. To
# know whether its standing is information or protocol, the same two feature sets that carry the
# main claim are re-fit under P1's EXACT protocol and reported beside it.
# REMOVED 2026-09-01: the within-setting, cross-seed protocol is LEAKY and must not be
# reported. The three bug-split seeds share ONE frozen backbone checkpoint (seed only
# re-shuffles the bug/clean partition), and their clean_test sets overlap ~80% at the image
# level (measured: gtsrb/rn50 8017/10027, tt100k/dn121 5709/7155, lisa/cvt 1522/1899). So the
# pre-repair features of a scored row are bit-identical to a row the model was fitted on, and
# W1's parity with the LOBO gate was never evidence that LOBO is unnecessary.
# P1 (the latent probe) sat on the same protocol and is removed with it.
WITHIN: dict[str, list[str]] = {}

QGRID = np.clip(np.concatenate([np.arange(0.0, 0.30, 0.01),
                                np.arange(0.30, 1.0001, 0.025)]), 0.0, 1.0)
RGRID = np.arange(0.05, 0.96, 0.05)
CURVE_CSV = f"outputs/gate_zoo_curves{L.DUMP_TAG}.csv"
# the tuple operating_points() returns: N_CURVES q-indexed arrays, then N_FIELDS-N_CURVES scalars
CURVE_KEYS = ("Reg", "CReg", "Risk", "RR")
SCALAR_KEYS = ("Reg_base", "CReg_base", "Risk_base", "RR_base",
               "n_crit", "n_creg", "n_clean", "n_held")
N_CURVES, N_FIELDS = len(CURVE_KEYS), len(CURVE_KEYS) + len(SCALAR_KEYS)


# ---- criticality --------------------------------------------------------------------------


def critical_set(ds: str) -> set[int]:
    j = json.loads((ROOT / f"artifacts/risk/{ds}_safety_risk_matrix.json").read_text())
    return {int(i) for v in j["critical_signs"].values() for i in v}


def risk_matrix(ds: str) -> np.ndarray:
    """Cost-weighted confusion matrix W[y_true, y_pred], committed in configs/risk/*.yaml.

    Zero diagonal, off-diagonal 1-6, hand-specified from sign semantics (a speed limit read as a
    less conservative speed limit costs 4, any error on a critical sign costs 6, ...). These
    weights are NOT invented by this analysis: they predate it and live in the repo config. They
    are heuristic, and every weighted number below is reported as such.
    """
    j = json.loads((ROOT / f"artifacts/risk/{ds}_safety_risk_matrix.json").read_text())
    return np.asarray(j["risk_matrix"], dtype=np.float64)


def clean_rows(seed: int, ds: str, bb: str) -> list[dict] | None:
    """clean_test rows of the effect dump, in dump order (the order the features are in)."""
    # MUST carry L.DUMP_TAG: these rows supply the criticality labels for the cached features,
    # so reading them from a different patch than the features came from mixes two systems.
    # attach_criticality's guard catches it (it did, on 127 rows), but the fix belongs here.
    p = (ROOT / f"outputs/effect_dump{L.DUMP_TAG}_v8_s{seed}/{ds}/{bb}/deploy_direct/predictions"
         / "clean_eval_predictions.csv")
    if not p.exists():
        return None
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    keep = G.clean_test_idx(seed, ds, bb)
    return [r for r in rows if keep is None or int(r["dataset_index"]) in keep]


def attach_idx(cells: dict) -> dict:
    """Re-attach the dataset indices build_cells() drops, so latents can be aligned by index.

    The clean side is re-filtered with the same clean_test mask build_cells() applies, and both
    sides are length-checked against the feature arrays.
    """
    harm, help_ = L.collect("harm"), L.collect("help")
    for k, cell in cells.items():
        ci = harm[k]["idx"]
        keep = G.clean_test_idx(*k)
        if keep is not None:
            ci = ci[np.fromiter((int(i) in keep for i in ci), dtype=bool, count=len(ci))]
        if len(ci) != cell["clean"]["n"] or len(help_[k]["idx"]) != cell["held"]["n"]:
            raise SystemExit(f"index re-attach failed for {k}")
        cell["clean"]["idx"] = ci
        cell["held"]["idx"] = help_[k]["idx"]
    return cells


def attach_criticality(cells: dict) -> dict:
    """Add label / base_pred / patched_pred / is_critical to the clean side of every cell.

    Alignment is NOT assumed: the row count and the regression indicator must both agree with
    the cached feature arrays, or the cell is refused. A silent misalignment here would put the
    criticality verdict on the wrong rows, which is exactly the class of bug that has bitten this
    project before (note/PITFALLS.md).
    """
    for k, cell in list(cells.items()):
        rows = clean_rows(*k)
        c = cell["clean"]
        if rows is None or len(rows) != c["n"]:
            raise SystemExit(f"criticality alignment failed for {k}: "
                             f"{0 if rows is None else len(rows)} csv rows vs {c['n']} features")
        reg = np.array([r["regressed"] == "True" for r in rows])
        if not np.array_equal(reg, c["y"].astype(bool)):
            raise SystemExit(f"criticality alignment failed for {k}: `regressed` column disagrees "
                             f"with the cached label on {int((reg != c['y']).sum())} rows")
        crit = critical_set(k[1])
        c["label"] = np.array([int(r["label"]) for r in rows])
        c["patched_pred"] = np.array([int(r["patched_pred"]) for r in rows])
        c["crit"] = np.isin(c["label"], list(crit))
        # per-row cost of committing this flip. W has a zero diagonal and every clean_test row is
        # base-correct, so this is automatically 0 on the rows that do not regress.
        c["w"] = risk_matrix(k[1])[c["label"], c["patched_pred"]]
    return cells


# ---- latent gate --------------------------------------------------------------------------


def latent_for(seed: int, ds: str, bb: str) -> dict | None:
    """DISABLED. Router latent features for the two populations.

    P1 could only ever be fitted within setting, cross-fitted over the three bug-split seeds --
    and that protocol was removed on 2026-09-01 as leaky (see WITHIN). Returning None here
    keeps the call sites intact while guaranteeing no P1 row can reach a curve file.
    """
    return None

    # The routefeat tree exists only for the shipped patch. Under a DUMP_TAG the latent gate
    # would be fit on one system's latents against another system's labels, so it is skipped
    # (P1 is a rival gate, not ours; its absence does not affect the pre+post curves).
    if L.DUMP_TAG:
        return None
    out = {}
    for side, split in (("clean", "clean_eval"), ("held", "repair_holdout_unseen")):
        rd = ROOT / f"outputs/effect_dump_routefeat_v8_s{seed}/{ds}/{bb}/deploy_direct/predictions"
        rf, ri = rd / f"route_features_{split}.npy", rd / f"dataset_indices_{split}.npy"
        if not (rf.exists() and ri.exists()):
            return None
        out[side] = (np.load(rf, mmap_mode="r"), np.load(ri))
    return out


def latent_matrix(entry, idx: np.ndarray) -> np.ndarray | None:
    arr, ridx = entry
    order = {int(v): i for i, v in enumerate(ridx)}
    if not set(int(v) for v in idx) <= set(order):
        return None
    return np.asarray(arr[[order[int(v)] for v in idx]], dtype=np.float64)


# ---- frontier -----------------------------------------------------------------------------


def add_derived(cells: dict) -> dict:
    for k in cells:
        for side in ("clean", "held"):
            f = cells[k][side]["f"]
            if "H_patched" not in f:
                f["H_patched"] = f["H_base"] + f["dH"]
    return cells


def operating_points(c: dict, h: dict, sc: np.ndarray, sh: np.ndarray) -> tuple:
    """Exact (Reg, CReg, RR_held) at every veto fraction q. Vetoing reverts the row to base."""
    pool = np.concatenate([sc[c["flip"]], sh[h["flip"]]])
    if not len(pool):
        return None
    cd = int(c["crit"].sum())
    reg, creg, risk, rr = [], [], [], []
    for q in QGRID:
        t = -np.inf if q <= 0 else np.quantile(pool, q)
        kc, kh = ~c["flip"] | (sc > t), ~h["flip"] | (sh > t)
        reg.append(c["y"][kc].sum() / c["n"])
        creg.append((c["y"] & kc & c["crit"]).sum() / max(cd, 1))
        risk.append(c["w"][kc].sum() / c["n"])
        rr.append(h["y"][kh].sum() / h["n"])
    return (np.array(reg), np.array(creg), np.array(risk), np.array(rr),
            c["y"].sum() / c["n"], (c["y"] & c["crit"]).sum() / max(cd, 1),
            c["w"].sum() / c["n"], h["y"].sum() / h["n"],
            cd, int((c["y"] & c["crit"]).sum()), c["n"], h["n"])


def _pool_seeds(per_setting: dict) -> dict:
    """Average the three seeds of a setting point-wise on the q grid."""
    out = {}
    for st, lst in per_setting.items():
        out[st] = tuple(np.mean([x[i] for x in lst], axis=0) for i in range(N_CURVES)) + tuple(
            float(np.mean([x[i] for x in lst])) for i in range(N_CURVES, N_FIELDS))
    return out


def curves(cells: dict) -> dict:
    """{(gate, (ds, bb)): (Reg, CReg, RR, Reg0, CReg0, RR0, n_crit, n_creg)}."""
    out: dict = {}

    # --- learned gates: leave one backbone out
    for gate, names in LEARNED.items():
        per_setting: dict = {}
        for held_bb in BBS:
            tr = [k for k in cells if k[2] != held_bb]
            te = [k for k in cells if k[2] == held_bb]
            X, g = [], []
            for k in tr:
                for side in ("clean", "held"):
                    d = cells[k][side]
                    X.append(G.mat(d["f"], names, d["flip"])); g.append(d["gain"][d["flip"]])
            mdl = G.fit_gain(np.vstack(X), np.concatenate(g))
            if mdl is None:
                continue
            for k in te:
                c, h = cells[k]["clean"], cells[k]["held"]
                op = operating_points(c, h, G.score(mdl, G.mat(c["f"], names)),
                                      G.score(mdl, G.mat(h["f"], names)))
                if op:
                    per_setting.setdefault((k[1], k[2]), []).append(op)
        for st, v in _pool_seeds(per_setting).items():
            out[(gate, st)] = v

    # --- fixed rules: nothing is fitted, so every cell is scored directly
    for gate, (name, sign) in FIXED.items():
        per_setting = {}
        for k, cell in cells.items():
            c, h = cell["clean"], cell["held"]
            op = operating_points(c, h, sign * c["f"][name], sign * h["f"][name])
            if op:
                per_setting.setdefault((k[1], k[2]), []).append(op)
        for st, v in _pool_seeds(per_setting).items():
            out[(gate, st)] = v

    # --- within-setting gates, cross-fitted over the three seeds (the EASIER protocol)
    for gate, names in WITHIN.items():
        per_setting = {}
        for ds, bb in itertools.product(DSS, BBS):
            if any((s, ds, bb) not in cells for s in SEEDS):
                continue
            for s in SEEDS:
                X, g = [], []
                for t in (t for t in SEEDS if t != s):
                    for side in ("clean", "held"):
                        d = cells[(t, ds, bb)][side]
                        X.append(G.mat(d["f"], names, d["flip"])); g.append(d["gain"][d["flip"]])
                mdl = G.fit_gain(np.vstack(X), np.concatenate(g))
                if mdl is None:
                    continue
                c, h = cells[(s, ds, bb)]["clean"], cells[(s, ds, bb)]["held"]
                op = operating_points(c, h, G.score(mdl, G.mat(c["f"], names)),
                                      G.score(mdl, G.mat(h["f"], names)))
                if op:
                    per_setting.setdefault((ds, bb), []).append(op)
        for st, v in _pool_seeds(per_setting).items():
            out[(gate, st)] = v

    # --- latent gate: within setting, cross-fitted over seeds (the EASIER protocol)
    per_setting = {}
    for ds, bb in itertools.product(DSS, BBS):
        lat = {s: latent_for(s, ds, bb) for s in SEEDS}
        if any(v is None for v in lat.values()):
            continue
        mats = {}
        ok = True
        for s in SEEDS:
            k = (s, ds, bb)
            if k not in cells:
                ok = False
                break
            for side in ("clean", "held"):
                m = latent_matrix(lat[s][side], cells[k][side]["idx"])
                if m is None or len(m) != cells[k][side]["n"]:
                    ok = False
                mats[(s, side)] = m
        if not ok:
            continue
        for s in SEEDS:
            tr = [t for t in SEEDS if t != s]
            X, g = [], []
            for t in tr:
                for side in ("clean", "held"):
                    d = cells[(t, ds, bb)][side]
                    X.append(mats[(t, side)][d["flip"]]); g.append(d["gain"][d["flip"]])
            mdl = G.fit_gain(np.vstack(X), np.concatenate(g))
            if mdl is None:
                continue
            c, h = cells[(s, ds, bb)]["clean"], cells[(s, ds, bb)]["held"]
            op = operating_points(c, h, G.score(mdl, mats[(s, "clean")]),
                                  G.score(mdl, mats[(s, "held")]))
            if op:
                per_setting.setdefault((ds, bb), []).append(op)
    for st, v in _pool_seeds(per_setting).items():
        out[(LATENT, st)] = v
    return out


def at_removed(x: np.ndarray, y: np.ndarray, x0: float, r: np.ndarray) -> np.ndarray:
    """y interpolated at 'fraction r of the baseline x removed'."""
    if x0 <= 0:
        return np.full_like(r, np.nan, dtype=float)
    removed = (x0 - x) / x0
    o = np.argsort(removed)
    rem, yy = removed[o], y[o]
    keep = np.concatenate([[True], np.diff(rem) > 0])
    return np.interp(r, rem[keep], yy[keep], left=np.nan, right=np.nan)


# ---- reporting ----------------------------------------------------------------------------


def gates_in_order() -> list[str]:
    """The zoo as reported: every gate under its own native protocol."""
    return list(LEARNED) + list(FIXED)


def protocol_matched() -> list[str]:
    """Empty since the within-setting protocol was removed as leaky (see WITHIN above)."""
    return []


def write_csv(cv: dict, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gate", "dataset", "backbone", "q", "Reg", "CReg", "Risk", "RR_held",
                    "Reg_base", "CReg_base", "Risk_base", "RR_base",
                    "n_crit", "n_creg", "n_clean", "n_held",
                    "frac_Reg_removed", "frac_CReg_removed", "frac_Risk_removed"])
        for (gate, (ds, bb)), v in sorted(cv.items()):
            reg, creg, risk, rr, reg0, creg0, risk0, rr0, ncrit, ncreg, nclean, nheld = v
            for i in range(len(QGRID)):
                w.writerow([gate, ds, bb, f"{QGRID[i]:.4f}", f"{reg[i]:.6f}", f"{creg[i]:.6f}",
                            f"{risk[i]:.6f}", f"{rr[i]:.6f}", f"{reg0:.6f}", f"{creg0:.6f}",
                            f"{risk0:.6f}", f"{rr0:.6f}",
                            f"{ncrit:.1f}", f"{ncreg:.1f}", f"{nclean:.1f}", f"{nheld:.1f}",
                            f"{((reg0 - reg[i]) / reg0 if reg0 else 0):.4f}",
                            f"{((creg0 - creg[i]) / creg0 if creg0 else 0):.4f}",
                            f"{((risk0 - risk[i]) / risk0 if risk0 else 0):.4f}"])


def frontier_table(cv: dict, settings: list, r_at: float) -> dict:
    """RR_held retained by every gate at a fixed fraction of baseline Reg removed."""
    gates = gates_in_order()
    print(f"\n== r = {r_at:.2f} of baseline Reg removed  (RR_held retained; higher is better)")
    print(f"{'setting':<22}" + "".join(f"{g.split(' ', 1)[1][:11]:>13}" for g in gates))
    print("-" * (22 + 13 * len(gates)))
    best = {g: 0 for g in gates}
    vals_by_setting = {}
    for st in settings:
        vals = {}
        for g in gates:
            e = cv.get((g, st))
            if not e:
                continue
            v = at_removed(e[0], e[3], e[4], np.array([r_at]))[0]
            if np.isfinite(v):
                vals[g] = float(v)
        if not vals:
            continue
        vals_by_setting[st] = vals
        top = max(vals.values())
        for g, v in vals.items():
            if v >= top - 1e-9:
                best[g] += 1
        print(f"{st[0].replace('_signs', '') + '/' + st[1]:<22}"
              + "".join((f"{vals[g]:>12.3f}" + ("*" if vals[g] >= top - 1e-9 else " "))
                        if g in vals else f"{'--':>13}" for g in gates))
    print("-" * (22 + 13 * len(gates)))
    print(f"{'best-in-row':<22}" + "".join(f"{best[g]:>13}" for g in gates))
    return vals_by_setting


def collapse_table(cv: dict, settings: list) -> None:
    """The RQ1 claim, stated as a number: what fraction of its own low-assurance repair rate
    does each gate still hold at r = 0.9?"""
    gates = gates_in_order()
    print("\n\nHIGH-ASSURANCE COLLAPSE: RR_held at r=0.90 as a fraction of RR_held at r=0.40")
    print("(1.00 = the gate gives up nothing extra to go from lax to strict; "
          "'--' = the curve never reaches r)")
    print(f"\n{'setting':<22}" + "".join(f"{g.split(' ', 1)[1][:11]:>13}" for g in gates))
    print("-" * (22 + 13 * len(gates)))
    ratios = {g: [] for g in gates}
    for st in settings:
        cells = {}
        for g in gates:
            e = cv.get((g, st))
            if not e:
                continue
            lo = at_removed(e[0], e[3], e[4], np.array([0.40]))[0]
            hi = at_removed(e[0], e[3], e[4], np.array([0.90]))[0]
            if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
                cells[g] = hi / lo
                ratios[g].append(hi / lo)
        print(f"{st[0].replace('_signs', '') + '/' + st[1]:<22}"
              + "".join(f"{cells[g]:>13.2f}" if g in cells else f"{'--':>13}" for g in gates))
    print("-" * (22 + 13 * len(gates)))
    print(f"{'median':<22}" + "".join(
        f"{np.median(ratios[g]):>13.2f}" if ratios[g] else f"{'--':>13}" for g in gates))
    print(f"{'n settings':<22}" + "".join(f"{len(ratios[g]):>13}" for g in gates))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CURVE_CSV)
    a = ap.parse_args()

    cells = attach_criticality(attach_idx(add_derived(G.build_cells())))
    print(f"cells: {len(cells)}  ({len(SEEDS)} seeds x 12 settings expected)")
    cv = curves(cells)
    settings = sorted({k[1] for k in cv})
    print(f"gates: {len(gates_in_order())}   curves: {len(cv)}")

    (ROOT / a.csv).parent.mkdir(parents=True, exist_ok=True)
    write_csv(cv, ROOT / a.csv)
    print(f"wrote {ROOT / a.csv}")

    print("\n" + "=" * 100)
    print("RQ1  THE GATE ZOO ON ONE FRONTIER   (every learned gate leave-one-backbone-out)")
    print("=" * 100)
    for r_at in (0.60, 0.80, 0.90):
        frontier_table(cv, settings, r_at)
    collapse_table(cv, settings)

    # REMOVED 2026-09-01 with the within-setting protocol. This block existed only to give
    # P1 a protocol-matched control; both P1 and the W* controls were fitted on rows that
    # overlap the reported rows at the image level, so nothing here was reportable.

    keys = CURVE_KEYS + SCALAR_KEYS
    with (ROOT / f"outputs/gate_zoo_curves{L.DUMP_TAG}.pkl").open("wb") as fh:
        pickle.dump({k: dict(zip(keys, v)) for k, v in cv.items()}, fh)
    json.dump({f"{g}|{s[0]}/{s[1]}": dict(zip(SCALAR_KEYS, v[N_CURVES:]))
               for (g, s), v in cv.items()},
              (ROOT / f"outputs/gate_zoo_meta{L.DUMP_TAG}.json").open("w"), indent=1)
    print(f"\nwrote {ROOT / f'outputs/gate_zoo_curves{L.DUMP_TAG}.pkl'} and gate_zoo_meta{L.DUMP_TAG}.json")


if __name__ == "__main__":
    main()
