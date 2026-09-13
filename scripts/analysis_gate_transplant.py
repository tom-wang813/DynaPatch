#!/usr/bin/env python3
"""What happens if NN-Patching/PatchNAS's own patch output is routed by OUR gate instead of
their own error estimator?

Motivation (2026-09-04 session): raw, always-apply repair capability is close between DynaPatch
and NN-Patching/PatchNAS (0.575 vs ~0.55/0.54), but DEPLOYED repair rate is not (0.575 vs
0.375/0.32, per note/DRAFT_RESULTS_NEW_RQS.md RQ2.10) because their own estimator declines to
route 51-62% of held-out failures to keep regression low. This asks: is that gap the ESTIMATOR's
conservatism, or does DynaPatch's own patch mechanism also matter once you swap in a stronger
gate? If NN-Patching/PatchNAS + our gate closes most of the gap to DynaPatch's own r=0.60/0.80/0.90
numbers, the gap was mostly about gate quality. If it does not, the generator matters too.

Mechanism: our gate's feature computation (scripts/analyze_response_gate_lobo.py:build_features)
needs only base_logits, patched_logits, and a ground-truth label per row -- verified generic
except `dnorm_r` (rank of a DynaPatch-specific 2048-d patch_vec norm), substituted here with
rank(||patched_logits - base_logits||), the same Delta z(x) norm used throughout the M1 series.
NN-Patching/PatchNAS's own base_logits/patched_logits already exist (raw, un-routed -- their own
router's decision is discarded entirely; that is the point of this experiment) in
outputs/prior_patch_persample/<setting>_s<seed>/<method>/tau/.

Gate fit and scoring reuses the SAME functions the shipped gate uses
(scripts/probe_gonogo_pre_vs_prepost.py: fit_gain = 3-class logistic regression over
gain in {-1,0,+1}, score = P(+1)-P(-1); feature set = names.SHIPPED_GATE = "4 pre+post", 12
features) -- not a reimplementation, the identical model class and feature list.

Training population: this experiment's failure side is `repair_support_seen` (their own
patch-training evidence, the "seen" population) and clean side is `clean_calib` -- the "dgen"
variant of scripts/gate_protocol_b.py's --train-on options. NN-Patching/PatchNAS have no
generator/gate evidence SPLIT the way DynaPatch does (no separate "bug_val" slice held out from
patch training specifically for the gate), so "dgen" (train on the same evidence the patch itself
was fit on, plus clean_calib) is the closest available analogue, not an exact match to DynaPatch's
shipped "both" (bug_train + bug_val) protocol. Reported on `repair_holdout_unseen` (held) and
`clean_test` (clean) -- the SAME reporting population as every other RQ2/RQ4 table.

Threshold: theta_for_r, identical construction to scripts/gate_report.py's (smallest veto that
removes >= fraction r of the ungated clean_test regression), reimplemented here (not imported)
because scripts/gate_report.py's version is coupled to gate_zoo's DynaPatch-specific cell cache.

    .venv/bin/python scripts/analysis_gate_transplant.py
    .venv/bin/python scripts/analysis_gate_transplant.py --feature-set "2 pre-strong"

--feature-set lets this same experiment answer a second question (2026-09-04, RQ3
robustness check): does post-proposal information help NN-Patching/PatchNAS's gate the
same way it helps DynaPatch's own (RQ3's Finding 3), or is that a property of DynaPatch's
specific patch mechanism? "2 pre-strong" = pre-information only (no post features); "4
pre+post" (default) = the full shipped feature set. Both are LAYERS keys from
scripts/probe_gonogo_pre_vs_prepost.py, the same pre/post split RQ3 itself uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                             # noqa: E402
from analyze_response_gate_lobo import build_features             # noqa: E402
from probe_gonogo_pre_vs_prepost import fit_gain, score, mat, LAYERS  # noqa: E402

SEEDS = (101, 202, 303)
SETTINGS = [(d, b) for d in ("gtsrb", "tt100k_signs", "lisa_signs")
            for b in ("resnet50", "convnext_tiny", "densenet121", "vgg16")]
METHODS = ("NN-Patching", "PatchNAS")
FEATURE_SETS = {"4 pre+post": LAYERS["4 pre+post"], "2 pre-strong": LAYERS["2 pre-strong"]}


def critical_classes(dataset: str) -> set[int]:
    path = ROOT / "artifacts/risk" / f"{dataset}_safety_risk_matrix.json"
    if not path.is_file():
        return set()
    raw = json.loads(path.read_text())
    return {int(c) for cs in raw["critical_signs"].values() for c in cs}


def load_pop(dirp: Path, pop: str) -> dict | None:
    bl, pl, pc = (dirp / f"base_logits_{pop}.npy", dirp / f"patched_logits_{pop}.npy",
                 dirp / f"{pop}_predictions.csv")
    if not (bl.is_file() and pl.is_file() and pc.is_file()):
        return None
    base = np.load(bl).astype(np.float64)
    patched = np.load(pl).astype(np.float64)
    t = pd.read_csv(pc)
    y = t.label.to_numpy()
    dnorm = np.linalg.norm(patched - base, axis=1)
    feats, _ = build_features({"base": base, "patched": patched, "dnorm": dnorm, "truth": y})
    flip = feats["flip"] > 0.5
    argmax_patched = patched.argmax(1)
    correct_after = argmax_patched == y
    base_correct = base.argmax(1) == y
    return {"f": feats, "flip": flip, "y": y, "n": len(y),
           "correct_after": correct_after, "base_correct": base_correct,
           "idx": t.dataset_index.to_numpy()}


def metrics_at(clean: dict, held: dict, theta: float, critical: set[int]) -> dict:
    """RR_held / Reg / CReg after applying the veto at this threshold. Deployment simulation:
    a vetoed row reverts to the base prediction -- exact, no re-run needed (same construction as
    scripts/analyze_response_gate_lobo.py's docstring)."""
    keep_h = ~held["flip"] | (held["score"] > theta)
    repaired = keep_h & held["correct_after"]              # held pop is all base-wrong
    rr_held = float(repaired.mean())

    keep_c = ~clean["flip"] | (clean["score"] > theta)
    final_correct_c = np.where(keep_c, clean["correct_after"], clean["base_correct"])
    reg = float((clean["base_correct"] & ~final_correct_c).sum() / clean["n"])
    return {"RR_held": rr_held, "Reg": reg,
           "CReg": _creg(clean, final_correct_c, critical)}


def _creg(clean: dict, final_correct_c: np.ndarray, critical: set[int]) -> float:
    if not critical:
        return float("nan")
    y = clean["y"]
    crit_mask = np.isin(y, list(critical)) & clean["base_correct"]
    if not crit_mask.any():
        return float("nan")
    return float((crit_mask & ~final_correct_c).sum() / crit_mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="4 pre+post", choices=list(FEATURE_SETS),
                    help="'4 pre+post' = shipped gate (default); '2 pre-strong' = pre-only, "
                         "for the RQ3 robustness check (does post-info help NN-Patching/"
                         "PatchNAS's gate the way it helps DynaPatch's own?)")
    a = ap.parse_args()
    FEATS = FEATURE_SETS[a.feature_set]
    tag = a.feature_set.replace(" ", "_").replace("-", "")   # "4pre+post" / "2prestrong"

    rows = []
    for method in METHODS:
        for ds, bb in SETTINGS:
            crit = critical_classes(ds)
            for seed in SEEDS:
                dirp = ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/{method}/tau"
                seen = load_pop(dirp, "bug_train")
                held = load_pop(dirp, "bug_eval")
                calib = load_pop(dirp, "clean_calib")
                clean_test = load_pop(dirp, "clean_test")
                if any(p is None for p in (seen, held, calib, clean_test)):
                    rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                               "note": "missing per-sample artefact"})
                    continue

                # training gain labels: failure side = repair_support_seen (their own evidence,
                # the "dgen" variant -- see module docstring), clean side = clean_calib
                seen_flip = seen["flip"]
                seen_gain = np.where(seen["correct_after"], 1, 0).astype(int)  # all base-wrong
                calib_flip = calib["flip"]
                calib_regressed = calib["base_correct"] & ~calib["correct_after"]
                calib_gain = np.where(calib_regressed, -1, 0).astype(int)

                if seen_flip.sum() < 2 or calib_flip.sum() < 2:
                    rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                               "note": f"too few flipped training rows "
                                       f"(seen={int(seen_flip.sum())}, calib={int(calib_flip.sum())})"})
                    continue

                X = np.vstack([mat(seen["f"], FEATS, seen_flip), mat(calib["f"], FEATS, calib_flip)])
                g = np.concatenate([seen_gain[seen_flip], calib_gain[calib_flip]])
                mdl = fit_gain(X, g)
                if mdl is None:
                    rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                               "note": "gate fit failed (single class)"})
                    continue

                held["score"] = score(mdl, mat(held["f"], FEATS))
                clean_test["score"] = score(mdl, mat(clean_test["f"], FEATS))

                # theta=0: commit iff the fitted model's own score favours beneficial over
                # harmful (P(gain=+1) > P(gain=-1)). No target r, no threshold search, no
                # calibration split touched -- the model's raw class decision, evaluated
                # directly on the report population. Replaces the r-grid/theta_for_r sweep
                # (2026-09-04, user decision: drop the r-targeting/calibration machinery
                # entirely rather than debate whether its calib split is clean).
                m = metrics_at(clean_test, held, 0.0, crit)
                rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                           "theta": 0.0, "n_train_flip_seen": int(seen_flip.sum()),
                           "n_train_flip_calib": int(calib_flip.sum()), **m, "note": ""})

    cell = pd.DataFrame(rows)
    ok = cell[cell.note.eq("")] if "note" in cell.columns else cell

    RM.write_section(
        f"GateTransplant_{tag}", f"Gate transplant ({a.feature_set}) — NN-Patching/PatchNAS's "
        "raw patch output routed by OUR gate instead of their own estimator, at the "
        "no-calibration natural threshold (raw)",
        f"""
Motivation and method in full in the script docstring (`scripts/analysis_gate_transplant.py`).
Short version: same gate model class, feature set = **{a.feature_set}** (`LAYERS["{a.feature_set}"]`
in `scripts/probe_gonogo_pre_vs_prepost.py`, {len(FEATS)} features) -- refit per (method,
setting, seed) on NN-Patching/PatchNAS's OWN raw (always-apply) `base_logits`/`patched_logits`,
using their own `repair_support_seen` (failure side) + `clean_calib` (clean side) as training
evidence (the "dgen" analogue -- they have no held-out gate-only evidence slice the way
DynaPatch does), reported on `repair_holdout_unseen` + `clean_test`.

Threshold: theta=0 (commit iff the fitted model's own score favours beneficial over harmful,
P(gain=+1) > P(gain=-1)) -- no target r, no threshold search, no calibration split of any
kind. 2026-09-04 (user decision): replaced the old r-grid/theta_for_r sweep, which searched
`clean_test` (the report population) for a threshold hitting each target r -- an eval-set
selection bug -- rather than debate whether a calibration split can ever be clean of the gate's
own fitting data.

One row per (method, setting, seed); `note` explains a dropped row instead of a silent omission
(e.g. too few flipped training rows -- a real, recorded limitation of small failure sets, not a
script bug).

Compare `RR_held`/`Reg`/`CReg` here against:
- their own estimator's numbers at the tau operating point (`note/RQ2_DATA.md` RQ2.10)
- DynaPatch's own natural-threshold numbers (`note/RQ4_DATA.md` RQ4.1)
- the OTHER feature-set run of this same script (`--feature-set` "2 pre-strong" vs "4 pre+post")
  -- the RQ3 robustness question: does adding post-information help NN-Patching/PatchNAS's gate
  the way `note/RQ3_DATA.md`'s Finding 3 shows it helps DynaPatch's own gate?

If our gate closes most of the RR_held gap to DynaPatch at matched Reg, the gap was mostly gate
quality; if a gap remains even with our gate, the generator's own patch quality matters too.
""",
        [("by_cell", cell.sort_values(["method", "setting", "seed"]))],
    )

    out = ROOT / "outputs" / "gate_transplant"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / f"per_cell_{tag}.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/per_cell_{tag}.csv  ({len(cell)} rows, "
         f"{len(cell) - len(ok)} dropped with a note)")

    if len(ok):
        summ = ok.groupby("method")[["RR_held", "Reg", "CReg"]].mean()
        print(f"\n[{a.feature_set}] Setting-balanced mean (mean of per-cell values; NOT weighted "
             f"by setting size):")
        print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
