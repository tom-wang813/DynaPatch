#!/usr/bin/env python3
"""Fit the gate WITHIN the setting, on the repair split only -- no leave-one-backbone-out.

SHIPPED DEFAULT: --train-on both, i.e. the failure side is bug_train + bug_val and the clean
side is clean_calib. Decision of 2026-09-01: the 80/20 generator/gate split is kept, and the
gate is allowed to reuse the generator's own evidence. That is legitimate -- bug_train is the
deployer's own repair evidence, and the reporting set (bug_eval + clean_test) is untouched by
every variant here -- but it is worth knowing WHY it buys so little: on bug_train the
generator has memorised, so the flipped-row count and the beneficial-row count are equal in
every cell, and those rows contribute no negative examples at all. All the "do not commit
this" signal comes from clean_calib.

The paper says the gate is trained on a disjoint slice of the repair data:

    D^f_repair = D^f_gen (80%, patch generator)  U  D^f_gate (20%, gate)
    D^c_repair = D^c_gen             U  D^c_gate

On disk those two slices are `bug_val` and `clean_calib`. What ships today instead is
leave-one-backbone-out: the gate is fitted on the OTHER THREE backbones. LOBO is not in the
paper, and D_gate is not in the code. This script runs D_gate so the two can be compared on
identical features, identical head and identical reporting rows.

What changes and what does NOT
------------------------------
CHANGED   the rows the 3-class model is fitted on:
            failure side  bug_val      (deploy_direct_calib pass: its `repair_holdout_unseen`
                                        IS the 20% gate slice)
            clean side    clean_calib  (the clean_eval dump, filtered to clean_calib)
UNCHANGED features (the same 12), head (3-class over gain in {-1,0,+1}), score
          (P(+1) - P(-1)), threshold convention (the veto quantile attaining target r), and
          reporting rows (S_held + S_clean^test).

Known limitation, stated not hidden
-----------------------------------
D_gate is small, and it is the POSITIVE class that starves, not the training set as a whole.
Measured per cell (--min-pos 1 prints the full inventory):

  GTSRB / TT100K   6-41 positives, 5-79 negatives   -> fits
  LISA             0-5  positives, 1-17 negatives   -> the negatives are fine, the
                                                       positives are not

A 12-parameter model on 1-5 positives is separable, so the direction is set by the L2 prior
rather than by the data, and it does not reproduce across bug-split seeds. Measured at r=0.90
with --min-pos 1: LISA/ResNet scores F1 .737 / .894 / .000 on the three seeds -- the third
fold vetoes EVERY beneficial application. The three-seed spread under LOBO on the same cells
is .875 / .875 / .692. So LISA cells are refused by default rather than reported as noise.

This is a property of the 80/20 split, not of the protocol: at 50/50 the LISA failure slice
would carry roughly 15-35 rows. Fixing it means retraining every generator.

Second-order caveat: the rank features (m_base_r, dm_r, dnorm_r) are ranked WITHIN a cell, so
ranking 10 calibration rows and then scoring 48 reporting rows puts the two on different rank
grids. LOBO has the same issue but between comparably sized populations.

Zero GPU. Usage:
  DUMP_TAG=_ep40ns .venv/bin/python scripts/gate_protocol_b.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_response_gate_lobo as L  # noqa: E402
import gate_report as R  # noqa: E402
import gate_zoo as Z  # noqa: E402
import names as N  # noqa: E402
import probe_gonogo_pre_vs_prepost as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "outputs" / "gate_protocol_b.csv"

# vgg16/convnext_tiny need the last_affine patch-site dump, not the shipped deep_feat
# (_ep40ns) tree -- see scripts/gate_lambda_sweep.py's own header comment and
# note/PITFALLS.md's patch_site entry. Duplicated here (not imported) rather than
# `from gate_lambda_sweep import _load_split`, since gate_lambda_sweep.py itself imports
# `train_cells` FROM this module -- importing back would be circular.
LAST_AFFINE_BACKBONES = {"vgg16", "convnext_tiny"}
SHIPPED_TAG = "_ep40ns"


def _load_split(pipeline):
    """Run PIPELINE once per patch-site tag and merge by backbone (see gate_lambda_sweep.py's
    identical helper for the full rationale)."""
    saved = (L.DUMP_TAG, L.CACHE, dict(L._MEM))
    merged: dict = {}
    try:
        for tag, keep in ((SHIPPED_TAG, lambda bb: bb not in LAST_AFFINE_BACKBONES),
                          ("_lastaffine", lambda bb: bb in LAST_AFFINE_BACKBONES)):
            L.DUMP_TAG = tag
            L.CACHE = ROOT / "outputs" / f"_response_gate_cache{tag}"
            L._MEM.clear()
            for k, v in pipeline().items():
                if keep(k[2]):
                    merged[k] = v
    finally:
        L.DUMP_TAG, L.CACHE = saved[0], saved[1]
        L._MEM.clear()
        L._MEM.update(saved[2])
    return merged
CALIB_PASS = "deploy_direct_calib"
CT_TAG = "_ct_ep40ns"      # the tree holding the clean_train dump
MIN_POS = 1      # with --train-on both the failure side is never empty


def calib_idx(seed: int, ds: str, bb: str) -> set[int] | None:
    f = ROOT / f"artifacts/bug_sets/shuffled_split_seed{seed}/{ds}_{bb}/{ds}_clean_calib_indices.json"
    return set(json.loads(f.read_text())["indices"]) if f.exists() else None


def _side(entry: dict, keep: set[int] | None, gain_pos: int) -> dict | None:
    """One training population: features, the flip mask, and the gain label."""
    if entry is None:
        return None
    idx = entry["idx"]
    m = (np.fromiter((int(i) in keep for i in idx), dtype=bool, count=len(idx))
         if keep is not None else np.ones(len(idx), dtype=bool))
    f = {k: v[m] for k, v in entry["feats"].items()}
    y = entry["y"][m]
    return {"f": f, "flip": f["flip"] > 0.5, "gain": np.where(y, gain_pos, 0).astype(int),
            "idx": np.asarray(idx)[m]}


def bugtrain_idx(seed: int, ds: str, bb: str) -> set[int] | None:
    f = ROOT / f"artifacts/bug_sets/shuffled_split_seed{seed}/{ds}_{bb}/{ds}_bug_train_indices.json"
    return set(json.loads(f.read_text())["indices"]) if f.exists() else None


def clean_train_side() -> dict:
    """`clean_train` rows per cell, from the 2026-09-01 dump tree.

    This split is the dataset's TRAIN partition: disjoint from `bug_eval` AND `clean_test`,
    both of which come from the test partition. Adding it enlarges the gate's NEGATIVE class
    (clean inputs the patch breaks), which is the binding constraint -- clean_calib carries
    only 1-79 of them per cell while the post-repair block needs 7 dimensions estimated.

    CAVEAT to carry into any write-up: the generator's clean-replay was itself optimised on
    clean_train, so the patch regresses there LESS than on held-out clean data (measured
    2026-09-01: 6-15 negatives on two GTSRB cells versus 109-317 on TT100K). These negatives
    are therefore both fewer per row and not an unbiased sample of deployment regressions.
    """
    saved = (L.PASS_DIR, L.CACHE, L.DUMP_TAG, dict(L._MEM))
    L.DUMP_TAG = CT_TAG
    L.PASS_DIR = "deploy_direct"
    L.CACHE = ROOT / "outputs" / f"_response_gate_cache{CT_TAG}"
    L._MEM.clear()
    try:
        return L.collect("harm_train")
    finally:
        L.PASS_DIR, L.CACHE, L.DUMP_TAG = saved[0], saved[1], saved[2]
        L._MEM.clear(); L._MEM.update(saved[3])


def train_cells(source: str = "dgate", clean_source: str = "calib") -> dict:
    """Training rows per (seed, ds, bb). `source` selects the failure side:

      dgate  bug_val only            -- the paper's Eq.(2): disjoint from the generator
      dgen   bug_train only          -- the generator's OWN evidence, which it has memorised
      both   bug_train + bug_val     -- everything labelled in the repair split

    The clean side is clean_calib in every case. VERIFIED on disk (gtsrb/resnet50 s101):
    `repair_support_seen` holds 38 rows == bug_train exactly, NOT the full 48-row support,
    so `both` has to union it with the calib pass's `repair_holdout_unseen` (== bug_val).
    """
    # clean side comes from the pass already cached; only the failure side needs the calib pass.
    harm = L.collect("harm")
    seen = L.collect("seen") if source in ("dgen", "both") else None

    help_calib: dict = {}
    if source in ("dgate", "both"):
        saved_pass, saved_cache, saved_mem = L.PASS_DIR, L.CACHE, dict(L._MEM)
        L.PASS_DIR = CALIB_PASS
        L.CACHE = ROOT / "outputs" / f"_response_gate_cache{L.DUMP_TAG}_calibpass"
        L._MEM.clear()
        try:
            help_calib = L.collect("help")   # repair_holdout_unseen of calib pass == bug_val
        finally:
            L.PASS_DIR, L.CACHE = saved_pass, saved_cache
            L._MEM.clear()
            L._MEM.update(saved_mem)

    ct = clean_train_side() if clean_source == "calib+train" else {}
    out = {}
    keys = set(harm) & (set(help_calib) if source == "dgate" else set(seen))
    if source == "both":
        keys &= set(help_calib)
    if clean_source == "calib+train":
        keys &= set(ct)
    for k in sorted(keys):
        c = _side(harm[k], calib_idx(*k), -1)
        if source == "dgate":
            h = _side(help_calib[k], None, +1)
        elif source == "dgen":
            h = _side(seen[k], None, +1)          # repair_support_seen IS bug_train
        else:
            a_, b_ = _side(seen[k], None, +1), _side(help_calib.get(k), None, +1)
            if a_ is None or b_ is None:
                continue
            h = {"f": {kk: np.concatenate([a_["f"][kk], b_["f"][kk]]) for kk in a_["f"]},
                 "flip": np.concatenate([a_["flip"], b_["flip"]]),
                 "gain": np.concatenate([a_["gain"], b_["gain"]]),
                 "idx": np.concatenate([a_["idx"], b_["idx"]])}
        if c is None or h is None:
            continue
        if clean_source == "calib+train":
            t = _side(ct[k], None, -1)
            if t is None:
                continue
            c = {"f": {kk: np.concatenate([c["f"][kk], t["f"][kk]]) for kk in c["f"]},
                 "flip": np.concatenate([c["flip"], t["flip"]]),
                 "gain": np.concatenate([c["gain"], t["gain"]]),
                 "idx": np.concatenate([c["idx"], t["idx"]])}
        out[k] = {"clean": c, "held": h}
    return out


def eval_at_theta(rc: dict, rh: dict, sc: np.ndarray, sh: np.ndarray, t: float) -> dict:
    """Reg/CReg/RR_held on the report population (rc/rh) at a FIXED, externally-chosen veto
    threshold t. Used with a `t` selected from calib data only, so the resulting numbers are a
    genuine held-out evaluation, not a value optimised against rc/rh."""
    kc, kh = ~rc["flip"] | (sc > t), ~rh["flip"] | (sh > t)
    reg0 = rc["y"].sum() / rc["n"]
    cd = int(rc["crit"].sum())
    reg = rc["y"][kc].sum() / rc["n"]
    return {
        "theta": t,
        "Reg": reg,
        "CReg": (rc["y"] & kc & rc["crit"]).sum() / max(cd, 1),
        "RR_held": rh["y"][kh].sum() / rh["n"],
        "realised_r": (reg0 - reg) / reg0 if reg0 else float("nan"),
        "n_held": int(rh["n"]), "n_clean": int(rc["n"]), "n_crit": cd,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=N.SHIPPED_GATE)
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--train-on", choices=["dgate", "dgen", "both"], default="both",
                    help="which failure rows the gate is fitted on")
    ap.add_argument("--per-sample", default=None,
                    help="directory for per-sample gate decisions: one CSV per (setting, seed) "
                         "carrying dataset_index, the gate score, theta and the apply/veto "
                         "decision at every reported r. Needed to analyse the SHIPPED "
                         "gate+generator combination at sample level; the scores already exist "
                         "in memory, nothing is refitted.")
    ap.add_argument("--emit-curves", default=None,
                    help="also write the FULL Reg/CReg/RR_held frontier in gate_zoo's curve "
                         "format, so the RQ3/RQ4 tables can be rebuilt on this protocol")
    ap.add_argument("--emit-natural-point", default=None,
                     help="one row per (setting, seed): Reg/CReg/RR_held (plus the confusion "
                          "stats n/prevalence/accuracy/precision/recall/f1) at theta=0 -- "
                          "commit iff the fitted model's own score favours beneficial over "
                          "harmful (P(gain=+1) > P(gain=-1)). No r target, no calibration "
                          "split, no threshold search of any kind -- the model's raw class "
                          "decision, evaluated directly on the report population. The ONLY "
                          "admissible source for a reported gated-DynaPatch number "
                          "(names.py:GATE_NATURAL_POINT); see note/RESEARCH_STATE.md.")
    ap.add_argument("--clean-source", choices=["calib", "calib+train"], default="calib",
                    help="clean_calib alone, or plus the clean_train dump (more negatives)")
    ap.add_argument("--min-pos", type=int, default=MIN_POS,
                    help="refuse a cell with fewer than this many of either class")
    a = ap.parse_args()
    key = N.assert_gate_admissible(a.features)
    feats = Z.LEARNED[key]

    report = _load_split(lambda: Z.attach_criticality(Z.attach_idx(Z.add_derived(G.build_cells()))))
    train = _load_split(lambda: train_cells(a.train_on, a.clean_source))

    print(f"gate: {N.gate_display(key)}   {len(feats)} features, 3-class, "
          f"train-on={a.train_on} + clean={a.clean_source}, same setting, same seed")
    print(f"report cells {len(report)}   train cells {len(train)}\n")

    print("D_gate size per cell (the entire training set of the gate):")
    hdr = f"{'setting':<10}{'seed':>6}{'flip_bugval':>13}{'pos(+1)':>9}{'reg(-1)':>9}{'total':>7}  status"
    print(hdr); print("-" * len(hdr))
    rows, refused = [], []
    natural_rows: list = []
    curve_per_setting: dict = {}
    for k in sorted(train):
        seed, ds, bb = k
        c, h = train[k]["clean"], train[k]["held"]
        npos = int((h["gain"][h["flip"]] == 1).sum())
        nneg = int((c["gain"][c["flip"]] == -1).sum())
        ntot = int(h["flip"].sum() + c["flip"].sum())
        ok = npos >= a.min_pos and nneg >= a.min_pos
        print(f"{N.SETTING_LABEL[f'{ds}/{bb}']:<10}{seed:>6}{int(h['flip'].sum()):>13}"
              f"{npos:>9}{nneg:>9}{ntot:>7}  {'fit' if ok else 'REFUSED (<%d)' % MIN_POS}")
        if not ok:
            refused.append(k)
            continue
        X = np.vstack([G.mat(d["f"], feats, d["flip"]) for d in (c, h)])
        g = np.concatenate([d["gain"][d["flip"]] for d in (c, h)])
        mdl = G.fit_gain(X, g)
        if mdl is None:
            refused.append(k); continue
        rc, rh = report[k]["clean"], report[k]["held"]
        sc, sh = G.score(mdl, G.mat(rc["f"], feats)), G.score(mdl, G.mat(rh["f"], feats))
        op = Z.operating_points(rc, rh, sc, sh)
        if op is not None:
            curve_per_setting.setdefault((ds, bb), []).append(op)
        u = np.concatenate([rc["gain"][rc["flip"]] == 1,
                            rh["gain"][rh["flip"]] == 1]).astype(int)
        s = np.concatenate([sc[rc["flip"]], sh[rh["flip"]]])
        if a.emit_natural_point:
            ev = eval_at_theta(rc, rh, sc, sh, 0.0)
            natural_rows.append({"setting": f"{ds}/{bb}", "seed": seed,
                                 "n_train_pos": npos, "n_train_neg": nneg,
                                 **ev, **R.confusion(s > 0.0, u)})
        thetas: dict[float, float] = {}
        for r in R.RS:
            t = R.theta_for_r(rc, rh, sc, sh, r)
            if t is None:
                continue
            rows.append({"setting": f"{ds}/{bb}", "seed": seed, "r": r, "theta": t,
                         "n_train_pos": npos, "n_train_neg": nneg, **R.confusion(s > t, u)})
            thetas[r] = t

        if a.per_sample:
            # One row per REPORTED input, both populations, with the decision at every r. The
            # `side` column matters: `clean` rows are clean_test (where Reg is measured) and
            # `held` rows are bug_eval (where RR_held is). Merging them without it reproduces
            # the population-identity mistake this project has made repeatedly.
            recs = []
            for side, rep_, sco in (("clean", rc, sc), ("held", rh, sh)):
                d = pd.DataFrame({"setting": f"{ds}/{bb}", "seed": seed, "side": side,
                                  "dataset_index": rep_["idx"], "gate_score": sco,
                                  "flip": rep_["flip"], "gain": rep_["gain"]})
                # the natural (theta=0) decision -- no target r, no threshold search, no
                # calibration split; the ONLY admissible per-sample decision now (2026-09-05,
                # user decision). r-indexed columns below are kept ONLY for scripts not yet
                # migrated off them (note/RESEARCH_STATE.md's "not fixed, explicitly deferred").
                d["apply_natural"] = sco > 0.0
                for r, t in thetas.items():
                    d[f"apply_r{r:.2f}"] = sco > t
                    d[f"theta_r{r:.2f}"] = t
                recs.append(d)
            od = Path(a.per_sample); od.mkdir(parents=True, exist_ok=True)
            pd.concat(recs, ignore_index=True).to_csv(
                od / f"{ds}_{bb}_s{seed}.csv", index=False)

    print(f"\nfitted {len(train) - len(refused)}/{len(train)} cells; "
          f"refused {len(refused)} for having fewer than {a.min_pos} of a class")

    settings = [f"{d}/{b}" for d, b, _ in N.SETTING_ORDER]
    for r in R.RS:
        print(f"\n=== PROTOCOL B   r = {r:.2f} ===")
        hdr = (f"{'setting':<10}{'seeds':>7}{'n':>7}{'prev':>7}{'acc':>8}"
               f"{'prec':>8}{'rec':>8}{'F1':>8}")
        print(hdr); print("-" * len(hdr))
        agg = []
        for stg in settings:
            p = R.pooled(rows, stg, r)
            if p is None:
                print(f"{N.SETTING_LABEL[stg]:<10}{'-- no cell had enough D_gate':>37}")
                continue
            agg.append(p)
            print(f"{N.SETTING_LABEL[stg]:<10}{p['seeds']:>7}{p['n']:>7}{p['prevalence']:>7.2f}"
                  f"{p['accuracy']:>8.3f}{p['precision']:>8.3f}{p['recall']:>8.3f}{p['f1']:>8.3f}")
        if agg:
            med = lambda kk: float(np.median([x[kk] for x in agg]))  # noqa: E731
            print("-" * len(hdr))
            print(f"{'MEDIAN':<10}{'':>7}{'':>7}{med('prevalence'):>7.2f}{med('accuracy'):>8.3f}"
                  f"{med('precision'):>8.3f}{med('recall'):>8.3f}{med('f1'):>8.3f}"
                  f"   ({len(agg)}/12 settings estimable)")

    if a.emit_curves and curve_per_setting:
        # pooled point-wise across seeds on the q grid, exactly as gate_zoo does, so the two
        # curve files are directly comparable and the same downstream readers work on both
        cv = {(N.gate_display(key), st): v
              for st, v in Z._pool_seeds(curve_per_setting).items()}
        Z.write_csv(cv, Path(a.emit_curves))
        print(f"\nwrote {a.emit_curves}  ({len(cv)} curves)")

    if rows:
        with Path(a.out).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.out}  ({len(rows)} rows)")

    if a.emit_natural_point and natural_rows:
        with Path(a.emit_natural_point).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(natural_rows[0].keys()))
            w.writeheader(); w.writerows(natural_rows)
        print(f"\nwrote {a.emit_natural_point}  ({len(natural_rows)} rows, theta=0, no calibration)")


if __name__ == "__main__":
    main()
