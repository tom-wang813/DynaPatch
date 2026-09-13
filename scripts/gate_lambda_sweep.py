#!/usr/bin/env python3
"""Cost-weighted gate threshold, as an alternative to the plain theta=0 rule and to the
quantile-q sweep in `analysis_gate_transplant_curve.py`/`outputs/gate_curves_protocolC_ep40ns.csv`.

theta=0 ("commit iff P(gain=+1) > P(gain=-1)") is not an arbitrary cutoff: it is the Bayes-optimal
decision under a utility function that gives +1 for a correct repair, -1 for a caused regression,
and 0 for a no-effect outcome or a veto either way. That equal-weighting of "one repair" against
"one regression" is an assumption, not a derived fact, and safety-critical settings usually treat
a caused regression as costing more than a missed repair is worth. This script makes that
assumption explicit and adjustable: commit iff P(gain=+1) > lambda * P(gain=-1), for a FIXED,
a-priori grid of lambda values. Unlike the q-quantile sweep, lambda touches the report
population's score distribution NOT AT ALL -- it is a pure modelling choice applied identically
everywhere, so this is if anything a step further from any eval-set selection concern, not a step
closer. lambda=1 recovers the shipped theta=0 rule exactly.

Covers both \\DPGate's own protocol-C fit (gate_protocol_b.py's "both"/"calib" default) and the
gate transplanted onto NN-Patching/PatchNAS's raw patches (analysis_gate_transplant.py's "dgen"
protocol), at the SAME lambda grid, so the two sides are directly joinable into one table.

    .venv/bin/python scripts/gate_lambda_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402
import analyze_response_gate_lobo as L  # noqa: E402
import gate_zoo as Z  # noqa: E402
import names as N  # noqa: E402
import probe_gonogo_pre_vs_prepost as G  # noqa: E402
from gate_protocol_b import train_cells  # noqa: E402
from analysis_gate_transplant import METHODS, SETTINGS, SEEDS, critical_classes, load_pop  # noqa: E402
from probe_gonogo_pre_vs_prepost import fit_gain, mat, LAYERS  # noqa: E402

LAMBDAS = (0.5, 1.0, 2.0, 4.0)
FEATS = LAYERS["4 pre+post"]

# 2026-09-05: vgg16/convnext_tiny adopted repair.patch_site=last_affine (see RQ1_DATA.md's
# header); resnet50/densenet121 are untouched (last_affine is bit-identical to deep_feat there).
# `analyze_response_gate_lobo.DUMP_TAG` selects ONE tag for every setting at once, so getting
# both halves right requires running the same builder twice and merging by backbone -- the same
# split `sample_frame.py`'s `from_tree(..., settings=...)` applies to DynaPatch-NoGate. Do not
# read `dp_curve()`'s cells under a single global DUMP_TAG; a prior run that did (and one that
# forced DUMP_TAG=_lastaffine for all 12 settings) both produced wrong per-cell numbers -- see
# note/PITFALLS.md.
#
# The non-last_affine tag is "_ep40ns", NOT the module default "" -- `outputs/effect_dump_v8_*`
# (the bare 12-epoch shipped patch DUMP_TAG="" points to) no longer exists on disk; every other
# RQ1-4 script reads the 40-epoch no-early-stop build (`outputs/effect_dump_ep40ns_v8_*`) as the
# "shipped" baseline now (see gate_protocol_b.py's own module docstring). Confirmed on disk
# 2026-09-05: `effect_dump_v8_s101/` is absent, `effect_dump_ep40ns_v8_s101/` is present.
LAST_AFFINE_BACKBONES = {"vgg16", "convnext_tiny"}
SHIPPED_TAG = "_ep40ns"


def _load_split(pipeline):
    """Run a zero-arg builder+transform PIPELINE (reads via `analyze_response_gate_lobo`'s
    module globals) once per patch-site tag and merge by backbone: last_affine for
    vgg16/convnext_tiny, the shipped deep_feat tree for everything else.

    `pipeline` must include every step that re-reads per-cell files keyed on the CURRENTLY set
    `L.DUMP_TAG` -- e.g. `gate_zoo.attach_criticality`, which re-reads `clean_eval_predictions.csv`
    from `outputs/effect_dump{DUMP_TAG}_v8_*`. Running such a transform AFTER this function
    merges the two tags and restores `L.DUMP_TAG` reads every cell from whichever tag happened to
    be restored, silently mixing (or, when that tag's tree does not exist, crashing on) cells
    that were actually built under the OTHER tag. Pass the full per-tag pipeline in, not just the
    raw cell-builder.
    """
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


def posneg(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = model.predict_proba(X)
    cls = list(model.classes_)
    pos = p[:, cls.index(1)] if 1 in cls else np.zeros(len(X))
    neg = p[:, cls.index(-1)] if -1 in cls else np.zeros(len(X))
    return pos, neg


# --------------------------------------------------------------- DP's own gate (protocol C)

MIN_POS = 1  # matches gate_protocol_b.py's shipped default -- see the check below


def dp_curve() -> pd.DataFrame:
    report = _load_split(lambda: Z.attach_criticality(Z.attach_idx(Z.add_derived(G.build_cells()))))
    train = _load_split(lambda: train_cells("both", "calib"))
    rows = []
    for k in sorted(set(train) & set(report)):
        seed, ds, bb = k
        c, h = train[k]["clean"], train[k]["held"]
        # gate_protocol_b.py refuses a cell with fewer than MIN_POS of EITHER gain class, not
        # merely "at least 2 distinct classes present" (fit_gain's own, weaker guard). Without
        # this check a cell whose clean side has ZERO regressions (gain=-1 never observed, e.g.
        # lisa_signs/vgg16 seeds 101/303 under last_affine) still fits -- on gain in {0,+1} only
        # -- producing a gate that has never seen a single harmful example. That silently
        # disagreed with `outputs/gate_natural_point_protocolC_ep40ns.csv` (names.py's ONLY
        # admissible source for a reported gated-DynaPatch number), which refuses exactly these
        # cells; replicate its refusal here so the two stay comparable.
        npos = int((h["gain"][h["flip"]] == 1).sum())
        nneg = int((c["gain"][c["flip"]] == -1).sum())
        if npos < MIN_POS or nneg < MIN_POS:
            continue
        X = np.vstack([G.mat(d["f"], FEATS, d["flip"]) for d in (c, h)])
        g = np.concatenate([d["gain"][d["flip"]] for d in (c, h)])
        mdl = fit_gain(X, g)
        if mdl is None:
            continue
        rc, rh = report[k]["clean"], report[k]["held"]
        pos_c, neg_c = posneg(mdl, G.mat(rc["f"], FEATS))
        pos_h, neg_h = posneg(mdl, G.mat(rh["f"], FEATS))
        cd = int(rc["crit"].sum())
        for lam in LAMBDAS:
            keep_h = ~rh["flip"] | (pos_h > lam * neg_h)
            keep_c = ~rc["flip"] | (pos_c > lam * neg_c)
            rr_held = float(rh["y"][keep_h].sum() / rh["n"])
            reg = float(rc["y"][keep_c].sum() / rc["n"])
            creg = float((rc["y"] & keep_c & rc["crit"]).sum() / max(cd, 1))
            rows.append({"setting": f"{ds}/{bb}", "seed": seed, "lambda": lam,
                         "RR_held": rr_held, "Reg": reg, "CReg": creg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- transplanted gate (dgen)

def transplant_curve() -> pd.DataFrame:
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
                    continue
                seen_flip, calib_flip = seen["flip"], calib["flip"]
                if seen_flip.sum() < 2 or calib_flip.sum() < 2:
                    continue
                seen_gain = np.where(seen["correct_after"], 1, 0).astype(int)
                calib_regressed = calib["base_correct"] & ~calib["correct_after"]
                calib_gain = np.where(calib_regressed, -1, 0).astype(int)
                X = np.vstack([mat(seen["f"], FEATS, seen_flip), mat(calib["f"], FEATS, calib_flip)])
                g = np.concatenate([seen_gain[seen_flip], calib_gain[calib_flip]])
                mdl = fit_gain(X, g)
                if mdl is None:
                    continue
                pos_h, neg_h = posneg(mdl, mat(held["f"], FEATS))
                pos_c, neg_c = posneg(mdl, mat(clean_test["f"], FEATS))
                cd_mask = np.isin(clean_test["y"], list(crit)) & clean_test["base_correct"] \
                    if crit else np.zeros(clean_test["n"], dtype=bool)
                for lam in LAMBDAS:
                    keep_h = ~held["flip"] | (pos_h > lam * neg_h)
                    rr_held = float((keep_h & held["correct_after"]).mean())
                    keep_c = ~clean_test["flip"] | (pos_c > lam * neg_c)
                    final_correct_c = np.where(keep_c, clean_test["correct_after"],
                                                clean_test["base_correct"])
                    reg = float((clean_test["base_correct"] & ~final_correct_c).sum()
                                / clean_test["n"])
                    creg = float((cd_mask & ~final_correct_c).sum() / cd_mask.sum()) \
                        if cd_mask.any() else float("nan")
                    rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                                 "lambda": lam, "RR_held": rr_held, "Reg": reg, "CReg": creg})
    return pd.DataFrame(rows)


def main() -> None:
    dp = dp_curve()
    dp["method"] = "DynaPatch"
    tr = transplant_curve()
    cell = pd.concat([dp, tr], ignore_index=True)

    out = ROOT / "outputs" / "gate_lambda_sweep"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_cell.csv", index=False)

    # per-setting mean first, THEN mean across settings -- never a grand mean over raw cells.
    # A direct cell-level mean silently reweights by each setting's seed count (DynaPatch has an
    # uneven 2-3 seeds/setting here: some last_affine fits are refused for too few flips, see
    # RQ3_DATA.md's header) and, even at even seed counts, is not what "the 12-setting mean"
    # anywhere else in this repo means (analysis_p1_regression.py, RQ4_DATA.md's RQ4.2-4.4, ...).
    per_setting = cell.groupby(["method", "setting", "lambda"], as_index=False)[
        ["RR_held", "Reg", "CReg"]].mean()
    pooled = per_setting.groupby(["method", "lambda"], as_index=False)[
        ["RR_held", "Reg", "CReg"]].mean()
    RM.write_section(
        "GateLambdaSweep",
        "Cost-weighted gate threshold sweep (raw): commit iff P(gain=+1) > lambda * P(gain=-1)",
        f"""
Alternative to the plain theta=0 rule (lambda=1 recovers it exactly) and to the quantile-q
sweep (`outputs/gate_curves_protocolC_ep40ns.csv`,
`outputs/gate_transplant/curve_4pre+post.csv`). lambda in {{{', '.join(str(l) for l in LAMBDAS)}}}
is fixed a priori and never touches the report population's score distribution -- a strictly
weaker leakage exposure than the quantile-q sweep, which at least reads a quantile off that
distribution. `DynaPatch` rows use the shipped protocol-C fit (`gate_protocol_b.train_cells`,
"both"/"calib"); `NN-Patching`/`PatchNAS` rows use the gate-transplant "dgen" fit
(`analysis_gate_transplant.py`), both scored with the SAME shipped pre+post feature set.
""",
        [("by_cell", cell.sort_values(["method", "setting", "seed", "lambda"])),
         ("pooled", pooled)],
    )
    print(f"[written] outputs/gate_lambda_sweep/per_cell.csv  ({len(cell)} rows)")
    print(pooled.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
