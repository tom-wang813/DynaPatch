#!/usr/bin/env python3
"""RQ1.12: alignment of each method's own correction Delta z(x) to the IDEAL MARGIN direction
d* = e_y - e_yhat, on held-out failures, for all FOUR RQ1 methods (FixedPatch, DynaPatch-NoGate,
NN-Patching, PatchNAS).

`cos_to_ideal(x) = Delta m(x) / (||Delta z(x)|| * sqrt(2))` is algebraically
`cos(Delta z(x), e_y - e_yhat)` -- a true cosine, so, unlike raw `||Delta z||`/`Delta m` (not
cross-method comparable: NN-Patching/PatchNAS's freshly-trained head has no constraint tying its
logit scale to DynaPatch's/FixedPatch's frozen one), this metric is legitimately comparable
across all four methods. It is the exact per-input quantity
`scripts/analysis_m1_aim_vs_magnitude.py` already computes to bucket repaired/aimed-not-
repaired/misaimed; this aggregates it into a mean per (method, setting) instead of a three-way
bucket count.

NOTE ON "ideal": d* = e_y - e_yhat is an OPERATIONAL PROXY for the desired repair direction (it
only asks whether the true-vs-wrong-class margin increases), not a mathematically optimal
correction -- actual classification success also depends on every other class's logit. Call it
the "ideal margin direction" / "target-margin direction" in prose, never "optimal repair
direction".

FixedPatch requires a one-off redeploy (2026-09-06): its `outputs/patch_ablation/.../deploy/
predictions/` trees only ever stored argmax-level predictions, never full logit vectors, unlike
DynaPatch's `effect_dump` trees. Redeployed via `src/experiment/deploy_eval.py`
(`deployment.save_route_features=true`, `method.name=fixed_patch`, the per-setting winning
learning-rate arm `scripts/sample_frame.py`'s FixedPatch selection rule already uses, checkpoint
`repair_last.pt` -- NOT `repair_best.pt`, verified: `repair_best.pt` reproduces a DIFFERENT,
wrong RR_held) into `outputs/effect_dump_fixedpatch_v8_s{seed}/{ds}/{bb}/deploy_direct/`, same
file layout as DynaPatch's own `effect_dump` trees. Verified against the already-published
RQ1.1 FixedPatch RR_held numbers before trusting: 11/12 settings reproduce to 4 decimal places
exactly; the 12th (`tt100k_signs/resnet50`) matches on 2/3 seeds and differs by exactly 1/258
rows on the third (floating-point forward-pass nondeterminism, not a config mismatch).

    .venv/bin/python scripts/analysis_rq1_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                                       # noqa: E402
from analysis_m1_correction_direction import METHODS, SEEDS, SETTINGS, load_cell  # noqa: E402

SQRT2 = float(np.sqrt(2.0))

# FixedPatch's per-setting winning learning-rate arm, same selection rule
# scripts/sample_frame.py applies (highest mean RR_held over 3 seeds on repair_holdout_unseen).
FIXEDPATCH_BEST_ARM = {
    "gtsrb/resnet50": "fixed60_lr1e1", "gtsrb/convnext_tiny": "fixed60_lr1e2",
    "gtsrb/vgg16": "fixed60_lr1e2", "gtsrb/densenet121": "fixed60_lr1e1",
    "tt100k_signs/resnet50": "fixed60_lr1e2", "tt100k_signs/convnext_tiny": "fixed60_lr1e2",
    "tt100k_signs/vgg16": "fixed60_lr1e2", "tt100k_signs/densenet121": "fixed60_lr1e2",
    "lisa_signs/resnet50": "fixed60_lr1e1", "lisa_signs/convnext_tiny": "fixed60_lr1e2",
    "lisa_signs/vgg16": "fixed60_lr1e2", "lisa_signs/densenet121": "fixed60_lr1e2",
}


def load_fixedpatch_cell(ds: str, bb: str, seed: int) -> dict | None:
    d = ROOT / f"outputs/effect_dump_fixedpatch_v8_s{seed}/{ds}/{bb}/deploy_direct/predictions"
    bl, pl, pc = (d / "base_logits_repair_holdout_unseen.npy",
                 d / "patched_logits_repair_holdout_unseen.npy",
                 d / "repair_holdout_unseen_predictions.csv")
    if not (bl.is_file() and pl.is_file() and pc.is_file()):
        return None
    base = np.load(bl).astype(np.float64)
    patched = np.load(pl).astype(np.float64)
    t = pd.read_csv(pc)
    fail = (t.base_pred != t.label).to_numpy()
    base, patched = base[fail], patched[fail]
    y = t.label.to_numpy()[fail]
    yhat = t.base_pred.to_numpy()[fail]
    n = len(y)
    ar = np.arange(n)
    dz = patched - base
    dm = (patched[ar, y] - patched[ar, yhat]) - (base[ar, y] - base[ar, yhat])
    return {"dz": dz, "dm": dm, "n": n}


def main() -> None:
    rows = []
    for method, dir_fn in METHODS.items():
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                cell, err = load_cell(dir_fn, ds, bb, seed, "held")
                if cell is None:
                    continue
                cos = cell["dm"] / (np.linalg.norm(cell["dz"], axis=1) * SQRT2 + 1e-12)
                rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                            "mean_cos_to_ideal": float(cos.mean()),
                            "median_cos_to_ideal": float(np.median(cos)), "n": cell["n"]})
    for ds, bb in SETTINGS:
        for seed in SEEDS:
            cell = load_fixedpatch_cell(ds, bb, seed)
            if cell is None:
                continue
            cos = cell["dm"] / (np.linalg.norm(cell["dz"], axis=1) * SQRT2 + 1e-12)
            rows.append({"method": "FixedPatch", "setting": f"{ds}/{bb}", "seed": seed,
                        "mean_cos_to_ideal": float(cos.mean()),
                        "median_cos_to_ideal": float(np.median(cos)), "n": cell["n"]})

    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "rq2" / "alignment"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_cell.csv", index=False)

    st = df.groupby(["method", "setting"]).mean_cos_to_ideal.mean().reset_index()
    st.to_csv(out / "per_setting.csv", index=False)
    summary = st.groupby("method").mean_cos_to_ideal.agg(["mean", "median", "size"]).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    piv = st.pivot(index="setting", columns="method", values="mean_cos_to_ideal")
    order = [f"{d}/{b}" for d, b in SETTINGS]
    piv = piv.loc[order][["FixedPatch", "DynaPatch-NoGate", "NN-Patching", "PatchNAS"]]

    RM.write_section(
        "RQ1Alignment",
        "RQ1.12 -- alignment of each method's own correction to the ideal margin direction, all four methods (raw)",
        f"""
`cos_to_ideal(x) = Delta m(x) / (||Delta z(x)|| * sqrt(2))`, algebraically
`cos(Delta z(x), e_y - e_yhat)` -- a true, scale-free cosine, comparable across methods despite
their differently-scaled logit heads (unlike raw `Delta z` norm / `Delta m`). Mean per
(method, setting), averaged over 3 seeds' held-out failures. FixedPatch required a one-off
redeploy this session (`outputs/effect_dump_fixedpatch_v8_s{{seed}}/`, see script docstring for
the checkpoint gotcha: `repair_last.pt`, not `repair_best.pt`).
""",
        [("per_setting", piv.reset_index()), ("summary", summary)],
    )
    pd.set_option("display.width", 150)
    print(piv.round(4).to_string())
    print()
    print(summary.round(4).to_string(index=False))
    print(f"\nDynaPatch > FixedPatch: {(piv['DynaPatch-NoGate'] > piv['FixedPatch']).sum()}/12")
    beats_all = ((piv["DynaPatch-NoGate"] > piv["FixedPatch"])
                & (piv["DynaPatch-NoGate"] > piv["NN-Patching"])
                & (piv["DynaPatch-NoGate"] > piv["PatchNAS"]))
    print(f"DynaPatch beats all three others: {beats_all.sum()}/12")
    print(f"\n[written] {out.relative_to(ROOT)}/{{per_cell,per_setting,summary}}.csv")


if __name__ == "__main__":
    main()
