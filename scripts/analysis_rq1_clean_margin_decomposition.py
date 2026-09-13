#!/usr/bin/env python3
"""RQ1.14: does the same scale/alignment mechanism that explains repair on held-out FAILURES
(RQ1.12/RQ1.13: NN-Patching/PatchNAS have larger ||Delta z||, lower alignment) also show up in
how they damage originally-CORRECT (clean) inputs?

Gap this closes (user, 2026-09-06): RQ1.12/RQ1.13 characterise corrections on held-out failures
only. RQ1.10 separately shows NN-Patching/PatchNAS have catastrophic ungated Reg (0.82-0.86).
Those two facts are consistent with "large, poorly-aimed corrections repair some failures but
also damage clean inputs," but nothing computed so far actually measures what the correction
DOES to clean inputs -- the failure-side scale/alignment numbers cannot, by themselves, prove
they are what causes the clean-side regression. This script computes the parallel quantities on
the CLEAN population instead of the failure population, so the two sides can be read side by
side rather than inferred from one to the other.

For an originally-correct input (base_pred == y), the natural threat to correctness is its
STRONGEST BASE COMPETITOR c* = argmax_{k != y} base_logits(x)[k] (the runner-up class before any
patch is applied -- analogous to RQ1.12's d* = e_y - e_yhat, but anchored on PRESERVING y against
its closest rival rather than correcting y against the wrong prediction). Define, in exact
parallel to RQ1.12/RQ1.13:

    Delta z(x)            = patched_logits(x) - base_logits(x)
    Delta m'(x)            = [z'_y - z'_c*] - [z_y - z_c*]      (margin change vs the base runner-up)
    cos_to_preserve(x)     = Delta m'(x) / (||Delta z(x)|| * sqrt(2))

Delta m'(x) < 0 means the correction erodes the true class's margin over its closest competitor
(the exact mechanism a "regression" requires, though not sufficient by itself -- the competitor
could still lose to a THIRD class, which this pairwise metric does not see, same caveat RQ1.12's
d* already carries for the failure side).

Compares REGRESSED vs STAYED-CORRECT rows, same FOUR methods as RQ1.12/RQ1.13 (FixedPatch
included -- the zero-input-conditioning floor, mechanistically the most important comparison
point, same as RQ1.12/RQ1.13), raw (always-apply) `patched_logits` -- matching every other
M1/RQ1 sub-analysis's "what does the patch mechanism itself produce" framing, not the
routed/estimator-gated outcome.

Data sources (verified present):
  FixedPatch         outputs/effect_dump_fixedpatch_v8_s{seed}/<ds>/<bb>/deploy_direct/
                     predictions/{base,patched}_logits_clean_eval.npy, clean_eval_predictions.csv
                     (same redeployed tree RQ1.12's load_fixedpatch_cell() reads; arm selection
                     already happened at deploy time, per setting)
  DynaPatch-NoGate   outputs/effect_dump{_ep40ns,_lastaffine}_v8_s{seed}/<ds>/<bb>/deploy_direct/
                     predictions/{base,patched}_logits_clean_eval.npy, clean_eval_predictions.csv
  NN-Patching/       outputs/prior_patch_persample/<ds>_<bb>_s<seed>/<method>/tau/
  PatchNAS           {base,patched}_logits_clean_test.npy, clean_test_predictions.csv

**Evidence-strength note (explicit, per user instruction)**: this is a correlational,
cross-sectional comparison (regressed vs stayed-correct rows), not a controlled ablation that
fixes direction and varies magnitude, or vice versa. Report with "consistent with"/"these results
suggest", never "large scale CAUSES regression" -- that claim needs a fixed-direction,
varied-magnitude intervention this script does not run.

    .venv/bin/python scripts/analysis_rq1_clean_margin_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                                     # noqa: E402
from analysis_m1_correction_direction import SEEDS, SETTINGS, dynapatch_dir  # noqa: E402

SQRT2 = float(np.sqrt(2.0))
METHOD_ORDER = ["FixedPatch", "DynaPatch-NoGate", "NN-Patching", "PatchNAS"]


def load_clean_cell(method: str, ds: str, bb: str, seed: int):
    if method == "FixedPatch":
        # Same redeployed tree RQ1.12/analysis_rq1_alignment.py's load_fixedpatch_cell() reads
        # (deployment.save_route_features=true also dumped clean_eval, not just the two repair
        # splits that function uses) -- arm selection already happened at deploy time, per
        # setting, so no FIXEDPATCH_BEST_ARM lookup is needed here.
        d = ROOT / f"outputs/effect_dump_fixedpatch_v8_s{seed}/{ds}/{bb}/deploy_direct/predictions"
        pop = "clean_eval"
    elif method == "DynaPatch-NoGate":
        d, _ = dynapatch_dir(ds, bb, seed)
        pop = "clean_eval"
    else:
        d = ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/{method}/tau"
        pop = "clean_test"
    bl, pl, pc = (d / f"base_logits_{pop}.npy", d / f"patched_logits_{pop}.npy",
                 d / f"{pop}_predictions.csv")
    if not (bl.is_file() and pl.is_file() and pc.is_file()):
        return None, "missing artefact"
    base = np.load(bl).astype(np.float64)
    patched = np.load(pl).astype(np.float64)
    t = pd.read_csv(pc)
    if not (len(t) == len(base) == len(patched)):
        return None, f"row mismatch csv={len(t)} base={len(base)} patched={len(patched)}"
    correct = (t.base_pred == t.label).to_numpy()
    if correct.sum() < 6:
        return None, f"too few base-correct rows ({int(correct.sum())})"
    base, patched, t = base[correct], patched[correct], t[correct].reset_index(drop=True)
    y = t.label.to_numpy()
    n = len(t)
    ar = np.arange(n)

    base_for_competitor = base.copy()
    base_for_competitor[ar, y] = -np.inf
    cstar = base_for_competitor.argmax(1)

    dz = patched - base
    dm = (patched[ar, y] - patched[ar, cstar]) - (base[ar, y] - base[ar, cstar])
    regressed = patched.argmax(1) != y
    return {"dz": dz, "dm": dm, "regressed": regressed, "n": n}, None


def main() -> None:
    rows = []
    for method in METHOD_ORDER:
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                cell, err = load_clean_cell(method, ds, bb, seed)
                if err is not None:
                    rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                                "outcome": "-", "note": err})
                    continue
                dz_norm = np.linalg.norm(cell["dz"], axis=1)
                cos = cell["dm"] / (dz_norm * SQRT2 + 1e-12)
                reg = cell["regressed"]
                for outcome, mask in (("regressed", reg), ("stayed_correct", ~reg)):
                    if mask.sum() == 0:
                        continue
                    rows.append({
                        "method": method, "setting": f"{ds}/{bb}", "seed": seed,
                        "outcome": outcome, "n": int(mask.sum()),
                        "mean_dz_norm": float(dz_norm[mask].mean()),
                        "median_dz_norm": float(np.median(dz_norm[mask])),
                        "mean_dm": float(cell["dm"][mask].mean()),
                        "median_dm": float(np.median(cell["dm"][mask])),
                        "mean_cos_to_preserve": float(cos[mask].mean()),
                        "median_cos_to_preserve": float(np.median(cos[mask])),
                        "note": "",
                    })

    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "rq1_clean_margin_decomposition"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_cell.csv", index=False)

    valid = df[df.outcome != "-"]
    st = valid.groupby(["method", "outcome", "setting"])[
        ["mean_dz_norm", "median_dz_norm", "mean_dm", "median_dm",
         "mean_cos_to_preserve", "median_cos_to_preserve"]].mean().reset_index()
    st.to_csv(out / "per_setting.csv", index=False)
    summary = st.groupby(["method", "outcome"])[
        ["mean_dz_norm", "median_dz_norm", "mean_dm", "median_dm",
         "mean_cos_to_preserve", "median_cos_to_preserve"]].median().reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    RM.write_section(
        "RQ1CleanMarginDecomposition",
        "RQ1.14 -- scale/direction decomposition of Delta z(x) on CLEAN (originally-correct) "
        "inputs, regressed vs stayed-correct, all three per-input methods (raw)",
        f"""
Parallel construction to RQ1.12/RQ1.13's failure-side `cos_to_ideal`, anchored on PRESERVING the
true class against its strongest BASE-MODEL competitor instead of correcting it: `c* =
argmax_{{k != y}} base_logits(x)[k]`, `Delta m'(x) = [z'_y - z'_c*] - [z_y - z_c*]`,
`cos_to_preserve(x) = Delta m'(x) / (||Delta z(x)|| * sqrt(2))`. Rows restricted to inputs the
base model got RIGHT (a regression target), split into `regressed` (patch flips the argmax away
from y) vs `stayed_correct`. Median (not mean) reported for `||Delta z||`/`Delta m'`, same
reasoning as RQ1.13 (outlier-sensitive cell(s) possible in a freshly-trained head).

**Evidence-strength note**: this is a correlational, cross-sectional comparison, not a
controlled ablation isolating scale from direction. Report as "consistent with"/"these results
suggest", not "large scale CAUSES regression".
""",
        [("per_setting", st), ("summary", summary)],
    )

    pd.set_option("display.width", 160)
    piv_norm = st.pivot_table(index="setting", columns=["method", "outcome"],
                              values="median_dz_norm")
    piv_dm = st.pivot_table(index="setting", columns=["method", "outcome"], values="median_dm")
    print("median ||Delta z||, regressed vs stayed_correct:")
    print(piv_norm.round(2).to_string())
    print("\nmedian Delta m' (vs base runner-up), regressed vs stayed_correct:")
    print(piv_dm.round(3).to_string())
    print("\nsummary (median over settings):")
    print(summary.round(4).to_string(index=False))
    print(f"\n[written] {out.relative_to(ROOT)}/{{per_cell,per_setting,summary}}.csv")


if __name__ == "__main__":
    main()
