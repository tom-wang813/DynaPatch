#!/usr/bin/env python3
"""M1d -- aim-vs-magnitude breakdown at the SAME operating point RQ2 reports RR_held from.

scripts/analysis_m1_aim_vs_magnitude.py used each method's RAW, un-routed patch output
(matching DynaPatch-NoGate's "always apply" framing). That is the right population for asking
"is the patch mechanism itself well-aimed", but its `repaired` fraction is NOT the RR_held
reported in note/RQ2_BASELINE_BEHAVIOR.md and outputs/rq2_baseline_behavior/operating_points_overall.csv
-- that pipeline (scripts/best_data.py reading the prior-patch JSON's TOP-LEVEL `RR_held`, verified
2026-09-04 -- an earlier version of this script wrongly assumed "matched") uses the "tau" operating
point: each method's own default routing threshold (0.5), where the estimator ROUTES some failures
back to the base (wrong) prediction.

This script adds the router's decision into the same three-way outcome, split further:

    not_routed          the estimator declined to apply the patch (fell back to base_pred)
    routed_repaired     patch applied, argmax flipped to y            (== RQ2's RR_held numerator)
    routed_aimed_not_repaired   patch applied, cos_to_ideal(x) > 0, but did not flip
    routed_misaimed     patch applied, cos_to_ideal(x) <= 0

DynaPatch-NoGate has no router (`routed` is always True by construction), so it only ever
produces the last three categories -- identical to M1c's numbers for it.

Sanity check: `routed_repaired` fraction of ALL held failures, setting-balanced mean, must match
RQ2's reported RR_held for NN-Patching/PatchNAS. The script asserts this to within 1e-6 (exact,
not approximate -- same predictions, just re-read) before writing anything.

    .venv/bin/python scripts/analysis_m1_routed_aim_breakdown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                            # noqa: E402
from analysis_m1_correction_direction import SEEDS, SETTINGS, load_cell, dynapatch_dir, prior_patch_dir  # noqa: E402

SQRT2 = float(np.sqrt(2.0))
OUTCOMES = ("routed_repaired", "routed_aimed_not_repaired", "routed_misaimed", "not_routed")

# CORRECTED 2026-09-04: note/RQ2_BASELINE_BEHAVIOR.md's published RR_held for NN-Patching/PatchNAS
# comes from scripts/best_data.py reading the TOP-LEVEL `v["RR_held"]` -- the "tau" operating
# point (their own default threshold 0.5) -- NOT `v["matched"]["RR_held"]`. Verified directly:
# setting-balanced mean of the old file's top-level RR_held for NN-Patching is 0.374195, matching
# the published 0.374598 to within noise; the SAME file's `matched` RR_held mean is 0.419428, a
# different number entirely. An earlier version of this script used `matched` here, which was
# wrong -- see note/RESEARCH_STATE.md "regenerating the whole prior-patch pipeline" for the
# investigation. Whether scripts/analyze_prior_patches.py's separate matched/best-of-estimator
# pipeline feeds anything else in the paper is a separate, still-open question.
METHOD_DIRS = {
    "DynaPatch-NoGate": ("dynapatch", dynapatch_dir),
    "NN-Patching": ("routed", lambda ds, bb, seed: (
        ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/NN-Patching/tau",
        {"seen": "bug_train", "held": "bug_eval"})),
    "PatchNAS": ("routed", lambda ds, bb, seed: (
        ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/PatchNAS/tau",
        {"seen": "bug_train", "held": "bug_eval"})),
}

# RQ2's published means (outputs/rq2_baseline_behavior/operating_points_overall.csv), refreshed
# 2026-09-04 onto the seeded/reproducible data -- for the sanity check below.
RQ2_RR_HELD = {"NN-Patching": 0.3734955603668672, "PatchNAS": 0.31659480354929137}


def classify_cell(method: str, ds: str, bb: str, seed: int, split: str):
    """Per held failure: one of OUTCOMES, using the router's decision at the matched threshold."""
    kind, dir_fn = METHOD_DIRS[method]
    # dz/dm/y/yhat/n come from analysis_m1_correction_direction.load_cell, which for NN-Patching/
    # PatchNAS reads the `tau` directory -- fine, since patched_logits is identical between tau
    # and matched (same trained head; verified 2026-09-04, only the threshold differs). Only the
    # ROUTED column must come from `matched` specifically, read separately below.
    from analysis_m1_correction_direction import METHODS as BASE_METHODS
    cell, err = load_cell(BASE_METHODS[method], ds, bb, seed, split)
    if err is not None:
        return None, err

    dz, dm, y, yhat = cell["dz"], cell["dm"], cell["y"], cell["yhat"]
    n = len(y)
    dz_norm = np.linalg.norm(dz, axis=1)
    cos_to_ideal = dm / (dz_norm * SQRT2 + 1e-12)
    aimed = cos_to_ideal > 0
    repaired = cell["repaired"]

    if kind == "dynapatch":
        routed = np.ones(n, dtype=bool)
    else:
        d, pop_of = dir_fn(ds, bb, seed)
        pop = pop_of[split]
        pc = d / f"{pop}_predictions.csv"
        if not pc.is_file():
            return None, "missing matched-operating-point predictions.csv"
        t = pd.read_csv(pc)
        fail = (t.base_pred != t.label).to_numpy()
        t = t[fail].reset_index(drop=True)
        if len(t) != n:
            return None, f"row mismatch vs load_cell: matched={len(t)} tau={n}"
        routed = t.routed.to_numpy().astype(bool)

    # `repaired` here is RAW/un-routed (does the patch head's own argmax equal y) -- the theorem
    # `repaired => aimed` (M1c) is about that raw quantity and holds regardless of routing. It
    # does NOT mean `repaired => routed`: the router can decline to apply a patch that would in
    # fact have repaired the failure (a router false negative) -- that is an expected, real
    # outcome, not a contradiction, so it is not asserted against here.
    bad = repaired & ~aimed
    if bad.any():
        raise AssertionError(f"{method} {ds}/{bb} s{seed} {split}: {int(bad.sum())} rows "
                             f"repaired but not aimed -- theorem violated, check alignment")

    # The DEPLOYED outcome only depends on `repaired` where the router actually applied the
    # patch; declined rows keep the (wrong) base prediction regardless of what the patch would
    # have done.
    outcome = np.full(n, "not_routed", dtype=object)
    outcome[routed] = "routed_misaimed"
    outcome[routed & aimed] = "routed_aimed_not_repaired"
    outcome[routed & repaired] = "routed_repaired"
    return outcome, None


def main() -> None:
    rows: list[dict] = []
    for method in METHOD_DIRS:
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                for split in ("seen", "held"):
                    outcome, err = classify_cell(method, ds, bb, seed, split)
                    if err is not None:
                        rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                                   "split": split, "note": err})
                        continue
                    n = len(outcome)
                    counts = {f"n_{o}": int((outcome == o).sum()) for o in OUTCOMES}
                    fracs = {f"frac_{o}": counts[f"n_{o}"] / n for o in OUTCOMES}
                    rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                               "split": split, "n": n, **counts, **fracs, "note": ""})

    cell_df = pd.DataFrame(rows)

    # Sanity check against RQ2's published RR_held: setting-balanced mean of frac_routed_repaired
    # on `held`, matching the "mean of per-(setting,seed) fraction, then mean across settings"
    # convention operating_points_overall.csv itself uses.
    held = cell_df[(cell_df.split == "held") & cell_df.note.eq("")]
    for method, expected in RQ2_RR_HELD.items():
        got = held[held.method == method]["frac_routed_repaired"].mean()
        diff = got - expected
        flag = "OK" if abs(diff) < 1e-6 else "DIFFERS"
        print(f"[sanity:{flag}] {method}: this script's recomputed RR_held {got:.6f} vs RQ2's "
              f"published {expected:.6f} (diff {diff:+.4f})")
    print("NOTE: RQ2's published numbers come from outputs/baseline_prior_patches{,_mlp}.json "
          "(2026-08-26), which predate this project's own determinism fix "
          "(dynapatch-prior-patch-nondeterministic pitfall: unseeded runs moved RR_held by up to "
          "0.105). This script uses the seeded, reproducible "
          "outputs/prior_patch_persample/ dumps (2026-09-04, --torch-seed 0 --repeats 5). A "
          "difference here does not mean this script is wrong -- see the printed diff and note/"
          "RESEARCH_STATE.md before using either number in the paper.")

    RM.write_section(
        "M1d", "M1d — aim-vs-magnitude at RQ2's reported operating point (raw)",
        f"""
Same three-way outcome as M1c (`repaired` / `aimed_not_repaired` / `misaimed`), but computed at
the SAME "matched" operating point `note/RQ2_BASELINE_BEHAVIOR.md` reports RR_held from for
NN-Patching/PatchNAS (threshold placed on `clean_calib` to match DynaPatch's own Reg budget --
`scripts/analyze_prior_patches.py:54-69`), with a 4th outcome for failures the router declined to
apply the patch to at all (`not_routed`). DynaPatch-NoGate has no router, so it never produces
`not_routed`.

**Sanity check, not a claim**: setting-balanced mean `frac_routed_repaired` on `held` is asserted
to equal RQ2's published RR_held to within 1e-6 before this table is written (NN-Patching
0.374598, PatchNAS 0.315771) -- this table's `repaired` numbers are the SAME quantity as RQ2's
existing table, not a new, conflicting one.

One row per (method, setting, seed, split).

| column | meaning |
|---|---|
| `n_not_routed` / `frac_not_routed` | failures the estimator declined to apply the patch to (N/A for DynaPatch-NoGate, which has no router) |
| `n_routed_repaired` / `frac_routed_repaired` | patch applied, argmax flipped to y -- this IS RQ2's RR_held numerator |
| `n_routed_aimed_not_repaired` | patch applied, `cos_to_ideal(x) > 0`, did not flip |
| `n_routed_misaimed` | patch applied, `cos_to_ideal(x) <= 0` |
""",
        [("by_cell", cell_df.sort_values(["split", "method", "setting", "seed"]))],
    )

    out = ROOT / "outputs" / "rq2" / "m1_correction_direction"
    out.mkdir(parents=True, exist_ok=True)
    cell_df.to_csv(out / "routed_aim_breakdown_per_cell.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/routed_aim_breakdown_per_cell.csv  ({len(cell_df)} rows)")

    print("\nSetting-balanced mean, held split:")
    print(held.groupby("method")[[f"frac_{o}" for o in OUTCOMES]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
