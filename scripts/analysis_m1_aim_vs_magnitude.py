#!/usr/bin/env python3
"""M1c -- when a failure is NOT repaired, was the correction aimed wrong, or not strong enough?

Every failure input x gets one correction Delta z(x) from one method. `repaired` (already in the
per-sample CSVs: does argmax(patched_logits) == y) says whether it worked. This script asks WHY
when it did not: aimed wrong, or aimed right but insufficient to cross the decision boundary.

"Aimed right" is `cos_to_ideal(x) > 0` (scripts/analysis_m1_correction_direction.py's scale-free
cosine between Delta z(x) and the ideal margin direction e_y - e_yhat -- unlike raw magnitude,
this stays comparable across methods despite NN-Patching/PatchNAS using a differently-scaled,
freshly trained head). This gives each failure exactly one of three outcomes, no clustering, no
permutation null, no cross-method magnitude comparison, and it is provably NOT a geometric
identity: `repaired => cos_to_ideal(x) > 0` is a theorem (repaired means z'_y is now the max over
ALL classes including yhat, so z'_y - z'_yhat > 0; the base model was wrong so z_y - z_yhat < 0;
Delta m(x) = (z'_y - z'_yhat) - (z_y - z_yhat) is therefore strictly positive), so the three
outcomes below are exhaustive and mutually exclusive by construction, and NOT an artefact the way
the deleted M1b personalisation-template comparison was (see note/PITFALLS.md).

    repaired            argmax flipped to y                          (aimed AND strong enough)
    aimed_not_repaired  cos_to_ideal(x) > 0 but argmax did not flip   (aimed right, not enough / another class won)
    misaimed            cos_to_ideal(x) <= 0                         (did not even favour y over yhat)

No win/loss framing: each method's own three-way split of its own failures is reported side by
side, not compared against another method's count.

    .venv/bin/python scripts/analysis_m1_aim_vs_magnitude.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                            # noqa: E402
from analysis_m1_correction_direction import METHODS, SEEDS, SETTINGS, load_cell   # noqa: E402

SQRT2 = float(np.sqrt(2.0))
OUTCOMES = ("repaired", "aimed_not_repaired", "misaimed")


def classify(cell: dict) -> np.ndarray:
    """One of OUTCOMES per failure input. Asserts the `repaired => aimed` theorem holds."""
    dz, dm, repaired = cell["dz"], cell["dm"], cell["repaired"]
    dz_norm = np.linalg.norm(dz, axis=1)
    cos_to_ideal = dm / (dz_norm * SQRT2 + 1e-12)
    aimed = cos_to_ideal > 0
    bad = repaired & ~aimed
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} rows are repaired but not aimed -- "
                             f"the repaired=>aimed theorem should make this impossible; "
                             f"check load_cell's argmax/label alignment")
    out = np.full(len(repaired), "misaimed", dtype=object)
    out[aimed] = "aimed_not_repaired"
    out[repaired] = "repaired"
    return out


def plot_breakdown(cell_df: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"repaired": "#2F6F73", "aimed_not_repaired": "#C9A227", "misaimed": "#C66A45"}
    held = cell_df[cell_df.split == "held"]
    pooled = held.groupby("method")[[f"n_{o}" for o in OUTCOMES]].sum()
    pooled = pooled.loc[list(METHODS.keys())]
    frac = pooled.div(pooled.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    left = np.zeros(len(frac))
    y_pos = np.arange(len(frac))
    for o in OUTCOMES:
        vals = frac[f"n_{o}"].to_numpy()
        ax.barh(y_pos, vals, left=left, color=colors[o], label=o.replace("_", " "), height=0.6)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.03:
                ax.text(l + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                       color="white")
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frac.index)
    ax.set_xlabel("fraction of held-out failures (all settings/seeds pooled)")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8, frameon=False)
    ax.set_title("Why a failure stayed unrepaired: wrong aim vs not enough (held-out)",
                fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"aim_vs_magnitude.{suffix}", dpi=220, bbox_inches="tight",
                   facecolor="white")
    plt.close(fig)


def main() -> None:
    cell_rows: list[dict] = []
    for method, dir_fn in METHODS.items():
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                for split in ("seen", "held"):
                    cell, err = load_cell(dir_fn, ds, bb, seed, split)
                    if err is not None:
                        cell_rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                                         "split": split, "note": err})
                        continue
                    outcome = classify(cell)
                    n = len(outcome)
                    counts = {f"n_{o}": int((outcome == o).sum()) for o in OUTCOMES}
                    fracs = {f"frac_{o}": counts[f"n_{o}"] / n for o in OUTCOMES}
                    cell_rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                                     "split": split, "n": n, **counts, **fracs, "note": ""})

    cell_df = pd.DataFrame(cell_rows)

    RM.write_section(
        "M1c", "M1c — why an unrepaired failure stayed unrepaired: wrong aim vs not enough (raw)",
        f"""
Every failure input gets exactly one outcome, by construction (see script docstring for the
`repaired => aimed` proof, and note/PITFALLS.md for why this is NOT the same shape of mistake as
the deleted M1b personalisation-template comparison):

| outcome | meaning |
|---|---|
| `repaired` | argmax(patched_logits) == true label |
| `aimed_not_repaired` | `cos_to_ideal(x) > 0` (correction favours the true class over the base model's wrong prediction) but the argmax did not flip to it -- not strong enough, or another class won instead |
| `misaimed` | `cos_to_ideal(x) <= 0` -- the correction did not even favour the true class over the wrong one |

One row per (method, setting, seed, split): raw counts (`n_repaired`, `n_aimed_not_repaired`,
`n_misaimed`) and fractions of that cell's failures. Not a win/loss table -- each method's own
three-way split of its own failures, nothing compared against another method's count.

🚨 **`repaired` here is NOT the RR_held reported in `note/RQ2_BASELINE_BEHAVIOR.md`.** Every M1
sub-analysis uses each method's RAW, un-routed patch output (`patched_logits`), matching
DynaPatch-NoGate's "always apply" framing -- NOT the estimator-routed prediction RQ2 reports for
NN-Patching/PatchNAS. Setting-mean `frac_repaired` on `held` here is DynaPatch-NoGate 0.575
(exactly matches RQ2's reported RR_held=0.5750 -- a consistency check, not a coincidence, since
DynaPatch-NoGate has no router either way) but NN-Patching 0.552 and PatchNAS 0.541 (versus RQ2's
routed 0.3746 / 0.3158). The gap is the router/estimator suppressing patch application it does
not trust, not a discrepancy in either number. This table measures blind repair capability if the
patch were always applied; it does not measure deployed repair rate.

![Aim vs magnitude](figures/m1_correction_direction/aim_vs_magnitude.png)

Figure: held-out failures pooled across all settings and seeds, one bar per method, stacked by
outcome fraction.
""",
        [("by_cell", cell_df.sort_values(["split", "method", "setting", "seed"]))],
    )

    out = ROOT / "outputs" / "m1_correction_direction"
    out.mkdir(parents=True, exist_ok=True)
    cell_df.to_csv(out / "aim_vs_magnitude_per_cell.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/aim_vs_magnitude_per_cell.csv  ({len(cell_df)} rows)")

    figure_dir = ROOT / "note" / "figures" / "m1_correction_direction"
    plot_breakdown(cell_df, figure_dir)
    print(f"[written] {figure_dir.relative_to(ROOT)}/aim_vs_magnitude.{{png,pdf}}")


if __name__ == "__main__":
    main()
