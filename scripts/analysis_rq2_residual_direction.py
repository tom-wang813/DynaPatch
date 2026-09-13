#!/usr/bin/env python3
"""Residual check for tab:rq1d / tab:rq2_failure_type: does the within/between failure-type
cosine gap on Delta z(x) survive once the component along the ideal margin direction
d*(x) = e_y - e_yhat is projected out?

WHY THIS SCRIPT EXISTS (2026-09-11, user-raised concern, confirmed valid before writing this):
`failure_type` is defined as the string "y->yhat" (`analysis_m1_correction_direction.py`,
`analysis_m2_within_type_alignment.py`), and the "ideal margin direction" used for
`cos_to_ideal` (tab:rq2_norm) is `d*(x) = e_y - e_yhat` -- the SAME (y, yhat) pair, just
vectorised instead of stringified. `Delta m(x) = <Delta z(x), d*(x)>` exactly (by construction,
not by approximation -- verified algebraically in analysis_m1_correction_direction.py's own
docstring). So if `Delta z(x)` is well-aligned with `d*(x)` (high cos_to_ideal, tab:rq2_norm's
finding), then within a `failure_type` group `d*` is a CONSTANT vector, and between groups `d*`
vectors are generically near-orthogonal (distinct one-hot pairs in a >2-class label space) --
which by itself is enough to mechanically produce "low within-type cosine distance, high
between-type cosine distance" (tab:rq1d / tab:rq2_failure_type's finding), with no need for the
generator to encode anything about failure type BEYOND what cos_to_ideal already says. The two
results could therefore be the same fact measured twice, not two independent pieces of evidence.

THE CHECK: project each Delta z(x) onto the subspace orthogonal to d*(x) (remove the aligned
component `(Delta m(x) / 2) * d*(x)` -- see derivation below), then re-run the EXACT SAME
within/between-failure-type cosine-gap test (`analysis_p4_patchvec.pair_stats`, same population,
same label-shuffle null) on the RESIDUAL. If the gap survives on the residual, the direction
carries failure-type structure beyond alignment to d* -- tab:rq1d is independent evidence. If the
residual gap collapses towards the null, tab:rq1d was mechanically implied by tab:rq2_norm and
should not be cited as separate evidence.

Projection derivation: d*(x) = e_y - e_yhat has ||d*||^2 = 2 (y != yhat, distinct one-hot
indices). Delta m(x) = <Delta z(x), d*(x)> exactly. The component of Delta z(x) along d*(x) is
  proj(x) = (<Delta z(x), d*(x)> / ||d*(x)||^2) * d*(x) = (Delta m(x) / 2) * d*(x).
Residual: Delta z_resid(x) = Delta z(x) - proj(x). By construction <Delta z_resid(x), d*(x)> = 0
for every x -- verified with an assertion below before trusting any downstream number.

    .venv/bin/python scripts/analysis_rq2_residual_direction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analysis_m1_correction_direction import METHODS, SETTINGS, SEEDS, MAX_N, load_cell  # noqa: E402
from analysis_p4_patchvec import pair_stats  # noqa: E402

SPLIT = "held"  # tab:rq1d/tab:rq2_failure_type is held-split only


def d_star_matrix(y: np.ndarray, yhat: np.ndarray, n_classes: int) -> np.ndarray:
    d = np.zeros((len(y), n_classes))
    d[np.arange(len(y)), y] += 1.0
    d[np.arange(len(y)), yhat] -= 1.0
    return d


def main() -> None:
    rows = []
    for method, dir_fn in METHODS.items():
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                cell, err = load_cell(dir_fn, ds, bb, seed, SPLIT)
                if err is not None:
                    continue
                dz, dm, y, yhat, n = cell["dz"], cell["dm"], cell["y"], cell["yhat"], cell["n"]
                n_classes = dz.shape[1]
                dstar = d_star_matrix(y, yhat, n_classes)

                # sanity: Delta m(x) == <Delta z(x), d*(x)> exactly (algebraic identity this
                # whole projection depends on) -- verified before trusting anything downstream.
                dm_check = (dz * dstar).sum(1)
                assert np.allclose(dm_check, dm, atol=1e-6), \
                    f"{method} {ds}/{bb} s{seed}: Delta m(x) != <Delta z(x), d*(x)>, max err " \
                    f"{np.abs(dm_check - dm).max():.3e}"

                proj = (dm / 2.0)[:, None] * dstar
                dz_resid = dz - proj
                # residual must be orthogonal to d*(x) for every row, by construction
                resid_dot = (dz_resid * dstar).sum(1)
                assert np.abs(resid_dot).max() < 1e-6, \
                    f"{method} {ds}/{bb} s{seed}: residual not orthogonal to d*, max dot " \
                    f"{np.abs(resid_dot).max():.3e}"

                fail_type = (pd.Series(y).astype(str) + "->" + pd.Series(yhat).astype(str)).to_numpy()

                dz_c, dzr_c, lab_c = dz, dz_resid, fail_type
                if n > MAX_N:
                    sel = np.random.default_rng(0).choice(n, MAX_N, replace=False)
                    dz_c, dzr_c, lab_c = dz[sel], dz_resid[sel], fail_type[sel]

                st_full = pair_stats(dz_c, lab_c, np.random.default_rng(0))
                st_resid = pair_stats(dzr_c, lab_c, np.random.default_rng(0))
                if st_full is None or st_resid is None:
                    continue
                rows.append({
                    "method": method, "setting": f"{ds}/{bb}", "seed": seed,
                    "n_vec": st_full["n_vec"], "n_groups": st_full["n_groups"],
                    "cos_within_full": st_full["cos_within"], "cos_between_full": st_full["cos_between"],
                    "cos_gap_full": st_full["cos_gap"],
                    "cos_within_resid": st_resid["cos_within"], "cos_between_resid": st_resid["cos_between"],
                    "cos_gap_resid": st_resid["cos_gap"],
                    "cos_gap_resid_p_perm": st_resid.get("cos_p_perm"),
                    "cos_gap_resid_z": st_resid.get("cos_z"),
                })

    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "rq2_residual_direction"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_cell.csv", index=False)

    summary = df.groupby("method")[
        ["cos_within_full", "cos_between_full", "cos_gap_full",
         "cos_within_resid", "cos_between_resid", "cos_gap_resid"]].mean()
    n_sig = df.groupby("method").apply(
        lambda g: (g["cos_gap_resid_p_perm"] < 0.05).sum(), include_groups=False)
    n_cells = df.groupby("method").size()

    summary.to_csv(out / "summary.csv")
    print(f"[written] {out.relative_to(ROOT)}/per_cell.csv  ({len(df)} rows)")
    print(f"[written] {out.relative_to(ROOT)}/summary.csv")
    print()
    print("Full Delta z(x) (reproduces tab:rq1d) vs residual after removing the d*(x) component:")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print("Cells with residual gap significant at p<0.05 (label-shuffle null):")
    for m in summary.index:
        print(f"  {m}: {n_sig.get(m, 0)}/{n_cells.get(m, 0)}")


if __name__ == "__main__":
    main()
