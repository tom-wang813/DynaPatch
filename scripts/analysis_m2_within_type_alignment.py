#!/usr/bin/env python3
"""M2 -- within a single failure type, does input-level variation in correction DIRECTION (not
just magnitude) predict repair, tested in a way that does not repeat the deleted M1b mistake?

Background (note/PITFALLS.md "own-direction vs shared-per-type-template is a geometric
identity", 2026-09-04): comparing each input's own direction against its failure type's AVERAGE
direction is a geometric identity (the average of unit vectors always has norm <= 1, so the
renormalized template is mechanically favoured regardless of whether personalisation helps) and
was deleted for exactly that reason. Its re-entry condition: link `cos_to_ideal(x)` directly to
the actual `repaired` outcome per input, not to a within-type average direction.

This script follows that condition. For each failure input x (in a failure type = (true label,
base prediction) with >= MIN_TYPE_N members, same floor `analysis_m1_correction_direction.py`
uses for "estimable"):
  cos_to_ideal(x) = Delta m(x) / (||Delta z(x)|| * sqrt(2))   -- scale-free aim quality
                                                                  (analysis_m1_aim_vs_magnitude.py)
  dz_norm(x)      = ||Delta z(x)||                             -- magnitude
Both are DEMEANED WITHIN their failure type (subtract the type's own mean), which removes
between-type average differences entirely -- no group average is compared to anything, only the
input's deviation from its own type's mean is used, and that deviation is regressed directly
against `repaired`, never against another vector's norm. This cannot reproduce the M1b identity:
demeaning does not involve renormalizing an averaged vector, and the regression target is a real
outcome label, not a score built from the same directions being compared.

Model: standardized logistic regression, `repaired ~ cos_to_ideal_centered + dz_norm_centered`,
fit per (method, setting) pooling seeds x held-out failures, both regressors z-scored after
centering so coefficients are comparable within a fit and (as mean-|coef| across settings)
across methods. Both predictors together separate "was this input's correction better-AIMED than
its type's average" from "was it just BIGGER than its type's average" -- the same direction-vs-
magnitude question RQ1 section 3 already answered BETWEEN failure types (destroying direction
collapses RR, destroying magnitude alone does not), asked here WITHIN a type instead.

    .venv/bin/python scripts/analysis_m2_within_type_alignment.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                                            # noqa: E402
from analysis_m1_correction_direction import METHODS, SEEDS, SETTINGS, MIN_TYPE_N, load_cell  # noqa: E402

SQRT2 = float(np.sqrt(2.0))
MIN_ROWS = 20        # need at least this many eligible rows to attempt a 2-covariate fit
MIN_MINORITY = 5     # need at least this many of the rarer `repaired` class


def cell_frame(dir_fn, ds: str, bb: str, seed: int, split: str) -> pd.DataFrame | None:
    cell, err = load_cell(dir_fn, ds, bb, seed, split)
    if cell is None:
        return None
    dz, dm, y, yhat, repaired = cell["dz"], cell["dm"], cell["y"], cell["yhat"], cell["repaired"]
    dz_norm = np.linalg.norm(dz, axis=1)
    cos_to_ideal = dm / (dz_norm * SQRT2 + 1e-12)
    fail_type = pd.Series(y).astype(str) + "->" + pd.Series(yhat).astype(str)
    bad = repaired & ~(cos_to_ideal > 0)
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} rows repaired but cos_to_ideal<=0 -- "
                             f"the repaired=>aimed theorem should make this impossible")
    df = pd.DataFrame({"fail_type": fail_type.to_numpy(), "cos": cos_to_ideal,
                       "mag": dz_norm, "repaired": repaired})
    counts = df.fail_type.value_counts()
    keep = counts[counts >= MIN_TYPE_N].index
    df = df[df.fail_type.isin(keep)]
    return df if len(df) else None


def within_type_center(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("cos", "mag"):
        df[f"{col}_c"] = df.groupby("fail_type")[col].transform(lambda s: s - s.mean())
        sd = df[f"{col}_c"].std()
        df[f"{col}_c"] = df[f"{col}_c"] / sd if sd > 1e-9 else 0.0
    return df


def fit_cell(df: pd.DataFrame) -> dict | None:
    n_pos = int(df.repaired.sum())
    n_neg = int((~df.repaired).sum())
    if len(df) < MIN_ROWS or min(n_pos, n_neg) < MIN_MINORITY:
        return None
    X = df[["cos_c", "mag_c"]].to_numpy()
    y = df["repaired"].to_numpy().astype(int)
    mdl = LogisticRegression(max_iter=2000)
    mdl.fit(X, y)
    return {"coef_cos": float(mdl.coef_[0, 0]), "coef_mag": float(mdl.coef_[0, 1]),
           "n": len(df), "n_pos": n_pos, "n_neg": n_neg,
           "n_types": int(df.fail_type.nunique()),
           "train_acc": float((mdl.predict(X) == y).mean())}


def main() -> None:
    per_setting_rows = []
    for method, dir_fn in METHODS.items():
        for ds, bb in SETTINGS:
            parts = []
            for seed in SEEDS:
                d = cell_frame(dir_fn, ds, bb, seed, "held")
                if d is not None:
                    parts.append(d)
            if not parts:
                continue
            pooled = within_type_center(pd.concat(parts, ignore_index=True))
            fit = fit_cell(pooled)
            if fit is None:
                continue
            per_setting_rows.append({"method": method, "setting": f"{ds}/{bb}", **fit})

    df = pd.DataFrame(per_setting_rows)
    out = ROOT / "outputs" / "m2_within_type_alignment"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_setting.csv", index=False)

    summary = df.groupby("method").agg(
        mean_coef_cos=("coef_cos", "mean"), mean_coef_mag=("coef_mag", "mean"),
        n_settings_cos_pos=("coef_cos", lambda s: int((s > 0).sum())),
        n_settings_mag_pos=("coef_mag", lambda s: int((s > 0).sum())),
        n_settings=("coef_cos", "size"),
        mean_train_acc=("train_acc", "mean")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    RM.write_section(
        "M2WithinTypeAlignment",
        "M2 -- within-failure-type input-level direction vs repair, corrected for the M1b identity (raw)",
        f"""
Per (method, setting): pool the 3 seeds' held-out failures, restrict to failure types with
>= {MIN_TYPE_N} members, demean `cos_to_ideal(x)` and `dz_norm(x)` WITHIN each failure type (so
between-type average differences are removed entirely -- no group average is compared to
anything else, only each input's own deviation from its type's mean is used), z-score both
demeaned columns, then fit standardized logistic regression `repaired ~ cos_to_ideal_centered +
dz_norm_centered`. `coef_cos` isolates whether being better-AIMED than one's own failure type's
average (at matched magnitude) predicts repair; `coef_mag` isolates the same for being BIGGER
than the type average (at matched aim). A settings with fewer than {MIN_ROWS} eligible rows or
fewer than {MIN_MINORITY} of the rarer `repaired` class after the type-size floor is dropped
(too little signal for a 2-covariate fit), noted as absent from `per_setting.csv` rather than
forced.

This is NOT the deleted M1b comparison (`note/PITFALLS.md`, 2026-09-04): M1b renormalized an
averaged direction vector and was mechanically biased by the `||mean of units|| <= 1` identity
regardless of whether personalisation helps. Here no vector is renormalized or averaged into a
template -- only within-type demeaning (a linear operation with no such bound) and a direct
regression against the real `repaired` label.
""",
        [("per_setting", df), ("summary", summary)],
    )
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print("\nSummary (mean over settings):")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
