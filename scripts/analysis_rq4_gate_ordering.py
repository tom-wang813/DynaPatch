#!/usr/bin/env python3
"""RQ4 mechanism: does the gate's own score order actual patch outcomes, independently of any
particular lambda -- and does tightening lambda simply move a cutoff through that pre-existing
ordering, rather than lambda itself creating the repair/regression asymmetry?

This is the analysis that should anchor RQ4, not the lambda-vs-no-gate percentage comparison:
raising lambda in `commit iff P(gain=+1) > lambda * P(gain=-1)` is DESIGNED to penalize
regression more as lambda grows, so "RR falls slower than Reg as lambda rises" is close to the
decision rule doing its job by construction, not an independent finding. What is NOT guaranteed
by construction is whether the score itself -- a quantity that exists independently of any
lambda, log P(gain=+1) - log P(gain=-1) -- actually separates repaired / no-benefit / regression
rows in the right order on held-out data. That is an empirical question about the fitted model,
and lambda's only role, if the ordering holds, is to slide a cutoff through an already-structured
axis.

Reuses `outputs/gate_lambda_mechanism/per_sample.csv` (scripts/analysis_gate_lambda_mechanism.py)
and `outputs/rq4_lambda_normalized/*.csv` (scripts/analysis_rq4_lambda_normalized.py) -- no new
gate fit, no new labels, only new views of already-computed per-sample scores/outcomes:

  A) setting-balanced weighted distribution of the score by true outcome (same weighting
     scheme as `scripts/build_results_draft.py`'s RQ3.3 confidence-distribution ECDF: equal
     total mass per setting, equal mass per seed-cell within a setting, equal mass per sample
     within a cell) -- for the ECDF figure.
  B) outcome composition of the patch applications newly rejected at each lambda step (held-side
     successful-repair/ineffective-change plus clean-side regression), computed per setting and
     then averaged equally across the 12 settings (a setting with zero newly-rejected rows at a
     given step is excluded from that step's mean, not counted as 0%) -- for the
     stacked-composition figure.
  C) per-setting separation between the repair and regression score distributions (AUC / rank
     statistic, i.e. P(score_repair > score_regression) for a random pair from that setting) --
     to check whether the settings with the best repair-regression trade-off (RQ4's earlier
     panel) are exactly the settings whose gate best separates the two outcomes.

    .venv/bin/python scripts/analysis_rq4_gate_ordering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402

N_SETTINGS = 12
OUTCOME_ORDER = ["regression", "ineffective change", "successful repair"]


def weighted_distribution(cell: pd.DataFrame) -> pd.DataFrame:
    """Equal total mass per setting, equal mass per (setting,seed) cell within a setting, equal
    mass per sample within a cell -- identical construction to build_results_draft.py's RQ3.3
    confidence-distribution ECDF, applied here to the gate's score instead of confidence change."""
    cell_sizes = cell.groupby(["setting", "seed", "outcome"], as_index=False).size().rename(
        columns={"size": "cell_n"})
    cells_per_setting = cell_sizes.groupby(["setting", "outcome"], as_index=False).size().rename(
        columns={"size": "cells_in_setting"})
    d = cell.merge(cell_sizes, on=["setting", "seed", "outcome"])
    d = d.merge(cells_per_setting, on=["setting", "outcome"])
    d["weight"] = 1 / (N_SETTINGS * d.cells_in_setting * d.cell_n)
    assert np.allclose(d.groupby("outcome").weight.sum().to_numpy(), 1), \
        "each outcome's weights must sum to 1 -- equal total mass per setting is the point"
    return d


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile of a weighted empirical distribution (linear interpolation on the weighted
    CDF) -- a descriptive statistic of the already-weighted distribution `weighted_distribution`
    builds, not a new evaluation metric."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, v))


def score_summary_table(dist: pd.DataFrame) -> pd.DataFrame:
    """Table A: median/IQR gate score and the fraction rejected at lambda=1 (score <= 0), by true
    outcome, on the SAME 12-setting-equal-weight distribution as the (now-dropped) ECDF figure --
    this is a descriptive readout of that distribution, not a separate computation."""
    rows = []
    for outcome in OUTCOME_ORDER:
        g = dist[dist.outcome == outcome]
        v, w = g.score.to_numpy(), g.weight.to_numpy()
        rows.append({
            "outcome": outcome,
            "median_score": weighted_quantile(v, w, 0.5),
            "q25_score": weighted_quantile(v, w, 0.25),
            "q75_score": weighted_quantile(v, w, 0.75),
            "pct_rejected_at_lambda1": float(w[v <= 0].sum() / w.sum() * 100),
            "n_raw": int(len(g)),
        })
    return pd.DataFrame(rows)


def rank_auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a_i > b_i) for a uniformly random pair (a_i, b_i) -- the Mann-Whitney U statistic
    normalized to [0, 1]. 0.5 = no separation, 1.0 = perfect separation (every repair scores
    above every regression in this setting)."""
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def main() -> None:
    cell = pd.read_csv(ROOT / "outputs/gate_lambda_mechanism/per_sample.csv")
    out = ROOT / "outputs" / "rq4_gate_ordering"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- A) score distribution
    dist = weighted_distribution(cell)
    dist.to_csv(out / "score_distribution_weighted.csv", index=False)
    score_table = score_summary_table(dist)
    score_table.to_csv(out / "score_summary_table.csv", index=False)

    # ---------------------------------------------------------------- B) removal composition
    # Setting-balanced: compute each setting's own outcome-share of its newly-rejected batch at
    # each lambda step, THEN average those shares equally across the 12 settings -- matching
    # this project's "per-setting mean first, then mean across settings, never a raw pooled sum"
    # convention (gate_lambda_sweep.py's own module comment). A prior version pooled raw counts
    # across settings before taking shares, which silently reweights by each setting's sample
    # count (caught by user review, 2026-09-07) -- tt100k's much larger populations dominated the
    # pooled composition and overstated how much the late steps shift toward the held side.
    removal_setting = pd.read_csv(ROOT / "outputs/gate_lambda_mechanism/nested_removal_per_setting.csv").copy()
    removal_setting["removed_total"] = (removal_setting.removed_successful_repair
                                         + removal_setting.removed_ineffective_change
                                         + removal_setting.removed_regression)
    OUTCOME_COLS = ["removed_successful_repair", "removed_ineffective_change", "removed_regression"]
    for col in OUTCOME_COLS:
        removal_setting[col.replace("removed_", "share_")] = np.where(
            removal_setting.removed_total > 0, removal_setting[col] / removal_setting.removed_total * 100,
            np.nan)
    removal_setting.to_csv(out / "removal_composition_per_setting.csv", index=False)

    SHARE_COLS = [c.replace("removed_", "share_") for c in OUTCOME_COLS]
    TRANSITION_ORDER = ["No gate→0.5", "0.5→1.0", "1.0→2.0", "2.0→4.0"]
    rows = []
    for t in TRANSITION_ORDER:
        sub = removal_setting[removal_setting.transition == t]
        used = sub[sub.removed_total > 0]  # exclude vacuous (0/0) settings from this transition's mean
        rows.append({
            "transition": t, "n_settings_used": len(used), "n_settings_excluded": len(sub) - len(used),
            **{c: float(used[c].mean()) for c in SHARE_COLS},
        })
    removal = pd.DataFrame(rows).set_index("transition").loc[TRANSITION_ORDER].reset_index()
    removal.to_csv(out / "removal_composition.csv", index=False)

    # ---------------------------------------------------------------- C) per-setting separation
    sep_rows = []
    for setting, sub in cell.groupby("setting"):
        rep = sub[sub.outcome == "successful repair"].score.to_numpy()
        reg = sub[sub.outcome == "regression"].score.to_numpy()
        ineff = sub[sub.outcome == "ineffective change"].score.to_numpy()
        if len(rep) == 0 or len(reg) == 0:
            continue
        sep_rows.append({
            "setting": setting,
            "auc_repair_vs_regression": rank_auc(rep, reg),
            "auc_repair_vs_ineffective": rank_auc(rep, ineff) if len(ineff) else float("nan"),
            "n_repair": len(rep), "n_ineffective": len(ineff), "n_regression": len(reg),
        })
    sep = pd.DataFrame(sep_rows).sort_values("auc_repair_vs_regression", ascending=False)
    sep.to_csv(out / "per_setting_separation.csv", index=False)

    RM.write_section(
        "RQ4GateOrdering",
        "RQ4 mechanism: score distribution by outcome, removal composition across lambda, "
        "per-setting separation -- reusing the already-fitted gate, no new labels",
        f"""
Reuses `outputs/gate_lambda_mechanism/per_sample.csv` (scripts/analysis_gate_lambda_mechanism.py)
-- the gate's own score log P(gain=+1) - log P(gain=-1) is independent of any lambda; lambda only
selects a cutoff on it. This checks whether the score orders true outcomes correctly (a property
of the fitted gate, not of the decision rule), and whether tightening lambda's rejections track
that ordering, rather than reporting the lambda-vs-no-gate percentage comparison as if it were
itself the mechanism.

**A) Score distribution weights** sum to 1 per outcome, equal mass per setting per seed-cell per
sample (identical construction to `build_results_draft.py`'s RQ3.3 confidence ECDF weighting).
`score_summary_table` reads median/IQR and the fraction rejected at lambda=1 (score <= 0) off
this SAME weighted distribution -- descriptive statistics, not a separate computation or a new
metric.

**B) Removal composition** is a per-setting share averaged equally across settings, NOT a pooled
count (`n_settings_used`/`n_settings_excluded` record how many of the 12 settings had at least
one newly-rejected row at that step; the rest are vacuous 0/0 and excluded from that step's mean,
not counted as 0%).

**C) Per-setting separation** (rank-based AUC, P(score_repair > score_regression) for a random
pair from that setting; 0.5 = no separation, 1.0 = perfect):
""",
        [("score_summary_table", score_table), ("removal_composition", removal),
         ("removal_composition_per_setting", removal_setting), ("per_setting_separation", sep)],
    )
    print(f"[written] {out.relative_to(ROOT)}/*.csv")
    print("\nA) Score summary by outcome (12-setting equal-weight distribution):")
    print(score_table.round(3).to_string(index=False))
    print("\nB) Removal composition (share %, 12-setting equal-weight mean):")
    print(removal[["transition", "share_successful_repair", "share_ineffective_change",
                    "share_regression"]].round(1).to_string(index=False))
    print("\nC) Per-setting separation (AUC repair vs regression):")
    print(sep[["setting", "auc_repair_vs_regression", "n_repair", "n_regression"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
