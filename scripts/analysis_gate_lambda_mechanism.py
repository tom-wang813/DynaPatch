#!/usr/bin/env python3
"""Why does the regression cost lambda remove regression faster than repair?

RQ4.7 (`scripts/gate_lambda_sweep.py`, `note/RQ4_DATA.md`) shows RR_held and Reg both fall as
lambda increases -- but that is close to guaranteed by construction: raising lambda in
    commit iff  P(gain=+1) > lambda * P(gain=-1)
can only shrink the accepted set (RQ4.6 already proves this nested-set monotonicity on the
quantile-q axis; the same holds here by the same argument). Reporting "RR/Reg both go down as
lambda goes up" is therefore a restatement of the decision rule, not a finding. What lambda's
*effectiveness* actually rests on is whether the gate's score ranks harmful patch applications
below beneficial ones in the first place -- if it does not, tightening lambda would remove
repairs and regressions in equal proportion, not preferentially.

This script tests that directly, with the SAME gate fit `gate_lambda_sweep.py::dp_curve()` uses
(protocol-C, "both"/"calib", pre+post features, theta=0 threshold family), on two questions:

  A) Score distribution by true outcome. Every flipped (patch actually proposed a change) row
     has an eventual ground-truth outcome: "successful repair"/"ineffective change" on the held
     (failure) side, "regression"/"no-effect" on the clean side -- same four-way split
     `scripts/analysis_gate_response_baselines.py` already uses for confidence/entropy (mirrors
     `note/RQ3_DATA.md` RQ3.3's three named outcomes, plus the un-named "no-effect on clean"
     complement needed to fully partition the clean side). Does the fitted score
     log P(gain=+1) - log P(gain=-1) separate these groups in the expected order (regression <
     ineffective change < successful repair)?

  B) Nested removal by outcome. lambda's accepted sets are nested (RQ4.6): raising lambda can only
     move a row from "kept" to "rejected", never the reverse. For each step in
     No-gate -> 0.5 -> 1.0 -> 2.0 -> 4.0, which rows newly flip from kept to rejected, and what
     were their true outcomes? If tightening lambda mechanically explains RQ4.7's Reg-falls-
     faster-than-RR pattern, the newly-rejected clean rows at each step should be enriched for
     actual regressions relative to the clean population's regression rate BEFORE that step (and
     symmetrically the held side should be enriched for ineffective changes over successful
     repairs).

No retraining, no new labels: reuses the exact cached features/labels/gate-fit machinery
`gate_lambda_sweep.py` already verified reproduces `note/RQ4_DATA.md`'s numbers. Only the
per-sample scores and outcomes -- discarded by `dp_curve()`, which keeps only the aggregated
RR/Reg/CReg rate -- are newly saved here.

    .venv/bin/python scripts/analysis_gate_lambda_mechanism.py
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
import probe_gonogo_pre_vs_prepost as G  # noqa: E402
from gate_protocol_b import train_cells  # noqa: E402
from probe_gonogo_pre_vs_prepost import fit_gain  # noqa: E402
from gate_lambda_sweep import (  # noqa: E402
    LAMBDAS, FEATS, MIN_POS, SHIPPED_TAG, LAST_AFFINE_BACKBONES, _load_split, posneg,
)

def _label(a: float | None, b: float) -> str:
    return f"{'No gate' if a is None else a}→{b}"


TRANSITIONS = [(_label(a, b), (a, b)) for a, b in zip((None,) + LAMBDAS[:-1], LAMBDAS)]
EPS = 1e-6


def _outcome_masks(clean: dict, held: dict) -> dict[str, tuple[dict, np.ndarray]]:
    """Same four-way partition as `analysis_gate_response_baselines.py` (which itself mirrors
    RQ3.3), plus the "no-effect" clean complement RQ3.3 does not name (it only reports the three
    outcomes worth a confidence story; here the clean side must be fully partitioned so removal
    counts add up)."""
    return {
        "successful repair": (held, held["flip"] & held["y"]),
        "ineffective change": (held, held["flip"] & ~held["y"]),
        "regression":         (clean, clean["flip"] & clean["y"]),
        "no-effect":          (clean, clean["flip"] & ~clean["y"]),
    }


def per_sample_rows() -> pd.DataFrame:
    report = _load_split(lambda: Z.attach_criticality(Z.attach_idx(Z.add_derived(G.build_cells()))))
    train = _load_split(lambda: train_cells("both", "calib"))
    rows = []
    for k in sorted(set(train) & set(report)):
        seed, ds, bb = k
        c, h = train[k]["clean"], train[k]["held"]
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
        for outcome, (data, mask) in _outcome_masks(rc, rh).items():
            pos, neg = (pos_c, neg_c) if data is rc else (pos_h, neg_h)
            idx = np.flatnonzero(mask)
            for i in idx:
                rows.append({
                    "setting": f"{ds}/{bb}", "seed": seed, "outcome": outcome,
                    "population": "clean" if data is rc else "held",
                    "pos": float(pos[i]), "neg": float(neg[i]),
                    "score": float(np.log(pos[i] + EPS) - np.log(neg[i] + EPS)),
                    "crit": bool(rc["crit"][i]) if data is rc else False,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- A) score distribution

def score_by_outcome(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_setting = cell.groupby(["setting", "outcome"], as_index=False).agg(
        n=("score", "size"), mean_score=("score", "mean"), median_score=("score", "median"))
    overall = per_setting.groupby("outcome", as_index=False).agg(
        n=("n", "sum"), mean_score=("mean_score", "mean"), median_score=("median_score", "mean"),
        settings=("setting", "nunique"))
    order = {o: i for i, o in enumerate(
        ["regression", "no-effect", "ineffective change", "successful repair"])}
    overall["order"] = overall.outcome.map(order)
    overall = overall.sort_values("order").drop(columns="order")
    return per_setting, overall


# ---------------------------------------------------------------- B) nested removal

def keep_mask(cell: pd.DataFrame, lam: float | None) -> np.ndarray:
    if lam is None:
        return np.ones(len(cell), dtype=bool)
    return (cell["pos"] > lam * cell["neg"]).to_numpy()


def nested_removal(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for setting, sub in cell.groupby("setting"):
        for label, (lo, hi) in TRANSITIONS:
            keep_lo = keep_mask(sub, lo)
            keep_hi = keep_mask(sub, hi)
            newly_rejected = keep_lo & ~keep_hi
            row = {"setting": setting, "transition": label,
                   "n_pre_held": int((sub.population.eq("held") & keep_lo).sum()),
                   "n_pre_clean": int((sub.population.eq("clean") & keep_lo).sum())}
            for outcome, col in (("successful repair", "removed_successful_repair"),
                                 ("ineffective change", "removed_ineffective_change"),
                                 ("regression", "removed_regression"),
                                 ("no-effect", "removed_no_effect")):
                m = newly_rejected & sub.outcome.eq(outcome).to_numpy()
                row[col] = int(m.sum())
            rows.append(row)
    per_setting = pd.DataFrame(rows)
    assert (per_setting.removed_no_effect == 0).all(), (
        "a clean (base-correct) row whose top-1 prediction changed (flip=True) is a regression "
        "BY CONSTRUCTION -- there is no 'flip but stayed correct' case on the clean side. A "
        "non-zero count here means the flip/y semantics assumed above are wrong, not that this "
        "outcome is real.")
    num_cols = [c for c in per_setting.columns if c.startswith(("n_pre", "removed_"))]
    order = {lab: i for i, (lab, _) in enumerate(TRANSITIONS)}
    pooled = per_setting.groupby("transition", as_index=False)[num_cols].sum()
    pooled["order"] = pooled.transition.map(order)
    pooled = pooled.sort_values("order").drop(columns="order").drop(columns="removed_no_effect")
    # The clean side has only one possible outcome for a flip (regression, by construction -- see
    # the assert above), so there is no enrichment question to ask there: removed_regression IS
    # the clean removal count, full stop (it is the same event RQ4.7's falling Reg already counts,
    # not new evidence). The only non-trivial mix-shift question is on the HELD side, where a
    # flip can genuinely go either way (successful repair or ineffective change).
    held_removed_total = pooled.removed_successful_repair + pooled.removed_ineffective_change
    pooled["frac_removed_is_ineffective"] = pooled.removed_ineffective_change / held_removed_total.replace(0, np.nan)
    return per_setting.sort_values(["setting", "transition"]), pooled


def pre_transition_rates(cell: pd.DataFrame) -> pd.DataFrame:
    """Ineffective-change share of the HELD population still ELIGIBLE to be rejected immediately
    before each transition -- the baseline `frac_removed_is_ineffective` above must beat to count
    as enrichment rather than just reflecting the ambient class mix. (No clean-side analogue: a
    clean flip is a regression by construction, see the assert in `nested_removal()`.)"""
    rows = []
    for label, (lo, _hi) in TRANSITIONS:
        keep_lo = keep_mask(cell, lo)
        held = cell[keep_lo & cell.population.eq("held")]
        rows.append({
            "transition": label,
            "pre_held_ineffective_rate": float((held.outcome == "ineffective change").mean()) if len(held) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    cell = per_sample_rows()
    out = ROOT / "outputs" / "gate_lambda_mechanism"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_sample.csv", index=False)

    score_setting, score_overall = score_by_outcome(cell)
    score_setting.to_csv(out / "score_by_outcome_per_setting.csv", index=False)
    score_overall.to_csv(out / "score_by_outcome_overall.csv", index=False)

    removal_setting, removal_pooled = nested_removal(cell)
    baseline = pre_transition_rates(cell)
    removal_pooled = removal_pooled.merge(baseline, on="transition", how="left")
    removal_setting.to_csv(out / "nested_removal_per_setting.csv", index=False)
    removal_pooled.to_csv(out / "nested_removal_pooled.csv", index=False)

    RM.write_section(
        "GateLambdaMechanism",
        "Why lambda removes regression faster than repair: score-by-outcome and nested removal",
        f"""
Same gate fit as `scripts/gate_lambda_sweep.py::dp_curve()` (protocol-C, both/calib, pre+post
features, {len(LAMBDAS)} lambda values {LAMBDAS}); only the per-sample score and true outcome are
newly saved (dp_curve() only kept the aggregated RR/Reg/CReg rate). Outcome partition matches
`scripts/analysis_gate_response_baselines.py` (itself matching RQ3.3): successful repair /
ineffective change on the held (failure) side, regression on the clean side. There is no
"no-effect on clean" case: a clean (base-correct) row whose top-1 prediction changes IS a
regression by construction (asserted in `nested_removal()`), so the clean side of the removal
table is a magnitude count only (ties to RQ4.7's Reg column, not new evidence); the only
non-trivial mix-shift question is on the held side, where a flip can genuinely go either way.

**A) Score by outcome** (setting-balanced: per-setting mean, then mean across settings):
score = log P(gain=+1) - log P(gain=-1), the same log-odds `dp_curve()` thresholds against
lambda. Expected order if lambda's effectiveness is real rather than an artefact of the
construction: regression < ineffective change < successful repair.

**B) Nested removal by outcome** (pooled across 12 settings; accepted sets are nested per
setting/seed, RQ4.6, so a transition's "newly rejected" rows strictly subtract from the prior
step): `frac_removed_is_ineffective` (share of newly-rejected HELD rows that were ineffective
changes) compared against `pre_held_ineffective_rate` (ineffective share of the held population
still eligible to be rejected just before that step) is the enrichment test -- if lambda's
ordering is doing real work, the former should sit well above the latter at every step.
""",
        [("score_by_outcome_overall", score_overall),
         ("score_by_outcome_per_setting", score_setting),
         ("nested_removal_pooled", removal_pooled),
         ("nested_removal_per_setting", removal_setting)],
    )
    print(f"[written] {out.relative_to(ROOT)}/*.csv")
    print("\nA) Score by outcome (setting-balanced overall):")
    print(score_overall.round(4).to_string(index=False))
    print("\nB) Nested removal by outcome (pooled across 12 settings):")
    print(removal_pooled.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
