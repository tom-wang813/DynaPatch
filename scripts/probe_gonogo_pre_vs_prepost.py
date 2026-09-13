#!/usr/bin/env python3
"""GO / NO-GO: does post-repair evidence add deployment value beyond a STRONG pre-repair gate?

Why the previous experiments do not answer this
-----------------------------------------------
RQ2/RQ3 asked "given that the base is already known to be wrong, can the response tell whether
the flip landed right?" -- the answer is yes, but the population is oracle-conditioned. Section 10
then showed the shipped post-only gate loses to a bare p^B_max heuristic on the real deployment
mixture. Neither settles the question, because a single confidence scalar is a WEAK stand-in for
"pre-repair information": beating it proves nothing, and losing to it proves nothing either. That
is the same mistake the magnitude null made in RQ4.

So this script builds a genuinely strong LEARNED pre-only gate and asks the nested question:

    G_pre        vs        G_pre + post-repair evidence

on the full deployment population, same calibration, same folds, same repair methods.

The decision rule the gate is actually for
------------------------------------------
On a flipped row the choice is commit-or-rollback, and the outcome is three-valued:

    base_correct   -> committing COSTS one correct prediction   (gain = -1)
    patched_correct-> committing GAINS one                      (gain = +1)
    neither        -> committing changes nothing                (gain =  0)

(base_correct and patched_correct cannot both hold on a flip -- section 8.) So the gate should
rank by expected gain, and we fit a 3-class model and score

    s(x) = P(gain=+1) - P(gain=-1)

rather than the P(repair succeeds) head we shipped, which only ever modelled the first term.
This is NOT proposed as a novelty: separating the repaired-side and broken-side predictions is
already what Ishimoto et al. do with two models, and learning-to-defer already frames the gate as
a meta-classifier over "who is more likely right". It is used here because it is the correct
decision-theoretic target for the comparison, not because it is new.

Verdict is a DEPLOYMENT quantity, not AUROC
-------------------------------------------
Gate discrimination and deployment gain are different things, so every conclusion below is read
off the exact suppression simulation (dReg, dRR_held, cost) at matched suppression rates.

Feature availability, stated up front
-------------------------------------
Full logits exist only for OUR repair (outputs/effect_dump_v8_s*). The baseline trees ship two
scalars per row (base_conf/patched_conf). So:
  * `--repair ours`      full four-layer hierarchy: mag / pre-strong / post-only / pre+post
  * `--repair external`  reduced hierarchy on {p^B_max} vs {p^B_max, p^P_max} only
The reduced arm cannot refute a strong-pre claim; it is reported as a coverage check, not as the
head-to-head.

Usage:
  .venv/bin/python scripts/probe_gonogo_pre_vs_prepost.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import analyze_response_gate_lobo as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DSS, BBS, SEEDS = L.DSS, L.BBS, L.SEEDS
QS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80)

# The four layers. Every feature is label-free: nothing here needs the ground truth of the input
# being judged, so all of it is computable by a runtime gate.
#
# Fixed 2026-09-07: "4 pre+post" (== gate_zoo.LEARNED["L4 pre+post"] == names.SHIPPED_GATE) had
# its 3 within-split rank features (m_base_r, dm_r, dnorm_r) removed -- see gate_zoo.py's LEARNED
# comment for why they are not legitimately per-input/deployable. The pre-fix 12-feature list is
# kept under "4-legacy pre+post" so analysis_gate_norank_ablation.py/analysis_gate_norank_full.py
# can still compute the before/after comparison explicitly. "2 pre-strong" (contains m_base_r,
# dnorm_r) and "1 magnitude" (pure dnorm_r) share the same issue and have NOT been fixed -- flagged,
# not silently inconsistent, since neither is the paper's shipped gate.
LAYERS = {
    "1 magnitude":  ["dnorm_r"],
    "2 pre-strong": ["pB_max", "pB_margin", "H_base", "m_base_r", "dnorm_r"],
    "3 post-only":  ["pP_max", "kl", "dH", "dp_c", "rho", "rho_worst", "dm_r"],
    "4-legacy pre+post": ["pB_max", "pB_margin", "H_base", "m_base_r", "dnorm_r",
                          "pP_max", "kl", "dH", "dp_c", "rho", "rho_worst", "dm_r"],
    "4 pre+post":   ["pB_max", "pB_margin", "H_base",
                     "pP_max", "kl", "dH", "dp_c", "rho", "rho_worst"],
}
NESTED = ("4 pre+post", "2 pre-strong")     # the go/no-go comparison


def fit_gain(X: np.ndarray, g: np.ndarray):
    """3-class model over gain in {-1,0,+1}; score = P(+1) - P(-1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(g)) < 2:
        return None
    # The three classes are far from balanced: gain=-1 (clean inputs the patch breaks) is the
    # rarest by a wide margin -- 7.1:1 against gain=+1 at the median when the clean side is
    # clean_calib alone, 43:1 at worst. GATE_CLASS_WEIGHT=balanced reweights inversely with
    # frequency, so "is the imbalance what makes post evidence look useless?" can be answered
    # instead of assumed. Default stays unweighted: that is what every shipped number used.
    cw = os.environ.get("GATE_CLASS_WEIGHT") or None
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, random_state=0,
                                            class_weight=cw)).fit(X, g)


def score(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    cls = list(model.classes_)
    pos = p[:, cls.index(1)] if 1 in cls else np.zeros(len(X))
    neg = p[:, cls.index(-1)] if -1 in cls else np.zeros(len(X))
    return pos - neg


def clean_test_idx(seed: int, ds: str, bb: str) -> set[int] | None:
    f = ROOT / f"artifacts/bug_sets/v8_splits_seed{seed}/{ds}_{bb}/{ds}_clean_test_indices.json"
    return set(json.loads(f.read_text())["indices"]) if f.exists() else None


def build_cells() -> dict:
    """Per (seed, ds, bb): the clean and held populations with features, flip and gain."""
    harm = L.collect("harm")          # clean_eval,            label=regressed
    help_ = L.collect("help")         # repair_holdout_unseen, label=repaired
    out = {}
    for k in sorted(set(harm) & set(help_)):
        c, h = harm[k], help_[k]
        # clean_eval == clean_calib U clean_test, and clean_calib was the early-stopping order
        # key, so ~20% of these rows took part in model selection. Reg is only honest on
        # clean_test (note/RESULTS_ALL_RQ.md section 1.2).
        keep = clean_test_idx(*k)
        if keep is not None:
            m = np.fromiter((int(i) in keep for i in c["idx"]), dtype=bool, count=len(c["idx"]))
            c = {"feats": {kk: vv[m] for kk, vv in c["feats"].items()},
                 "y": c["y"][m], "idx": c["idx"][m], "dnorm": c["dnorm"][m]}
        # clean side: base_correct  <=>  lps == pB_max  (exact, see probe_commit_arm_confound)
        cbc = np.isclose(c["feats"]["lps"], c["feats"]["pB_max"], rtol=0, atol=1e-12)
        cflip = c["feats"]["flip"] > 0.5
        # regressed == flip AND base_correct, so gain = -1 exactly on the regressions
        cgain = np.where(c["y"], -1, 0).astype(int)
        hflip = h["feats"]["flip"] > 0.5
        hgain = np.where(h["y"], 1, 0).astype(int)   # held-out failures are base-wrong
        out[k] = {"clean": {"f": c["feats"], "flip": cflip, "gain": cgain,
                            "y": c["y"], "bc": cbc, "n": len(c["y"])},
                  "held": {"f": h["feats"], "flip": hflip, "gain": hgain,
                           "y": h["y"], "n": len(h["y"])}}
    return out


def mat(f: dict, names: list[str], m: np.ndarray | None = None) -> np.ndarray:
    X = np.column_stack([f[n] for n in names])
    return X if m is None else X[m]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/gonogo_pre_vs_prepost.json")
    a = ap.parse_args()

    cells = build_cells()
    print("GO/NO-GO: does post-repair evidence add value beyond a STRONG learned pre-repair gate?")
    print("population: clean_test + held-out failures; only FLIPPED rows are actionable")
    print("target: expected gain, score = P(gain=+1) - P(gain=-1); protocol: LOBO")
    print("verdict is read off the exact suppression simulation, not AUROC\n")

    res: dict = {}
    for lname, names in LAYERS.items():
        rows = []
        for held_bb in BBS:
            tr = [k for k in cells if k[2] != held_bb]
            te = [k for k in cells if k[2] == held_bb]
            X, g = [], []
            for k in tr:
                for side in ("clean", "held"):
                    d = cells[k][side]
                    X.append(mat(d["f"], names, d["flip"])); g.append(d["gain"][d["flip"]])
            X, g = np.vstack(X), np.concatenate(g)
            mdl = fit_gain(X, g)
            if mdl is None:
                continue
            for k in te:
                c, h = cells[k]["clean"], cells[k]["held"]
                sc = score(mdl, mat(c["f"], names))
                sh = score(mdl, mat(h["f"], names))
                pool = np.concatenate([sc[c["flip"]], sh[h["flip"]]])
                if not len(pool):
                    continue
                reg0 = c["y"].sum() / c["n"]
                rr0 = h["y"].sum() / h["n"]
                for q in QS:
                    t = np.quantile(pool, q)
                    kc, kh = ~c["flip"] | (sc > t), ~h["flip"] | (sh > t)
                    reg = c["y"][kc].sum() / c["n"]
                    rr = h["y"][kh].sum() / h["n"]
                    rows.append((k, q, reg0, reg - reg0, rr0, rr - rr0))
        res[lname] = rows

    print(f"{'layer':<15}{'q':>6}{'Reg base':>10}{'dReg':>10}{'dRR_held':>11}"
          f"{'cost':>8}{'Reg cut':>9}{'cells':>7}")
    print("-" * 76)
    agg = {}
    for lname in LAYERS:
        for q in QS:
            r = [x for x in res[lname] if x[1] == q]
            if not r:
                continue
            m = lambda i: float(np.mean([x[i] for x in r]))
            cost = abs(m(5) / m(3)) if m(3) else float("inf")
            agg[(lname, q)] = (m(3), m(5), cost)
            print(f"{lname:<15}{q:>6.2f}{m(2):>10.4f}{m(3):>+10.4f}{m(5):>+11.4f}"
                  f"{cost:>8.1f}{(-m(3) / m(2) if m(2) else 0):>8.1%}{len(r):>7}")

    print(f"\nGO/NO-GO  --  {NESTED[0]}  minus  {NESTED[1]}  (same folds, same rows, same q)")
    print(f"{'q':>6}{'dReg pre':>11}{'dReg both':>11}{'dRR pre':>10}{'dRR both':>11}"
          f"{'cost pre':>10}{'cost both':>11}   verdict")
    print("-" * 78)
    verdict = {}
    for q in QS:
        b, p = agg.get((NESTED[0], q)), agg.get((NESTED[1], q))
        if not b or not p:
            continue
        # cost = |dRR|/|dReg| is undefined when a gate gives up no repair at all (dRR == 0),
        # which is the best possible outcome, not an error. Report the pair and a verdict.
        f2 = lambda v: ("  inf" if not np.isfinite(v) else f"{v:>5.1f}")
        tag = ("pre+post better" if (b[0] < p[0] - 1e-6 and b[1] >= p[1] - 1e-6)
               else "pre-only better" if (p[0] < b[0] - 1e-6 and p[1] >= b[1] - 1e-6)
               else "tie")
        print(f"{q:>6.2f}{p[0]:>+11.4f}{b[0]:>+11.4f}{p[1]:>+10.4f}{b[1]:>+11.4f}"
              f"{f2(p[2]):>10}{f2(b[2]):>11}   {tag}")
        verdict[str(q)] = {"dReg_pre": p[0], "dReg_both": b[0],
                           "dRR_pre": p[1], "dRR_both": b[1], "verdict": tag}

    # per-setting sign test on the cost, so one big cell cannot carry the verdict
    print(f"\nper-setting: is {NESTED[0]} cheaper than {NESTED[1]}?  (cost, lower is better)")
    print(f"{'q':>6}{'settings where pre+post dominates':>36}{'exact ties':>12}")
    for q in QS:
        bw = {x[0]: x for x in res[NESTED[0]] if x[1] == q}
        pw = {x[0]: x for x in res[NESTED[1]] if x[1] == q}
        win = tot = und = 0
        for k in sorted(set(bw) & set(pw)):
            # dominance: strictly more Reg removed and no more RR given up
            db, dp = bw[k][3], pw[k][3]
            rb, rp = bw[k][5], pw[k][5]
            if db == dp and rb == rp:
                und += 1
                continue
            tot += 1
            win += (db < dp - 1e-9 and rb >= rp - 1e-9) or (db <= dp + 1e-9 and rb > rp + 1e-9)
        print(f"{q:>6.2f}{f'{win}/{tot}':>36}{und:>12}")
        verdict.setdefault(str(q), {}).update({"per_cell_win": win, "per_cell_n": tot})

    (ROOT / a.json).write_text(json.dumps(
        {"aggregate": {f"{k[0]}|{k[1]}": v for k, v in agg.items()}, "gonogo": verdict}, indent=1, default=float))
    print(f"\nwrote {ROOT / a.json}")


if __name__ == "__main__":
    main()
