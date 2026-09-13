#!/usr/bin/env python3
"""P3 -- what post-proposal evidence changes about the gate's DECISIONS.

AUROC says how well a score ranks; it does not say which mistakes the gate stops making. This
splits every input the patch would act on into the four outcomes that actually matter, and
reports each arm's apply rate inside each one.

    A  beneficial   base wrong -> patched right      committing gains a correct answer
    B  ineffective  base wrong -> patched wrong      committing costs nothing and gains nothing
                    (split into `same` and `different` wrong answer)
    C  safe         base right -> patched right      committing is harmless
    D  harmful      base right -> patched wrong      committing loses a correct answer

The two numbers the introduction's second claim rests on:

    BR = P(apply | beneficial)     beneficial retention
    HR = 1 - P(apply | harmful)    harmful rejection

The population
--------------
The NATURAL STREAM: `clean_test` u `bug_eval`, the two splits no generator, gate or threshold
was ever fitted on, with the importance weights that restore the real test-set composition

    w_clean = |clean_eval| / |clean_test|      (clean_calib was a model-selection key)
    w_fail  = (|bug| + |bug_eval|) / |bug_eval|  (bug was spent on patch fitting)

so weighted counts are what a deployment would see. Unweighted counts are printed too, because
the weights change prevalence but not any conditional rate.

This is NOT optional bookkeeping. On the raw dumps the harmful cases live entirely in
`clean_eval` and the beneficial ones entirely in the repair splits, so an arm compared across
the pooled set is being scored on population identity. `split` is therefore carried in every row.

Three arms, identical learner, identical training rows (protocol C: bug_train + bug_val +
clean_calib + clean_train of the same setting and seed), differing only in the feature block:

    pre       what is knowable BEFORE the patch acts
    post      what the patch DID
    pre+post  the union -- the shipped gate

    .venv/bin/python scripts/analysis_p3_postinfo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import names as N  # noqa: E402
import raw_md as RM  # noqa: E402

ARMS = {"pre": "outputs/gate_persample_pre",
        "post": "outputs/gate_persample_post",
        "pre+post": "outputs/gate_persample"}


def weights(seed: int, ds: str, bb: str) -> dict[str, float]:
    sp = ROOT / f"artifacts/bug_sets/v8_splits_seed{seed}/{ds}_{bb}"
    n = lambda f: len(json.loads((sp / f).read_text())["indices"])  # noqa: E731
    try:
        return {"clean": n(f"{ds}_clean_eval_indices.json")
                         / max(n(f"{ds}_clean_test_indices.json"), 1),
                "held": (n(f"{ds}_bug_indices.json") + n(f"{ds}_bug_eval_indices.json"))
                        / max(n(f"{ds}_bug_eval_indices.json"), 1)}
    except FileNotFoundError:
        return {"clean": 1.0, "held": 1.0}


def load_arm(tag: str, path: str) -> pd.DataFrame | None:
    d = ROOT / path
    if not d.is_dir():
        return None
    g = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*.csv"))], ignore_index=True)
    g = g.rename(columns={"side": "split"})
    rs = sorted(float(c.split("_r")[1]) for c in g.columns if c.startswith("apply_r"))
    keep = ["setting", "seed", "split", "dataset_index", "gate_score"]
    out = []
    for r in rs:
        out.append(g[keep + [f"apply_r{r:.2f}"]]
                   .rename(columns={f"apply_r{r:.2f}": "applied"})
                   .assign(arm=tag, r=r))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    df = pd.read_csv(ROOT / "outputs/sample_frame.csv.gz", low_memory=False)
    ung = df[(df.method == N.display("DPNoGate")) & df.split.isin(("clean", "held"))].copy()
    if ung.empty:
        raise SystemExit("ungated DynaPatch rows missing from the frame")

    # the four outcomes, defined on what the patch WOULD do if committed
    same = ung.base_pred == ung.patched_pred
    ung["outcome"] = np.select(
        [(~ung.base_correct) & ung.patched_correct,
         (~ung.base_correct) & (~ung.patched_correct) & same,
         (~ung.base_correct) & (~ung.patched_correct) & (~same),
         ung.base_correct & ung.patched_correct,
         ung.base_correct & (~ung.patched_correct)],
        ["A_beneficial", "B_ineffective_same", "B_ineffective_diff", "C_safe", "D_harmful"],
        default="?")
    # one lookup per (seed, setting), not per row
    wmap = {(sd, stg): weights(sd, *stg.split("/"))
            for sd in ung.seed.unique() for stg in ung.setting.unique()}
    ung["w"] = [wmap[(sd, stg)][sp]
                for sd, stg, sp in zip(ung.seed.to_numpy(), ung.setting.to_numpy(),
                                       ung.split.to_numpy())]

    arms = [a for a in (load_arm(t, p) for t, p in ARMS.items()) if a is not None]
    if not arms:
        raise SystemExit("no per-sample gate dumps; run gate_protocol_b.py --per-sample")
    g = pd.concat(arms, ignore_index=True)

    m = ung[["setting", "seed", "split", "dataset_index", "outcome", "w"]].merge(
        g, on=["setting", "seed", "split", "dataset_index"], how="inner")

    # per (arm, r, setting, seed, outcome): the apply rate, weighted and not
    rows = []
    for (arm, r, stg, sd, oc), d in m.groupby(["arm", "r", "setting", "seed", "outcome"]):
        rows.append({"arm": arm, "r": r, "setting": stg, "seed": sd, "outcome": oc,
                     "split": d.split.iloc[0] if d.split.nunique() == 1 else "mixed",
                     "n": len(d), "n_applied": int(d.applied.sum()),
                     "apply_rate": float(d.applied.mean()),
                     "w_n": float(d.w.sum()),
                     "w_apply_rate": float((d.applied * d.w).sum() / d.w.sum()),
                     "score_mean": float(d.gate_score.mean()),
                     "score_median": float(d.gate_score.median())})
    per = pd.DataFrame(rows)

    # BR / HR side by side per (arm, r, setting, seed)
    p = per.pivot_table(index=["arm", "r", "setting", "seed"], columns="outcome",
                        values=["apply_rate", "n"])
    p.columns = [f"{a}__{b}" for a, b in p.columns]
    p = p.reset_index()
    if "apply_rate__A_beneficial" in p:
        p["BR"] = p["apply_rate__A_beneficial"]
    if "apply_rate__D_harmful" in p:
        p["HR"] = 1 - p["apply_rate__D_harmful"]

    RM.write_section("P3", "P3 — post-proposal evidence and the gate's decisions (raw)", """
Population: the **natural stream** `clean_test` u `bug_eval` — the two splits nothing was ever
fitted on. `w` restores the real test-set composition
(`w_clean = |clean_eval|/|clean_test|`, `w_fail = (|bug|+|bug_eval|)/|bug_eval|`); `apply_rate`
is unweighted and `w_apply_rate` is weighted. Weights change prevalence, not conditional rates.

The four outcomes are defined by what committing the patch WOULD do:

| outcome | base | patched | committing |
|---|---|---|---|
| `A_beneficial` | wrong | right | gains a correct answer |
| `B_ineffective_same` | wrong | wrong, same label | no effect |
| `B_ineffective_diff` | wrong | wrong, different label | no effect |
| `C_safe` | right | right | harmless |
| `D_harmful` | right | wrong | loses a correct answer |

Arms differ ONLY in the feature block; same learner, same protocol-C training rows
(`bug_train + bug_val + clean_calib + clean_train`), same threshold rule.

`BR = apply_rate(A_beneficial)`, `HR = 1 - apply_rate(D_harmful)`.

**`split` is carried on every row.** On the raw dumps the harmful cases sit almost entirely in
`clean_test` and the beneficial ones in `bug_eval`; an arm compared on the pooled set without
this column is being scored on population identity, not on gating.
""", [("Block A — apply rate per (arm, r, setting, seed, outcome)",
       per.sort_values(["arm", "r", "setting", "seed", "outcome"])),
      ("Block B — the same cells with BR and HR side by side",
       p.sort_values(["arm", "r", "setting", "seed"]))])

    out = ROOT / "outputs" / "p3_postinfo"
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_outcome.csv", index=False)
    p.to_csv(out / "br_hr.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/  ({len(per)} rows)")


if __name__ == "__main__":
    main()
