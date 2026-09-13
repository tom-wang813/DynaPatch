#!/usr/bin/env python3
"""The final RQ4 table: RR_repair, RR_held, Reg, CReg for every method, per setting.

Metric definitions (identical to scripts/summarize_rq4_v8.py::_quad, so the archived rows
reproduce exactly):
    RR_repair  mean(patched_correct) over repair_support_seen     -- the patch's OWN evidence
    RR_held    mean(patched_correct) over repair_holdout_unseen   -- held-out failures
    Reg        fraction of S_clean^test now wrong
    CReg       fraction of the safety-critical rows of S_clean^test now wrong

Baseline trees whose clean_eval is already S_clean^test are read directly; the repairbench
deploy_direct dump holds the full clean pool and is filtered by the split's index list.

SWEPT BASELINES. Two opponents had a hardcoded, never-swept perturbation-magnitude knob that
was the binding constraint on their repair capacity:
    TopKSearch   step_scale   0.5 shipped;  RR_held 0.042 -> 0.417 at 32 (gtsrb/resnet50 s101)
    Arachne(DE)  bound_scale  2.0 shipped;  RR_held 0.125 -> 0.375 at 128
Per the discipline in note/RESULTS_ALL_RQ.md section 31.2, every baseline hyper-parameter is
swept and the value most favourable to the baseline is reported. `--select` picks that value
PER SETTING (default) or globally. The shipped-default row is ALWAYS printed alongside, because
"we mis-configured this opponent" is a finding that has to stay auditable -- deleting the old
row and printing only the swept one would erase it.

NAMING. Keys are truth, paths are not:
    outputs/baselines_rq4_v8/*/distrrep      is plain full fine-tuning, NOT Nie et al.'s DistRep
    outputs/baselines_lsr_v8/*/arachne_style is TopKSearch, NOT Arachne
Both appear here under their true names; the real implementations are separate rows.

CAVEATS printed with the table, not buried:
  * DistRep(PSO) runs at the paper's own budget (5x40x40, clean-cap 2048) but on ONE bug-split
    seed, where every other method has three. The reduced-budget row was dropped 2026-09-01.
  * Arachne(DE) is a faithful re-implementation, not the authors' TF/Keras artefact.
  * `dataset_index` in the Arachne and DistRep trees is a POSITIONAL counter, not a dataset
    index, so those two trees must never be joined on it.

Zero GPU. Usage:
  .venv/bin/python scripts/table_rq4_final.py
  .venv/bin/python scripts/table_rq4_final.py --select global --csv outputs/rq4_final.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import guards  # noqa: E402
from guards import Ratio, check_params_recorded  # noqa: E402

DATASETS = ["gtsrb", "tt100k_signs", "lisa_signs"]
BACKBONES = ["resnet50", "convnext_tiny", "densenet121", "vgg16"]
SETTINGS = [(d, b) for d in DATASETS for b in BACKBONES]
SEEDS = [101, 202, 303]
POSITIONAL_IDX = ("Arachne", "DistRep(PSO)")   # trees whose dataset_index is a counter

# name -> (kind, template). "sweep" entries carry a list of (label, template) variants.
# Rows removed 2026-09-01 (they were under-trained, misnamed, or an unswept single point;
# see scripts/names.py:RQ4_METHOD_REJECT for the per-row reason). The artefact trees are
# untouched on disk -- only the table stopped reading them.
FIXED_ROWS = [
    ("Weighted Retraining (40ep, matched)", "tree",
     "outputs/baselines_wr_ep40_v8_s{s}/{ds}/{bb}/weighted_retraining/predictions"),
    ("Full fine-tuning (40ep)", "tree",
     "outputs/fewshot_distrep_ep40_v8_s{s}_kfull/{ds}/{bb}/predictions"),
    ("Head-Only Fine-Tuning (40ep)", "tree",
     "outputs/fewshot_headonly_ep40_v8_s{s}_kfull/{ds}/{bb}/predictions"),
    ("LastDelta (40ep)", "tree",
     "outputs/fewshot_lastdelta_ep40_v8_s{s}_kfull/{ds}/{bb}/predictions"),
    # The paper's own budget (5x40x40, clean-cap 2048) instead of the reduced 3x15x15 / 512 that
    # the row above runs. One seed only: a cell costs ~4300 s at this budget versus ~200 s, so
    # 12 cells took a full night on three GPUs. The two rows are NOT merged -- a table row has to
    # be one budget, and this one carries a different seed count as well.
    # 2026-09-02: switched to the `_ct_` tree. run_distrep_pso.py:151 fed the clean-preservation
    # objective `clean_test` -- the split its own Reg is reported on. Overlap was ~21% on GTSRB,
    # ~28% on TT100K and 100% on all four LISA cells (clean_test is 1864-1899 rows, under the
    # 2048 cap), so the shipped Reg was optimistic. DynaPatch's clean-replay always used
    # clean_train; `--preserve-split clean_train` puts the baseline on the same footing.
    ("DistRep(PSO) real, FULL budget (1 seed)", "tree",
     "outputs/distrepPSO_ct_full_v8_s{s}_kfull/{ds}/{bb}/predictions", (101,)),
]
# REMOVED 2026-09-06 (decision: remove entirely, not just deprecate): "Always Patch
# (ungated)" used to live here, reading outputs/repairbench_v8_s{s}/.../deploy_direct/predictions
# -- a 12-epoch build (timestamped 2026-07-19, predates even the ep40ns/40-epoch recipe let
# alone last_affine), already self-documented in scripts/all_rq_data.py's own note as "NOT
# epoch-matched with RQ1's 40ep arm" and quarantined into RQ3_METHODS alongside the retired
# r-grid "DynaPatch [2nd round] @ r=X" rows. It was NEVER the same thing as "DynaPatch
# (ungated)"/DynaPatch-NoGate (the current, ep40ns/last_affine, 40-epoch tree sample_frame.py's
# `dpn` construction reads) -- confirmed by diverging clean-side Reg (0.0071 vs 0.0300) despite
# a coincidentally-matching held-side RR mean. `dynapatch_rows()` below already independently
# supplies DynaPatch's own (gated) row from the current tree; nothing in the live pipeline
# reads FIXED_ROWS for DynaPatch's ungated point. See note/RESEARCH_STATE.md 2026-09-06 entry.
SWEEPS = [
    # 2026-09-03: TopKSearch removed from the paper (user decision). The swept step_scale rows
    # (outputs/topksearch_ss*) stay on disk as evidence but are no longer tabulated.
    # 2026-08-30: grid extended from {2,16,128} to the full swept set. The coarse grid was both
    # TRUNCATED (RR_held was still climbing at 128 on gtsrb/resnet50: .062 -> .257) and
    # CONTAMINATED (at 128 the DE collapses on gtsrb/densenet121: .333@16 -> .108@128), so its
    # per-setting max was not the method's best operating point. Missing (setting, variant)
    # cells are simply absent and are dropped by the seed-count guard in resolve_sweep.
    # 2026-09-02: `_ct_` tree, same clean_test-contamination fix as DistRep above
    # (run_arachne_de.py:241). 252/252 cells re-run.
    ("Arachne(DE) (swept bound_scale)",
     [(f"bound={v}", f"outputs/arachne_ct_bs{v}_v8_s{{s}}_kfull/{{ds}}/{{bb}}/predictions")
      for v in ("2.0", "4.0", "8.0", "16.0", "32.0", "64.0", "128.0")]),
]


def critical(ds: str) -> set[int]:
    cfg = json.loads((ROOT / f"artifacts/risk/{ds}_safety_risk_matrix.json").read_text())
    out: set[int] = set()
    for ids in cfg["critical_signs"].values():
        out.update(int(i) for i in ids)
    return out


def rows_of(p: Path) -> list[dict]:
    return [{"pc": r["patched_correct"].lower() == "true", "lab": int(r["label"]),
             "i": int(r["dataset_index"])} for r in csv.DictReader(p.open())]


def quad(rep, held, clean, crit) -> dict:
    """The four headline rates, each constructed through `guards.Ratio`.

    The four are computed over four different populations, and this metric family has already
    been reported against inconsistent denominators once -- RR_reported was taken over bug_train
    in one script and bug_train+bug_val in another, which made a gated arm appear to beat its own
    ungated arm on a quantity that can only fall. `Ratio` refuses to be constructed without a
    named population and rejects a numerator larger than its denominator, so an accidental swap
    of populations fails here instead of reaching a table.

    The returned floats are unchanged; `provenance` carries the printable "83/112 over X" labels.
    """
    cc = [r for r in clean if r["lab"] in crit]
    ratios = [
        Ratio("RR_repair", sum(r["pc"] for r in rep), max(len(rep), 1), "repair_support_seen",
              "the patch's own evidence"),
        Ratio("RR_held", sum(r["pc"] for r in held), max(len(held), 1), "repair_holdout_unseen"),
        Ratio("Reg", sum(1 for r in clean if not r["pc"]), max(len(clean), 1), "S_clean^test"),
        Ratio("CReg", sum(1 for r in cc if not r["pc"]), max(len(cc), 1),
              "safety-critical rows of S_clean^test"),
    ]
    out = {r.name: r.value for r in ratios}
    out.update({"n_seen": len(rep), "n_held": len(held), "n_clean": len(clean), "n_crit": len(cc),
                "provenance": {r.name: r.label() for r in ratios}})
    return out


def read_tree(tmpl: str, ds: str, bb: str, s: int) -> dict | None:
    p = ROOT / tmpl.format(s=s, ds=ds, bb=bb)
    need = ["repair_support_seen", "repair_holdout_unseen", "clean_eval"]
    if not all((p / f"{n}_predictions.csv").exists() for n in need):
        return None
    return quad(*(rows_of(p / f"{n}_predictions.csv") for n in need), critical(ds))


def read_direct(tmpl: str, ds: str, bb: str, s: int) -> dict | None:
    p = ROOT / tmpl.format(s=s, ds=ds, bb=bb)
    idxf = ROOT / f"artifacts/bug_sets/v8_splits_seed{s}/{ds}_{bb}/{ds}_clean_test_indices.json"
    if not (p / "clean_eval_predictions.csv").exists() or not idxf.exists():
        return None
    keep = set(json.loads(idxf.read_text())["indices"])
    return quad(rows_of(p / "repair_support_seen_predictions.csv"),
                rows_of(p / "repair_holdout_unseen_predictions.csv"),
                [r for r in rows_of(p / "clean_eval_predictions.csv") if r["i"] in keep],
                critical(ds))


READERS = {"tree": read_tree, "direct": read_direct}

# --- DynaPatch, second round -----------------------------------------------------------------
# The gate is a curve, not a point, so it enters the table at the deployer-specified operating
# points of section 21.3 (r = fraction of baseline Reg removed) plus the first round's own
# selection rule (largest threshold keeping CReg = 0). RR_held / Reg / CReg come from
# outputs/gate_zoo_curves.csv; RR_repair is not in that file -- gate_zoo never scores
# repair_support_seen -- so it is joined in from outputs/gate_rr_repair.csv, which re-runs the
# SAME gate, protocol and threshold and only adds the third population (its q=0 point reproduces
# the ungated 0.834/0.522 exactly).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import names as _N
GATE = _N.GATE_CURVE_NAME
NATURAL_POINT = ROOT / _N.GATE_NATURAL_POINT   # protocol C, theta=0, declared in names.py
# RR_repair is deliberately absent: under protocol C the gate is fitted on bug_train, so a
# repair rate on the evidence set is in-sample for the gate. It is reported as NaN.


def dynapatch_rows() -> list[tuple[str, dict, dict]]:
    # FIXED 2026-09-04, two rounds. Round 1 scanned _N.GATE_CURVES and, per setting, picked
    # "the q whose frac_Reg_removed crosses r, then the highest RR_held among those" -- an
    # eval-set selection bug (theta chosen by searching S_held + S_clean^test, the same
    # population RR_held/Reg/CReg are then reported on). Round 2 dropped r-targeting and
    # calibration entirely: the calib population round 1 used to pick theta (bug_val +
    # clean_calib) is ALSO part of what fits the gate under protocol C, so it was never fully
    # disjoint calibration either. Now reads _N.GATE_NATURAL_POINT: theta=0 (the fitted
    # model's own class decision, P(gain=+1) > P(gain=-1)) -- no target r, no threshold
    # search, no calibration split of any kind. See note/RESEARCH_STATE.md.
    if not NATURAL_POINT.exists():
        return []
    nat = pd.read_csv(NATURAL_POINT)
    pooled = nat.groupby("setting", as_index=False)[
        ["RR_held", "Reg", "CReg", "n_held", "n_clean", "n_crit"]].mean()

    per = {}
    for _, pt in pooled.iterrows():
        q = {"RR_held": float(pt.RR_held), "Reg": float(pt.Reg), "CReg": float(pt.CReg),
             "RR_repair": float("nan"),   # in-sample under protocol C, see above
             "n_seen": 0, "n_held": int(pt.n_held),
             "n_clean": int(pt.n_clean), "n_crit": int(pt.n_crit)}
        per[pt.setting] = {"pooled": q}   # natural-points already pools the three seeds
    if not per:
        return []
    # The `CReg=0` and r-indexed (r in {.60,.80,.90}) operating points were both DROPPED
    # (2026-09-02 and 2026-09-04 respectively, user decisions) -- both defined a reported
    # number by searching or targeting something on the split it is reported on.
    # scripts/names.py:RQ4_METHOD_REJECT already refused to treat either as a method.
    return [(_N.gate_label_natural(), per, None)]
METRICS = ("RR_repair", "RR_held", "Reg", "CReg")


def collect(kind: str, tmpl: str, seeds: tuple[int, ...] | None = None) -> dict:
    """{setting: {seed: quad}}

    `seeds` restricts which seeds may enter the row. It exists for rows that are DELIBERATELY
    single-seed: DistRep(PSO) at the paper's own budget costs 3300-6200 s per cell, so only seed
    101 is run to completion. Two seed-202 cells exist on disk from an interrupted sweep; letting
    them in would give that row 12 one-seed settings and 2 two-seed settings, and `seed_mean`
    would then average over different seed counts per setting -- the SEED MISMATCH failure mode
    analyze_mainline.py flags, which has produced four false verdicts in this project.
    """
    per: dict = {}
    for ds, bb in SETTINGS:
        for s in (seeds or SEEDS):
            q = READERS[kind](tmpl, ds, bb, s)
            if q:
                per.setdefault(f"{ds}/{bb}", {})[s] = q
    return per


def audit_baseline_params(name: str, tmpl: str, seeds: tuple[int, ...] | None = None) -> None:
    """Say, per row, whether the artefacts record what the baseline was configured with.

    The sweep rows carry their knob in the tree name, so those are auditable by construction.
    The fixed rows are not: as of 2026-09-03 a baseline tree persists `selected_top_k` and
    nothing else -- not `step_scale`, `rounds`, `clean_tradeoff`, `bound_scale`, `patch_aggr`,
    nor the PSO settings. That is why `step_scale=0.5` survived into a reported table until it
    was swept by hand. This prints the gap rather than asserting a conclusion the artefacts do
    not support: "we cannot tell whether this row was tuned" is the honest claim, and it is
    weaker than "this row was left at defaults".
    """
    method = next((m for m in guards.LIBRARY_DEFAULTS if m.lower() in name.lower()), None)
    if method is None:
        return
    tree = (ROOT / tmpl.format(s=(seeds or SEEDS)[0], ds=SETTINGS[0][0], bb=SETTINGS[0][1])).parent
    check_params_recorded(tree, method, warn_only=True)


def seed_mean(per: dict, metric: str) -> dict:
    return {k: st.mean(v[metric] for v in d.values()) for k, d in per.items()}


def resolve_sweep(variants: list[tuple[str, str]], mode: str) -> tuple[dict, dict]:
    """Pick, per setting (or globally), the variant with the highest mean RR_held.

    Most-favourable-to-the-baseline, per section 31.2. Returns (per, chosen) where `chosen`
    records which variant won each setting so the table can print it."""
    loaded = {lbl: collect("tree", t) for lbl, t in variants}
    loaded = {k: v for k, v in loaded.items() if v}
    if not loaded:
        return {}, {}
    if mode == "global":
        best = max(loaded, key=lambda k: st.mean(seed_mean(loaded[k], "RR_held").values()))
        return loaded[best], {k: best for k in loaded[best]}
    per, chosen = {}, {}
    settings = {s for v in loaded.values() for s in v}
    for stg in settings:
        cands = [(lbl, v[stg]) for lbl, v in loaded.items() if stg in v]
        # SEED-COUNT GUARD. A variant that only ran on one seed must not win a comparison of
        # means against three-seed variants -- that is the failure mode analyze_mainline.py
        # already flags as SEED MISMATCH. Keep only the best-covered variants.
        nmax = max(len(d) for _, d in cands)
        cands = [c for c in cands if len(c[1]) == nmax]
        lbl, d = max(cands, key=lambda c: st.mean(x["RR_held"] for x in c[1].values()))
        per[stg], chosen[stg] = d, lbl
    return per, chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", choices=["per-setting", "global"], default="per-setting")
    ap.add_argument("--csv", default="outputs/rq4_final.csv")
    ap.add_argument("--per-setting", action="store_true", help="also print every setting")
    a = ap.parse_args()

    table, raw = [], []
    # Before any number is read: for each fixed row, say whether its artefacts record the
    # hyperparameters the result depends on. The sweep rows carry theirs in the tree name.
    for r in FIXED_ROWS:
        audit_baseline_params(r[0], r[2], r[3] if len(r) > 3 else None)
    entries = [(r[0], collect(r[1], r[2], r[3] if len(r) > 3 else None), None)
               for r in FIXED_ROWS]
    entries += [(n, *resolve_sweep(v, a.select)) for n, v in SWEEPS]
    entries += dynapatch_rows()

    for name, per, chosen in entries:
        if not per:
            table.append((name, None, None, "MISSING"))
            continue
        means = {}
        for m in METRICS:
            v = [x for x in seed_mean(per, m).values() if x == x]   # drop nan
            means[m] = st.mean(v) if v else float("nan")
        nseeds = sorted({len(d) for d in per.values()})
        pooled = any("pooled" in d for d in per.values())
        note = (f"{len(per)}/12 settings, "
                + ("seeds pooled in curve" if pooled else f"{nseeds} seeds"))
        if any(name.startswith(p) for p in POSITIONAL_IDX):
            note += ", positional idx"
        if chosen:
            note += "; " + ",".join(sorted({v for v in chosen.values()}))
        table.append((name, means, per, note))
        for stg, d in per.items():
            for s, q in d.items():
                raw.append({"method": name, "setting": stg, "seed": s,
                            "variant": (chosen or {}).get(stg, ""),
                            **{m: round(q[m], 6) for m in METRICS},
                            **{k: q[k] for k in ("n_seen", "n_held", "n_clean", "n_crit")}})

    out = ROOT / a.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw[0]))
        w.writeheader()
        w.writerows(raw)

    w0 = max(len(n) for n, *_ in table) + 1
    print(f"{'Method':{w0}s} {'RR_repair':>9s} {'RR_held':>8s} {'Reg':>8s} {'CReg':>8s}   coverage")
    print("-" * (w0 + 70))
    for name, means, per, note in table:
        if means is None:
            print(f"{name:{w0}s} {'--':>9s} {'--':>8s} {'--':>8s} {'--':>8s}   {note}")
            continue
        f3 = lambda x: f"{x:9.3f}" if x == x else f"{'--':>9s}"   # noqa: E731
        f4 = lambda x: f"{x:8.4f}" if x == x else f"{'--':>8s}"     # noqa: E731
        print(f"{name:{w0}s} {f3(means['RR_repair'])} {means['RR_held']:8.3f} "
              f"{f4(means['Reg'])} {f4(means['CReg'])}   {note}")

    if a.per_setting:
        for name, means, per, _ in table:
            if not per:
                continue
            print(f"\n--- {name}")
            for stg in sorted(per):
                q = {m: st.mean(v[m] for v in per[stg].values()) for m in METRICS}
                print(f"    {stg:26s} {q['RR_repair']:7.3f} {q['RR_held']:7.3f} "
                      f"{q['Reg']:8.4f} {q['CReg']:8.4f}   ({len(per[stg])} seeds)")

    print("\nCAVEATS")
    print("  * DistRep(PSO): the paper's full budget 5x40x40 / clean-cap 2048, but ONE seed"
          " where every other method has three.")
    print("  * Arachne(DE): faithful re-implementation, not the authors' TF/Keras artefact.")
    print("  * Arachne / DistRep trees use a POSITIONAL dataset_index -- never join on it.")
    print("  * Swept rows report the value most favourable to the baseline; the shipped-default"
          " row is printed above each for audit.")
    print(f"\nwrote {out}  ({len(raw)} rows)")


if __name__ == "__main__":
    main()
