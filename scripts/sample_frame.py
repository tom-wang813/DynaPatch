#!/usr/bin/env python3
"""P0 -- ONE sample-level table behind every mechanism analysis, plus its audit.

Every downstream question ("which failure types does each method repair", "where does the
regression come from", "what does the gate veto") needs the same thing: one row per (method,
setting, seed, input) with the base and patched predictions side by side. This assembles it from
the artefacts that already exist and refuses to paper over what is missing.

Three rules this file exists to enforce
---------------------------------------
1. **The reported configuration only.** Sweeps are resolved with the SAME rule the paper table
   uses (`table_rq4_final.resolve_sweep`, per-setting max mean RR_held, deliberately favourable
   to the baseline) so the frame and the table can never disagree about which run a method is.
2. **`split` is never dropped.** `clean_test` is where regression is measured and `bug_eval` is
   where repair is; the two carry disjoint label mass (on flip rows `gain=-1 <=> base_correct`
   is a theorem). Any analysis that pools them without conditioning is measuring population
   identity. The column is mandatory and the audit prints the composition.
3. **Missing is reported, never silently dropped.** A method absent from a cell shows up in
   AUDIT.md as a gap, because a per-failure-type comparison over ragged coverage is a
   comparison of coverage.

Outputs
-------
    outputs/sample_frame.csv.gz      the long table
    outputs/analysis_audit.csv       per (method, setting, seed): row counts and coverage
    note/AUDIT.md                    the human-readable version, with the gaps called out

    .venv/bin/python scripts/sample_frame.py
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
import table_rq4_final as T  # noqa: E402

SEEDS = (101, 202, 303)
SETTINGS = [(d, b) for d in ("gtsrb", "tt100k_signs", "lisa_signs")
            for b in ("resnet50", "convnext_tiny", "densenet121", "vgg16")]
OBJ = T.OBJ
# the three reported splits. `repair_support_seen` is the patch's OWN evidence: kept, because the
# seen/unseen contrast is the whole of P2, but never to be pooled with the other two.
SPLITS = {"clean_eval": "clean", "repair_holdout_unseen": "held",
          "repair_support_seen": "seen"}
MANIFEST_SPLIT = {"clean": "clean_test", "held": "bug_eval", "seen": "bug_train"}
POSITIONAL_INDEX_METHODS = {N.display("Arachne"), N.display("DistrRep")}

COLS = ["method", "dataset", "backbone", "setting", "seed", "split", "dataset_index",
        "label", "base_pred", "patched_pred", "base_correct", "patched_correct",
        "repaired", "regressed", "routed", "gate_score", "gate_applied", "operating_point"]
BASE_AUDIT_COLS = ["source_base_pred", "source_base_correct", "base_evidence_mismatch"]


def _read(p: Path, split: str) -> pd.DataFrame | None:
    f = p / f"{split}_predictions.csv"
    if not f.is_file() or f.stat().st_size == 0:
        return None
    d = pd.read_csv(f)
    need = {"dataset_index", "label", "base_pred", "patched_pred"}
    if not need <= set(d.columns):
        return None
    for c in ("base_correct", "patched_correct"):
        if c in d:
            d[c] = d[c].astype(str).str.lower() == "true"
    d["base_correct"] = d.get("base_correct", d.base_pred == d.label)
    d["patched_correct"] = d.get("patched_correct", d.patched_pred == d.label)
    d["repaired"] = (~d.base_correct) & d.patched_correct
    d["regressed"] = d.base_correct & (~d.patched_correct)
    return d


def split_indices(dataset: str, backbone: str, seed: int, tag: str) -> list[int]:
    """Return the frozen original dataset identities for one reported population."""
    split_name = MANIFEST_SPLIT[tag]
    path = (
        ROOT / f"artifacts/bug_sets/v8_splits_seed{seed}/{dataset}_{backbone}"
        / f"{dataset}_{split_name}_indices.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen split manifest: {path}")
    payload = json.loads(path.read_text())
    indices = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(indices, list) or len(indices) != len(set(indices)):
        raise AssertionError(f"invalid or duplicate split identities in {path}")
    return [int(index) for index in indices]


def normalise_population(
    frame: pd.DataFrame,
    method: str,
    dataset: str,
    backbone: str,
    seed: int,
    tag: str,
) -> pd.DataFrame:
    """Map positional baseline IDs and restrict every source to the frozen test split.

    Arachne and DistrRep write 0..N-1 counters although their rows follow the manifest order.
    Direct FixedPatch/DynaPatch dumps write the full clean_eval pool.  Both formats are valid
    locally but must be normalised before methods can share a sample-level evidence table.
    """
    expected = split_indices(dataset, backbone, seed, tag)
    frame = frame.copy()
    if method in POSITIONAL_INDEX_METHODS:
        # Their seen/support dumps do not preserve manifest order. They are retained for the
        # aggregate seen-vs-held table but never identity-joined or used by mechanism analyses.
        if tag == "seen":
            return frame
        positional = frame.dataset_index.astype(int).to_numpy()
        if not np.array_equal(positional, np.arange(len(frame))):
            raise AssertionError(
                f"{method} is declared positional but is not 0..N-1: "
                f"{dataset}/{backbone} s{seed} {tag}"
            )
        if len(frame) != len(expected):
            raise AssertionError(
                f"{method} positional rows do not match manifest: "
                f"{dataset}/{backbone} s{seed} {tag}: {len(frame)} vs {len(expected)}"
            )
        frame["dataset_index"] = expected
    else:
        expected_set = set(expected)
        frame = frame[frame.dataset_index.astype(int).isin(expected_set)].copy()
        observed = set(frame.dataset_index.astype(int))
        if observed != expected_set:
            missing = len(expected_set - observed)
            extra = len(observed - expected_set)
            raise AssertionError(
                f"{method} cannot reconstruct frozen {tag} population for "
                f"{dataset}/{backbone} s{seed}: missing={missing}, extra={extra}"
            )
    if frame.dataset_index.duplicated().any():
        raise AssertionError(
            f"duplicate dataset identity after normalisation: "
            f"{method} {dataset}/{backbone} s{seed} {tag}"
        )
    return frame


def from_tree(method: str, tpl: str, seeds: tuple[int, ...] | None = None,
              settings: list[tuple[str, str]] | None = None
              ) -> list[pd.DataFrame]:
    out = []
    for ds, bb in (settings or SETTINGS):
        for s in (seeds or SEEDS):
            base = ROOT / tpl.format(s=s, ds=ds, bb=bb)
            for split, tag in SPLITS.items():
                d = _read(base, split)
                if d is None:
                    continue
                d = normalise_population(d, method, ds, bb, s, tag)
                d = d.assign(method=method, dataset=ds, backbone=bb,
                             setting=f"{ds}/{bb}", seed=s, split=tag)
                out.append(d)
    return out


def resolved_sweeps() -> dict[str, dict[str, str]]:
    """setting -> template, using the paper table's own per-setting choice."""
    chosen: dict[str, dict[str, str]] = {}
    for name, variants in T.SWEEPS:
        tpl = dict(variants)
        _, pick = T.resolve_sweep(variants, "per-setting")
        chosen[name] = {stg: tpl[lbl] for stg, lbl in pick.items()}
    return chosen


def gated_dynapatch(ungated: pd.DataFrame) -> list[pd.DataFrame]:
    """DynaPatch, gated: the ungated predictions with the gate's veto applied at the natural
    threshold (theta=0 -- commit iff the fitted gate's own score favours beneficial over
    harmful; no target r, no threshold search, no calibration split -- 2026-09-05, user
    decision, see note/RESEARCH_STATE.md).

    A vetoed input keeps the DEPLOYED model's answer -- that is what the gate does at runtime.
    Joined on dataset_index within (setting, seed, side); `seen` rows have no gate decision
    (the gate is never asked about the patch's own evidence) and are dropped from these rows.
    """
    gp = ROOT / "outputs/gate_persample"
    if not gp.is_dir():
        return []
    g = pd.concat([pd.read_csv(f) for f in sorted(gp.glob("*.csv"))], ignore_index=True)
    g = g.rename(columns={"side": "split"})
    if "apply_natural" not in g.columns:
        return []
    gg = g[["setting", "seed", "split", "dataset_index", "gate_score",
            "apply_natural"]].rename(columns={"apply_natural": "gate_applied"})
    m = ungated[ungated.split.isin(("clean", "held"))].merge(
        gg, on=["setting", "seed", "split", "dataset_index"], how="inner")
    if m.empty:
        return []
    # veto => fall back to the deployed model
    m["patched_pred"] = np.where(m.gate_applied, m.patched_pred, m.base_pred)
    m["patched_correct"] = m.patched_pred == m.label
    m["repaired"] = (~m.base_correct) & m.patched_correct
    m["regressed"] = m.base_correct & (~m.patched_correct)
    label = N.gate_label_natural() if hasattr(N, "gate_label_natural") else "DynaPatch (gated)"
    return [m.assign(method=label, operating_point="natural")]


def canonicalise_base_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Use one deployed-model prediction to define every shared evaluation row.

    Repair methods should differ only in ``patched_pred``.  Two fine-tuning dumps classify one
    borderline G-DN held input differently in their recorded *base* forward pass, which otherwise
    changes the failure denominator and failure type for those methods.  Preserve those source
    values for audit, but define the common population from DynaPatch's frozen deployed backbone.
    """
    keys = ["setting", "seed", "split", "dataset_index"]
    test = frame[frame.split.isin(("clean", "held"))].copy()
    seen = frame[~frame.split.isin(("clean", "held"))].copy()
    reference_method = N.display("DPNoGate")
    reference = test[test.method == reference_method][
        keys + ["label", "base_pred", "base_correct"]
    ].rename(columns={
        "label": "reference_label",
        "base_pred": "reference_base_pred",
        "base_correct": "reference_base_correct",
    })
    if reference.duplicated(keys).any():
        raise AssertionError("duplicate DynaPatch base-evidence reference rows")
    test["source_base_pred"] = test.base_pred
    test["source_base_correct"] = test.base_correct
    test = test.merge(reference, on=keys, how="left", validate="many_to_one")
    if test.reference_label.isna().any():
        missing = test.loc[test.reference_label.isna(), ["method", *keys]].head()
        raise AssertionError(f"rows lack common deployed-base evidence:\n{missing}")
    label_mismatch = test.label.astype(int) != test.reference_label.astype(int)
    if label_mismatch.any():
        bad = test.loc[label_mismatch, [
            "method", *keys, "label", "reference_label"
        ]].head()
        raise AssertionError(
            f"method rows disagree on the true label after identity alignment:\n{bad}"
        )
    test["base_evidence_mismatch"] = (
        test.source_base_pred.astype(int) != test.reference_base_pred.astype(int)
    )
    test["base_pred"] = test.reference_base_pred.astype(int)
    test["base_correct"] = test.reference_base_correct.astype(bool)
    test["repaired"] = (~test.base_correct) & test.patched_correct
    test["regressed"] = test.base_correct & (~test.patched_correct)
    test = test.drop(columns=[
        "reference_label", "reference_base_pred", "reference_base_correct"
    ])
    seen["source_base_pred"] = seen.base_pred
    seen["source_base_correct"] = seen.base_correct
    seen["base_evidence_mismatch"] = False
    return pd.concat([test, seen], ignore_index=True)


def prior_patches() -> list[pd.DataFrame]:
    root = ROOT / "outputs/prior_patch_persample"
    if not root.is_dir():
        return []
    out = []
    for cell in sorted(root.iterdir()):
        ds, bb, sd = cell.name.rsplit("_s", 1)[0].rsplit("_", 1)[0], None, None
        parts = cell.name.rsplit("_s", 1)
        seed = int(parts[1])
        stem = parts[0]
        for d_, b_ in SETTINGS:
            if stem == f"{d_}_{b_}":
                ds, bb = d_, b_
        if ds is None:
            continue
        for meth in sorted(x for x in cell.iterdir() if x.is_dir()):
            for op in sorted(x for x in meth.iterdir() if x.is_dir()):
                for pop, tag in (("clean_test", "clean"), ("bug_eval", "held"),
                                 ("bug_train", "seen")):
                    f = op / f"{pop}_predictions.csv"
                    if not f.is_file():
                        continue
                    d = pd.read_csv(f)
                    # The operating point goes in the METHOD NAME, not only in its own column.
                    # These two methods dump two thresholds each (`tau` = their default 0.5,
                    # `matched` = calibrated on clean_calib to our own Reg). Sharing one method
                    # label let a groupby on (method, setting, seed) pool them, which doubled the
                    # population -- 273 held rows against everyone else's 136.5 -- and made their
                    # repair COUNTS look like twice what they are.
                    out.append(d.assign(method=f"{meth.name} [{op.name}]", dataset=ds,
                                        backbone=bb, setting=f"{ds}/{bb}", seed=seed,
                                        split=tag, operating_point=op.name))
    return out


def main() -> None:
    frames: list[pd.DataFrame] = []

    for row in T.FIXED_ROWS:
        name, kind, tpl = row[0], row[1], row[2]
        seeds = row[3] if len(row) > 3 else None   # DistRep is deliberately single-seed
        key = N.RQ4_METHOD_TO_KEY.get(name)
        label = N.display(key) if key else name
        frames += from_tree(label, tpl, seeds)

    for name, per_setting in resolved_sweeps().items():
        key = N.RQ4_METHOD_TO_KEY.get(name)
        label = N.display(key) if key else name
        for stg, tpl in per_setting.items():
            ds, bb = stg.split("/")
            for s in SEEDS:
                base = ROOT / tpl.format(s=s, ds=ds, bb=bb)
                for split, tag in SPLITS.items():
                    d = _read(base, split)
                    if d is None:
                        continue
                    d = normalise_population(d, label, ds, bb, s, tag)
                    frames.append(d.assign(method=label, dataset=ds, backbone=bb,
                                           setting=stg, seed=s, split=tag))

    # DynaPatch, ungated (the 40ep no-early-stop build the gated rows also come from).
    # 2026-09-05: vgg16/convnext_tiny adopted repair.patch_site=last_affine (see
    # note/RESEARCH_STATE.md and note/RQ1_DATA.md's header for the measured trade-off);
    # resnet50/densenet121 are untouched (last_affine is bit-identical to deep_feat there,
    # so there is no separate tree for them -- reading ep40ns_s{s}_kfull is correct either way).
    LASTAFFINE_BACKBONES = {"vgg16", "convnext_tiny"}
    deepfeat_settings = [(ds, bb) for ds, bb in SETTINGS if bb not in LASTAFFINE_BACKBONES]
    lastaffine_settings = [(ds, bb) for ds, bb in SETTINGS if bb in LASTAFFINE_BACKBONES]
    dpn = from_tree(N.display("DPNoGate"),
                    "outputs/patch_ablation/{ds}_{bb}/ep40ns_s{s}_kfull/deploy/predictions",
                    settings=deepfeat_settings)
    dpn += from_tree(N.display("DPNoGate"),
                     "outputs/patch_ablation/{ds}_{bb}/ep40ns_lastaffine_s{s}_kfull/deploy/predictions",
                     settings=lastaffine_settings)
    frames += dpn

    # FixedPatch: per-setting best lr on mean RR_held, the same rule all_rq_data.py applies
    fp: list[pd.DataFrame] = []
    for ds, bb in SETTINGS:
        cand = {}
        for arm in ("fixed60_lr1e2", "fixed60_lr1e1"):
            got = {}
            for s in SEEDS:
                d = _read(ROOT / f"outputs/patch_ablation/{ds}_{bb}/{arm}_s{s}_kfull/deploy"
                                 "/predictions", "repair_holdout_unseen")
                if d is not None:
                    got[s] = d
            if len(got) == len(SEEDS):
                cand[arm] = got
        if not cand:
            continue
        best = max(cand, key=lambda a: np.mean([c.patched_correct.mean()
                                                for c in cand[a].values()]))
        for s in SEEDS:
            for split, tag in SPLITS.items():
                d = _read(ROOT / f"outputs/patch_ablation/{ds}_{bb}/{best}_s{s}_kfull/deploy"
                                 "/predictions", split)
                if d is not None:
                    d = normalise_population(d, N.display("FP"), ds, bb, s, tag)
                    fp.append(d.assign(method=N.display("FP"), dataset=ds, backbone=bb,
                                       setting=f"{ds}/{bb}", seed=s, split=tag,
                                       operating_point=best))
    frames += fp

    frames += prior_patches()

    if not frames:
        raise SystemExit("no artefacts found")
    df = pd.concat(frames, ignore_index=True)
    df = df.assign(**{c: df.get(c) for c in COLS if c not in df})
    # gated DynaPatch rows are built from the ungated ones
    if dpn:
        df = pd.concat([df] + gated_dynapatch(pd.concat(dpn, ignore_index=True)),
                       ignore_index=True)
    df["operating_point"] = df["operating_point"].fillna("-")
    df = canonicalise_base_evidence(df)

    # failure_type is defined ONLY on inputs the deployed model got wrong
    df["failure_type"] = np.where(df.base_correct, pd.NA,
                                  df.label.astype(str) + "->" + df.base_pred.astype(str))
    df["true_class"] = df.label
    df = df[COLS + BASE_AUDIT_COLS + ["failure_type", "true_class"]]

    # csv.gz, not parquet: this repo ships as the paper's supplement and must not need
    # pyarrow to be readable.
    out = ROOT / "outputs/sample_frame.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    print(f"[written] {out.relative_to(ROOT)}   {len(df):,} rows, "
          f"{df.method.nunique()} methods")

    # ---------------------------------------------------------------- audit
    a = (df.groupby(["method", "setting", "seed", "split"])
           .agg(n=("dataset_index", "size"),
                n_base_wrong=("base_correct", lambda x: int((~x).sum())),
                n_repaired=("repaired", "sum"), n_regressed=("regressed", "sum"))
           .reset_index())
    a.to_csv(ROOT / "outputs/analysis_audit.csv", index=False)

    cov = a.pivot_table(index="method", columns="split", values="n",
                        aggfunc=lambda x: len(x)).fillna(0).astype(int)
    full = len(SETTINGS) * len(SEEDS)
    L = ["# Sample-frame audit", "",
         f"`outputs/sample_frame.csv.gz` — {len(df):,} rows, {df.method.nunique()} methods, "
         f"built {pd.Timestamp.now():%Y-%m-%d %H:%M}.", "",
         "## Coverage: (setting, seed) cells present per split",
         "",
         f"A complete method has **{full}** cells (12 settings x 3 seeds) in each split it "
         "reports. `seen` is the patch's own evidence and is present only for patch-type "
         "methods; weight-editing baselines report it too but it is NOT comparable across "
         "method families and must never be pooled with `held`.", "",
         "| method | clean | held | seen | gap |", "|---|---|---|---|---|"]
    for m in cov.index:
        r = cov.loc[m]
        g = [f"{s} {int(r.get(s,0))}/{full}" for s in ("clean", "held") if r.get(s, 0) < full]
        L.append(f"| {m} | {int(r.get('clean',0))} | {int(r.get('held',0))} | "
                 f"{int(r.get('seen',0))} | {'; '.join(g) if g else 'complete'} |")

    L += ["", "## Failure-type granularity", "",
          "`failure_type` is the ordered pair (true label -> the deployed model's prediction), "
          "defined only where the deployed model is wrong. It fragments badly on the repair "
          "side: the counts below are why the primary unit of the per-type analyses is the "
          "TRUE CLASS, with the ordered pair reserved for the regression side where the "
          "populations are large.", "",
          "| setting | base-wrong on `held` (median over seeds) | distinct failure types | "
          "types with n>=5 |", "|---|---|---|---|"]
    h = df[(df.split == "held") & (df.method == N.display("DPNoGate"))]
    for stg in sorted(h.setting.unique()):
        x = h[h.setting == stg]
        per = x.groupby("seed").size().median()
        ft = x.failure_type.dropna()
        vc = ft.value_counts()
        L.append(f"| {stg} | {per:.0f} | {ft.nunique()} | {int((vc >= 5).sum())} |")

    L += ["", "## Split composition (why `split` may never be dropped)", "",
          "On flip rows `gain=-1 <=> base_correct` is a theorem, so the two reporting "
          "populations carry disjoint label mass. Pooling them measures which split a row came "
          "from, not whether the gate works.", "",
          "| split | source | rows | base-wrong |", "|---|---|---|---|"]
    for sp, src in (("clean", "clean_test — where Reg/CReg are measured"),
                    ("held", "bug_eval — where RR_held is measured"),
                    ("seen", "repair_support_seen — the patch's OWN evidence")):
        x = df[df.split == sp]
        if len(x):
            L.append(f"| {sp} | {src} | {len(x):,} | {int((~x.base_correct).sum()):,} |")

    miss = [m for m in cov.index if cov.loc[m].get("held", 0) < full]
    L += ["", "## Gaps", ""]
    L += ([f"- **{m}**: {int(cov.loc[m].get('held',0))}/{full} held cells — any per-type "
           f"comparison including it is over ragged coverage." for m in miss]
          or ["- none: every method covers all 12 settings x 3 seeds."])
    (ROOT / "note/AUDIT.md").write_text("\n".join(L) + "\n")
    print(f"[written] note/AUDIT.md, outputs/analysis_audit.csv")


if __name__ == "__main__":
    main()
