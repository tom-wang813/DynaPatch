#!/usr/bin/env python3
"""Every RQ's per-setting numbers, in ONE tidy file, at the best available configuration.

Why this exists
---------------
The four RQs live in four different artefact trees with four different conventions. This script
is the single export: one long-format CSV (one row per rq/method/setting/seed/metric) plus a
readable wide dump. Nothing is aggregated away -- per-seed rows are kept, so any mean or median
can be recomputed downstream.

"Best available configuration" means, per RQ:
  RQ1  the fixed-patch arm is the best of its lr sweep {1e-2,1e-1} per setting (chosen on
       RR_held, i.e. most favourable to the baseline); the conditioned arm is ep40ns
       (40 epochs, no early stop). Both from outputs/patch_ablation/.
  RQ2  the shipped gate feature set (pre_mag_response = pre+post), protocol `within`, arm
       `help_flipped`; the three weaker feature sets are exported too, tagged.
  RQ3  the gate operating points r in {.60,.80,.90}, plus the ungated reference. CReg=0 was
       dropped 2026-09-02 -- its threshold was selected by a value read off the reporting split.
  RQ4  every prior-art method; the two search baselines at their swept optimum
       (Arachne: per-setting best of bound_scale {2,4,8,16,32,64,128};
        TopKSearch: per-setting best of step_scale {8,32,128}), DistRep at the paper's full
       budget. Shipped-default rows are exported as well, tagged `shipped_default`.

CAVEAT carried in the `note` column, not buried: RQ3/RQ4's DynaPatch rows come from the shipped
12-epoch patch, while RQ1's conditioned arm is 40-epoch. They are NOT the same training budget.

Zero GPU. Usage:  .venv/bin/python scripts/all_rq_data.py
"""
from __future__ import annotations
import csv, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT_CSV, OUT_TXT = ROOT / "outputs/ALL_RQ_DATA.csv", ROOT / "outputs/ALL_RQ_DATA.txt"
DSS = ["gtsrb", "tt100k_signs", "lisa_signs"]
BBS = ["resnet50", "convnext_tiny", "densenet121", "vgg16"]
SEEDS = [101, 202, 303]
SHORT = {"gtsrb": "G", "tt100k_signs": "T", "lisa_signs": "L"}
BB = {"resnet50": "RN", "convnext_tiny": "CN", "densenet121": "DN", "vgg16": "VG"}
PAPER = [(d, b) for d in DSS for b in ("resnet50", "convnext_tiny", "vgg16", "densenet121")]
rows: list[dict] = []


def add(rq, method, ds, bb, seed, metrics, variant="", note="", source=""):
    for k, v in metrics.items():
        if v is None or v != v:
            continue
        rows.append(dict(rq=rq, method=method, setting=f"{ds}/{bb}",
                         label=f"{SHORT[ds]}-{BB[bb]}", dataset=ds, backbone=bb,
                         seed=seed, variant=variant, metric=k, value=round(float(v), 6),
                         note=note, source=source))


# ---------------------------------------------------------------- RQ1: patch synthesis
import contextlib, io  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):        # the module prints its own report on import
    import table_rq1_ablation as A  # noqa: E402  (its cell() is the canonical per_setting metric)

# lr1e3 was only ever run on 3 settings x 1 seed; the len(cs)==3 guard below drops it.
FIXED_ARMS = ["fixed60_lr1e3", "fixed60_lr1e2", "fixed60_lr1e1"]
for ds in DSS:
    cr = A.crit(ds)
    for bb in BBS:
        # 2026-09-05: vgg16/convnext_tiny adopted repair.patch_site=last_affine (see
        # note/RQ1_DATA.md's header and note/RESEARCH_STATE.md); resnet50/densenet121 stay
        # on the shipped "ep40ns" tree (last_affine is bit-identical to deep_feat there).
        arm = "ep40ns_lastaffine" if bb in ("vgg16", "convnext_tiny") else "ep40ns"
        for s in SEEDS:
            c = A.cell(arm, ds, bb, s, cr)
            if c:
                add("RQ1", "DynaPatch-NoGate (40ep, no early stop)", ds, bb, s,
                    {"RR_seen": c["seen"], "RR_held": c["held"], "Reg": c["reg"], "CReg": c["creg"]},
                    variant=arm, source="outputs/patch_ablation")
        # fixed patch: best lr per setting on mean RR_held (most favourable to the baseline)
        cand = {}
        for arm in FIXED_ARMS:
            cs = {s: c for s in SEEDS if (c := A.cell(arm, ds, bb, s, cr))}
            if len(cs) == 3:
                cand[arm] = cs
        if cand:
            best = max(cand, key=lambda a: st.mean(c["held"] for c in cand[a].values()))
            for s, c in cand[best].items():
                add("RQ1", "FixedPatch (best lr)", ds, bb, s,
                    {"RR_seen": c["seen"], "RR_held": c["held"], "Reg": c["reg"], "CReg": c["creg"]},
                    variant=best, note="best of lr {1e-2,1e-1} on RR_held",
                    source="outputs/patch_ablation")

# ---------------------------------------------------------------- RQ2: gate as a classifier
GP = ROOT / "outputs/gate_performance.csv"
SHIPPED_FEAT = "pre_mag_response"
if GP.exists():
    for r in csv.DictReader(GP.open()):
        if r["protocol"] != "within" or r["arm"] != "help_flipped":
            continue
        ds, bb = r["setting"].split("/")
        ds = {"tt100k": "tt100k_signs", "lisa": "lisa_signs"}.get(ds, ds)
        m = {"n_test": r["n_test"], "pos": r["pos"], "prevalence": r["prevalence"],
             "AUROC": r["auroc"], "AUPRC": r["auprc"], "ECE_raw": r["ece_raw"],
             "ECE_platt": r["ece_platt"], "Brier_platt": r["brier_platt"]}
        for rr_ in ("0.60", "0.80", "0.90"):
            for k in ("accuracy", "precision", "recall", "f1", "coverage"):
                m[f"r{rr_}_{k}"] = r[f"r{rr_}_{k}"]
        add("RQ2", "Gate (" + r["features"] + ")", ds, bb, "pooled",
            {k: float(v) for k, v in m.items() if v not in ("", None)},
            variant=r["features"],
            note="SHIPPED" if r["features"] == SHIPPED_FEAT else "weaker feature set",
            source="outputs/gate_performance.csv")

# ---------------------------------------------------------------- RQ3 + RQ4: one CSV
# REMOVED 2026-09-06 (user decision, root out entirely): RQ3_METHODS used to route "Always
# Patch (ungated)" and the retired r-grid rows ("DynaPatch [2nd round] @ r=0.60/0.80/0.90")
# into RQ3 with a "12-epoch patch -- NOT epoch-matched with RQ1's 40ep arm" note. All four
# entries are gone from the live pipeline: the r-grid was dropped project-wide for theta=0
# (note/RESEARCH_STATE.md, 2026-09-04/05) and "Always Patch (ungated)" was a stale 12-epoch
# tree (outputs/repairbench_v8, timestamped 2026-07-19) never the same thing as the current
# 40-epoch "DynaPatch (ungated)"/"DynaPatch (gated)" rows -- see scripts/table_rq4_final.py's
# FIXED_ROWS removal note. The old `r["method"].startswith("DynaPatch")` condition below had
# ALSO been silently mislabeling the CURRENT "DynaPatch (gated)" row (which does start with
# "DynaPatch") with the same wrong "12-epoch, not epoch-matched" note -- found and fixed here,
# not just the entry it was originally written for.
SHIPPED_DEFAULT = ("TopKSearch [shipped as 'LSR'] step=0.5", "Weighted Retraining (12ep, shipped)",
                   "Full fine-tuning [shipped as 'DistrRep'] (12ep)",
                   "Head-Only Fine-Tuning (12ep, shipped)", "DistRep(PSO) real, reduced budget")
F = ROOT / "outputs/rq4_final.csv"
for r in csv.DictReader(F.open()):
    ds, bb = r["setting"].split("/")
    rq = "RQ4"
    note = "shipped_default (audit row, not the best config)" if r["method"] in SHIPPED_DEFAULT else ""
    add(rq, r["method"], ds, bb, r["seed"],
        {k: r[k] for k in ("RR_repair", "RR_held", "Reg", "CReg") if r[k] not in ("", None)},
        variant=r["variant"], note=note, source="outputs/rq4_final.csv")

# 5 draws (median, with __min/__max kept alongside). See scripts/baseline_prior_patches.py
# --repeats: one draw of these methods is not reportable.
PP = next(p for p in (ROOT / "outputs/baseline_prior_patches_mlp_r5.json",
                      ROOT / "outputs/baseline_prior_patches_mlp.json") if p.is_file())
if PP.exists():
    for name, d in json.load(PP.open()).items():
        for stg, per_seed in d.items():
            ds, bb = stg.split("/")
            for s, v in per_seed.items():
                add("RQ4", name, ds, bb, s,
                    {"RR_repair": v["RR_seen"], "RR_held": v["RR_held"],
                     "Reg": v["Reg"], "CReg": v["CReg"]},
                    note="error estimator swept, MLP", source=PP.name)

# ---------------------------------------------------------------- write
FIELDS = ["rq", "method", "variant", "setting", "label", "dataset", "backbone", "seed",
          "metric", "value", "note", "source"]
with OUT_CSV.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in FIELDS})

# readable wide dump: seed-mean per (rq, method, setting, metric)
agg: dict = {}
chosen: dict = {}
for r in rows:
    agg.setdefault((r["rq"], r["method"], r["metric"]), {}).setdefault(r["setting"], []).append(r["value"])
    chosen.setdefault((r["rq"], r["method"]), {})[r["setting"]] = r["variant"]
lines = [__doc__.strip().split("\n\n")[0], ""]
for rq in ("RQ1", "RQ2", "RQ3", "RQ4"):
    for metric in ("RR_held", "RR_seen", "RR_repair", "Reg", "CReg", "AUROC", "AUPRC",
                   "r0.80_f1", "r0.80_precision", "r0.80_recall", "r0.80_accuracy"):
        keys = [k for k in agg if k[0] == rq and k[2] == metric]
        if not keys:
            continue
        lines += ["", f"### {rq}  {metric}   (seed mean; per-seed rows are in ALL_RQ_DATA.csv)"]
        hdr = f"{'method':52s}" + "".join(f"{SHORT[d]+'-'+BB[b]:>9s}" for d, b in PAPER) + f"{'MEAN':>9s}"
        lines += [hdr, "-" * len(hdr)]
        for k in sorted(keys, key=lambda x: x[1]):
            per = agg[k]
            vals = [st.mean(per[f"{d}/{b}"]) if f"{d}/{b}" in per else None for d, b in PAPER]
            got = [v for v in vals if v is not None]
            dec = 4 if metric in ("Reg", "CReg") else 3
            vs = sorted({v for v in chosen.get((k[0], k[1]), {}).values() if v})
            name = k[1] + (f" [{','.join(vs)}]" if vs else "")
            # NaN is not missing data: under protocol C the gate is fitted within the setting,
            # so a repair rate on `repair_support_seen` -- the patch's own evidence, part of the
            # gate's training pool -- would be IN-SAMPLE. table_rq4_final.py writes NaN on
            # purpose. Printing a bare "nan" reads like a crashed run, so it prints as n/a.
            f1 = lambda v: (f"{'n/a':>9s}" if v != v else f"{v:9.{dec}f}") if v is not None \
                else f"{'--':>9s}"
            got = [v for v in got if v == v]
            lines.append(f"{name[:52]:52s}" + "".join(f1(v) for v in vals)
                         + (f"{st.mean(got):9.{dec}f}" if got else f"{'n/a':>9s}"))
            if any(v is not None and v != v for v in vals):
                lines.append("    n/a = in-sample under protocol C (the gate is fitted on "
                             "bug_train, which the seen split feeds); not a missing run.")
OUT_TXT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT_CSV}  ({len(rows)} rows)")
print(f"wrote {OUT_TXT}")
