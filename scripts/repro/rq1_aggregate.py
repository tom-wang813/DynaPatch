#!/usr/bin/env python3
"""Drive all 7 methods (FullFT, HeadFT, Arachne, DistrRep, NNPatch, PatchNAS, DynaPatch) across
the 12 (dataset, backbone) settings from shipped checkpoints, and assemble
outputs/repro/{rq1_rr,rq1_summary,rq2_ungated_persetting,rq2_ungated_summary}.csv.

Each method is invoked as a subprocess against its own scripts/checkpoint_eval/<method>.py CLI
(kept as separate processes rather than in-process imports so each method's own sys.path/global
state -- e.g. prior_patch_common.CRIT -- never leaks across methods).

NNPatch/PatchNAS's Table rq1_rr operating point is tau=0.5, their own estimator's natural
decision boundary (paper.tex describes them as "uses an error estimator to decide whether to
apply the patch", no recalibration) -- FIXED 2026-09-22, previously used a Reg-matched calib_tau
(prior_patch_common.calib_tau, still computed and kept as an unused diagnostic in summary.json's
"matched" key) which gave RQ1 means far below paper's (0.133/0.173 vs paper's 0.373/0.317); tau=0.5
reproduces much closer (0.330/0.259). DynaPatch is still run first per setting to populate
outputs/repro/dynapatch_reg_lookup.json, now only consumed by that unused diagnostic.

FixedPatch is deliberately NOT included here yet -- no shipped checkpoint exists for it (see the
plan). scripts/checkpoint_eval/fixedpatch.py + scripts/repro/rq2_train_fixedpatch.py add it in a
later step, at which point rq2_ungated_persetting/summary become fillable.

Usage:
  uv run python scripts/repro/rq1_aggregate.py --settings gtsrb/resnet50          # one setting
  uv run python scripts/repro/rq1_aggregate.py                                    # all 12
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/checkpoint_eval"))
import common  # noqa: E402

SEED = 101
OUT_ROOT = ROOT / "outputs/repro/ckpt_eval"
REPORT_DIR = ROOT / "outputs/repro"
REG_LOOKUP = REPORT_DIR / "dynapatch_reg_lookup.json"

# setting label per paper.tex convention: {G,T,L}-{RN,CN,DN,VG}
DATASETS = [("gtsrb", "G"), ("tt100k_signs", "T"), ("lisa_signs", "L")]
BACKBONES = [("resnet50", "RN"), ("convnext_tiny", "CN"), ("densenet121", "DN"), ("vgg16", "VG")]
ALL_SETTINGS = [(ds, bb) for ds, _ in DATASETS for bb, _ in BACKBONES]


def setting_label(dataset: str, backbone: str) -> str:
    d = dict(DATASETS)[dataset] if False else {ds: lab for ds, lab in DATASETS}[dataset]
    b = {bb: lab for bb, lab in BACKBONES}[backbone]
    return f"{d}-{b}"


def run(cmd: list[str]) -> str:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def py(*args: str) -> list[str]:
    return ["uv", "run", "python", *args]


def run_dynapatch(dataset: str, backbone: str) -> dict:
    out_root = OUT_ROOT / "DynaPatch_driver"
    run(py("scripts/checkpoint_eval/dynapatch.py", "--dataset", dataset, "--backbone", backbone,
            "--seed", str(SEED), "--mode", "checkpoint", "--output-root", str(out_root), "--gate"))
    base = out_root / dataset / backbone / f"s{SEED}"
    ungated = common.summarize(base / "deploy" / "predictions", dataset)
    gated = common.summarize(base / "deploy_gated" / "predictions", dataset)
    return {"ungated": ungated, "gated": gated}


def run_headft(dataset: str, backbone: str, baseline: str, method: str) -> dict:
    out_root = OUT_ROOT / method
    run(py("scripts/checkpoint_eval/headft.py", "--baseline", baseline, "--dataset", dataset,
            "--backbone", backbone, "--seed", str(SEED), "--mode", "checkpoint",
            "--output-root", str(out_root)))
    return common.summarize(out_root / method / dataset / backbone / f"s{SEED}" / "predictions", dataset)


def run_arachne(dataset: str, backbone: str) -> dict:
    out_root = OUT_ROOT / "Arachne"
    run(py("scripts/checkpoint_eval/arachne.py", "--dataset", dataset, "--backbone", backbone,
            "--seed", str(SEED), "--mode", "checkpoint", "--output-root", str(out_root)))
    return common.summarize(out_root / "Arachne" / dataset / backbone / f"s{SEED}" / "predictions", dataset)


def run_fixedpatch(dataset: str, backbone: str) -> dict | None:
    ckpt = ROOT / f"artifacts/checkpoints/baselines/FixedPatch/{dataset}_{backbone}_s{SEED}/repair_best.pt"
    if not ckpt.exists():
        print(f"[FixedPatch] no checkpoint for {dataset}/{backbone} -- skipping")
        return None
    out_root = OUT_ROOT / "FixedPatch"
    run(py("scripts/checkpoint_eval/fixedpatch.py", "--dataset", dataset, "--backbone", backbone,
            "--seed", str(SEED), "--mode", "checkpoint", "--output-root", str(out_root)))
    return common.summarize(out_root / dataset / backbone / f"s{SEED}" / "deploy" / "predictions", dataset)


def run_distrep(dataset: str, backbone: str) -> dict | None:
    ckpt = common.checkpoint_dir("DistrRep", dataset, backbone, SEED) / "distrep_repaired.pt"
    if not ckpt.exists():
        print(f"[DistrRep] no checkpoint for {dataset}/{backbone} -- skipping")
        return None
    out_root = OUT_ROOT / "DistrRep"
    run(py("scripts/checkpoint_eval/distrep.py", "--dataset", dataset, "--backbone", backbone,
            "--seed", str(SEED), "--mode", "checkpoint", "--output-root", str(out_root)))
    return common.summarize(out_root / "DistrRep" / dataset / backbone / f"s{SEED}" / "predictions", dataset)


def run_prior_patch(script: str, method: str, dataset: str, backbone: str) -> dict:
    out_root = OUT_ROOT / method
    # no --tau: default is -inf (always route / unconditional patch), matching paper.tex's
    # ungated NNPatch/PatchNAS operating point -- see nnpatch.py/patchnas.py's --tau docstring.
    run(py(f"scripts/checkpoint_eval/{script}", "--dataset", dataset, "--backbone", backbone,
            "--seed", str(SEED), "--mode", "checkpoint",
            "--output-root", str(out_root)))
    summary = json.loads((out_root / method / dataset / backbone / f"s{SEED}" / "summary.json").read_text())
    return summary  # {"ungated": {...tau=-inf, always route...}, "natural": {...tau=0.5, RQ1 op point...}, "matched": {...Reg-matched diagnostic, unused...}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", default=None,
                     help="dataset/backbone pairs, e.g. gtsrb/resnet50. Default: all 12.")
    a = ap.parse_args()
    settings = [tuple(s.split("/")) for s in a.settings] if a.settings else ALL_SETTINGS

    reg_lookup: dict[str, float] = json.loads(REG_LOOKUP.read_text()) if REG_LOOKUP.exists() else {}
    rows: list[dict] = []  # one row per (method, setting): RR_seen/RR_held/Reg/CReg

    for dataset, backbone in settings:
        label = setting_label(dataset, backbone)
        print(f"\n=== {label} ({dataset}/{backbone}) ===")

        dp = run_dynapatch(dataset, backbone)
        reg_lookup[f"{dataset}/{backbone}"] = dp["ungated"]["reg"] if dp["ungated"] else 0.016
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REG_LOOKUP.write_text(json.dumps(reg_lookup, indent=2))
        rows.append({"method": "DynaPatch-NoGate", "setting": label, "dataset": dataset,
                      "backbone": backbone, **{k.upper(): v for k, v in (dp["ungated"] or {}).items()}})
        rows.append({"method": "DynaPatch", "setting": label, "dataset": dataset,
                      "backbone": backbone, **{k.upper(): v for k, v in (dp["gated"] or {}).items()}})

        for baseline, method in (("head_ft", "HeadFT"), ("full_ft", "FullFT")):
            s = run_headft(dataset, backbone, baseline, method)
            rows.append({"method": method, "setting": label, "dataset": dataset, "backbone": backbone,
                         **{k.upper(): v for k, v in (s or {}).items()}})

        s = run_arachne(dataset, backbone)
        rows.append({"method": "Arachne", "setting": label, "dataset": dataset, "backbone": backbone,
                     **{k.upper(): v for k, v in (s or {}).items()}})

        s = run_distrep(dataset, backbone)
        if s is not None:
            rows.append({"method": "DistrRep", "setting": label, "dataset": dataset, "backbone": backbone,
                         **{k.upper(): v for k, v in s.items()}})

        s = run_fixedpatch(dataset, backbone)
        if s is not None:
            rows.append({"method": "FixedPatch", "setting": label, "dataset": dataset, "backbone": backbone,
                         **{k.upper(): v for k, v in s.items()}})

        for script, method in (("nnpatch.py", "NNPatch"), ("patchnas.py", "PatchNAS")):
            s = run_prior_patch(script, method, dataset, backbone)
            rows.append({"method": f"{method} (always-route, ungated op point)", "setting": label,
                         "dataset": dataset, "backbone": backbone,
                         "RR_SEEN": s["ungated"]["RR_seen"], "RR_HELD": s["ungated"]["RR_held"],
                         "REG": s["ungated"]["Reg"], "CREG": s["ungated"]["CReg"]})
            rows.append({"method": f"{method} (tau=0.5 natural threshold, Table rq1_rr op point)", "setting": label,
                         "dataset": dataset, "backbone": backbone,
                         "RR_SEEN": s["natural"]["RR_seen"], "RR_HELD": s["natural"]["RR_held"],
                         "REG": s["natural"]["Reg"], "CREG": s["natural"]["CReg"]})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / "rq1_raw_cells.csv"
    fieldnames = ["method", "setting", "dataset", "backbone", "RR_SEEN", "RR_HELD", "REG", "CREG", "N_SEEN", "N_HELD", "N_CLEAN", "N_CRIT"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
