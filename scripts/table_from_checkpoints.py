#!/usr/bin/env python3
"""RR_seen/RR_held/Reg/CReg summary table for DynaPatch, computed from the deploy-eval
predictions scripts/deploy_from_checkpoints.sh produces against the checkpoints listed in
artifacts/checkpoints/manifest.json. Same metric definitions as scripts/table_rq1_ablation.py:

    RR_seen  mean(patched_correct) over repair_support_seen
    RR_held  mean(patched_correct) over repair_holdout_unseen
    Reg      fraction of clean_eval that was correct pre-patch and wrong post-patch
    CReg     the same, restricted to the dataset's critical-class rows

Usage:
    bash scripts/deploy_from_checkpoints.sh
    uv run python scripts/table_from_checkpoints.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tf(x: object) -> bool:
    return str(x).lower() == "true"


def crit(ds: str) -> set[int]:
    cfg = json.loads((ROOT / f"artifacts/risk/{ds}_safety_risk_matrix.json").read_text())
    out: set[int] = set()
    for ids in cfg["critical_signs"].values():
        out.update(int(i) for i in ids)
    return out


def cell(out_root: Path, ds: str, bb: str, seed: int, cr: set[int]) -> dict | None:
    pred_dir = out_root / ds / bb / f"s{seed}" / "deploy" / "predictions"
    files = [pred_dir / f for f in (
        "repair_support_seen_predictions.csv",
        "repair_holdout_unseen_predictions.csv",
        "clean_eval_predictions.csv",
    )]
    if not all(f.exists() for f in files):
        return None
    sn, hd, cl = (list(csv.DictReader(f.open())) for f in files)
    cc = [r for r in cl if int(r["label"]) in cr]
    return {
        "seen": sum(_tf(r["patched_correct"]) for r in sn) / max(len(sn), 1),
        "held": sum(_tf(r["patched_correct"]) for r in hd) / max(len(hd), 1),
        "reg": sum(1 for r in cl if _tf(r["base_correct"]) and not _tf(r["patched_correct"])) / max(len(cl), 1),
        "creg": sum(1 for r in cc if _tf(r["base_correct"]) and not _tf(r["patched_correct"])) / max(len(cc), 1),
        "n_seen": len(sn), "n_held": len(hd), "n_clean": len(cl), "n_crit": len(cc),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/from_checkpoints",
                     help="Root that scripts/deploy_from_checkpoints.sh wrote predictions under.")
    ap.add_argument("--csv", default=None, help="Where to write the summary CSV "
                     "(default: <outdir>/summary.csv).")
    args = ap.parse_args()
    out_root = ROOT / args.outdir

    manifest = json.loads((ROOT / "artifacts/checkpoints/manifest.json").read_text())
    settings = sorted({(e["dataset"], e["backbone"]) for e in manifest["experiment_checkpoints"]})
    seeds = manifest["seeds"]

    per_setting: dict[tuple[str, str], list[dict]] = {}
    missing: list[str] = []
    for ds, bb in settings:
        cr = crit(ds)
        cells = []
        for seed in seeds:
            c = cell(out_root, ds, bb, seed, cr)
            if c is None:
                missing.append(f"{ds}/{bb} s{seed}")
            else:
                cells.append(c)
        per_setting[(ds, bb)] = cells

    print(f"{'setting':30}{'RR_seen':>10}{'RR_held':>10}{'Reg':>10}{'CReg':>10}{'seeds':>8}")
    out_rows = []
    for (ds, bb), cells in per_setting.items():
        label = f"{ds}/{bb}"
        if not cells:
            print(f"{label:30}  MISSING (run scripts/deploy_from_checkpoints.sh first)")
            continue
        mean = lambda k: st.mean(c[k] for c in cells)  # noqa: B023
        print(f"{label:30}{mean('seen'):>10.4f}{mean('held'):>10.4f}{mean('reg'):>10.4f}"
              f"{mean('creg'):>10.4f}{len(cells):>8}")
        out_rows.append({
            "setting": label, "RR_seen": mean("seen"), "RR_held": mean("held"),
            "Reg": mean("reg"), "CReg": mean("creg"), "n_seeds": len(cells),
        })

    if missing:
        print(f"\n{len(missing)} (setting, seed) cell(s) missing predictions -- run "
              "scripts/deploy_from_checkpoints.sh first (needs the checkpoints; see README.md "
              "\"Using checkpoints directly\"):")
        for m in missing:
            print(f"  {m}")

    if out_rows:
        csv_path = Path(args.csv) if args.csv else out_root / "summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\n[written] {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
