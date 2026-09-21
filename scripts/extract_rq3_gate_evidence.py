#!/usr/bin/env python3
"""RQ3: DynaPatch's own gate, pre-only vs pre+post evidence -- classification quality
(accuracy/precision/recall/f1) and deployed RR_held/Reg/CReg, at the natural threshold
(theta=0, "both" protocol, 9 honest features for pre+post).

Replaces the old note/RQ3_DATA.md-derived extract_rq2_gate_dynapatch.py: that note was a
frozen snapshot written before the 2026-09-07 rank-feature-leakage fix (see gate.py's
FEATURE_NAMES / gate_zoo.py's LEARNED["L4 pre+post"]) and never refreshed, so its numbers no
longer match the paper. This script re-derives both evidence arms directly from
gate_protocol_b.py --emit-natural-point, which IS regenerable end-to-end from the raw
effect_dump* trees, and pools per (dataset, backbone) exactly like every other RQ table here.

"pre" uses the "L2 pre-strong" feature set (pB_max, pB_margin, H_base, m_base_r, dnorm_r --
still carries the 2 within-split rank features; that arm was explicitly NOT part of the
2026-09-07 fix, see gate_zoo.py's own header note, so this is a secondary/framing comparison,
not the paper's headline number). "pre+post" is the shipped gate (names.SHIPPED_GATE).

    .venv/bin/python scripts/extract_rq3_gate_evidence.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ARMS = [("pre", "L2 pre-strong"), ("pre+post", None)]  # None = --features default (shipped)


def run_arm(features: str | None, out_csv: Path) -> pd.DataFrame:
    cmd = [sys.executable, str(ROOT / "scripts" / "gate_protocol_b.py"),
           "--emit-natural-point", str(out_csv), "--out", "/dev/null"]
    if features:
        cmd += ["--features", features]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    d = pd.read_csv(out_csv)
    d[["dataset", "backbone"]] = d.setting.str.split("/", n=1, expand=True)
    return d.groupby(["dataset", "backbone", "setting"], as_index=False)[
        ["RR_held", "Reg", "CReg", "accuracy", "precision", "recall", "f1"]].mean()


def main() -> None:
    tmp = ROOT / "outputs" / "_tmp_gate_evidence"
    tmp.mkdir(parents=True, exist_ok=True)
    frames = []
    for evidence, features in ARMS:
        d = run_arm(features, tmp / f"{evidence.replace('+', '_')}.csv")
        d["evidence"] = evidence
        frames.append(d)
    all_ = pd.concat(frames, ignore_index=True)

    clf = all_[["setting", "evidence", "accuracy", "precision", "recall", "f1"]]
    clf_out = ROOT / "outputs" / "rq3" / "gate_classification.csv"
    clf.to_csv(clf_out, index=False)

    dep = all_.rename(columns={"RR_held": "rr_held", "Reg": "reg", "CReg": "creg"})[
        ["setting", "evidence", "rr_held", "reg", "creg"]]
    dep_out = ROOT / "outputs" / "rq3" / "gate_deployment.csv"
    dep.to_csv(dep_out, index=False)

    for name, arm in (("classification", clf), ("deployment", dep)):
        print(f"\n{name} mean by evidence:")
        print(arm.groupby("evidence").mean(numeric_only=True))
    print(f"\nwrote {clf_out}\nwrote {dep_out}")

    for f in tmp.glob("*.csv"):
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
