#!/usr/bin/env python3
"""RQ3 (comparison with existing methods): extract the two "why do others win/lose" tables from
the ALREADY-PUBLISHED, last_affine-corrected markdown tables in note/RQ2_BASELINE_BEHAVIOR.md
(recomputed end-to-end 2026-09-05 per that file's own header note) -- no new computation.

T3 (failure-type coverage/strength, section 3): merges "Method | Estimable settings | RR held |
Coverage@.5 | Type-RR P10 | Type-RR median" with "Method | Coverage@.5 | Partial types
(0<RR<1) | Zero-repair types | Fully repaired types" (same Coverage@.5 column, sanity-checked
equal before merging) -- explains why FullFT is strong (high coverage, mostly fully-repaired
types) and why Arachne/DistrRep are weak (low coverage, mostly zero-repair types).

T4 (regression distribution, section 4): "Method | Mean Reg | Mean worst-class Reg | Fraction of
clean classes hit" -- explains why some methods "look safe" on mean Reg alone but are not:
Arachne has low mean Reg but high worst-class Reg (localized spikes); DistrRep is worst on all
three (broad, diffuse damage); DynaPatch's gate is lowest on all three.

    .venv/bin/python scripts/extract_rq3_baseline_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "note" / "RQ2_BASELINE_BEHAVIOR.md"


def _table_after(lines: list[str], header_line: str) -> pd.DataFrame:
    start = next(i for i, l in enumerate(lines) if l.strip() == header_line)
    rows, header = [], None
    for line in lines[start:]:
        if not line.strip().startswith("|"):
            if header is not None:
                break
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(cells)
    return pd.DataFrame(rows, columns=header)


def main() -> None:
    lines = NOTE.read_text().splitlines()

    t3a = _table_after(lines, "| Method | Estimable settings | RR held | Coverage@.5 | "
                              "Type-RR P10 | Type-RR median |")
    for c in ("RR held", "Coverage@.5", "Type-RR P10", "Type-RR median"):
        t3a[c] = t3a[c].astype(float)

    t3b = _table_after(lines, "| Method | Coverage@.5 | Partial types (`0<RR<1`) | "
                              "Zero-repair types | Fully repaired types |")
    for c in ("Coverage@.5", "Partial types (`0<RR<1`)", "Zero-repair types",
             "Fully repaired types"):
        t3b[c] = t3b[c].astype(float)

    merged = t3a.merge(t3b[["Method", "Partial types (`0<RR<1`)", "Zero-repair types",
                            "Fully repaired types"]], on="Method")
    mismatch = (merged["Coverage@.5"] - t3b.set_index("Method").loc[merged.Method,
               "Coverage@.5"].values).abs()
    assert (mismatch < 1e-6).all(), "Coverage@.5 disagrees between the two source tables"
    merged.columns = [c.lower().replace(" ", "_").replace("@.5", "_at_0.5")
                      .replace("(`0<rr<1`)", "").replace("`", "").strip("_")
                      for c in merged.columns]
    merged.to_csv(ROOT / "outputs" / "rq3_baseline_failure_type.csv", index=False)
    print(f"T3: {len(merged)} methods -> outputs/rq3_baseline_failure_type.csv")
    print(merged.to_string(index=False))

    t4 = _table_after(lines, "| Method | Mean Reg | Mean worst-class Reg | "
                             "Fraction of clean classes hit |")
    for c in ("Mean Reg", "Mean worst-class Reg", "Fraction of clean classes hit"):
        t4[c] = t4[c].astype(float)
    t4.columns = [c.lower().replace(" ", "_") for c in t4.columns]
    t4.to_csv(ROOT / "outputs" / "rq3_baseline_regression_profile.csv", index=False)
    print(f"\nT4: {len(t4)} methods -> outputs/rq3_baseline_regression_profile.csv")
    print(t4.to_string(index=False))


if __name__ == "__main__":
    main()
