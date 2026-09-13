#!/usr/bin/env python3
"""Shared emitter for the raw-data appendix.

The analyses in this project must ship their PER-CELL numbers -- one row per
(method, setting, seed) -- not medians, not win counts. Aggregation is the reader's job:
every aggregate in this repository that was computed before the raw table existed has at some
point hidden a pathological cell, a seed mismatch or a sign reversal.

Each analysis appends one section. `write_section` rewrites its own section in place, so
re-running a single analysis does not disturb the others.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "note" / "ANALYSIS_RAW.md"
HEAD = ("# Raw analysis data\n\n"
        "One row per (method, setting, seed). **No aggregation anywhere in this file** — no "
        "means, no medians, no win counts. Every number is the value measured in that one cell.\n\n"
        "Sections are written by `scripts/analysis_*.py`; re-running one replaces only its own "
        "section.\n")


def _cell(v, floatfmt: str) -> str:
    """One value -> one markdown cell. Everything ends up a str, whatever the column dtype.

    Typing per COLUMN is not enough: an object column holding None beside floats (a value that
    only some arms report, e.g. the requested capacity of a run that has no cap) leaves floats
    untouched and the row join then fails.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float):
        return floatfmt.format(v)
    return str(v)


def table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        d[c] = d[c].map(lambda v: _cell(v, floatfmt))
    out = ["| " + " | ".join(str(c) for c in d.columns) + " |",
           "|" + "---|" * len(d.columns)]
    out += ["| " + " | ".join(r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(out)


def write_section(key: str, title: str, note: str, blocks: list[tuple[str, pd.DataFrame]]) -> None:
    """Replace (or append) the section tagged `key`."""
    body = [f"<!--BEGIN {key}-->", f"## {title}", "", note.strip(), ""]
    for sub, df in blocks:
        if sub:
            body += [f"### {sub}", ""]
        body += [f"`{len(df)} rows`", "", table(df), ""]
    body += [f"<!--END {key}-->"]
    new = "\n".join(body)

    RAW.parent.mkdir(parents=True, exist_ok=True)
    cur = RAW.read_text() if RAW.exists() else HEAD
    pat = re.compile(rf"<!--BEGIN {re.escape(key)}-->.*?<!--END {re.escape(key)}-->", re.S)
    RAW.write_text(pat.sub(lambda _: new, cur) if pat.search(cur)
                   else cur.rstrip() + "\n\n" + new + "\n")
    print(f"[{key}] {sum(len(d) for _, d in blocks)} rows -> {RAW.relative_to(ROOT)}")
