#!/usr/bin/env python3
"""Freeze the artefact set every paper table is allowed to read.

Why this exists
---------------
Numbers in this project have moved for reasons that had nothing to do with the method: a row
silently pointing at an under-trained tree, a baseline optimising against its own reporting
split, a re-implementation with no torch seed, a sweep whose grid was truncated. Each was found
by accident. This script makes the input set explicit and checkable instead.

It does three things and nothing else:

  1. RESOLVE   For every reported row, list the exact prediction CSVs that feed it.
  2. VERIFY    Every (setting, seed) cell present? Same seed count across rows? Nothing empty?
  3. RECORD    sha256 of every file, into outputs/FROZEN.json.

Re-running it later re-checks the recorded hashes: any file that changed underneath a frozen
table is reported as DRIFT. That is the whole point -- "the numbers moved and nobody knows when"
stops being possible.

This does NOT decide which configuration a method is reported at; scripts/names.py does, and the
tree templates live in scripts/table_rq4_final.py. This only pins what is on disk.

    .venv/bin/python scripts/freeze_results.py            # verify against outputs/FROZEN.json
    .venv/bin/python scripts/freeze_results.py --write    # (re)write the freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import names as N  # noqa: E402
from table_rq4_final import FIXED_ROWS, SWEEPS  # noqa: E402

FROZEN = ROOT / "outputs" / "FROZEN.json"
SEEDS = (101, 202, 303)
SPLITS = ("repair_holdout_unseen", "clean_eval")   # the two RR_held / Reg are read from


def cells(template: str, seeds: tuple[int, ...] | None = None) -> dict[str, dict]:
    """(setting, seed) -> {split: (relpath, sha256, nbytes)} for every cell that exists.

    `seeds` narrows the freeze to the seeds a row is REPORTED at, so cells left on disk by an
    interrupted sweep are not silently pinned into a row that does not use them.
    """
    out: dict[str, dict] = {}
    for ds, bb in ((d, b) for d in ("gtsrb", "tt100k_signs", "lisa_signs")
                   for b in ("resnet50", "convnext_tiny", "densenet121", "vgg16")):
        for s in (seeds or SEEDS):
            d = ROOT / template.format(s=s, ds=ds, bb=bb)
            got = {}
            for sp in SPLITS:
                f = d / f"{sp}_predictions.csv"
                if f.is_file() and f.stat().st_size > 0:
                    got[sp] = [str(f.relative_to(ROOT)),
                               hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size]
            if len(got) == len(SPLITS):
                out[f"{ds}/{bb}|s{s}"] = got
    return out


def resolve() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in FIXED_ROWS:
        name, kind, tpl = row[0], row[1], row[2]
        rows[name] = {"kind": kind, "seeds": list(row[3]) if len(row) > 3 else list(SEEDS),
                      "variants": {"-": cells(tpl, row[3] if len(row) > 3 else None)}}
    for name, variants in SWEEPS:
        rows[name] = {"kind": "sweep",
                      "variants": {lbl: cells(tpl) for lbl, tpl in variants}}
    return rows


def verify(rows: dict, prev: dict | None) -> int:
    bad = 0
    print(f"{'row':44s} {'variants':>9s} {'cells':>7s} {'seeds':>6s}  status")
    print("-" * 92)
    for name, r in sorted(rows.items()):
        allc = {c for v in r["variants"].values() for c in v}
        seeds = {c.split("|")[1] for c in allc}
        stgs = {c.split("|")[0] for c in allc}
        # a row is COMPLETE when every setting it covers has the same seed count
        per = {s: sum(1 for c in allc if c.startswith(s + "|")) for s in stgs}
        note = []
        if len(stgs) < 12:
            note.append(f"only {len(stgs)}/12 settings")
        if len(set(per.values())) > 1:
            note.append(f"ragged seeds {sorted(set(per.values()))}")
        if not allc:
            note.append("EMPTY")
        st = "ok" if not note else "; ".join(note)
        bad += bool(note)
        print(f"{name[:44]:44s} {len(r['variants']):9d} {len(allc):7d} "
              f"{','.join(sorted(seeds)):>6s}  {st}")
    if prev:
        print("\n--- drift against the recorded freeze ---")
        drift = 0
        for name, r in rows.items():
            p = prev.get("rows", {}).get(name)
            if p is None:
                print(f"  NEW ROW      {name}"); drift += 1; continue
            for lbl, cs in r["variants"].items():
                pc = p["variants"].get(lbl, {})
                for cell, files in cs.items():
                    if cell not in pc:
                        print(f"  NEW CELL     {name} [{lbl}] {cell}"); drift += 1; continue
                    for sp, meta in files.items():
                        if pc[cell].get(sp, [None, None])[1] != meta[1]:
                            print(f"  CHANGED      {name} [{lbl}] {cell} {sp}"); drift += 1
                for cell in pc:
                    if cell not in cs:
                        print(f"  DISAPPEARED  {name} [{lbl}] {cell}"); drift += 1
        print(f"  {drift} drifted" if drift else "  no drift: every frozen file is byte-identical")
        bad += drift
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="(re)write outputs/FROZEN.json")
    a = ap.parse_args()
    rows = resolve()
    prev = json.loads(FROZEN.read_text()) if FROZEN.exists() and not a.write else None
    bad = verify(rows, prev)
    if a.write:
        nfiles = sum(len(f) for r in rows.values() for v in r["variants"].values()
                     for f in v.values())
        FROZEN.write_text(json.dumps({"rows": rows, "n_files": nfiles}, indent=1))
        print(f"\n[written] {FROZEN.relative_to(ROOT)}  ({nfiles} files pinned)")
        if bad:
            print("WARNING: froze an INCOMPLETE set -- the rows flagged above are still moving.")
    raise SystemExit(0 if a.write else min(bad, 1))


if __name__ == "__main__":
    main()
