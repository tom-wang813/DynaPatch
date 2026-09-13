#!/usr/bin/env python3
"""P5 -- is Arachne narrow because of its idea, or because of its budget?

Arachne's bidirectional localisation returns the Pareto front of (forward impact, gradient loss).
Measured over 84 shipped cells that front is **1 to 14 weight positions**; seven cells can move a
single weight. `--num-places` is a CAP on the front, so raising it changes nothing -- which is why
an earlier probe swept it and found no effect.

`--localise topn` removes the Pareto step and keeps the top-N positions under the SAME
bi-objective score. The front is a prefix-subset of that ranking, so this only ever ADDS capacity,
and N equal to the front's own size reproduces `pareto` byte-for-byte (verified in the sweep's
smoke gate: identical `best_fitness`, identical prediction CSVs).

    N = 16 / 64 / 256 / 1024      ~1.1x to ~1000x the front

Every N is swept over the FULL bound_scale grid, the same grid the reported Arachne row uses.
Cutting it for the relaxed arm would handicap the very control this exists to be honest about.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
It answers: does giving Arachne more free parameters make it better? If not, our advantage over
it is not a capacity advantage -- which is the reviewer's "isn't this just an implementation
difference?" question.

It does NOT make the relaxed arm Arachne. **These rows must be reported as
"Arachne (relaxed localisation, N=...)" and never as Arachne.** The published algorithm is the
`pareto` row.

Raw output only: one row per (arm, bound_scale, setting). Selecting a bound_scale is left to the
reader, because the table's own rule (per-setting max RR_held, deliberately favourable to the
baseline) is a selection on the reported metric and should be applied knowingly, not buried here.

    .venv/bin/python scripts/analysis_p5_arachne_capacity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402

SETTINGS = [(d, b) for d in ("gtsrb", "tt100k_signs", "lisa_signs")
            for b in ("resnet50", "convnext_tiny", "densenet121", "vgg16")]
BOUNDS = ("2.0", "4.0", "8.0", "16.0", "32.0", "64.0", "128.0")
NS = (16, 64, 256, 1024)
SEED = 101


def read_cell(d: Path) -> dict | None:
    p = d / "predictions"
    try:
        held = pd.read_csv(p / "repair_holdout_unseen_predictions.csv")
        clean = pd.read_csv(p / "clean_eval_predictions.csv")
    except (FileNotFoundError, OSError):
        return None
    for t in (held, clean):
        for c in ("base_correct", "patched_correct"):
            if c in t:
                t[c] = t[c].astype(str).str.lower() == "true"
    m = {}
    f = d / "metrics.json"
    if f.is_file():
        try:
            m = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    fail = held[~held.base_correct]
    return {"n_held": len(held), "n_held_fail": len(fail),
            "RR_held": float(fail.patched_correct.mean()) if len(fail) else float("nan"),
            "n_clean": len(clean),
            "Reg": float((~clean.patched_correct).mean()) if len(clean) else float("nan"),
            "n_regressed": int((clean.base_correct & ~clean.patched_correct).sum()),
            "num_places": m.get("num_places"), "iterations": m.get("iterations"),
            "best_fitness": m.get("best_fitness"),
            "repair_seconds": m.get("repair_seconds")}


def main() -> None:
    rows = []
    for ds, bb in SETTINGS:
        for bs in BOUNDS:
            for arm, tree in ([("pareto", f"arachne_ct_bs{bs}_v8_s{SEED}_kfull")]
                              + [(f"topn{n}", f"arachneTopN{n}_bs{bs}_v8_s{SEED}_kfull")
                                 for n in NS]):
                r = read_cell(ROOT / "outputs" / tree / ds / bb)
                if r is None:
                    continue
                rows.append({"arm": arm,
                             "N_requested": None if arm == "pareto" else int(arm[4:]),
                             "bound_scale": float(bs), "setting": f"{ds}/{bb}",
                             "seed": SEED, **r})
    if not rows:
        raise SystemExit("no Arachne cells found")
    d = pd.DataFrame(rows)

    RM.write_section("P5", "P5 — Arachne capacity control (raw)", f"""
`pareto` is Arachne as published: the localised set is the Pareto front of (forward impact,
gradient loss). `topnN` removes the Pareto step and keeps the top N positions under the SAME
bi-objective score, so it is a strict superset and N equal to the front's size reproduces
`pareto` exactly.

**The `topn` rows are NOT Arachne.** Report them as "Arachne (relaxed localisation, N=...)".

Seed {SEED} only. One row per (arm, bound_scale, setting) — no bound_scale is selected here.
The paper table's rule is per-setting max RR_held, which is a selection on the reported metric
and favours the baseline; apply it knowingly rather than inheriting it from this file.

| column | meaning |
|---|---|
| `N_requested` | the cap asked for; blank for `pareto` |
| `num_places` | the weights the run ACTUALLY moved. On `pareto` this is the front's own size (1-14 across the shipped cells), which is why raising `--num-places` alone is a no-op |
| `RR_held` | repair rate among the deployed model's failures in `bug_eval` |
| `Reg` | error rate on `clean_eval`; `n_regressed` counts inputs that were right and became wrong |
| `best_fitness` / `iterations` / `repair_seconds` | the DE's own record, for checking that a larger search space did not simply fail to converge |

`iterations` matters here: `max_unchanged` early-stops the DE, so a relaxed arm that stops early
has not been given a fair chance at its larger space, and that is visible in this column rather
than having to be assumed.
""", [("", d.sort_values(["setting", "arm", "bound_scale"]))])

    out = ROOT / "outputs" / "p5_arachne_capacity"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "per_cell.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/per_cell.csv  ({len(d)} rows)")
    print(d.groupby("arm").agg(cells=("setting", "size"),
                               places=("num_places", "median"),
                               RR=("RR_held", "median"),
                               Reg=("Reg", "median")).round(4).to_string())


if __name__ == "__main__":
    main()
