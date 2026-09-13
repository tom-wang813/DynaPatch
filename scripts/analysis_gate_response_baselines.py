#!/usr/bin/env python3
"""Does the RQ3.3 confidence/entropy response pattern (successful repairs raise confidence and
lower entropy; regressions do the opposite) hold for NN-Patching/PatchNAS too?

RQ3.3 (`scripts/build_results_draft.py`'s `rq3()`, `note/RQ3_DATA.md`) reports, for DynaPatch's
own patches, three outcome groups' mean base/patched confidence, confidence change, and entropy
change: successful repair (flipped AND now correct), ineffective change (flipped but still
wrong), regression (a clean, base-correct input flipped to wrong). This script computes the
identical three groups and the identical four statistics for NN-Patching/PatchNAS's own raw
(always-apply) patch output, reusing `scripts/analysis_gate_transplant.py:load_pop`, which already
runs the same `build_features()` DynaPatch's own RQ3.3 pipeline uses (`pB_max`, `pP_max`, `dp_max`,
`dH` are all fields of the SAME function, not a reimplementation).

    .venv/bin/python scripts/analysis_gate_response_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                          # noqa: E402
from analysis_gate_transplant import load_pop, METHODS, SETTINGS, SEEDS  # noqa: E402

OUTCOME_ORDER = ["successful repair", "ineffective change", "regression"]


def main() -> None:
    rows = []
    for method in METHODS:
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                dirp = ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/{method}/tau"
                held = load_pop(dirp, "bug_eval")
                clean = load_pop(dirp, "clean_test")
                if held is None or clean is None:
                    rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                               "outcome": "-", "note": "missing per-sample artefact"})
                    continue
                groups = [
                    ("successful repair", held, held["flip"] & held["correct_after"]),
                    ("ineffective change", held, held["flip"] & ~held["correct_after"]),
                    ("regression", clean,
                     clean["flip"] & clean["base_correct"] & ~clean["correct_after"]),
                ]
                for outcome, data, mask in groups:
                    if not mask.any():
                        rows.append({"method": method, "setting": f"{ds}/{bb}", "seed": seed,
                                   "outcome": outcome, "note": "no rows in this outcome"})
                        continue
                    f = data["f"]
                    dp_max = f["dp_max"][mask]
                    rows.append({
                        "method": method, "setting": f"{ds}/{bb}", "seed": seed,
                        "outcome": outcome, "n": int(mask.sum()),
                        "base_confidence": float(f["pB_max"][mask].mean()),
                        "patched_confidence": float(f["pP_max"][mask].mean()),
                        "confidence_change": float(dp_max.mean()),
                        "entropy_change": float(f["dH"][mask].mean()),
                        "frac_confidence_increased": float((dp_max > 0).mean()),
                        "note": "",
                    })

    cell = pd.DataFrame(rows)
    ok = cell[cell.note.eq("")] if "note" in cell.columns else cell
    setting_agg = ok.groupby(["method", "outcome", "setting"], as_index=False).agg(
        base_confidence=("base_confidence", "mean"), patched_confidence=("patched_confidence", "mean"),
        confidence_change=("confidence_change", "mean"), entropy_change=("entropy_change", "mean"),
        frac_confidence_increased=("frac_confidence_increased", "mean"), n=("n", "sum"))
    overall = setting_agg.groupby(["method", "outcome"], as_index=False).agg(
        base_confidence=("base_confidence", "mean"), patched_confidence=("patched_confidence", "mean"),
        confidence_change=("confidence_change", "mean"), entropy_change=("entropy_change", "mean"),
        frac_confidence_increased=("frac_confidence_increased", "mean"),
        settings=("setting", "nunique"), n=("n", "sum"))
    order = {o: i for i, o in enumerate(OUTCOME_ORDER)}
    overall["order"] = overall.outcome.map(order)
    overall = overall.sort_values(["method", "order"]).drop(columns="order")

    RM.write_section(
        "GateResponseBaselines", "Confidence/entropy response by outcome — NN-Patching/PatchNAS, "
        "compared to RQ3.3's DynaPatch pattern (raw)",
        f"""
Same construction as `note/RQ3_DATA.md` RQ3.3 (three outcome groups: successful repair,
ineffective change, regression; same four statistics: base/patched confidence, confidence
change, entropy change), computed for NN-Patching/PatchNAS's own raw (always-apply) patch
output instead of DynaPatch's. `note` explains a dropped row (e.g. an outcome group with zero
members in a cell -- some settings have very few clean regressions or ineffective changes)
rather than silently omitting it.

**Setting-balanced overall** (mirrors the manuscript's Table rq3_response, DynaPatch column
for comparison; DynaPatch's own numbers are `note/RQ3_DATA.md` RQ3.3, not restated here):
""",
        [("overall", overall), ("by_setting", setting_agg.sort_values(["method", "outcome", "setting"]))],
    )

    out = ROOT / "outputs" / "gate_response_baselines"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_cell.csv", index=False)
    setting_agg.to_csv(out / "per_setting.csv", index=False)
    overall.to_csv(out / "overall.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/*.csv")
    print("\nSetting-balanced overall:")
    print(overall.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
