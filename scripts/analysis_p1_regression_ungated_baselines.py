#!/usr/bin/env python3
"""P1's per-class repair/regression breakdown (RR_macro, cov@tau, repair_spread, Reg_*, reg_spread),
computed for NN-Patching/PatchNAS's RAW, UNGATED patch output instead of their own routed one.

`analysis_p1_regression.py` reads `outputs/sample_frame.csv.gz`, where NN-Patching/PatchNAS appear
as "NN-Patching [tau]"/"[matched]" -- their OWN gate's routed predictions (`scripts/sample_frame.py:
prior_patches()`), gated at their own paper-default threshold. RQ1's whole framing is
input-specific-vs-fixed WITHOUT any gate (`note/RQ1_INPUT_SPECIFICITY.md` §5), so a per-class table
meant to sit alongside FixedPatch/DynaPatch-NoGate in RQ1 must use the same ungated framing for
NN-Patching/PatchNAS too -- otherwise "this baseline's per-class coverage is worse" would partly be
measuring THEIR OWN ESTIMATOR'S conservatism, exactly the confound RQ1 §5 excludes baselines to
avoid.

No new run needed: `outputs/prior_patch_persample/{setting}_s{seed}/{method}/tau/{pop}_predictions.csv`
already carries `patch_pred_raw` (argmax of patched_logits, unconditional on routing) alongside the
routed `patched_pred` -- this script rebuilds a `sample_frame`-compatible per-sample table from that
column and calls `analysis_p1_regression.py`'s OWN `per_cell()`/`per_class_rows()` (not
reimplemented) so the metric definitions are identical to every other method's row in this table.

    .venv/bin/python scripts/analysis_p1_regression_ungated_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                             # noqa: E402
from analysis_p1_regression import per_cell, per_class_rows       # noqa: E402
from analysis_gate_transplant import METHODS, SETTINGS, SEEDS     # noqa: E402

POP_TO_SPLIT = {"bug_eval": "held", "clean_test": "clean"}


def build_frame(method: str, ds: str, bb: str, seed: int) -> pd.DataFrame | None:
    dirp = ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/{method}/tau"
    parts = []
    for pop, split in POP_TO_SPLIT.items():
        f = dirp / f"{pop}_predictions.csv"
        if not f.is_file():
            continue
        t = pd.read_csv(f)
        base_correct = t.base_pred == t.label
        repaired = (t.patch_pred_raw == t.label) & ~base_correct
        regressed = (t.patch_pred_raw != t.label) & base_correct
        parts.append(pd.DataFrame({
            "split": split, "true_class": t.label, "base_correct": base_correct,
            "repaired": repaired, "regressed": regressed,
        }))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    recs = []
    class_rows = []
    for method in METHODS:
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                d = build_frame(method, ds, bb, seed)
                if d is None:
                    continue
                setting = f"{ds}/{bb}"
                r = per_cell(d)
                if r:
                    recs.append({"method": f"{method} [ungated]", "setting": setting,
                                "seed": seed, **r})
                class_rows.extend(per_class_rows(d, f"{method} [ungated]", setting, seed))

    cell = pd.DataFrame(recs)
    out = ROOT / "outputs" / "p1_regression_ungated_baselines"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_cell.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "per_class.csv", index=False)

    st = cell.groupby(["method", "setting"]).median(numeric_only=True).reset_index()
    st.to_csv(out / "per_setting.csv", index=False)

    cols = ["RR_macro", "RR_median_class", "cov0.3", "cov0.5", "cov0.7", "repair_spread",
           "n_repaired", "Reg_overall", "Reg_max_class", "reg_spread", "n_regressed"]
    summary = st.groupby("method")[cols].median().reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    RM.write_section(
        "P1UngatedBaselines",
        "P1's per-class breakdown, NN-Patching/PatchNAS at their RAW ungated (always-apply) "
        "output instead of their own gate (raw)",
        f"""
Same metric definitions as `scripts/analysis_p1_regression.py` (`per_cell()`/`per_class_rows()`,
reused not reimplemented), applied to a per-sample frame built from `patch_pred_raw` (argmax of
patched_logits, unconditional on the method's own routing) instead of the routed `patched_pred`
`outputs/sample_frame.csv.gz` carries. This is the ungated framing RQ1 uses throughout for
FixedPatch/DynaPatch-NoGate -- see `note/RQ1_INPUT_SPECIFICITY.md` §5 for why RQ1 needs baselines
at THIS operating point (their own gate's conservatism is a separate question, RQ2's).
""",
        [("per_setting", st), ("summary", summary)],
    )
    print(summary.round(4).to_string(index=False))
    print(f"\n[written] {out.relative_to(ROOT)}/{{per_cell,per_class,per_setting,summary}}.csv")


if __name__ == "__main__":
    main()
