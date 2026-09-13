#!/usr/bin/env python3
"""How good is NN-Patching/PatchNAS's OWN error estimator as a classifier, next to our gate
transplanted onto their raw patch output (already reported by analysis_gate_cv_importance.py)?

Uses the IDENTICAL ground-truth label analysis_gate_cv_importance.py's pooled_baseline() scores
the transplanted gate against (gain in {-1,0,+1}, restricted to rows the raw patch actually
flips: +1 if applying it unconditionally corrects a held-out failure, -1 if it regresses a
clean-correct input, 0 otherwise) -- same label, so this is a same-question comparison, not a
new metric definition that could disagree with RQ3.8 for uninteresting reasons. Pools all 4
populations (bug_train, bug_eval, clean_calib, clean_test) across the 3 seeds per (dataset,
backbone), same as pooled_baseline().

Structural caveat, stated rather than hidden: their own estimator is a fitted BINARY router
(p(error) > tau -> apply the patch); it was never fit to separate "harmful" from "no-effect"
the way our 3-class gate is. So only binary "commit" metrics are reported here (AUC,
precision/recall/F1 at their own paper-default tau=0.5) -- there is no macro-F1(3-class) row
for their own estimator, and printing one would compare unlike things.

`auc_commit` is threshold-free (ranks by the raw estimator score) and is therefore the number
directly comparable to outputs/gate_cv_importance/cv_auc_per_cell.csv's `mean_auc_commit` for
method=NN-Patching/PatchNAS, feature_set="4 pre+post" (our gate refit on the same raw patch
output, same label, same pooled data).

    .venv/bin/python scripts/analysis_gate_own_estimator_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                                       # noqa: E402
from analysis_gate_transplant import load_pop, METHODS, SETTINGS, SEEDS  # noqa: E402

TAU_DEFAULT = 0.5


def pooled(method: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for ds, bb in SETTINGS:
        g_parts, es_parts = [], []
        for seed in SEEDS:
            dirp = ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/{method}/tau"
            for pop_name, kind in (("bug_train", "fail"), ("bug_eval", "fail"),
                                   ("clean_calib", "clean"), ("clean_test", "clean")):
                csv_path = dirp / f"{pop_name}_predictions.csv"
                pop = load_pop(dirp, pop_name)
                if pop is None or not csv_path.is_file():
                    continue
                es_all = pd.read_csv(csv_path)["estimator_score"].to_numpy()
                m = pop["flip"]
                if kind == "fail":
                    g_sub = np.where(pop["correct_after"][m], 1, 0).astype(int)
                else:
                    regressed = pop["base_correct"] & ~pop["correct_after"]
                    g_sub = np.where(regressed[m], -1, 0).astype(int)
                g_parts.append(g_sub)
                es_parts.append(es_all[m])
        if not g_parts:
            continue
        out[(ds, bb)] = {"g": np.concatenate(g_parts), "es": np.concatenate(es_parts)}
    return out


def cell_metrics(g: np.ndarray, es: np.ndarray, tau: float = TAU_DEFAULT) -> dict | None:
    y_bin = (g == 1).astype(int)
    if len(np.unique(y_bin)) < 2:
        return None
    pred_bin = (es > tau).astype(int)
    n = len(g)
    return {
        "n_pos": int((g == 1).sum()), "n_neg": int((g == -1).sum()), "n_zero": int((g == 0).sum()),
        "auc_commit": float(roc_auc_score(y_bin, es)),
        "accuracy_commit": float((pred_bin == y_bin).mean()),
        "error_rate_commit": float((pred_bin != y_bin).mean()),
        "precision_commit": float(precision_score(y_bin, pred_bin, zero_division=0)),
        "recall_commit": float(recall_score(y_bin, pred_bin, zero_division=0)),
        "f1_commit": float(f1_score(y_bin, pred_bin, zero_division=0)),
        "route_rate": float(pred_bin.mean()),
        # full 2 (decision) x 3 (true gain) confusion, so "success/error" has a visible
        # denominator instead of collapsing no-effect and harmful into one "negative" bucket
        "commit_on_beneficial": int(((pred_bin == 1) & (g == 1)).sum()),
        "commit_on_no_effect": int(((pred_bin == 1) & (g == 0)).sum()),
        "commit_on_harmful": int(((pred_bin == 1) & (g == -1)).sum()),
        "decline_on_beneficial": int(((pred_bin == 0) & (g == 1)).sum()),
        "decline_on_no_effect": int(((pred_bin == 0) & (g == 0)).sum()),
        "decline_on_harmful": int(((pred_bin == 0) & (g == -1)).sum()),
    }


def main() -> None:
    rows = []
    for method in METHODS:
        pb = pooled(method)
        for (ds, bb), d in pb.items():
            m = cell_metrics(d["g"], d["es"])
            if m is None:
                continue
            rows.append({"method": method, "setting": f"{ds}/{bb}", **m})
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "gate_own_estimator_quality"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_setting.csv", index=False)
    summary = df.groupby("method")[["auc_commit", "accuracy_commit", "error_rate_commit",
                                    "precision_commit", "recall_commit",
                                    "f1_commit", "route_rate"]].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    RM.write_section(
        "GateOwnEstimatorQuality",
        "NN-Patching/PatchNAS's own error estimator, scored as a classifier (raw)",
        f"""
Same ground-truth label as analysis_gate_cv_importance.py's `pooled_baseline()` (gain in
{{-1,0,+1}}, restricted to rows the raw patch actually flips): +1 if applying the patch
unconditionally corrects a held-out failure, -1 if it regresses a clean-correct input, 0
otherwise. Pools bug_train+bug_eval+clean_calib+clean_test across the 3 seeds per (dataset,
backbone) -- same pooling as the transplanted-gate table this is meant to sit next to.

Their own estimator is a fitted BINARY router (p(error) > tau -> apply), not a 3-class model
like our gate, so only binary "commit" metrics are reported: `auc_commit` (threshold-free,
directly comparable to `outputs/gate_cv_importance/cv_auc_per_cell.csv`'s `mean_auc_commit`
for the same method/setting, feature_set="4 pre+post") and precision/recall/F1 at their own
paper-default operating point, tau={TAU_DEFAULT}. There is no macro-F1(3-class) analogue --
their estimator was never fit to separate "harmful" from "no-effect", only "apply or not".
""",
        [("per_setting", df), ("summary", summary)],
    )
    print(df.to_string(index=False))
    print("\nSummary (mean over settings):")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
