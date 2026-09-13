#!/usr/bin/env python3
"""Is pre+post really better than pre, and which features drive the gate's decision?

Two things Protocol C's single fixed train/test split cannot answer on its own (concern: the
sample count may simply be too small -- the split's test-side positive count is 6-41 for
GTSRB/TT100K and 0-5 for LISA, per scripts/gate_protocol_b.py's own documented limitation):

1. **Cross-validated AUC, pre-only vs pre+post, paired per fold.** Pools ALL flipped rows
   available for a (dataset, backbone) across its 3 seeds (not just Protocol C's one train/test
   split), then runs stratified k-fold CV with the SAME fold splits for both feature sets, so
   the pre-vs-pre+post AUC difference is measured on matched data, not confounded by which rows
   happened to land in which side of one fixed split.
2. **Feature importance.** `fit_gain`'s model is `StandardScaler -> LogisticRegression`
   (3-class, gain in {-1,0,+1}); the standardization means coefficients are already on a
   comparable scale. Reports the mean |coefficient| for the gain=+1 (commit) row, averaged
   across every (dataset, backbone) fit, ranked -- which features the gate actually leans on.

Both run for DynaPatch's own gate (the one that matters for the paper) AND for the gate-transplant
experiment (NN-Patching/PatchNAS, reusing scripts/analysis_gate_transplant.py's data loading) --
so "does post-info actually help, robustly" and "which features matter" are both checked for all
three methods, not just asserted from the single-split numbers already in RQ3.6/RQ3.7.

Caveat carried over from scripts/gate_protocol_b.py: this still is not a LOBO or truly held-out
generalization test -- CV folds are drawn from the SAME (dataset, backbone)'s own rows, pooled
across seeds. It answers "is the pre+post advantage stable within this cell's own data," not
"does it generalize to an unseen backbone" (that is what LOBO, scripts/gate_zoo.py, is for).

    .venv/bin/python scripts/analysis_gate_cv_importance.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                                                    # noqa: E402
from probe_gonogo_pre_vs_prepost import build_cells, fit_gain, score, mat, LAYERS  # noqa: E402
from analysis_gate_transplant import load_pop, METHODS as BASELINE_METHODS, SETTINGS, SEEDS  # noqa: E402

FEATURE_SETS = ["2 pre-strong", "4 pre+post"]
N_SPLITS = 5
MIN_PER_CLASS = 6          # need at least this many of the rarer class to attempt 5-fold CV


def cv_metrics(X: dict, g: np.ndarray, feats: list[str], rng_seed: int) -> list[dict]:
    """Stratified k-fold, both readings of the same 3-class model per fold:

    - 3-class: `mdl.predict()`'s own argmax over {-1,0,+1}, scored as macro-F1/accuracy over
      all three labels -- describes the model on the task it was actually fit on.
    - binary "commit or not": predicted_positive = (predicted class == 1), scored against
      actual gain==1 with precision/recall/F1, plus AUC ranking `score` = P(+1)-P(-1) against
      gain==1 -- describes the one decision that is actually deployed (apply the patch or not).
    """
    pos_ct, other_ct = int((g == 1).sum()), int((g != 1).sum())
    k = min(N_SPLITS, pos_ct, other_ct)
    if k < 2:
        return []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rng_seed)
    Xmat = mat(X, feats)
    out = []
    for tr, te in skf.split(Xmat, g):
        mdl = fit_gain(Xmat[tr], g[tr])
        if mdl is None:
            continue
        y_te = g[te]
        if len(np.unique(y_te)) < 2:
            continue
        pred_te = mdl.predict(Xmat[te])
        s_te = score(mdl, Xmat[te])
        y_bin = (y_te == 1).astype(int)
        pred_bin = (pred_te == 1).astype(int)
        row = {
            "macro_f1_3class": float(f1_score(y_te, pred_te, average="macro", zero_division=0)),
            "accuracy_3class": float((pred_te == y_te).mean()),
            "accuracy_commit": float((pred_bin == y_bin).mean()),
            "error_rate_commit": float((pred_bin != y_bin).mean()),
            "precision_commit": float(precision_score(y_bin, pred_bin, zero_division=0)),
            "recall_commit": float(recall_score(y_bin, pred_bin, zero_division=0)),
            "f1_commit": float(f1_score(y_bin, pred_bin, zero_division=0)),
        }
        if len(np.unique(y_bin)) == 2:
            row["auc_commit"] = float(roc_auc_score(y_bin, s_te))
        out.append(row)
    return out


def pooled_dynapatch() -> dict:
    """Per (ds, bb): pooled flipped-row features/gain across all 3 seeds' clean+held rows."""
    cells = build_cells()
    out: dict[tuple[str, str], dict] = {}
    for (seed, ds, bb), v in cells.items():
        key = (ds, bb)
        c, h = v["clean"], v["held"]
        parts = out.setdefault(key, {"f": None, "g": []})
        for side in (c, h):
            m = side["flip"]
            f_sub = {kk: vv[m] for kk, vv in side["f"].items()}
            g_sub = side["gain"][m]
            if parts["f"] is None:
                parts["f"] = {kk: [vv] for kk, vv in f_sub.items()}
            else:
                for kk, vv in f_sub.items():
                    parts["f"][kk].append(vv)
            parts["g"].append(g_sub)
    result = {}
    for key, parts in out.items():
        f = {kk: np.concatenate(vv) for kk, vv in parts["f"].items()}
        g = np.concatenate(parts["g"])
        result[key] = {"f": f, "g": g}
    return result


def pooled_baseline(method: str) -> dict:
    """Per (ds, bb): pooled flipped-row features/gain across all 3 seeds x 4 populations."""
    out: dict[tuple[str, str], dict] = {}
    for ds, bb in SETTINGS:
        parts = {"f": None, "g": []}
        any_data = False
        for seed in SEEDS:
            dirp = ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/{method}/tau"
            for pop_name, kind in (("bug_train", "fail"), ("bug_eval", "fail"),
                                   ("clean_calib", "clean"), ("clean_test", "clean")):
                pop = load_pop(dirp, pop_name)
                if pop is None:
                    continue
                any_data = True
                m = pop["flip"]
                if kind == "fail":
                    g_sub = np.where(pop["correct_after"][m], 1, 0).astype(int)
                else:
                    regressed = pop["base_correct"] & ~pop["correct_after"]
                    g_sub = np.where(regressed[m], -1, 0).astype(int)
                f_sub = {kk: vv[m] for kk, vv in pop["f"].items()}
                if parts["f"] is None:
                    parts["f"] = {kk: [vv] for kk, vv in f_sub.items()}
                else:
                    for kk, vv in f_sub.items():
                        parts["f"][kk].append(vv)
                parts["g"].append(g_sub)
        if not any_data:
            continue
        out[(ds, bb)] = {"f": {kk: np.concatenate(vv) for kk, vv in parts["f"].items()},
                         "g": np.concatenate(parts["g"])}
    return out


def importance_rows(method: str, pooled: dict) -> list[dict]:
    feats = LAYERS["4 pre+post"]
    rows = []
    for (ds, bb), d in pooled.items():
        Xmat = mat(d["f"], feats)
        g = d["g"]
        if (g == 1).sum() < 2 or (g == -1).sum() < 2:
            continue
        mdl = fit_gain(Xmat, g)
        if mdl is None:
            continue
        lr = mdl.named_steps["logisticregression"]
        classes = list(lr.classes_)
        if 1 not in classes:
            continue
        coef_row = lr.coef_[classes.index(1)]
        for feat, c in zip(feats, coef_row):
            rows.append({"method": method, "setting": f"{ds}/{bb}", "feature": feat,
                        "coef_gain1": float(c)})
    return rows


def main() -> None:
    cv_rows = []
    imp_rows = []

    dp = pooled_dynapatch()
    imp_rows += importance_rows("DynaPatch", dp)
    for (ds, bb), d in dp.items():
        for fset in FEATURE_SETS:
            for i, m in enumerate(cv_metrics(d["f"], d["g"], LAYERS[fset], rng_seed=0)):
                cv_rows.append({"method": "DynaPatch", "setting": f"{ds}/{bb}",
                              "feature_set": fset, "fold": i, **m,
                              "n_pos": int((d["g"] == 1).sum()), "n_neg": int((d["g"] == -1).sum()),
                              "n_zero": int((d["g"] == 0).sum())})

    for method in BASELINE_METHODS:
        pb = pooled_baseline(method)
        imp_rows += importance_rows(method, pb)
        for (ds, bb), d in pb.items():
            for fset in FEATURE_SETS:
                for i, m in enumerate(cv_metrics(d["f"], d["g"], LAYERS[fset], rng_seed=0)):
                    cv_rows.append({"method": method, "setting": f"{ds}/{bb}",
                                  "feature_set": fset, "fold": i, **m,
                                  "n_pos": int((d["g"] == 1).sum()), "n_neg": int((d["g"] == -1).sum()),
                                  "n_zero": int((d["g"] == 0).sum())})

    cv = pd.DataFrame(cv_rows)
    imp = pd.DataFrame(imp_rows)

    # per-(method,setting) mean metrics per feature set, then the paired delta
    cv_cell = cv.groupby(["method", "setting", "feature_set"], as_index=False).agg(
        mean_auc_commit=("auc_commit", "mean"), mean_f1_commit=("f1_commit", "mean"),
        mean_accuracy_commit=("accuracy_commit", "mean"), mean_error_rate_commit=("error_rate_commit", "mean"),
        mean_precision_commit=("precision_commit", "mean"), mean_recall_commit=("recall_commit", "mean"),
        mean_macro_f1_3class=("macro_f1_3class", "mean"), mean_accuracy_3class=("accuracy_3class", "mean"),
        n_folds=("f1_commit", "size"),
        n_pos=("n_pos", "first"), n_neg=("n_neg", "first"), n_zero=("n_zero", "first"))

    imp_cell = imp.groupby(["method", "feature"], as_index=False).agg(
        mean_abs_coef=("coef_gain1", lambda s: float(np.abs(s).mean())),
        mean_coef=("coef_gain1", "mean"), n_settings=("coef_gain1", "size"))
    imp_cell = imp_cell.sort_values(["method", "mean_abs_coef"], ascending=[True, False])

    RM.write_section(
        "GateCV", "Gate cross-validation and feature importance (raw)",
        f"""
Two checks, both for DynaPatch's own gate and the gate-transplant experiment
(NN-Patching/PatchNAS), addressing whether the single-fixed-split RQ3/RQ3.6 numbers are stable
given small failure counts (`scripts/gate_protocol_b.py`'s own documented D_gate size: 6-41
positives for GTSRB/TT100K, 0-5 for LISA).

**Cross-validated metrics** (`cv_auc` table, despite the name -- kept for continuity with the
earlier AUC-only version): stratified k-fold (k = min(5, positive count, other count)) on ALL
flipped rows pooled across a (dataset, backbone)'s 3 seeds -- more data per fold than Protocol
C's one fixed split, same fold splits used for both feature sets (paired). The underlying model
is **3-class** (`gain in {{-1,0,+1}}`, harmful/no-effect/beneficial; `StandardScaler ->
LogisticRegression`, multinomial). Two readings of the same fitted model, both reported:
- `mean_macro_f1_3class` / `mean_accuracy_3class`: the model scored on the 3-class task it was
  actually fit on (macro-F1 averages the per-class F1 equally, so the rare `-1`/harmful class is
  not swamped by the common classes).
- `mean_auc_commit` / `mean_precision_commit` / `mean_recall_commit` / `mean_f1_commit`: the
  same model collapsed to the **binary** decision that is actually deployed -- commit
  (predicted class == +1) or not. AUC ranks by `score` = P(gain=+1) - P(gain=-1); precision/
  recall/F1 use the model's own `predict()` argmax, not a re-tuned threshold.
This is still within-cell CV (folds share the same dataset/backbone/seeds pooled), not a
held-out-backbone generalization test -- that is LOBO (`scripts/gate_zoo.py`), not reproduced
here.

**Feature importance**: `importance_summary` averages the standardized logistic-regression
coefficient for the gain=+1 row across every (dataset, backbone) fit, **on the "4 pre+post"
feature set specifically** (all 12 features, not a subset) -- so this describes what the
FULL, shipped-style gate leans on, not the pre-only ablation. `importance_by_setting` is the
same coefficient, NOT averaged -- one row per (method, setting, feature), so a reader can see
whether a feature's importance is stable across settings or driven by one or two of them.
Features are standardized upstream (`StandardScaler` inside `fit_gain`), so coefficients are
on a comparable scale across features.
""",
        [("cv_auc_by_setting", cv_cell.sort_values(["method", "setting", "feature_set"])),
         ("importance_summary", imp_cell),
         ("importance_by_setting", imp.sort_values(["method", "setting", "feature"]))],
    )

    out = ROOT / "outputs" / "gate_cv_importance"
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "cv_auc_per_fold.csv", index=False)
    cv_cell.to_csv(out / "cv_auc_per_cell.csv", index=False)
    imp.to_csv(out / "importance_per_setting.csv", index=False)
    imp_cell.to_csv(out / "importance_summary.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/*.csv")

    print("\nCV metrics, mean over settings, per method x feature_set:")
    print(cv_cell.groupby(["method", "feature_set"])[
        ["mean_macro_f1_3class", "mean_accuracy_3class", "mean_auc_commit",
         "mean_precision_commit", "mean_recall_commit", "mean_f1_commit"]
    ].mean().round(4).to_string())

    print("\nFeature importance (mean |coef|), FULL ranking, feature set = 4 pre+post:")
    for m in imp_cell.method.unique():
        print(f"\n  {m}:")
        print(imp_cell[imp_cell.method == m][["feature", "mean_abs_coef", "mean_coef",
                                              "n_settings"]].to_string(index=False))


if __name__ == "__main__":
    main()
