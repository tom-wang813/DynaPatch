#!/usr/bin/env python3
"""Build a complete four-RQ per-setting data pack from frozen local artifacts.

The script performs analysis and plotting only. It does not train models or initialize a GPU.
All headline values follow one reporting rule: average seeds within each setting, then average
the 12 settings. The draft uses the metrics defined by the Experimental Design: RR, Reg, CReg,
and accuracy/precision/recall/F1 for the gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "csv"
OUT = ROOT / "outputs" / "rq_results_draft"
FIG = ROOT / "note" / "figures" / "rq_results_draft"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402

from plot_rq3_confidence_distribution import plot_confidence_distribution  # noqa: E402


COLORS = {
    "blue": "#2F6F73",
    "orange": "#C66A45",
    "gold": "#78838E",
    "ink": "#2C3137",
    "gray": "#AAB0B7",
    "light": "#E8EBEF",
}

SETTING_LABEL = {
    "gtsrb/resnet50": "G-RN",
    "gtsrb/convnext_tiny": "G-CN",
    "gtsrb/densenet121": "G-DN",
    "gtsrb/vgg16": "G-VG",
    "tt100k_signs/resnet50": "T-RN",
    "tt100k_signs/convnext_tiny": "T-CN",
    "tt100k_signs/densenet121": "T-DN",
    "tt100k_signs/vgg16": "T-VG",
    "lisa_signs/resnet50": "L-RN",
    "lisa_signs/convnext_tiny": "L-CN",
    "lisa_signs/densenet121": "L-DN",
    "lisa_signs/vgg16": "L-VG",
}

# Presentation order requested for all 12-setting result tables. Keep this
# separate from SETTING_LABEL's insertion order so the exported tables cannot
# silently change when the label map is edited.
REQUESTED_SETTINGS = [
    ("gtsrb/resnet50", "G-RN"),
    ("gtsrb/convnext_tiny", "G-CN"),
    ("gtsrb/vgg16", "G-VG"),
    ("gtsrb/densenet121", "G-DN"),
    ("tt100k_signs/resnet50", "T-RN"),
    ("tt100k_signs/convnext_tiny", "T-CN"),
    ("tt100k_signs/vgg16", "T-VG"),
    ("tt100k_signs/densenet121", "T-DN"),
    ("lisa_signs/resnet50", "L-RN"),
    ("lisa_signs/convnext_tiny", "L-CN"),
    ("lisa_signs/vgg16", "L-VG"),
    ("lisa_signs/densenet121", "L-DN"),
]

DISPLAY = {
    "DynaPatch (ungated)": "DynaPatch-NoGate",
    "FixedPatch": "FixedPatch",
    "NN-Patching": "NN-Patching",
    "PatchNAS": "PatchNAS",
    "Arachne": "Arachne",
    "DistrRep": "DistRep",
    "HeadFT": "HeadFT",
    "FullFT": "FullFT",
    "DynaPatch (gated)": "DynaPatch (gated)",
}


def f3(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}"


def f4(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.4f}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def base_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "text.color": COLORS["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLORS["light"],
        "grid.linewidth": .7,
        "axes.axisbelow": True,
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate_sources() -> dict:
    expected_settings = set(SETTING_LABEL)

    inp = pd.read_csv(CSV / "inputspecific_per_cell.csv")
    assert len(inp) == 216
    assert not inp.duplicated(["method", "setting", "seed", "split"]).any()
    assert set(inp.setting) == expected_settings
    assert inp.rate.dropna().between(0, 1).all()

    best = pd.read_csv(CSV / "all_metrics_best_config.csv")
    required = set(DISPLAY)
    assert required.issubset(set(best.method))
    assert best[best.method.isin(required)].groupby("method").setting.nunique().eq(12).all()
    for col in ("RR_test", "Reg", "CReg"):
        assert best[col].dropna().between(0, 1).all()

    reas = pd.read_csv(ROOT / "outputs" / "patch_reassignment_v1" / "per_cell.csv")
    assert len(reas) == 216
    assert not reas.duplicated(["setting", "seed", "condition"]).any()
    assert float(reas.max_reconstruction_error.max()) <= 1e-10
    assert reas.RR_held_mean.between(0, 1).all()

    # gate_ablation_raw's curves_*.csv are the ungated (q=0) baseline reference ONLY -- no
    # threshold search reads them for a reported number any more. gate_ablation_ct/bal moved
    # entirely to the natural-point (theta=0) files (2026-09-05, user decision: drop the old
    # r-grid machinery for these too, not just the main Protocol C arm).
    for arm in ("pre", "post", "pre_post"):
        d = pd.read_csv(ROOT / "outputs" / "gate_ablation_raw" / f"curves_{arm}.csv")
        settings = {f"{x}/{y}" for x, y in zip(d.dataset, d.backbone)}
        assert settings == expected_settings
        assert d.RR_held.dropna().between(0, 1).all()
        assert d.Reg.dropna().between(0, 1).all()
    for folder in ("gate_ablation_ct_natural", "gate_ablation_bal_natural"):
        for arm in ("pre", "post", "pre_post"):
            d = pd.read_csv(ROOT / "outputs" / folder / f"natural_points_{arm}.csv")
            settings = set(d.setting)
            assert settings == expected_settings
            assert d.RR_held.dropna().between(0, 1).all()
            assert d.Reg.dropna().between(0, 1).all()

    gate = pd.read_csv(CSV / "gate_performance.csv")
    assert gate.setting.nunique() == 12
    for col in ("r0.90_accuracy", "r0.90_precision", "r0.90_recall", "r0.90_f1"):
        assert gate[col].dropna().between(0, 1).all()

    return {
        "inputspecific_rows": len(inp),
        "best_config_rows": len(best),
        "reassignment_rows": len(reas),
        "gate_performance_rows": len(gate),
        "settings": len(expected_settings),
        "max_reconstruction_error": float(reas.max_reconstruction_error.max()),
    }


def method_means() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(CSV / "all_metrics_best_config.csv")
    metrics = ["RR_seen", "RR_test", "Reg", "CReg"]
    setting = raw.groupby(["method", "setting"], as_index=False)[metrics].mean()
    overall = setting.groupby("method", as_index=False)[metrics].mean()
    return setting, overall


def rq1(setting: pd.DataFrame, overall: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    methods = ["DynaPatch (ungated)", "FixedPatch", "FullFT", "HeadFT",
               "NN-Patching", "PatchNAS", "Arachne", "DistrRep", "DynaPatch (gated)"]
    summary = overall[overall.method.isin(methods)].copy()
    summary["order"] = summary.method.map({name: i for i, name in enumerate(methods)})
    summary = summary.sort_values("order").drop(columns="order")
    summary.to_csv(OUT / "rq1_method_summary.csv", index=False)

    inp = pd.read_csv(CSV / "inputspecific_per_cell.csv")
    st = inp.groupby(["setting", "split", "method"], as_index=False).rate.mean()
    pair = st.pivot(index=["setting", "split"], columns="method", values="rate").reset_index()
    pair = pair.rename(columns={"DynaPatch (ungated)": "DynaPatch-NoGate", "FixedPatch": "FixedPatch"})
    pair["difference"] = pair["DynaPatch-NoGate"] - pair["FixedPatch"]
    sizes = inp.groupby(["setting", "split"], as_index=False).n.mean().rename(columns={"n": "mean_n"})
    pair = pair.merge(sizes, on=["setting", "split"], validate="one_to_one")
    pair.to_csv(OUT / "rq1_setting_pairs.csv", index=False)

    dp = summary[summary.method == "DynaPatch (ungated)"].iloc[0]
    fixed = summary[summary.method == "FixedPatch"].iloc[0]
    info = {
        "rr_difference": float(dp.RR_test - fixed.RR_test),
        "reg_difference": float(dp.Reg - fixed.Reg),
        "creg_difference": float(dp.CReg - fixed.CReg),
    }
    return summary, pair, info


def rq2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict,
                   pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ROOT / "outputs" / "patch_reassignment_v1" / "per_cell.csv")
    order = ["original", "global_effect_shuffle", "global_direction_shuffle",
             "global_magnitude_shuffle", "within_true_class_shuffle", "within_failure_type_shuffle"]
    effect = raw.groupby("condition", as_index=False).agg(
        RR=("RR_held_mean", "mean"), moved_fraction=("moved_fraction", "mean"))
    original = float(effect.loc[effect.condition == "original", "RR"].iloc[0])
    effect["decrease_from_original"] = original - effect.RR
    effect["order"] = effect.condition.map({name: i for i, name in enumerate(order)})
    effect = effect.sort_values("order").drop(columns="order")
    effect.to_csv(OUT / "rq2_reassignment.csv", index=False)
    effect_setting = raw.groupby(["setting", "condition"], as_index=False).agg(
        RR=("RR_held_mean", "mean"), moved_fraction=("moved_fraction", "mean"),
        eligible_fraction=("eligible_fraction", "mean"),
        n=("n", "mean"), n_classes=("n_classes", "mean"),
        RR_eligible=("RR_held_mean_eligible", "mean"),
        original_RR_eligible=("original_RR_held_eligible", "mean"),
        n_eligible=("n_eligible", "mean"))

    per_class_all = pd.read_csv(CSV / "inputspecific_per_class.csv")
    detail_cell = per_class_all.groupby(
        ["split", "fixed_stratum", "setting", "seed"], as_index=False).agg(
            FixedPatch=("RR_FIX", "mean"), DynaPatch=("RR_GEN", "mean"),
            Delta=("delta", "mean"), class_cases=("delta", "size"))
    detail_setting = detail_cell.groupby(
        ["split", "fixed_stratum", "setting"], as_index=False).agg(
            FixedPatch=("FixedPatch", "mean"), DynaPatch=("DynaPatch", "mean"),
            Delta=("Delta", "mean"), class_cases=("class_cases", "sum"),
            seeds=("seed", "nunique"))
    per_class = per_class_all[per_class_all.split == "held"].copy()
    # The mechanism claim concerns transfer to failures not used to fit either patch.
    # Never collapse seen and held rows: doing so mixes two different populations.
    per_class = per_class[per_class.split == "held"].copy()
    cell = per_class.groupby(["fixed_stratum", "setting", "seed"], as_index=False).agg(
        FixedPatch=("RR_FIX", "mean"), DynaPatch=("RR_GEN", "mean"), Delta=("delta", "mean"))
    by_setting = cell.groupby(["fixed_stratum", "setting"], as_index=False)[
        ["FixedPatch", "DynaPatch", "Delta"]].mean()
    strata = by_setting.groupby("fixed_stratum", as_index=False).agg(
        FixedPatch=("FixedPatch", "mean"), DynaPatch=("DynaPatch", "mean"),
        Delta=("Delta", "mean"), settings=("setting", "nunique"))
    class_counts = per_class.groupby("fixed_stratum").size()
    strata["class_cases"] = strata.fixed_stratum.map(class_counts)
    strata["order"] = strata.fixed_stratum.map({"high": 0, "medium": 1, "low": 2})
    strata = strata.sort_values("order").drop(columns="order")
    strata.to_csv(OUT / "rq2_fixedpatch_strata.csv", index=False)

    cap = pd.read_csv(CSV / "arachne_capacity.csv")
    rows = []
    setting_rows = []
    for arm, group in cap.groupby("arm"):
        by_bound = group.groupby("bound_scale", as_index=False).agg(
            RR=("RR_held", "mean"), Reg=("Reg", "mean"), weights=("num_places", "mean"))
        best = by_bound.sort_values(["RR", "Reg"], ascending=[False, True]).iloc[0]
        rows.append({"arm": arm, "weights": best.weights, "bound_scale": best.bound_scale,
                     "RR": best.RR, "Reg": best.Reg})
        selected = group[group.bound_scale == best.bound_scale]
        for setting_name, cell in selected.groupby("setting"):
            setting_rows.append({"arm": arm, "setting": setting_name,
                                 "RR": cell.RR_held.mean(), "Reg": cell.Reg.mean()})
    capacity = pd.DataFrame(rows)
    capacity["order"] = capacity.arm.map(
        {"pareto": 0, "topn16": 1, "topn64": 2, "topn256": 3, "topn1024": 4})
    capacity = capacity.sort_values("order").drop(columns="order")

    capacity.to_csv(OUT / "rq2_arachne_capacity.csv", index=False)
    capacity_setting = pd.DataFrame(setting_rows)
    capacity_setting.to_csv(OUT / "rq2_arachne_by_setting.csv", index=False)

    arm_order = ["pareto", "topn16", "topn64", "topn256", "topn1024"]
    arm_labels = ["Arachne", "N=16", "N=64", "N=256", "N=1024"]
    pivot = capacity_setting.pivot(index="setting", columns="arm", values="RR").reindex(columns=arm_order)
    fig, axes = plt.subplots(3, 4, figsize=(13.6, 8.0), sharex=True, sharey=True)
    x = np.arange(len(arm_order))
    for ax, setting_name in zip(axes.flat, SETTING_LABEL):
        row = pivot.loc[setting_name]
        ax.plot(x, row.to_numpy(float), color=COLORS["orange"], linewidth=1.8,
                marker="o", markersize=4)
        ax.set_title(SETTING_LABEL[setting_name])
        ax.set_ylim(0, .75)
        ax.set_xticks(x, arm_labels, rotation=35, ha="right")
    for ax in axes[:, 0]:
        ax.set_ylabel("RR")
    fig.suptitle("Arachne capacity changes in each dataset–architecture setting",
                 x=.08, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "fig_rq2_arachne_by_setting")

    info = {"original_rr": original,
            "direction_shuffle_rr": float(effect.loc[effect.condition == "global_direction_shuffle", "RR"].iloc[0]),
            "magnitude_shuffle_rr": float(effect.loc[effect.condition == "global_magnitude_shuffle", "RR"].iloc[0]),
            "within_failure_type_rr": float(effect.loc[effect.condition == "within_failure_type_shuffle", "RR"].iloc[0])}
    return effect, strata, capacity, capacity_setting, info, effect_setting, detail_setting


def _gate_points(folder: str, variant: str) -> pd.DataFrame:
    """Picks, per (arm, r, setting), the veto quantile whose frac_Reg_removed on the REPORT
    population (S_held + S_clean^test) crosses r, then the highest RR_held among those. This
    selects theta by searching the same population RR_held/Reg/CReg are then reported on --
    an eval-set selection bug, confirmed 2026-09-04 to be exactly how the paper's canonical
    "DynaPatch @ r=X" numbers were produced (see note/RESEARCH_STATE.md). FIXED for the
    "Protocol C" variant via `_gate_points_deploy()` below; the "more clean negatives" and
    "class-balanced" sensitivity variants (RQ3.5 only, not the paper's headline numbers) still
    go through this function -- not yet fixed, flagged in RQ3.5's caption.
    """
    rs = (.50, .60, .70, .80, .90, .95)
    arm_files = {"pre": "pre", "post": "post", "pre+post": "pre_post"}
    rows = []
    for arm, filename in arm_files.items():
        data = pd.read_csv(ROOT / "outputs" / folder / f"curves_{filename}.csv")
        data["setting"] = data.dataset + "/" + data.backbone
        for r in rs:
            for setting, group in data.groupby("setting"):
                feasible = group[group.frac_Reg_removed >= r - 1e-9]
                assert len(feasible), f"no operating point reaches r={r:.2f} for {arm} in {setting}"
                row = feasible.loc[feasible.RR_held.idxmax()]
                rows.append({"variant": variant, "arm": arm, "r": r, "setting": setting,
                             "q": row.q, "realised_r": row.frac_Reg_removed,
                             "RR": row.RR_held, "Reg": row.Reg, "CReg": row.CReg,
                             "n_held": row.n_held, "n_clean": row.n_clean,
                             "n_crit": row.n_crit})
    return pd.DataFrame(rows)


def _gate_points_natural(folder: str, variant: str) -> pd.DataFrame:
    """No-calibration replacement for `_gate_points()`/`_gate_points_deploy()`: theta=0 (the
    fitted model's own class decision) needs no target r and no threshold search of any kind,
    so this only pools the natural-point CSVs' three seeds -- nothing here searches, targets,
    or argmaxes over anything. One row per (arm, setting); no "r" dimension any more (2026-09-04,
    user decision: drop r-targeting/calibration entirely, see note/RESEARCH_STATE.md)."""
    arm_files = {"pre": "pre", "post": "post", "pre+post": "pre_post"}
    rows = []
    for arm, filename in arm_files.items():
        data = pd.read_csv(ROOT / "outputs" / folder / f"natural_points_{filename}.csv")
        pooled = data.groupby("setting", as_index=False)[
            ["theta", "realised_r", "RR_held", "Reg", "CReg", "n_held", "n_clean", "n_crit"]
        ].mean()
        for _, row in pooled.iterrows():
            rows.append({"variant": variant, "arm": arm, "setting": row.setting,
                         "theta": row.theta, "realised_r": row.realised_r,
                         "RR": row.RR_held, "Reg": row.Reg, "CReg": row.CReg,
                         "n_held": row.n_held, "n_clean": row.n_clean,
                         "n_crit": row.n_crit})
    return pd.DataFrame(rows)


def rq3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict,
                   pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # No arm has an "r" column any more -- theta=0, no target r, no calibration split
    # (2026-09-05, user decision: this now applies to the "more clean negatives"/
    # "class-balanced" sensitivity variants too, not just the main Protocol C arm).
    points = pd.concat([
        _gate_points_natural("gate_ablation_natural", "Protocol C"),
        _gate_points_natural("gate_ablation_ct_natural", "more clean negatives"),
        _gate_points_natural("gate_ablation_bal_natural", "class-balanced"),
    ], ignore_index=True)
    summary = points.groupby(["variant", "arm"], as_index=False)[["RR", "Reg", "CReg"]].mean()
    summary.to_csv(OUT / "rq3_gate_summary.csv", index=False)

    # Classification quality at the natural (theta=0) decision -- no r any more (2026-09-04,
    # user decision: drop r-targeting/calibration entirely). Sourced from
    # outputs/gate_ablation_natural/natural_points_{arm}.csv, the SAME per-cell run that
    # produces RR_held/Reg/CReg (scripts/gate_protocol_b.py --emit-natural-point); "n"/prevalence/
    # accuracy/precision/recall/f1 come from R.confusion(score > 0, u) computed there.
    classifier_rows = []
    classifier_setting_rows = []
    for arm, filename in (("pre", "pre"), ("post", "post"), ("pre+post", "pre_post")):
        data = pd.read_csv(ROOT / "outputs" / "gate_ablation_natural" / f"natural_points_{filename}.csv")
        assert data.setting.nunique() == 12
        by_setting = data.groupby("setting", as_index=False)[
            ["n_train_pos", "n_train_neg", "n", "prevalence",
             "accuracy", "precision", "recall", "f1"]].mean()
        by_setting["arm"] = arm
        classifier_setting_rows.append(by_setting)
        row = by_setting[["accuracy", "precision", "recall", "f1"]].mean().to_dict()
        classifier_rows.append({"arm": arm, **row})
    classifier = pd.DataFrame(classifier_rows)
    classifier_setting = pd.concat(classifier_setting_rows, ignore_index=True)
    classifier.to_csv(OUT / "rq3_gate_classifier.csv", index=False)

    # Conditional response diagnostic from the frozen feature caches. Restrict failure inputs
    # to proposals that change the predicted class, matching the gate's actual decision problem.
    cache = ROOT / "outputs" / "_response_gate_cache_ep40ns"
    response_rows = []
    distribution_rows = []
    for setting_name in SETTING_LABEL:
        ds, bb = setting_name.split("/")
        for seed in (101, 202, 303):
            help_path = cache / f"help_{ds}_{bb}_s{seed}.npz"
            harm_path = cache / f"harm_{ds}_{bb}_s{seed}.npz"
            assert help_path.is_file() and harm_path.is_file()
            with np.load(help_path) as help_data, np.load(harm_path) as harm_data:
                help_y = help_data["__y"].astype(bool)
                help_flip = help_data["flip"] > .5
                groups = [
                    ("successful repair", help_data, help_flip & help_y),
                    ("ineffective change", help_data, help_flip & ~help_y),
                    ("regression", harm_data, harm_data["__y"].astype(bool)),
                ]
                for outcome, data, mask in groups:
                    if not mask.any():
                        continue
                    p_p_max = data["pP_max"] if "pP_max" in data.files else data["pB_max"] + data["dp_max"]
                    delta_confidence = np.asarray(data["dp_max"][mask], dtype=float)
                    response_rows.append({
                        "setting": setting_name, "seed": seed, "outcome": outcome,
                        "n": int(mask.sum()),
                        "base_confidence": float(data["pB_max"][mask].mean()),
                        "patched_confidence": float(p_p_max[mask].mean()),
                        "confidence_change": float(delta_confidence.mean()),
                        "entropy_change": float(data["dH"][mask].mean()),
                    })
                    distribution_rows.extend(
                        {"setting": setting_name, "seed": seed, "outcome": outcome,
                         "confidence_change": float(value)}
                        for value in delta_confidence)
    response_cell = pd.DataFrame(response_rows)
    response_setting = response_cell.groupby(["outcome", "setting"], as_index=False).agg(
        base_confidence=("base_confidence", "mean"),
        patched_confidence=("patched_confidence", "mean"),
        confidence_change=("confidence_change", "mean"),
        entropy_change=("entropy_change", "mean"), n=("n", "sum"))
    response = response_setting.groupby("outcome", as_index=False).agg(
        base_confidence=("base_confidence", "mean"),
        patched_confidence=("patched_confidence", "mean"),
        confidence_change=("confidence_change", "mean"),
        entropy_change=("entropy_change", "mean"),
        settings=("setting", "nunique"), n=("n", "sum"))
    response["order"] = response.outcome.map(
        {"successful repair": 0, "ineffective change": 1, "regression": 2})
    response = response.sort_values("order").drop(columns="order")
    assert response.settings.eq(12).all()
    response.to_csv(OUT / "rq3_response_outcomes.csv", index=False)

    # Give each setting equal total mass and each available seed equal mass within its setting.
    # This keeps the ECDF aligned with the paper's setting-level reporting convention.
    distribution = pd.DataFrame(distribution_rows)
    positive_by_setting = distribution.assign(
        confidence_increased=distribution.confidence_change > 0
    ).groupby(["setting", "seed", "outcome"], as_index=False).confidence_increased.mean(
    ).groupby(["setting", "outcome"], as_index=False).confidence_increased.mean()
    response_setting = response_setting.merge(
        positive_by_setting, on=["setting", "outcome"], validate="one_to_one")
    cell_sizes = distribution.groupby(["setting", "seed", "outcome"], as_index=False).size(
    ).rename(columns={"size": "cell_n"})
    cells_per_setting = cell_sizes.groupby(["setting", "outcome"], as_index=False).size(
    ).rename(columns={"size": "cells_in_setting"})
    distribution = distribution.merge(cell_sizes, on=["setting", "seed", "outcome"])
    distribution = distribution.merge(cells_per_setting, on=["setting", "outcome"])
    distribution["weight"] = 1 / (12 * distribution.cells_in_setting * distribution.cell_n)
    assert np.allclose(distribution.groupby("outcome").weight.sum().to_numpy(), 1)
    distribution.to_csv(OUT / "rq3_confidence_distribution.csv", index=False)
    positive_fraction = plot_confidence_distribution(distribution, FIG)

    # RQ4 figure: No gate vs the single natural-threshold gated point (2026-09-04, user
    # decision: dropped r-targeting/calibration, so there is no more "regression removal
    # requirement" curve to plot -- two points, not a frontier).
    main = points[(points.variant == "Protocol C") & (points.arm == "pre+post")].copy()
    ungated = pd.read_csv(CSV / "all_metrics_best_config.csv")
    ungated = ungated[ungated.method == "DynaPatch (ungated)"]
    ungated = ungated.groupby("setting", as_index=False)[["RR_test", "Reg", "CReg"]].mean()
    ungated = ungated[["RR_test", "Reg", "CReg"]].mean()
    no_gate = {"RR": float(ungated.RR_test), "Reg": float(ungated.Reg),
               "CReg": float(ungated.CReg)}
    gated = {"RR": float(main.RR.mean()), "Reg": float(main.Reg.mean()),
             "CReg": float(main.CReg.mean())}
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.0))
    panels = [("RR", "RR", None), ("Reg", "Reg (%)", None), ("CReg", "CReg (%)", None)]
    x = np.arange(2)
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        values = np.array([no_gate[metric], gated[metric]])
        if metric != "RR":
            values = 100 * values
        ax.bar(x, values, color=COLORS["blue"])
        ax.set_xticks(x, ["No gate", "Gated (natural)"], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(metric)
    fig.suptitle("RQ4 — DynaPatch, no gate vs the natural-threshold gated point",
                 x=.08, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "fig_rq4")

    # "more clean negatives"/"class-balanced" still carry an "r" dimension (not yet migrated
    # off the old r-grid machinery, see RQ3.5's caption) -- pivot Protocol C alone here, where
    # (variant, setting, arm) is already unique.
    proto_c = points[points.variant == "Protocol C"]
    wide = proto_c.pivot(index="setting", columns="arm", values="RR").reset_index()
    wide["difference_vs_pre"] = wide["pre+post"] - wide["pre"]
    info = {"Protocol C_difference_vs_pre": float(wide.difference_vs_pre.mean())}
    info.update({f"positive_confidence_change_{outcome.replace(' ', '_')}": value
                 for outcome, value in positive_fraction.items()})
    return summary, classifier, response, info, points, classifier_setting, response_setting


def build_data_markdown(setting_metrics: pd.DataFrame, r1p: pd.DataFrame,
                        effect_setting: pd.DataFrame, competence_setting: pd.DataFrame,
                        capacity: pd.DataFrame, capacity_setting: pd.DataFrame,
                        gate_points: pd.DataFrame, classifier_setting: pd.DataFrame,
                        response_setting: pd.DataFrame) -> str:
    """Emit the complete per-setting data pack, without Results prose or findings."""
    lines: list[str] = []
    add = lines.append

    def table(title: str, headers: list[str], rows: list[list[object]]) -> None:
        add(f"### {title}")
        add("")
        add(md_table(headers, rows))
        add("")

    def method_metric_table(title: str, methods: list[str], labels: list[str],
                            metric: str) -> None:
        selected = setting_metrics[setting_metrics.method.isin(methods)].copy()
        counts = selected.groupby("method").setting.nunique().reindex(methods)
        assert counts.eq(12).all(), counts.to_dict()
        assert not selected.duplicated(["method", "setting"]).any()
        pivot = selected.pivot(index="setting", columns="method", values=metric)
        assert not pivot.reindex(index=settings, columns=methods).isna().any().any()
        rows = [
            [SETTING_LABEL[setting]] + [f4(pivot.loc[setting, method]) for method in methods]
            for setting in settings
        ]
        rows.append(["Mean"] + [f4(pivot.loc[settings, method].mean()) for method in methods])
        table(title, ["Setting"] + labels, rows)

    settings = [setting for setting, _ in REQUESTED_SETTINGS]
    setting_rank = {name: i for i, name in enumerate(settings)}
    split_rank = {"seen": 0, "held": 1, "clean": 2}
    arm_rank = {"pre": 0, "post": 1, "pre+post": 2}

    add("# Four-RQ Per-Setting Data Pack")
    add("")
    add("This file contains tables only. It does not draft Results prose or findings.")
    add("")
    add("## Data conventions")
    add("")
    add("- Each reported value is kept at the dataset–architecture setting level; repeated seeds are averaged only within that setting.")
    add("- `seen`, `held`, and `clean` populations are never pooled. On `clean`, the reported rate is regression; on `seen` and `held`, it is repair rate.")
    add("- Gate rows marked `pooled` are the seed-pooled Protocol-C curves. DistRep uses seed 101 only. Arachne capacity uses seed 101 only.")
    add("- RQ4 thresholds are selected from the evaluation frontier and therefore describe an upper bound, not a deployable calibration procedure.")
    add("- `—` means that the source artifact does not define the metric for that row.")
    add("")
    add("## Source map")
    add("")
    add("- Index: `note/RAW_CSV.md`")
    add("- Coverage audit: `note/AUDIT.md`")
    add("- Experiment-tree status: `note/NAVIGATION.md` and `outputs/INDEX.md`")
    add("- Frozen-file manifest: `outputs/FROZEN.json`")
    add("")

    # RQ1 -----------------------------------------------------------------
    add("## RQ1 — Input-specific patches versus a fixed patch")
    add("")
    add("> How do input-specific patches affect repair effectiveness compared with a single fixed patch?")
    add("")

    pair = r1p.copy()
    assert len(pair) == 12 * 3
    assert not pair.duplicated(["setting", "split"]).any()
    pair["setting_rank"] = pair.setting.map(setting_rank)
    pair["split_rank"] = pair.split.map(split_rank)
    pair = pair.sort_values(["setting_rank", "split_rank"])
    metric_names = {"seen": "RR_seen", "held": "RR_held", "clean": "Reg"}
    rows = [[x.setting, x.split, metric_names[x.split], f"{x.mean_n:.1f}",
             f4(x.FixedPatch), f4(x["DynaPatch-NoGate"]), f"{x.difference:+.4f}"]
            for _, x in pair.iterrows()]
    table("RQ1.1 — FixedPatch vs DynaPatch-NoGate for every setting and split",
          ["Setting", "Split", "Metric", "Mean n", "FixedPatch", "DynaPatch-NoGate", "Difference"], rows)

    condition_order = ["original", "global_effect_shuffle", "global_direction_shuffle",
                       "global_magnitude_shuffle", "within_true_class_shuffle",
                       "within_failure_type_shuffle"]
    condition_names = {
        "original": "Original", "global_effect_shuffle": "Effect shuffle",
        "global_direction_shuffle": "Direction shuffle",
        "global_magnitude_shuffle": "Magnitude shuffle",
        "within_true_class_shuffle": "Within-class shuffle",
        "within_failure_type_shuffle": "Within-failure-type shuffle",
    }
    assert len(effect_setting) == 12 * len(condition_order)
    assert not effect_setting.duplicated(["setting", "condition"]).any()
    ep = effect_setting.pivot(index="setting", columns="condition", values="RR")
    mp = effect_setting.pivot(index="setting", columns="condition", values="moved_fraction")
    rows = []
    for setting_name in settings:
        rows.append([setting_name] + [f4(ep.loc[setting_name, c]) for c in condition_order])
    table("RQ1.2 — Patch reassignment RR_held for every setting",
          ["Setting"] + [condition_names[c] for c in condition_order], rows)
    rows = []
    for setting_name in settings:
        rows.append([setting_name] + [f4(mp.loc[setting_name, c]) for c in condition_order])
    table("RQ1.3 — Fraction of patch assignments moved in every setting",
          ["Setting"] + [condition_names[c] for c in condition_order], rows)

    # RQ1.3b: `eligible_fraction` -- for the two grouped shuffle conditions (within-class,
    # within-failure-type), the fraction of held-out failures whose group has >= 2 members at
    # all (a singleton group cannot be shuffled -- it is its own permutation). Read together with
    # RQ1.3's `moved_fraction`: within_failure_type_shuffle's near-zero RR change (Table RQ1.2) is
    # NOT by itself evidence that same-failure-type patches "behave similarly" -- many groups are
    # singletons that were never actually permuted. outputs/patch_reassignment_v1/SUMMARY.md's own
    # text: "The grouped shuffles retain singleton groups... they diagnose granularity rather than
    # provide a fair global null." global_effect_shuffle/global_direction_shuffle/
    # global_magnitude_shuffle are unconditionally eligible (1.0) since nothing is grouped.
    ep2 = effect_setting.pivot(index="setting", columns="condition", values="eligible_fraction")
    rows = []
    for setting_name in settings:
        rows.append([setting_name] + [f4(ep2.loc[setting_name, c]) for c in condition_order])
    table("RQ1.3b — Fraction of held-out failures whose group is eligible to be shuffled at all, "
          "for every setting",
          ["Setting"] + [condition_names[c] for c in condition_order], rows)

    # RQ1.5: fixes RQ1.2/1.3b's caveat directly (2026-09-05) instead of just disclosing it --
    # `original` and every shuffle draw's RR_held restricted to only the rows whose group had
    # >1 member and so were actually reassigned to a DIFFERENT failure's patch. A singleton's
    # one row is "shuffled" to itself (a no-op), which silently dilutes RQ1.2's
    # within_failure_type_shuffle row toward "no effect" regardless of whether direction
    # matters within a group; restricting both sides of the comparison to the same genuinely-
    # reassigned rows removes that confound. global_*_shuffle conditions are unaffected (no
    # grouping, so eligible == everyone already).
    eo = effect_setting.pivot(index="setting", columns="condition", values="original_RR_eligible")
    ep3 = effect_setting.pivot(index="setting", columns="condition", values="RR_eligible")
    en = effect_setting.pivot(index="setting", columns="condition", values="n_eligible")
    grouped_conditions = ["within_true_class_shuffle", "within_failure_type_shuffle"]
    rows = []
    for setting_name in settings:
        row = [setting_name]
        for c in grouped_conditions:
            row += [int(en.loc[setting_name, c]), f4(eo.loc[setting_name, c]),
                   f4(ep3.loc[setting_name, c]),
                   f"{eo.loc[setting_name, c] - ep3.loc[setting_name, c]:+.4f}"]
        rows.append(row)
    headers = ["Setting"]
    for c in grouped_conditions:
        label = condition_names[c]
        headers += [f"{label}: n eligible", f"{label}: original RR (eligible)",
                   f"{label}: shuffled RR (eligible)", f"{label}: original minus shuffled"]
    table("RQ1.5 — RQ1.2/1.3b's grouped shuffles, restricted to only the rows actually "
          "reassigned (singleton groups excluded from both sides of the comparison)",
          headers, rows)

    comp = competence_setting.copy()
    assert not comp.duplicated(["setting", "split", "fixed_stratum"]).any()
    assert set(comp.setting).issubset(set(settings))
    comp["setting_rank"] = comp.setting.map(setting_rank)
    comp["split_rank"] = comp.split.map(split_rank)
    comp["stratum_rank"] = comp.fixed_stratum.map({"high": 0, "medium": 1, "low": 2})
    comp = comp.sort_values(["split_rank", "setting_rank", "stratum_rank"])
    rows = [[x.setting, x.split, x.fixed_stratum, int(x.seeds), int(x.class_cases),
             f4(x.FixedPatch), f4(x.DynaPatch), f"{x.Delta:+.4f}"] for _, x in comp.iterrows()]
    table("RQ1.4 — FixedPatch-competence strata for every estimable setting and split",
          ["Setting", "Split", "Fixed stratum", "Seeds", "Class cases",
           "FixedPatch RR", "DynaPatch RR", "Difference"], rows)

    # RQ2 -----------------------------------------------------------------
    add("## RQ2 — Comparison with existing repair methods")
    add("")
    add("> How does DynaPatch compare with existing repair methods, and where do their behaviors differ?")
    add("")
    add("Detailed baseline-behavior evidence, operating-region figures, per-setting consistency, ")
    add("failure-type distributions, Arachne capacity boundaries, and the DistRep expert-oracle ")
    add("case study are collected in [RQ2_BASELINE_BEHAVIOR.md](RQ2_BASELINE_BEHAVIOR.md).")
    add("")
    add("`DynaPatch (gated)` rows (here and elsewhere): the gate commits iff its own fitted "
        "score favours beneficial over harmful (theta=0) -- no target r, no threshold search, "
        "no calibration split of any kind (2026-09-04, see `note/RESEARCH_STATE.md`: an "
        "earlier r-targeted version searched the report population for its threshold -- an "
        "eval-set selection bug -- and a fix that instead targeted r on `bug_val`/`clean_calib` "
        "was still in-sample for the gate, since that population is also part of what fits it "
        "under the shipped protocol; theta=0 needs no such split at all).")
    add("")

    method_order = ["FullFT", "HeadFT", "Arachne", "DistrRep", "NN-Patching",
                    "PatchNAS", "DynaPatch (gated)"]
    method_short = {
        "DynaPatch (gated)": "DP", "FullFT": "FullFT", "HeadFT": "HeadFT",
        "NN-Patching": "NN-Patch", "PatchNAS": "PatchNAS", "Arachne": "Arachne",
        "DistrRep": "DistrRep",
    }
    sm = setting_metrics[setting_metrics.method.isin(method_order)].copy()
    assert sm.groupby("method").setting.nunique().reindex(method_order).eq(12).all()
    assert not sm.duplicated(["method", "setting"]).any()
    for metric, title in (
        ("RR_test", "RQ2.2 — RR across 12 settings"),
        ("Reg", "RQ2.3 — Reg across 12 settings"),
        ("CReg", "RQ2.4 — CReg across 12 settings"),
    ):
        method_metric_table(title, method_order, [method_short[m] for m in method_order], metric)

    best = pd.read_csv(CSV / "all_metrics_best_config.csv")
    coverage_rows = []
    for method in method_order:
        group = best[best.method == method]
        seed_values = ",".join(str(x) for x in group.seed.drop_duplicates())
        coverage_rows.append([method_short[method], group.setting.nunique(), seed_values,
                              int(group.RR_seen.notna().sum()), int(group.RR_test.notna().sum())])
    table("RQ2.5 — Coverage of the compact comparison source",
          ["Method", "Settings", "Seed field", "RR_seen rows", "RR_held rows"], coverage_rows)

    cap_names = {"pareto": "Arachne", "topn16": "N=16", "topn64": "N=64",
                 "topn256": "N=256", "topn1024": "N=1024"}
    cap_order = ["pareto", "topn16", "topn64", "topn256", "topn1024"]
    cap = capacity.set_index("arm")
    assert set(cap.index) == set(cap_order)
    assert len(capacity_setting) == 12 * len(cap_order)
    assert not capacity_setting.duplicated(["setting", "arm"]).any()
    rows = [[cap_names[arm], int(round(cap.loc[arm, "weights"])),
             f4(cap.loc[arm, "bound_scale"]), f3(cap.loc[arm, "RR"]),
             f4(cap.loc[arm, "Reg"])] for arm in cap_order]
    table("RQ2.6 — Globally selected Arachne capacity configurations (seed 101)",
          ["Arm", "Edited weights", "Selected bound scale", "Mean RR_held", "Mean Reg"], rows)
    for metric, title, formatter in (
        ("RR", "RQ2.7 — Arachne capacity RR_held for every setting (seed 101)", f3),
        ("Reg", "RQ2.8 — Arachne capacity Reg for every setting (seed 101)", f4),
    ):
        pivot = capacity_setting.pivot(index="setting", columns="arm", values=metric)
        rows = [[setting_name] + [formatter(pivot.loc[setting_name, arm]) for arm in cap_order]
                for setting_name in settings]
        table(title, ["Setting"] + [cap_names[a] for a in cap_order], rows)

    params = pd.read_csv(CSV / "params_changed.csv")
    param_methods = ["DynaPatch", "FixedPatch", "FullFT", "HeadFT", "NN-Patching",
                     "PatchNAS", "Arachne", "DistRep"]
    params = params[params.method.isin(param_methods)].copy()
    assert params.groupby("method").setting.nunique().reindex(param_methods).eq(12).all()
    assert not params.duplicated(["method", "setting"]).any()
    params["setting_rank"] = params.setting.map(setting_rank)
    params["method_rank"] = params.method.map({m: i for i, m in enumerate(param_methods)})
    params = params.sort_values(["setting_rank", "method_rank"])
    rows = [[x.setting, x.method, int(x.n_changed), f3(x.n_bn_buffer_drift),
             f3(x.n_added), int(x.n_per_input), f3(x.n_total_model), int(x.n_tensors_touched)]
            for _, x in params.iterrows()]
    table("RQ2.9 — Parameter and buffer changes for every method and setting",
          ["Setting", "Method", "Changed params", "BN-buffer drift", "Added params",
           "Per-input params", "Total model params", "Tensors touched"], rows)

    # RQ2.10: does NN-Patching/PatchNAS's shortfall come from misaimed corrections, or from
    # their own error estimator declining to apply the patch? Computed at the "tau" operating
    # point -- verified 2026-09-04 that this, not "matched", is what note/RQ2_BASELINE_BEHAVIOR.md
    # actually reports RR_held from (scripts/best_data.py reads the prior-patch JSON's TOP-LEVEL
    # RR_held/Reg/CReg, not the nested "matched" arm; see note/RESEARCH_STATE.md "regenerating
    # the whole prior-patch pipeline" for the investigation -- an earlier version of this table
    # used "matched" and was wrong). Sourced from the seeded/reproducible
    # baseline_prior_patches_mlp_r5.json (2026-09-04, --torch-seed 0 --repeats 5) and
    # outputs/m1_correction_direction/routed_aim_breakdown_per_cell.csv (2026-09-04, also fixed
    # to read the `tau` per-sample directory). See note/RESEARCH_STATE.md "M1d" for the
    # aim/misaim definitions and the sanity check against RQ2_BASELINE_BEHAVIOR.md's published
    # RR_held (now within +-0.003, i.e. the two pipelines agree).
    aim = pd.read_csv(ROOT / "outputs/m1_correction_direction/routed_aim_breakdown_per_cell.csv")
    # pandas reads the CSV's empty `note` field (a valid row) back as NaN, not "" -- isna() is
    # the correct test here, matching how every note-column filter in this codebase is written.
    aim = aim[(aim.split == "held") & aim.note.isna()]
    aim_setting = aim.groupby(["method", "setting"], as_index=False)[
        ["frac_not_routed", "frac_routed_repaired", "frac_routed_aimed_not_repaired",
         "frac_routed_misaimed"]].mean()

    reg_creg_tau: dict[tuple[str, str], tuple[float, float]] = {}
    pp_r5 = ROOT / "outputs/baseline_prior_patches_mlp_r5.json"
    if pp_r5.is_file():
        pp_data = json.loads(pp_r5.read_text())
        for method in ("NN-Patching", "PatchNAS"):
            for stg, per_seed in pp_data[method].items():
                regs = [v["Reg"] for v in per_seed.values()]
                cregs = [v["CReg"] for v in per_seed.values()]
                reg_creg_tau[(method, stg)] = (float(np.mean(regs)), float(np.mean(cregs)))
    ungated_reg = setting_metrics[setting_metrics.method == "DynaPatch (ungated)"].set_index("setting")

    aim_indexed = aim_setting.set_index(["method", "setting"])
    method_order = ["NN-Patching", "PatchNAS", "DynaPatch (ungated)"]
    method_display = {"NN-Patching": "NN-Patching", "PatchNAS": "PatchNAS",
                      "DynaPatch (ungated)": "DynaPatch-NoGate"}
    rows = []
    for setting_name in settings:
        for method in method_order:
            key = (method if method != "DynaPatch (ungated)" else "DynaPatch-NoGate", setting_name)
            if key not in aim_indexed.index:
                continue
            r = aim_indexed.loc[key]
            if method == "DynaPatch (ungated)":
                reg = f4(ungated_reg.loc[setting_name, "Reg"])
                creg = f4(ungated_reg.loc[setting_name, "CReg"])
                not_routed = "—"
            else:
                rc = reg_creg_tau.get((method, setting_name))
                reg = f4(rc[0]) if rc else "—"
                creg = f4(rc[1]) if rc else "—"
                not_routed = f4(r.frac_not_routed)
            rows.append([setting_name, method_display[method], not_routed,
                       f4(r.frac_routed_repaired), f4(r.frac_routed_aimed_not_repaired),
                       f4(r.frac_routed_misaimed), reg, creg])
    table("RQ2.10 — NN-Patching/PatchNAS held-out failures split by routing and aim, vs "
          "DynaPatch-NoGate, at the tau (default threshold) operating point",
          ["Setting", "Method", "Not routed", "Routed & repaired", "Routed, aimed, not repaired",
           "Routed, misaimed", "Reg (tau)", "CReg (tau)"], rows)

    # RQ2.11: gate transplant -- NN-Patching/PatchNAS's own raw patch output, routed by OUR gate
    # (scripts/analysis_gate_transplant.py) instead of their own estimator, at the same
    # no-calibration natural threshold (theta=0) DynaPatch's own gated row now uses (2026-09-04,
    # user decision: drop r-targeting/calibration entirely). Full per-setting -- deliberately
    # NOT collapsed to a means-only summary (a mean can hide a setting where the swap does
    # something different). Compare against RQ2.10 (their own estimator, tau) and RQ2.2-2.4
    # (DynaPatch's own natural-threshold row).
    gt = pd.read_csv(ROOT / "outputs/gate_transplant/per_cell_4_pre+post.csv")
    gt = gt[gt.note.isna()] if "note" in gt.columns else gt
    gt_setting = gt.groupby(["method", "setting"], as_index=False)[
        ["RR_held", "Reg", "CReg"]].mean()
    gt_indexed = gt_setting.set_index(["method", "setting"])
    rows = []
    for setting_name in settings:
        for method in ("NN-Patching", "PatchNAS"):
            key = (method, setting_name)
            if key not in gt_indexed.index:
                continue
            x = gt_indexed.loc[key]
            rows.append([setting_name, method, f4(x.RR_held), f4(x.Reg), f4(x.CReg)])
    table("RQ2.11 — NN-Patching/PatchNAS's own raw patch output, routed by DynaPatch's gate "
          "instead of their own estimator, at the natural threshold (theta=0), for every "
          "setting",
          ["Setting", "Method", "RR_held", "Reg", "CReg"], rows)

    # RQ3 -----------------------------------------------------------------
    add("## RQ3 — Post-proposal information in gate decisions")
    add("")
    add("> How does post-proposal information contribute to gate decisions?")
    add("")

    # Classification quality at the natural (theta=0) decision -- no r any more (2026-09-04,
    # user decision: drop r-targeting/calibration entirely). RQ3.1 (wide, one row per setting)
    # and RQ3.2 (long, includes train-set sizes) are the same underlying numbers, two layouts.
    clf = classifier_setting.copy()
    assert len(clf) == 12 * 3
    assert not clf.duplicated(["setting", "arm"]).any()
    clf["setting_rank"] = clf.setting.map(setting_rank)
    clf["arm_rank"] = clf.arm.map(arm_rank)
    clf = clf.sort_values(["setting_rank", "arm_rank"])

    arms = ["pre", "post", "pre+post"]
    arm_labels = ["DP-Input", "DP-Response", "DP"]
    gate_metrics = ["accuracy", "precision", "recall", "f1"]
    gate_indexed = clf.set_index(["setting", "arm"])
    gate_headers = ["Setting"]
    for arm_label in arm_labels:
        gate_headers.extend([
            f"{arm_label} Acc", f"{arm_label} Prec",
            f"{arm_label} Rec", f"{arm_label} F1",
        ])
    gate_rows = []
    for setting_name in settings:
        row = [SETTING_LABEL[setting_name]]
        for arm in arms:
            row.extend(f4(gate_indexed.loc[(setting_name, arm), metric])
                       for metric in gate_metrics)
        gate_rows.append(row)
    mean_row = ["Mean"]
    for arm in arms:
        arm_data = clf[clf.arm == arm].set_index("setting").loc[settings]
        mean_row.extend(f4(arm_data[metric].mean()) for metric in gate_metrics)
    gate_rows.append(mean_row)
    table("RQ3.1 — DP-Input / DP-Response / DP at the natural threshold (theta=0)",
          gate_headers, gate_rows)

    rows = [[x.setting, x.arm, f"{x.n:.1f}", f4(x.prevalence),
             f"{x.n_train_pos:.1f}", f"{x.n_train_neg:.1f}", f4(x.accuracy),
             f4(x.precision), f4(x.recall), f4(x.f1)] for _, x in clf.iterrows()]
    table("RQ3.2 — Gate-classification metrics for every setting and evidence arm, at the "
          "natural threshold (theta=0)",
          ["Setting", "Evidence", "Mean n test", "Prevalence", "Train pos", "Train neg",
           "Accuracy", "Precision", "Recall", "F1"], rows)

    outcome_rank = {"successful repair": 0, "ineffective change": 1, "regression": 2}
    resp = response_setting.copy()
    assert len(resp) == 12 * 3
    assert not resp.duplicated(["setting", "outcome"]).any()
    resp["setting_rank"] = resp.setting.map(setting_rank)
    resp["outcome_rank"] = resp.outcome.map(outcome_rank)
    resp = resp.sort_values(["setting_rank", "outcome_rank"])
    rows = [[x.setting, x.outcome, int(x.n), f4(x.base_confidence), f4(x.patched_confidence),
             f"{x.confidence_change:+.4f}", f"{x.entropy_change:+.4f}",
             f4(x.confidence_increased)] for _, x in resp.iterrows()]
    table("RQ3.3 — Prediction response for every setting and patch outcome",
          ["Setting", "Outcome", "n", "Base confidence", "Patched confidence",
           "Confidence change", "Entropy change", "Fraction confidence increased"], rows)

    gp = gate_points.copy()
    gp["setting_rank"] = gp.setting.map(setting_rank)
    gp["arm_rank"] = gp.arm.map(arm_rank)
    main_gp = gp[gp.variant == "Protocol C"].sort_values(["setting_rank", "arm_rank"])
    assert len(main_gp) == 12 * 3
    assert not main_gp.duplicated(["setting", "arm"]).any()
    rows = [[x.setting, x.arm, f4(x.theta), f4(x.realised_r),
             f4(x.RR), f4(x.Reg), f4(x.CReg), int(x.n_held), int(x.n_clean),
             f"{x.n_crit:.1f}"] for _, x in main_gp.iterrows()]
    table("RQ3.4 — Protocol-C operating data for every setting and evidence arm, at the "
          "natural threshold (theta=0: commit iff the fitted model's own score favours "
          "beneficial over harmful) -- no target r, no threshold search, no calibration "
          "split of any kind (2026-09-04, user decision; see note/RESEARCH_STATE.md). "
          "'realised r' is informational only (the fraction of ungated regression this "
          "threshold happens to remove), not something targeted.",
          ["Setting", "Evidence", "theta", "Realised r (informational)", "RR_held", "Reg",
           "CReg", "n held", "n clean", "n critical"], rows)

    sensitivity = gp[gp.variant.isin(["more clean negatives", "class-balanced"])].copy()
    pre = sensitivity[sensitivity.arm == "pre"].set_index(["variant", "setting"])
    both = sensitivity[sensitivity.arm == "pre+post"].set_index(["variant", "setting"])
    delta = both[["RR", "Reg", "CReg"]] - pre[["RR", "Reg", "CReg"]]
    delta = delta.reset_index()
    assert len(delta) == 2 * 12
    assert not delta.duplicated(["variant", "setting"]).any()
    delta["setting_rank"] = delta.setting.map(setting_rank)
    delta = delta.sort_values(["variant", "setting_rank"])
    rows = [[x.variant, x.setting, f"{x.RR:+.4f}",
             f"{x.Reg:+.4f}", f"{x.CReg:+.4f}"] for _, x in delta.iterrows()]
    table("RQ3.5 — Sensitivity deltas: pre+post minus pre for every setting, at the natural "
          "threshold (theta=0, 2026-09-05 -- these two variants moved off the old r-grid "
          "machinery this round too, not just the main Protocol C arm). Not the paper's "
          "headline numbers; kept as a sensitivity check on whether pre+post beats "
          "pre-only under two alternative gate-fitting recipes.",
          ["Variant", "Setting", "Delta RR_held", "Delta Reg", "Delta CReg"], rows)

    # RQ3.6: does post-information help a DIFFERENT patch mechanism's gate the way Finding 3
    # says it helps DynaPatch's own (scripts/analysis_gate_transplant.py, run twice with
    # --feature-set "4 pre+post" and "2 pre-strong"), at the natural threshold (theta=0,
    # 2026-09-04 -- no target r, no calibration split). Delta = pre+post minus pre-only, for
    # NN-Patching/PatchNAS's patches routed by our gate, not DynaPatch's own.
    gt4 = pd.read_csv(ROOT / "outputs/gate_transplant/per_cell_4_pre+post.csv")
    gt2 = pd.read_csv(ROOT / "outputs/gate_transplant/per_cell_2_prestrong.csv")
    gt4 = gt4[gt4.note.isna()] if "note" in gt4.columns else gt4
    gt2 = gt2[gt2.note.isna()] if "note" in gt2.columns else gt2
    g4s = gt4.groupby(["method", "setting"], as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    g2s = gt2.groupby(["method", "setting"], as_index=False)[["RR_held", "Reg", "CReg"]].mean()
    g4i = g4s.set_index(["method", "setting"])
    g2i = g2s.set_index(["method", "setting"])
    rows = []
    for setting_name in settings:
        for method in ("NN-Patching", "PatchNAS"):
            key = (method, setting_name)
            if key not in g4i.index or key not in g2i.index:
                continue
            a, b = g4i.loc[key], g2i.loc[key]
            rows.append([setting_name, method, f"{a.RR_held - b.RR_held:+.4f}",
                       f"{a.Reg - b.Reg:+.4f}", f"{a.CReg - b.CReg:+.4f}"])

    # RQ3.6: the setting-balanced SUMMARY of the same delta, with DynaPatch's own pre+post-minus-
    # pre delta (from RQ3.4's "Protocol C" pre/pre+post arms) IN THE SAME TABLE -- so "does this
    # help DynaPatch" and "does this help NN-Patching/PatchNAS" are one direct comparison, not
    # two separately-shaped tables.
    dp_pre = main_gp[main_gp.arm == "pre"][["RR", "Reg", "CReg"]].mean()
    dp_both = main_gp[main_gp.arm == "pre+post"][["RR", "Reg", "CReg"]].mean()
    d = dp_both - dp_pre
    summary_rows = [["DynaPatch (own gate)", f"{d.RR:+.4f}", f"{d.Reg:+.4f}", f"{d.CReg:+.4f}"]]
    for method in ("NN-Patching", "PatchNAS"):
        a = g4s[g4s.method == method][["RR_held", "Reg", "CReg"]].mean()
        b = g2s[g2s.method == method][["RR_held", "Reg", "CReg"]].mean()
        summary_rows.append([f"{method} (our gate)", f"{a.RR_held - b.RR_held:+.4f}",
                            f"{a.Reg - b.Reg:+.4f}", f"{a.CReg - b.CReg:+.4f}"])
    table("RQ3.6 — Does post-information help, and does it help every patch mechanism? "
          "Setting-balanced summary at the natural threshold: Delta = (pre+post) minus "
          "(pre-only), DynaPatch's own gate alongside our gate applied to NN-Patching/PatchNAS",
          ["Arm", "Delta RR_held", "Delta Reg", "Delta CReg"], summary_rows)

    table("RQ3.7 — RQ3.6's NN-Patching/PatchNAS deltas, full per-setting (not averaged)",
          ["Setting", "Method", "Delta RR_held", "Delta Reg", "Delta CReg"], rows)

    # RQ3.8-3.10: stratified-K-fold CV classification metrics (AUC/F1/precision/recall on the
    # binary commit decision, plus macro-F1/accuracy on the underlying 3-class gain label),
    # 3-class gate-coefficient feature importance, and the RQ3.3 confidence/entropy-response
    # replication for NN-Patching/PatchNAS -- scripts/analysis_gate_cv_importance.py and
    # scripts/analysis_gate_response_baselines.py, 2026-09-04. Feeds the "is 3-class or binary?"
    # and "which feature matters, and is that stable across settings?" questions.
    cv_cell = pd.read_csv(ROOT / "outputs/gate_cv_importance/cv_auc_per_cell.csv")
    cv_cell = cv_cell[cv_cell.feature_set == "4 pre+post"].copy()
    cv_cell["setting_rank"] = cv_cell.setting.map(setting_rank)
    cv_cell = cv_cell.sort_values(["method", "setting_rank"])
    rows = [[x.method, x.setting, f"{x.mean_macro_f1_3class:.4f}", f"{x.mean_accuracy_3class:.4f}",
             f"{x.mean_precision_commit:.4f}", f"{x.mean_recall_commit:.4f}",
             f"{x.mean_f1_commit:.4f}", f"{x.mean_auc_commit:.4f}", int(x.n_pos), int(x.n_neg),
             int(x.n_zero)] for _, x in cv_cell.iterrows()]
    table("RQ3.8 — Stratified 5-fold CV, gate fitted on 4 pre+post features, for every setting "
          "(the fitted model is a 3-class classifier over gain in {harmful,no-effect,beneficial}; "
          "macro-F1/accuracy read it as 3-class, precision/recall/F1/AUC read it as the binary "
          "commit-vs-not decision the router actually uses)",
          ["Method", "Setting", "Macro-F1 (3-class)", "Accuracy (3-class)", "Precision (commit)",
           "Recall (commit)", "F1 (commit)", "AUC (commit)", "n commit-pos", "n commit-neg",
           "n no-effect"], rows)

    imp_summary = pd.read_csv(ROOT / "outputs/gate_cv_importance/importance_summary.csv")
    imp_summary = imp_summary.sort_values(["method", "mean_abs_coef"], ascending=[True, False])
    rows = [[x.method, x.feature, f"{x.mean_coef:+.4f}", f"{x.mean_abs_coef:.4f}", int(x.n_settings)]
             for _, x in imp_summary.iterrows()]
    table("RQ3.9a — Gate feature importance (coefficient for gain=+1 in the 3-class "
          "logistic-regression gate, 4 pre+post feature set), averaged across settings, ranked "
          "by |coefficient|",
          ["Method", "Feature", "Mean coefficient", "Mean |coefficient|", "n settings"], rows)

    imp_setting = pd.read_csv(ROOT / "outputs/gate_cv_importance/importance_per_setting.csv")
    imp_setting["setting_rank"] = imp_setting.setting.map(setting_rank)
    imp_setting = imp_setting.sort_values(["method", "setting_rank", "feature"])
    rows = [[x.method, x.setting, x.feature, f"{x.coef_gain1:+.4f}"]
             for _, x in imp_setting.iterrows()]
    table("RQ3.9b — RQ3.9a's coefficients, full per-(method, setting, feature) (not averaged -- "
          "shows whether a feature's importance is stable across settings or driven by one or two "
          "of them)",
          ["Method", "Setting", "Feature", "Coefficient (gain=+1)"], rows)

    resp_overall = pd.read_csv(ROOT / "outputs/gate_response_baselines/overall.csv")
    order = {"successful repair": 0, "ineffective change": 1, "regression": 2}
    resp_overall["order"] = resp_overall.outcome.map(order)
    resp_overall = resp_overall.sort_values(["method", "order"])
    rows = [[x.method, x.outcome, int(x.n), f"{x.base_confidence:.4f}", f"{x.patched_confidence:.4f}",
             f"{x.confidence_change:+.4f}", f"{x.entropy_change:+.4f}",
             f"{x.frac_confidence_increased:.4f}"] for _, x in resp_overall.iterrows()]
    table("RQ3.10a — Confidence/entropy response by outcome for NN-Patching/PatchNAS's own raw "
          "(always-apply) patches, setting-balanced (replicates RQ3.3's construction and "
          "statistics for DynaPatch; compare directly against RQ3.3's numbers)",
          ["Method", "Outcome", "n", "Base confidence", "Patched confidence",
           "Confidence change", "Entropy change", "Fraction confidence increased"], rows)

    resp_setting = pd.read_csv(ROOT / "outputs/gate_response_baselines/per_setting.csv")
    resp_setting["setting_rank"] = resp_setting.setting.map(setting_rank)
    resp_setting["order"] = resp_setting.outcome.map(order)
    resp_setting = resp_setting.sort_values(["method", "setting_rank", "order"])
    rows = [[x.method, x.setting, x.outcome, int(x.n), f"{x.base_confidence:.4f}",
             f"{x.patched_confidence:.4f}", f"{x.confidence_change:+.4f}", f"{x.entropy_change:+.4f}",
             f"{x.frac_confidence_increased:.4f}"] for _, x in resp_setting.iterrows()]
    table("RQ3.10b — RQ3.10a's response statistics, full per-setting (not averaged)",
          ["Method", "Setting", "Outcome", "n", "Base confidence", "Patched confidence",
           "Confidence change", "Entropy change", "Fraction confidence increased"], rows)


    # RQ4 -----------------------------------------------------------------
    add("## RQ4 — Repair–regression operating behavior")
    add("")
    add("> How effectively does DynaPatch control the repair–regression trade-off under different regression-control requirements?")
    add("")
    add("`theta`=0: commit iff the fitted gate's own score favours beneficial over harmful -- "
        "no target r, no threshold search, no calibration split of any kind (2026-09-04, user "
        "decision; see `note/RESEARCH_STATE.md`. An earlier r-targeted version searched the "
        "report population for its threshold -- an eval-set selection bug -- and a fix that "
        "instead targeted r on `bug_val`/`clean_calib` was still in-sample for the gate, since "
        "that population is also part of what fits it under the shipped protocol). 'realised "
        "r' is informational only -- the fraction of ungated regression this threshold happens "
        "to remove -- not something targeted.")
    add("")

    shipped = gp[(gp.variant == "Protocol C") & (gp.arm == "pre+post")].copy()
    assert len(shipped) == 12
    assert not shipped.duplicated(["setting"]).any()
    shipped["setting_rank"] = shipped.setting.map(setting_rank)
    shipped = shipped.sort_values("setting_rank")
    ungated = setting_metrics[setting_metrics.method == "DynaPatch (ungated)"].set_index("setting")
    rows = []
    for setting_name in settings:
        ref = shipped[shipped.setting == setting_name].iloc[0]
        base = ungated.loc[setting_name]
        rows.append([setting_name, "No gate", "0.0000", "0.0000", f4(base.RR_test),
                     f4(base.Reg), f4(base.CReg), int(ref.n_held), int(ref.n_clean), f"{ref.n_crit:.1f}"])
        rows.append([setting_name, "Gated (natural)", f4(ref.theta), f4(ref.realised_r),
                     f4(ref.RR), f4(ref.Reg), f4(ref.CReg), int(ref.n_held), int(ref.n_clean),
                     f"{ref.n_crit:.1f}"])
    table("RQ4.1 — No-gate and gated operating points for every setting",
          ["Setting", "Operating point", "theta", "Realised r (informational)", "RR_held",
           "Reg", "CReg", "n held", "n clean", "n critical"], rows)

    operating_methods = ["DynaPatch (ungated)", "DynaPatch (gated)"]
    operating_labels = ["NoGate", "Gated"]
    method_metric_table("RQ4.2 — RR across 12 settings", operating_methods,
                        operating_labels, "RR_test")
    method_metric_table("RQ4.3 — Reg across 12 settings", operating_methods,
                        operating_labels, "Reg")
    method_metric_table("RQ4.4 — CReg across 12 settings", operating_methods,
                        operating_labels, "CReg")

    # RQ4.5: DynaPatch's own gate vs the SAME gate applied to NN-Patching/PatchNAS's patches
    # (scripts/analysis_gate_transplant.py), all at the natural threshold (theta=0, 2026-09-04
    # -- no more r curve to compare shapes of, just one point per arm). Setting-balanced; full
    # 12-setting raw numbers already exist as RQ2.11 -- this table restates them alongside
    # DynaPatch's own point so the two are read together rather than from two different files.
    dp_point = setting_metrics[setting_metrics.method.isin(
        ["DynaPatch (ungated)", "DynaPatch (gated)"]
    )].groupby("method")[["RR_test", "Reg", "CReg"]].mean()
    rows = []
    for m in ("DynaPatch (ungated)", "DynaPatch (gated)"):
        x = dp_point.loc[m]
        rows.append([f"DynaPatch, own gate ({m.split('(')[1][:-1]})",
                   f4(x.RR_test), f4(x.Reg), f4(x.CReg)])
    gt_mean = g4s.groupby("method")[["RR_held", "Reg", "CReg"]].mean()
    for method in ("NN-Patching", "PatchNAS"):
        if method not in gt_mean.index:
            continue
        x = gt_mean.loc[method]
        rows.append([f"{method}, our gate", f4(x.RR_held), f4(x.Reg), f4(x.CReg)])
    table("RQ4.5 — DynaPatch's own gate vs the same gate applied to NN-Patching/PatchNAS's "
          "patches, at the natural threshold (setting-balanced summary; DynaPatch's own "
          "full 12-setting numbers are RQ4.1, NN-Patching/PatchNAS's are RQ2.11)",
          ["Arm", "RR_held", "Reg", "CReg"], rows)

    add("## Reproduction")
    add("")
    add("```bash")
    add(".venv/bin/python scripts/freeze_results.py")
    add(".venv/bin/python scripts/build_results_draft.py")
    add("```")
    add("")
    return "\n".join(lines)


RQ_FILES = {
    "RQ1": "RQ1_DATA.md", "RQ2": "RQ2_DATA.md", "RQ3": "RQ3_DATA.md", "RQ4": "RQ4_DATA.md",
}
RQ_TITLES = {
    "RQ1": "Input-specific patches versus a fixed patch",
    "RQ2": "Comparison with existing repair methods",
    "RQ3": "Post-proposal information in gate decisions",
    "RQ4": "Repair-regression operating behavior",
}


def split_data_pack(draft: str) -> None:
    """Split build_data_markdown's one big string into a slim index (note/DRAFT_RESULTS_NEW_RQS.md
    -- links and the high-level reporting convention only) plus one full-tables file per RQ
    (note/RQ{1,2,3,4}_DATA.md). Decision 2026-09-04: the monolithic file was unwieldy once RQ2
    alone carried 11 tables; each RQ now gets its own tables-only file, mirroring the split
    already used for RQ2's evidence pack (note/RQ2_BASELINE_BEHAVIOR.md, hand-maintained,
    references note/RQ2_DATA.md for the full tables rather than duplicating them).

    Pure string split on the "## RQ<N> — " headers `build_data_markdown` already emits -- no
    change to how any individual table is built, so a single table's content is identical
    whether read from the old monolithic file or the new per-RQ one.
    """
    import re
    NOTE = ROOT / "note"
    marker = re.compile(r"^## (RQ[1-4]) — ", re.MULTILINE)
    matches = list(marker.finditer(draft))
    if len(matches) != 4:
        raise SystemExit(f"expected 4 '## RQ<N> — ' section headers, found {len(matches)}")
    preamble = draft[:matches[0].start()].rstrip() + "\n"
    for i, m in enumerate(matches):
        rq = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(draft)
        section = draft[m.start():end].rstrip() + "\n"
        (NOTE / RQ_FILES[rq]).write_text(
            f"# {rq} — {RQ_TITLES[rq]} — full data pack\n\n"
            f"Tables only, no prose or findings (same convention as the old monolithic file). "
            f"Part of the four-RQ data pack; see [DRAFT_RESULTS_NEW_RQS.md](DRAFT_RESULTS_NEW_RQS.md) "
            f"for the index and reporting convention.\n\n" + section)

    index = [preamble, "## Where the tables live", "",
            "| RQ | Question | Full data pack | Evidence pack (organised findings) |",
            "|---|---|---|---|",
            "| RQ1 | How do input-specific patches affect repair effectiveness compared with a "
            "single fixed patch? | [RQ1_DATA.md](RQ1_DATA.md) | [RQ1_INPUT_SPECIFICITY.md]"
            "(RQ1_INPUT_SPECIFICITY.md) |",
            "| RQ2 | How does DynaPatch compare with existing repair methods, and where do their "
            "behaviors differ? | [RQ2_DATA.md](RQ2_DATA.md) | [RQ2_BASELINE_BEHAVIOR.md]"
            "(RQ2_BASELINE_BEHAVIOR.md) |",
            "| RQ3 | How does post-proposal information contribute to gate decisions? | "
            "[RQ3_DATA.md](RQ3_DATA.md) | -- not yet split out |",
            "| RQ4 | How effectively does DynaPatch control the repair-regression trade-off "
            "under different regression-control requirements? | [RQ4_DATA.md](RQ4_DATA.md) | "
            "-- not yet split out |",
            "",
            "Each `RQ<N>_DATA.md` is tables only (auto-generated by this script, never hand-"
            "edited). Each evidence pack is hand-maintained: it organises the raw tables into "
            "admissible findings and interpretation, is NOT polished submission prose, and must "
            "be updated by hand when the underlying data changes -- it is not regenerated by "
            "this script.", ""]
    (NOTE / "DRAFT_RESULTS_NEW_RQS.md").write_text("\n".join(index))
    print(f"[written] note/DRAFT_RESULTS_NEW_RQS.md (index)")
    for rq, fname in RQ_FILES.items():
        print(f"[written] note/{fname}")


def main() -> None:
    validation = validate_sources()
    base_style()
    setting, overall = method_means()
    r1s, r1p, r1i = rq1(setting, overall)
    r2e, r2x, r2c, r2cs, r2i, effect_setting, competence_setting = rq2()
    r3s, r3c, r3o, r3i, gate_points, classifier_setting, response_setting = rq3()
    draft = build_data_markdown(
        setting, r1p, effect_setting, competence_setting, r2c, r2cs,
        gate_points, classifier_setting, response_setting)
    split_data_pack(draft)
    summary = {
        "validation": validation,
        "reporting": "complete per-setting data pack; seeds averaged only within setting",
        "rq1": r1i,
        "rq2": r2i,
        "rq3": r3i,
        "rq4": {"source": "Protocol C pre+post operating curve"},
        "sources": [
            "outputs/csv/all_metrics_best_config.csv",
            "outputs/csv/inputspecific_per_cell.csv",
            "outputs/patch_reassignment_v1/per_cell.csv",
            "outputs/csv/arachne_capacity.csv",
            "outputs/gate_ablation_raw",
            "outputs/gate_ablation_natural",
            "outputs/gate_ablation_ct_natural",
            "outputs/gate_ablation_bal_natural",
            "outputs/csv/gate_performance.csv",
            "outputs/_response_gate_cache_ep40ns",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[written] {FIG} (supporting figures regenerated; not embedded in the data pack)")
    print(f"[written] {OUT} (auditable summaries + JSON)")
    print(f"[validated] {validation}")


if __name__ == "__main__":
    main()
