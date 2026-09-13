#!/usr/bin/env python3
"""Derive the final RQ2 baseline-behavior evidence from frozen test results.

This script does not train, tune, or select a model. It uses held-out failure
test rows for repair and clean-test rows for regression. All operating points
are setting-balanced: seeds are averaged within each dataset--architecture
setting, then the 12 settings are averaged equally.

The failure-type ECDF is also balanced hierarchically. Each valid setting has
equal mass, each seed within a setting has equal mass, and each estimable
failure type within a setting--seed cell has equal mass. Types with fewer than
the configured number of held-out failures remain in the upstream raw table
but are excluded from this distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def assert_rates(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = frame[column].dropna()
        if not values.between(0, 1).all():
            raise AssertionError(f"{column} contains a value outside [0, 1]")


def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def compare(left: float, right: float, tolerance: float) -> str:
    if left > right + tolerance:
        return "left_higher"
    if left < right - tolerance:
        return "left_lower"
    return "tie"


def summarize_comparisons(detail: pd.DataFrame) -> pd.DataFrame:
    """Collapse paired per-setting comparisons without counting missing cells."""
    rows: list[dict] = []
    for comparison_name, group in detail.groupby("comparison", sort=False):
        valid = group[group.outcome != "not_estimable"]
        preferred = group.preferred_direction.iloc[0]
        win_outcome = "left_lower" if preferred == "lower" else "left_higher"
        loss_outcome = "left_higher" if preferred == "lower" else "left_lower"
        rows.append({
            "comparison": comparison_name,
            "metric": group.metric.iloc[0],
            "preferred_direction": preferred,
            "n_settings_total": int(len(group)),
            "n_settings_estimable": int(len(valid)),
            "left_wins": int((valid.outcome == win_outcome).sum()),
            "ties": int((valid.outcome == "tie").sum()),
            "left_losses": int((valid.outcome == loss_outcome).sum()),
            "not_estimable": int((group.outcome == "not_estimable").sum()),
        })
    return pd.DataFrame(rows)


def main_metrics(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = resolve(config["sources"]["main_metrics"])
    raw = pd.read_csv(source)
    methods = config["baseline_methods"] + config["dynapatch_operating_points"]
    frame = raw[raw.method.isin(methods)].copy()
    expected_settings = set(config["settings"])
    if set(frame.setting) != expected_settings:
        raise AssertionError("main metrics do not contain exactly the configured settings")
    counts = frame.groupby("method").setting.nunique().reindex(methods)
    if counts.isna().any() or not counts.eq(len(expected_settings)).all():
        raise AssertionError(f"incomplete main metrics:\n{counts}")
    assert_rates(frame, ["RR_test", "Reg", "CReg"])

    # Seed means within settings; gate rows are already seed-pooled and remain one row.
    per_setting = (
        frame.groupby(["method", "setting"], as_index=False)
        .agg(
            n_seed_rows=("seed", "size"),
            RR_held=("RR_test", "mean"),
            Reg=("Reg", "mean"),
            CReg=("CReg", "mean"),
        )
    )
    if per_setting.duplicated(["method", "setting"]).any():
        raise AssertionError("duplicate method/setting main-metric row")
    overall = (
        per_setting.groupby("method", as_index=False)
        .agg(n_settings=("setting", "nunique"), RR_held=("RR_held", "mean"),
             Reg=("Reg", "mean"), CReg=("CReg", "mean"))
    )
    order = {method: index for index, method in enumerate(methods)}
    per_setting["_order"] = per_setting.method.map(order)
    overall["_order"] = overall.method.map(order)
    return (
        per_setting.sort_values(["_order", "setting"]).drop(columns="_order"),
        overall.sort_values("_order").drop(columns="_order"),
    )


def consistency_tables(config: dict, metrics: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    tolerance = float(config["comparison_tolerance"])
    rows: list[dict] = []

    pivot = metrics.pivot(index="setting", columns="method")
    dp_gated = "DynaPatch (gated)"
    for baseline in config["baseline_methods"]:
        for setting in config["settings"]:
            left = float(pivot.loc[setting, ("Reg", dp_gated)])
            right = float(pivot.loc[setting, ("Reg", baseline)])
            rows.append({
                "comparison": f"{dp_gated} Reg vs {baseline}",
                "metric": "Reg",
                "preferred_direction": "lower",
                "setting": setting,
                "left_method": dp_gated,
                "right_method": baseline,
                "left_value": left,
                "right_value": right,
                "delta_left_minus_right": left - right,
                "outcome": compare(left, right, tolerance),
            })

    for setting in config["settings"]:
        left = float(pivot.loc[setting, ("RR_held", "FullFT")])
        right = float(pivot.loc[setting, ("RR_held", dp_gated)])
        rows.append({
            "comparison": f"FullFT RR_held vs {dp_gated}",
            "metric": "RR_held",
            "preferred_direction": "higher",
            "setting": setting,
            "left_method": "FullFT",
            "right_method": dp_gated,
            "left_value": left,
            "right_value": right,
            "delta_left_minus_right": left - right,
            "outcome": compare(left, right, tolerance),
        })

    coverage = pd.read_csv(resolve(config["sources"]["failure_type_per_setting"]))
    coverage = coverage[coverage.method.isin(["DynaPatch (ungated)", "FixedPatch"])]
    coverage = coverage.pivot(
        index="setting", columns="method", values="coverage_at_0.5_seed_mean"
    )
    for setting in config["settings"]:
        left = coverage.loc[setting, "DynaPatch (ungated)"]
        right = coverage.loc[setting, "FixedPatch"]
        outcome = "not_estimable"
        delta = np.nan
        if pd.notna(left) and pd.notna(right):
            left, right = float(left), float(right)
            delta = left - right
            outcome = compare(left, right, tolerance)
        rows.append({
            "comparison": "DynaPatch (ungated) Coverage@0.5 vs FixedPatch",
            "metric": "Coverage@0.5",
            "preferred_direction": "higher",
            "setting": setting,
            "left_method": "DynaPatch (ungated)",
            "right_method": "FixedPatch",
            "left_value": left,
            "right_value": right,
            "delta_left_minus_right": delta,
            "outcome": outcome,
        })

    detail = pd.DataFrame(rows)
    return detail, summarize_comparisons(detail)


def failure_type_wtl_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Task 1: compare DynaPatch-NoGate to every RQ2 baseline by type behavior."""
    frame = pd.read_csv(resolve(config["sources"]["failure_type_per_setting"]))
    methods = ["DynaPatch (ungated)"] + config["baseline_methods"]
    frame = frame[frame.method.isin(methods)].copy()
    expected_settings = set(config["settings"])
    counts = frame.groupby("method").setting.nunique().reindex(methods)
    if counts.isna().any() or not counts.eq(len(expected_settings)).all():
        raise AssertionError(f"incomplete failure-type setting table:\n{counts}")
    metrics = {
        "Coverage@0.5": "coverage_at_0.5_seed_mean",
        "Type-RR P10": "RR_p10_estimable_seed_mean",
        "Type-RR median": "RR_median_estimable_seed_mean",
    }
    pivots = {
        label: frame.pivot(index="setting", columns="method", values=column)
        for label, column in metrics.items()
    }
    tolerance = float(config["comparison_tolerance"])
    rows: list[dict] = []
    for baseline in config["baseline_methods"]:
        for metric_label, pivot in pivots.items():
            for setting in config["settings"]:
                left = pivot.loc[setting, "DynaPatch (ungated)"]
                right = pivot.loc[setting, baseline]
                outcome = "not_estimable"
                delta = np.nan
                if pd.notna(left) and pd.notna(right):
                    left, right = float(left), float(right)
                    delta = left - right
                    outcome = compare(left, right, tolerance)
                rows.append({
                    "comparison": f"DynaPatch (ungated) {metric_label} vs {baseline}",
                    "metric": metric_label,
                    "preferred_direction": "higher",
                    "setting": setting,
                    "left_method": "DynaPatch (ungated)",
                    "right_method": baseline,
                    "left_value": left,
                    "right_value": right,
                    "delta_left_minus_right": delta,
                    "outcome": outcome,
                })
    detail = pd.DataFrame(rows)
    return detail, summarize_comparisons(detail)


def regression_behavior_tables(
    config: dict,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    """Tasks 2 and 4: clean-test class-level behavior and paired comparisons."""
    source = resolve(config["sources"]["regression_per_setting"])
    frame = pd.read_csv(source)
    methods = config["regression_plot_methods"]
    frame = frame[frame.method.isin(methods)].copy()
    expected_settings = set(config["settings"])
    counts = frame.groupby("method").setting.nunique().reindex(methods)
    if counts.isna().any() or not counts.eq(len(expected_settings)).all():
        raise AssertionError(f"incomplete regression setting table:\n{counts}")

    columns = {
        "overall Reg": "Reg_overall_seed_mean",
        "worst-class Reg": "Reg_max_class_seed_mean",
        "fraction of clean classes affected": "frac_classes_hit_seed_mean",
    }
    assert_rates(frame, list(columns.values()))
    per_setting = frame[["method", "setting", *columns.values()]].copy()
    per_setting = per_setting.rename(columns={value: key for key, value in columns.items()})
    if per_setting.duplicated(["method", "setting"]).any():
        raise AssertionError("duplicate method/setting regression-behavior row")
    overall = (
        per_setting.groupby("method", as_index=False)
        .agg(
            n_settings=("setting", "nunique"),
            overall_Reg=("overall Reg", "mean"),
            worst_class_Reg=("worst-class Reg", "mean"),
            fraction_clean_classes_affected=("fraction of clean classes affected", "mean"),
        )
    )
    order = {method: index for index, method in enumerate(methods)}
    per_setting["_order"] = per_setting.method.map(order)
    overall["_order"] = overall.method.map(order)
    per_setting = per_setting.sort_values(["_order", "setting"]).drop(columns="_order")
    overall = overall.sort_values("_order").drop(columns="_order")

    pivot = per_setting.pivot(index="setting", columns="method")
    tolerance = float(config["comparison_tolerance"])
    rows: list[dict] = []
    dp_gated = "DynaPatch (gated)"
    for baseline in config["baseline_methods"]:
        for metric_label in columns:
            for setting in config["settings"]:
                left = float(pivot.loc[setting, (metric_label, dp_gated)])
                right = float(pivot.loc[setting, (metric_label, baseline)])
                rows.append({
                    "comparison": f"{dp_gated} {metric_label} vs {baseline}",
                    "metric": metric_label,
                    "preferred_direction": "lower",
                    "setting": setting,
                    "left_method": dp_gated,
                    "right_method": baseline,
                    "left_value": left,
                    "right_value": right,
                    "delta_left_minus_right": left - right,
                    "outcome": compare(left, right, tolerance),
                })
    detail = pd.DataFrame(rows)
    summary = summarize_comparisons(detail)

    widest_rows: list[dict] = []
    for setting in config["settings"]:
        values = {
            method: float(pivot.loc[setting, ("fraction of clean classes affected", method)])
            for method in methods
        }
        maximum = max(values.values())
        at_max = [method for method, value in values.items() if np.isclose(value, maximum, atol=tolerance)]
        widest_rows.append({
            "setting": setting,
            "comparison_universe": ";".join(methods),
            "DistrRep_fraction": values["DistrRep"],
            "widest_fraction": maximum,
            "methods_at_widest": ";".join(at_max),
            "n_methods_tied_at_widest": len(at_max),
            "DistrRep_is_widest_including_ties": "DistrRep" in at_max,
            "DistrRep_is_unique_widest": at_max == ["DistrRep"],
        })
    widest = pd.DataFrame(widest_rows)

    arachne_rows: list[dict] = []
    comparators = [method for method in methods if method != "Arachne"]
    for comparator in comparators:
        for setting in config["settings"]:
            a_fraction = float(pivot.loc[setting, ("fraction of clean classes affected", "Arachne")])
            c_fraction = float(pivot.loc[setting, ("fraction of clean classes affected", comparator)])
            a_worst = float(pivot.loc[setting, ("worst-class Reg", "Arachne")])
            c_worst = float(pivot.loc[setting, ("worst-class Reg", comparator)])
            lower_fraction = a_fraction < c_fraction - tolerance
            higher_worst = a_worst > c_worst + tolerance
            arachne_rows.append({
                "setting": setting,
                "comparator": comparator,
                "Arachne_fraction": a_fraction,
                "comparator_fraction": c_fraction,
                "Arachne_worst_class_Reg": a_worst,
                "comparator_worst_class_Reg": c_worst,
                "Arachne_lower_affected_fraction": lower_fraction,
                "Arachne_higher_worst_class_Reg": higher_worst,
                "both_conditions": lower_fraction and higher_worst,
            })
    arachne_detail = pd.DataFrame(arachne_rows)
    arachne_summary = (
        arachne_detail.groupby("comparator", as_index=False)
        .agg(
            n_settings=("setting", "size"),
            lower_affected_fraction=("Arachne_lower_affected_fraction", "sum"),
            higher_worst_class_Reg=("Arachne_higher_worst_class_Reg", "sum"),
            both_conditions=("both_conditions", "sum"),
        )
    )
    return per_setting, overall, detail, summary, widest, arachne_detail, arachne_summary


def positive_type_strength_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Task 3: failure-type RR strength conditional on an observed positive repair rate."""
    raw = pd.read_csv(resolve(config["sources"]["failure_type_per_type"]))
    expected_support = int(config["min_type_support"])
    positive = raw[
        raw.estimable.astype(str).str.lower().eq("true")
        & raw.method.isin(config["type_distribution_methods"])
        & (raw.RR_type > 0)
    ].copy()
    if positive.empty or (positive.n_fail < expected_support).any():
        raise AssertionError("invalid positive failure-type population")
    assert_rates(positive, ["RR_type"])

    # Recompute the hierarchy after conditioning, so each valid setting, seed,
    # and positive-RR type has equal nested mass in the conditional population.
    positive["n_positive_types_in_cell"] = positive.groupby(
        ["method", "setting", "seed"]
    ).RR_type.transform("size")
    positive["n_positive_cells_in_setting"] = positive.groupby(
        ["method", "setting"]
    ).seed.transform("nunique")
    positive["n_valid_settings"] = positive.groupby("method").setting.transform("nunique")
    positive["conditional_weight"] = (
        1.0 / positive.n_valid_settings
        / positive.n_positive_cells_in_setting
        / positive.n_positive_types_in_cell
    )

    rows: list[dict] = []
    for method in config["type_distribution_methods"]:
        group = positive[positive.method == method]
        if group.empty or not np.isclose(group.conditional_weight.sum(), 1.0):
            raise AssertionError(f"invalid conditional hierarchy for {method}")
        values = group.RR_type.to_numpy(dtype=float)
        weights = group.conditional_weight.to_numpy(dtype=float)
        p25, median, p75 = weighted_quantile(
            values, weights, np.array([0.25, 0.50, 0.75])
        )
        rows.append({
            "method": method,
            "n_settings_total": len(config["settings"]),
            "n_settings_with_positive_types": int(group.setting.nunique()),
            "n_cells_with_positive_types": int(group.groupby(["setting", "seed"]).ngroups),
            "n_positive_type_rows": int(len(group)),
            "mean_RR_given_positive": float(np.average(values, weights=weights)),
            "p25_RR_given_positive": p25,
            "median_RR_given_positive": median,
            "p75_RR_given_positive": p75,
            "condition": f"estimable ordered-pair failure type, n_fail >= {expected_support}, RR_type > 0",
            "weighting": "equal valid setting, then equal positive seed-cell, then equal positive type",
        })
    keep = [
        "method", "source_method", "setting", "setting_label", "seed", "failure_type",
        "n_fail", "n_repaired", "RR_type", "conditional_weight",
        "n_positive_types_in_cell", "n_positive_cells_in_setting", "n_valid_settings",
    ]
    return positive[keep].copy(), pd.DataFrame(rows)


def type_distribution_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = resolve(config["sources"]["failure_type_per_type"])
    raw = pd.read_csv(source)
    assert_rates(raw, ["RR_type"])
    expected_support = int(config["min_type_support"])
    if not raw.min_support_for_summary.eq(expected_support).all():
        raise AssertionError("failure-type source uses a different support threshold")
    estimable = raw[raw.estimable.astype(str).str.lower().eq("true")].copy()
    estimable = estimable[estimable.method.isin(config["type_distribution_methods"])]
    if (estimable.n_fail < expected_support).any():
        raise AssertionError("an under-supported failure type entered the distribution")

    # Hierarchical weights: setting -> seed -> type. Each level is balanced.
    estimable["n_types_in_cell"] = estimable.groupby(
        ["method", "setting", "seed"]
    ).RR_type.transform("size")
    estimable["n_seeds_in_setting"] = estimable.groupby(
        ["method", "setting"]
    ).seed.transform("nunique")
    estimable["n_valid_settings"] = estimable.groupby("method").setting.transform("nunique")
    estimable["weight"] = (
        1.0 / estimable.n_valid_settings
        / estimable.n_seeds_in_setting
        / estimable.n_types_in_cell
    )

    quantile_rows: list[dict] = []
    ecdf_rows: list[dict] = []
    for method in config["type_distribution_methods"]:
        group = estimable[estimable.method == method].copy()
        if group.empty:
            raise AssertionError(f"no estimable failure types for {method}")
        if not np.isclose(group.weight.sum(), 1.0):
            raise AssertionError(f"hierarchical weights do not sum to one for {method}")
        values = group.RR_type.to_numpy(dtype=float)
        weights = group.weight.to_numpy(dtype=float)
        q10, q25, median, q75, q90 = weighted_quantile(
            values, weights, np.array([0.10, 0.25, 0.50, 0.75, 0.90])
        )
        quantile_rows.append({
            "method": method,
            "n_settings_total": len(config["settings"]),
            "n_settings_estimable": int(group.setting.nunique()),
            "n_cells_estimable": int(group.groupby(["setting", "seed"]).ngroups),
            "n_type_rows_estimable": int(len(group)),
            "q10": q10,
            "q25": q25,
            "median": median,
            "q75": q75,
            "q90": q90,
            "coverage_at_0.5": float(group.loc[group.RR_type >= config["coverage_threshold"], "weight"].sum()),
            "partial_share_0_lt_rr_lt_1": float(group.loc[(group.RR_type > 0) & (group.RR_type < 1), "weight"].sum()),
            "zero_share": float(group.loc[group.RR_type == 0, "weight"].sum()),
            "one_share": float(group.loc[group.RR_type == 1, "weight"].sum()),
            "population": f"held-out ordered-pair failure types with n_fail >= {expected_support}",
            "weighting": "equal setting, then equal seed, then equal type",
        })

        ordered = group.sort_values("RR_type", kind="stable")
        cumulative = ordered.weight.cumsum()
        for value, weight, cdf in zip(ordered.RR_type, ordered.weight, cumulative):
            ecdf_rows.append({
                "method": method,
                "RR_type": float(value),
                "weight": float(weight),
                "cumulative_weight": float(cdf),
            })

    return pd.DataFrame(quantile_rows), pd.DataFrame(ecdf_rows)


def plot_operating_region(config: dict, overall: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"blue": "#2F6F73", "orange": "#C66A45", "ink": "#2C3137",
              "gray": "#78838E", "light": "#E8EBEF"}
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    baselines = overall[overall.method.isin(config["baseline_methods"])]
    ax.scatter(baselines.Reg, baselines.RR_held, s=54, marker="o",
               facecolors="white", edgecolors=colors["orange"], linewidths=1.5, zorder=3)

    offsets = {
        "FullFT": (5, 7), "HeadFT": (5, 6), "Arachne": (5, -13),
        "DistrRep": (-55, 7), "NN-Patching": (5, 7), "PatchNAS": (-55, -14),
    }
    for row in baselines.itertuples():
        dx, dy = offsets[row.method]
        ax.annotate(row.method, (row.Reg, row.RR_held), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=colors["ink"])

    dyn_order = config["dynapatch_operating_points"]
    dyn = overall.set_index("method").loc[dyn_order].reset_index()
    ax.plot(dyn.Reg, dyn.RR_held, color=colors["blue"], linewidth=2.0,
            marker="o", markersize=5.5, zorder=4)
    dyn_labels = {"DynaPatch (ungated)": "NoGate", "DynaPatch (gated)": "Gated"}
    dyn_offsets = {"NoGate": (-8, 9), "Gated": (-2, -15)}
    for row in dyn.itertuples():
        label = dyn_labels[row.method]
        dx, dy = dyn_offsets[label]
        ax.annotate(label, (row.Reg, row.RR_held), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=colors["blue"],
                    fontweight="bold")

    fig.suptitle("Repair–regression operating points", x=0.125, y=0.985,
                 ha="left", fontsize=12, fontweight="bold", color=colors["ink"])
    fig.text(
        0.125, 0.938,
        "Setting-balanced means over 12 dataset–architecture settings; lower Reg and higher RR are preferable",
        ha="left", fontsize=8, color=colors["gray"],
    )
    ax.set_xlabel("Regression rate (Reg, clean test)")
    ax.set_ylabel("Repair rate (RR, held-out failure test)")
    ax.set_xlim(left=0)
    ax.set_ylim(0.14, 0.75)
    ax.grid(True, color=colors["light"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"operating_region.{suffix}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_type_ecdf(config: dict, ecdf: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "FullFT": ("#78838E", "--", 1.7),
        "NN-Patching": ("#C66A45", "-", 1.9),
        "PatchNAS": ("#C66A45", ":", 2.1),
        "DynaPatch (ungated)": ("#2F6F73", "-", 2.2),
    }
    labels = {"DynaPatch (ungated)": "DynaPatch-NoGate"}
    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    for method in config["ecdf_plot_methods"]:
        group = ecdf[ecdf.method == method]
        color, line_style, width = styles[method]
        x = np.r_[0.0, group.RR_type.to_numpy(), 1.0]
        y = np.r_[0.0, group.cumulative_weight.to_numpy(), 1.0]
        ax.step(x, y, where="post", label=labels.get(method, method), color=color,
                linestyle=line_style, linewidth=width)
    ax.axvline(0.5, color="#2C3137", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.text(0.507, 0.035, "RR=0.5", fontsize=8, color="#2C3137")
    fig.suptitle("Failure-type repair-rate distributions", x=0.125, y=0.985,
                 ha="left", fontsize=12, fontweight="bold", color="#2C3137")
    fig.text(
        0.125, 0.938,
        "Weighted ECDF over held-out ordered-pair types with n≥5; settings, seeds, and types contribute equally",
        ha="left", fontsize=8, color="#78838E",
    )
    ax.set_xlabel("Failure-type repair rate")
    ax.set_ylabel("Cumulative share of estimable failure types")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#E8EBEF", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"type_rr_ecdf.{suffix}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_regression_concentration(
    config: dict, overall: pd.DataFrame, figure_dir: Path
) -> None:
    """Task 4: show whether side effects are broad, severe, or both."""
    import matplotlib.pyplot as plt

    colors = {"blue": "#2F6F73", "orange": "#C66A45", "ink": "#2C3137",
              "gray": "#78838E", "light": "#E8EBEF"}
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    baselines = overall[overall.method.isin(config["baseline_methods"])]
    ax.scatter(
        baselines.fraction_clean_classes_affected, baselines.worst_class_Reg,
        s=54, marker="o", facecolors="white", edgecolors=colors["orange"],
        linewidths=1.5, zorder=3,
    )
    fixed = overall[overall.method == "FixedPatch"]
    ax.scatter(
        fixed.fraction_clean_classes_affected, fixed.worst_class_Reg,
        s=50, marker="D", facecolors="white", edgecolors=colors["gray"],
        linewidths=1.4, zorder=3,
    )
    dyn = overall[overall.method.isin(["DynaPatch (ungated)", "DynaPatch (gated)"])]
    ax.plot(
        dyn.fraction_clean_classes_affected, dyn.worst_class_Reg,
        color=colors["blue"], linewidth=1.6, zorder=2,
    )
    ax.scatter(
        dyn.fraction_clean_classes_affected, dyn.worst_class_Reg,
        s=56, marker="o", facecolors=colors["blue"], edgecolors=colors["blue"],
        linewidths=1.2, zorder=4,
    )

    labels = {
        "DynaPatch (ungated)": "DynaPatch-NoGate",
        "DynaPatch (gated)": "DynaPatch (gated)",
    }
    offsets = {
        "FullFT": (6, -14), "HeadFT": (-48, 8), "Arachne": (-38, 8),
        "DistrRep": (-58, 8), "FixedPatch": (6, -14),
        "NN-Patching": (6, 7), "PatchNAS": (-55, -14),
        "DynaPatch (ungated)": (-82, 8), "DynaPatch (gated)": (6, -14),
    }
    for row in overall.itertuples():
        dx, dy = offsets[row.method]
        color = colors["blue"] if row.method.startswith("DynaPatch") else colors["ink"]
        weight = "bold" if row.method.startswith("DynaPatch") else "normal"
        ax.annotate(
            labels.get(row.method, row.method),
            (row.fraction_clean_classes_affected, row.worst_class_Reg),
            xytext=(dx, dy), textcoords="offset points", fontsize=8,
            color=color, fontweight=weight,
        )

    fig.suptitle("Regression breadth and worst-class severity", x=0.125, y=0.985,
                 ha="left", fontsize=12, fontweight="bold", color=colors["ink"])
    fig.text(
        0.125, 0.938,
        "Setting-balanced clean-test class profiles over 12 dataset–architecture settings; lower is preferable on both axes",
        ha="left", fontsize=8, color=colors["gray"],
    )
    ax.set_xlabel("Fraction of clean classes affected")
    ax.set_ylabel("Worst-class regression rate")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=colors["light"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"regression_breadth_vs_severity.{suffix}", dpi=240,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/analysis/rq2_baseline_behavior.json",
        help="JSON config path, relative to the repository root unless absolute",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text())
    output_dir = resolve(config["output_dir"])
    figure_dir = resolve(config["figure_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    per_setting, overall = main_metrics(config)
    consistency_detail, consistency_summary = consistency_tables(config, per_setting)
    type_summary, type_ecdf = type_distribution_tables(config)
    failure_wtl_detail, failure_wtl_summary = failure_type_wtl_tables(config)
    (
        regression_per_setting,
        regression_overall,
        regression_wtl_detail,
        regression_wtl_summary,
        distrrep_widest,
        arachne_spike_detail,
        arachne_spike_summary,
    ) = regression_behavior_tables(config)
    positive_type_rows, positive_type_summary = positive_type_strength_tables(config)

    per_setting.to_csv(output_dir / "operating_points_per_setting.csv", index=False)
    overall.to_csv(output_dir / "operating_points_overall.csv", index=False)
    consistency_detail.to_csv(output_dir / "consistency_per_setting.csv", index=False)
    consistency_summary.to_csv(output_dir / "consistency_summary.csv", index=False)
    type_summary.to_csv(output_dir / "type_rr_distribution_summary.csv", index=False)
    type_ecdf.to_csv(output_dir / "type_rr_ecdf.csv", index=False)
    failure_wtl_detail.to_csv(output_dir / "failure_type_wtl_per_setting.csv", index=False)
    failure_wtl_summary.to_csv(output_dir / "failure_type_wtl_summary.csv", index=False)
    regression_per_setting.to_csv(output_dir / "regression_behavior_per_setting.csv", index=False)
    regression_overall.to_csv(output_dir / "regression_behavior_overall.csv", index=False)
    regression_wtl_detail.to_csv(output_dir / "regression_wtl_per_setting.csv", index=False)
    regression_wtl_summary.to_csv(output_dir / "regression_wtl_summary.csv", index=False)
    distrrep_widest.to_csv(output_dir / "distrrep_widest_affected_per_setting.csv", index=False)
    arachne_spike_detail.to_csv(output_dir / "arachne_localized_spike_per_setting.csv", index=False)
    arachne_spike_summary.to_csv(output_dir / "arachne_localized_spike_summary.csv", index=False)
    positive_type_rows.to_csv(output_dir / "positive_type_rr_weighted.csv", index=False)
    positive_type_summary.to_csv(output_dir / "positive_type_rr_summary.csv", index=False)

    plot_operating_region(config, overall, figure_dir)
    plot_type_ecdf(config, type_ecdf, figure_dir)
    plot_regression_concentration(config, regression_overall, figure_dir)

    sources = {key: resolve(value) for key, value in config["sources"].items()}
    provenance = {
        "analysis": "RQ2 baseline behavior",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"{sys.executable} scripts/analyze_rq2_baseline_behavior.py --config {args.config}",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "sources": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for key, path in sources.items()
        },
        "population": {
            "repair": "held-out base-wrong failures",
            "regression": "base-correct clean test",
            "failure_type": f"ordered true-label->base-prediction types with n_fail >= {config['min_type_support']}",
        },
        "aggregation": "mean seeds within setting, then mean 12 settings",
        "type_distribution_weighting": "equal setting, then equal seed, then equal estimable failure type",
        "selection_warning": "DynaPatch gate thresholds are evaluation-frontier upper bounds, not deployable calibration results",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[written] {output_dir.relative_to(ROOT)}")
    print(f"[written] {figure_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
