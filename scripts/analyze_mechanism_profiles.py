#!/usr/bin/env python3
"""Export test-only mechanism profiles for the four-RQ results package.

This is a reporting analysis, not a model-selection script.  It uses only
``held`` (bug_eval) for repair and ``clean`` (clean_test) for regression.
It never reads the ``seen`` rows when producing a metric.

Outputs retain one row per setting and seed before any setting-level summary.
The ordered-pair failure type (true label -> deployed-model prediction) is
often sparse, so every type remains in the raw table while Coverage@0.5 and
the low-type summaries use only types with at least ``--min-type-support``
held failures.  The denominator is printed in every row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import names as N  # noqa: E402


METHODS = {
    "FullFT": "FullFT",
    "HeadFT": "HeadFT",
    "Arachne": "Arachne",
    "DistrRep": "DistrRep",
    "FixedPatch": "FixedPatch",
    "NN-Patching [tau]": "NN-Patching",
    "PatchNAS [tau]": "PatchNAS",
    "DynaPatch (ungated)": "DynaPatch (ungated)",
    "DynaPatch (gated)": "DynaPatch (gated)",
}
ARMS = ("pre", "post", "pre+post")
RS = (0.60, 0.80, 0.90)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def check_rate(frame: pd.DataFrame, numerator: str, denominator: str,
               rate: str) -> None:
    valid = frame[denominator] > 0
    expected = frame.loc[valid, numerator] / frame.loc[valid, denominator]
    if not np.allclose(expected, frame.loc[valid, rate], equal_nan=True):
        raise AssertionError(f"{rate} does not equal {numerator}/{denominator}")
    if (frame.loc[valid, numerator] > frame.loc[valid, denominator]).any():
        raise AssertionError(f"{numerator} exceeds {denominator}")


def assert_paired_test_populations(sample: pd.DataFrame) -> None:
    """Prevent method-specific test rows or base predictions from reaching a comparison."""
    test = sample[sample.split.isin(("clean", "held"))]
    keys = ["setting", "seed", "split"]
    for cell, group in test.groupby(keys, sort=False):
        by_method = {
            method: set(rows.dataset_index.astype(int))
            for method, rows in group.groupby("source_method")
        }
        reference = next(iter(by_method.values()))
        unequal = {
            method: len(indices.symmetric_difference(reference))
            for method, indices in by_method.items()
            if indices != reference
        }
        if unequal:
            raise AssertionError(f"unpaired test identities in {cell}: {unequal}")
        if group.duplicated(["source_method", "dataset_index"]).any():
            raise AssertionError(f"duplicate method/input row in {cell}")
        evidence = group.pivot_table(
            index="dataset_index",
            columns="source_method",
            values=["label", "base_pred"],
            aggfunc="first",
        )
        if (evidence["label"].nunique(axis=1) > 1).any():
            raise AssertionError(f"true-label disagreement in paired cell {cell}")
        if (evidence["base_pred"].nunique(axis=1) > 1).any():
            raise AssertionError(f"base-prediction disagreement in paired cell {cell}")


def setting_mean(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [column for column in frame.select_dtypes(include="number").columns
               if column != "seed"]
    grouped = frame.groupby(keys, dropna=False)
    mean = grouped[numeric].mean().add_suffix("_seed_mean")
    median = grouped[numeric].median().add_suffix("_seed_median")
    valid = grouped[numeric].count().add_suffix("_n_valid")
    counts = grouped.seed.nunique().rename("n_seeds")
    return pd.concat([counts, valid, mean, median], axis=1).reset_index()


def setting_balanced_overall(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Collapse setting rows without sample- or seed-weighting them."""
    numeric = [
        column for column in frame.select_dtypes(include="number").columns
        if column not in {"n_seeds"} and not column.endswith("_n_valid")
    ]
    grouped = frame.groupby(keys, dropna=False)
    mean = grouped[numeric].mean().add_suffix("_setting_mean")
    median = grouped[numeric].median().add_suffix("_setting_median")
    valid = grouped[numeric].count().add_suffix("_n_settings_valid")
    counts = grouped.setting.nunique().rename("n_settings")
    return pd.concat([counts, valid, mean, median], axis=1).reset_index()


def paper_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the paper's short setting label and sort without leaking rank columns."""
    frame = frame.copy()
    sort_columns: list[str] = []
    if "setting" in frame:
        frame.insert(
            frame.columns.get_loc("setting") + 1,
            "setting_label",
            frame.setting.map(N.SETTING_LABEL),
        )
        frame["_setting_rank"] = frame.setting.map(
            {setting: index for index, setting in enumerate(N.SETTING_LABEL)}
        )
        sort_columns.append("_setting_rank")
    if "method" in frame:
        frame["_method_rank"] = frame.method.map(
            {method: index for index, method in enumerate(METHODS.values())}
        )
        sort_columns.append("_method_rank")
    if "r" in frame:
        sort_columns.append("r")
    if "arm" in frame:
        frame["_arm_rank"] = frame.arm.map(
            {arm: index for index, arm in enumerate(ARMS)}
        )
        sort_columns.append("_arm_rank")
    for column in ("seed", "failure_type", "outcome"):
        if column in frame:
            sort_columns.append(column)
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable")
    return frame.drop(columns=[c for c in frame if c.startswith("_")])


def failure_type_tables(sample: pd.DataFrame, min_support: int
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    held = sample[(sample.split == "held") & (~sample.base_correct)].copy()
    if held.failure_type.isna().any():
        raise AssertionError("base-wrong held rows must name a failure_type")
    expected_type = held.label.astype(str) + "->" + held.base_pred.astype(str)
    if not expected_type.equals(held.failure_type.astype(str)):
        raise AssertionError("failure_type is not label->base_pred")
    if held.duplicated(["source_method", "setting", "seed", "dataset_index"]).any():
        raise AssertionError("duplicate held input inside a method/setting/seed cell")

    per_type = (
        held.groupby(
            ["method", "source_method", "setting", "seed", "failure_type"],
            as_index=False,
        )
        .agg(n_fail=("repaired", "size"), n_repaired=("repaired", "sum"))
    )
    per_type["RR_type"] = per_type.n_repaired / per_type.n_fail
    per_type["estimable"] = per_type.n_fail >= min_support
    per_type["population"] = "base-wrong repair_holdout_unseen"
    per_type["min_support_for_summary"] = min_support
    check_rate(per_type, "n_repaired", "n_fail", "RR_type")

    rows: list[dict] = []
    for keys, group in per_type.groupby(
        ["method", "source_method", "setting", "seed"], sort=False
    ):
        estimable = group[group.estimable]
        n_fail = int(group.n_fail.sum())
        n_repaired = int(group.n_repaired.sum())
        rows.append({
            "method": keys[0],
            "source_method": keys[1],
            "setting": keys[2],
            "seed": keys[3],
            "population": "base-wrong repair_holdout_unseen",
            "n_fail": n_fail,
            "n_repaired": n_repaired,
            "RR_micro": n_repaired / n_fail,
            "n_types_total": int(len(group)),
            "n_types_repaired_any": int((group.n_repaired > 0).sum()),
            "frac_types_repaired_any": float((group.n_repaired > 0).mean()),
            "n_types_estimable": int(len(estimable)),
            "RR_macro_estimable": (
                float(estimable.RR_type.mean()) if len(estimable) else np.nan
            ),
            "RR_median_estimable": (
                float(estimable.RR_type.median()) if len(estimable) else np.nan
            ),
            "RR_p10_estimable": (
                float(estimable.RR_type.quantile(0.10)) if len(estimable) else np.nan
            ),
            "coverage_at_0.5": (
                float((estimable.RR_type >= 0.5).mean()) if len(estimable) else np.nan
            ),
            "min_support_for_summary": min_support,
        })
    per_cell = pd.DataFrame(rows)
    check_rate(per_cell, "n_repaired", "n_fail", "RR_micro")

    # Verify that grouping did not silently lose a held failure.
    expected = held.groupby(["method", "setting", "seed"]).size().sort_index()
    observed = per_cell.set_index(["method", "setting", "seed"]).n_fail.sort_index()
    if not expected.equals(observed):
        raise AssertionError("failure-type denominators do not reconstruct held cells")

    per_setting = setting_mean(per_cell, ["method", "source_method", "setting"])
    return per_type, per_cell, per_setting


def regression_tables(sample: pd.DataFrame, source: Path
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    regression = pd.read_csv(source)
    regression = regression[regression.method.isin(METHODS)].copy()
    regression["source_method"] = regression.method
    regression["method"] = regression.method.map(METHODS)
    if regression.duplicated(["method", "setting", "seed"]).any():
        raise AssertionError("duplicate regression profile cell")

    # Reconstruct the headline numerator and denominator from clean_test.
    clean = sample[(sample.split == "clean") & sample.base_correct]
    direct = (
        clean.groupby(["method", "setting", "seed"], as_index=False)
        .agg(n_clean=("regressed", "size"), n_regressed_check=("regressed", "sum"))
    )
    direct["Reg_check"] = direct.n_regressed_check / direct.n_clean
    regression = regression.merge(
        direct, on=["method", "setting", "seed"], validate="one_to_one"
    )
    if not np.allclose(regression.Reg_overall, regression.Reg_check):
        raise AssertionError("per-class Reg does not reconstruct from clean_test")
    if not (regression.n_regressed == regression.n_regressed_check).all():
        raise AssertionError("per-class regression counts do not reconstruct")
    regression["population"] = "base-correct clean_eval"
    per_setting = setting_mean(regression, ["method", "source_method", "setting"])
    return regression, per_setting


def postinfo_tables(outcome_source: Path, brhr_source: Path
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome = pd.read_csv(outcome_source)
    brhr = pd.read_csv(brhr_source)
    outcome = outcome[
        outcome.arm.isin(ARMS) & outcome.r.round(2).isin(RS)
    ].copy()
    brhr = brhr[brhr.arm.isin(ARMS) & brhr.r.round(2).isin(RS)].copy()
    if brhr.duplicated(["arm", "r", "setting", "seed"]).any():
        raise AssertionError("duplicate post-information BR/HR cell")

    beneficial = outcome[outcome.outcome == "A_beneficial"].set_index(
        ["arm", "r", "setting", "seed"]
    )
    harmful = outcome[outcome.outcome == "D_harmful"].set_index(
        ["arm", "r", "setting", "seed"]
    )
    check = brhr.set_index(["arm", "r", "setting", "seed"])
    common_b = check.index.intersection(beneficial.index)
    common_h = check.index.intersection(harmful.index)
    if not np.allclose(
        check.loc[common_b, "BR"], beneficial.loc[common_b, "apply_rate"]
    ):
        raise AssertionError("BR is not P(apply | beneficial)")
    if not np.allclose(
        check.loc[common_h, "HR"], 1 - harmful.loc[common_h, "apply_rate"]
    ):
        raise AssertionError("HR is not 1-P(apply | harmful)")

    brhr["beneficial_apply_rate"] = brhr["BR"]
    brhr["harmful_apply_rate"] = 1 - brhr["HR"]
    brhr["beneficial_population"] = "A_beneficial on repair_holdout_unseen"
    brhr["harmful_population"] = "D_harmful on clean_eval"
    setting_brhr = setting_mean(brhr, ["arm", "r", "setting"])
    setting_outcome = setting_mean(outcome, ["arm", "r", "setting", "outcome"])
    return outcome, brhr, setting_outcome, setting_brhr


def md_table(frame: pd.DataFrame, columns: list[str], decimals: dict[str, int]
             ) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("—")
            elif column in decimals:
                values.append(f"{float(value):.{decimals[column]}f}")
            elif isinstance(value, (float, np.floating)) and value.is_integer():
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(output: Path, ft: pd.DataFrame, reg: pd.DataFrame,
                   post: pd.DataFrame, min_support: int) -> None:
    rank = {setting: index for index, setting in enumerate(N.SETTING_LABEL)}
    method_rank = {method: index for index, method in enumerate(METHODS.values())}
    ft = ft.assign(
        setting_rank=ft.setting.map(rank), method_rank=ft.method.map(method_rank)
    ).sort_values(["setting_rank", "method_rank"])
    reg = reg.assign(
        setting_rank=reg.setting.map(rank), method_rank=reg.method.map(method_rank)
    ).sort_values(["setting_rank", "method_rank"])
    arm_rank = {arm: index for index, arm in enumerate(ARMS)}
    post = post.assign(
        setting_rank=post.setting.map(rank), arm_rank=post.arm.map(arm_rank)
    ).sort_values(["r", "setting_rank", "arm_rank"])

    lines = [
        "# Mechanism profiles — raw 12-setting tables",
        "",
        "This file contains data tables only. All repair quantities use held-out `bug_eval`; "
        "all regression quantities use `clean_test`. PatchPro is excluded.",
        "",
        "## Failure-type repair distribution",
        "",
        f"A failure type is `(true label -> deployed-model prediction)`. Coverage uses only "
        f"types with `n_fail >= {min_support}`; all types remain in "
        "`failure_type_per_type.csv`.",
        "",
    ]
    ft_columns = [
        "setting_label", "method", "n_seeds", "coverage_at_0.5_n_valid",
        "n_fail_seed_mean", "RR_micro_seed_mean",
        "n_types_total_seed_mean", "n_types_estimable_seed_mean",
        "n_types_repaired_any_seed_mean", "coverage_at_0.5_seed_mean",
        "RR_median_estimable_seed_mean", "RR_p10_estimable_seed_mean",
    ]
    lines += md_table(ft, ft_columns, {
        "coverage_at_0.5_n_valid": 0, "n_fail_seed_mean": 1, "RR_micro_seed_mean": 4,
        "n_types_total_seed_mean": 1, "n_types_estimable_seed_mean": 1,
        "n_types_repaired_any_seed_mean": 1, "coverage_at_0.5_seed_mean": 4,
        "RR_median_estimable_seed_mean": 4, "RR_p10_estimable_seed_mean": 4,
    })
    lines += ["", "## Regression distribution across clean true classes", ""]
    reg_columns = [
        "setting_label", "method", "n_seeds", "n_clean_seed_mean",
        "n_regressed_seed_mean", "Reg_overall_seed_mean", "Reg_macro_seed_mean",
        "Reg_max_class_seed_mean", "n_classes_hit_seed_mean",
        "frac_classes_hit_seed_mean", "reg_spread_seed_mean",
    ]
    lines += md_table(reg, reg_columns, {
        "n_clean_seed_mean": 1, "n_regressed_seed_mean": 1,
        "Reg_overall_seed_mean": 4, "Reg_macro_seed_mean": 4,
        "Reg_max_class_seed_mean": 4, "n_classes_hit_seed_mean": 1,
        "frac_classes_hit_seed_mean": 4, "reg_spread_seed_mean": 4,
    })
    lines += ["", "## Beneficial and harmful gate application", ""]
    post_columns = [
        "setting_label", "r", "arm", "n_seeds", "n__A_beneficial_seed_mean",
        "beneficial_apply_rate_seed_mean", "n__D_harmful_seed_mean",
        "harmful_apply_rate_seed_mean", "apply_rate__C_safe_seed_mean",
    ]
    lines += md_table(post, post_columns, {
        "r": 2, "n__A_beneficial_seed_mean": 1,
        "beneficial_apply_rate_seed_mean": 4, "n__D_harmful_seed_mean": 1,
        "harmful_apply_rate_seed_mean": 4, "apply_rate__C_safe_seed_mean": 4,
    })
    (output / "RAW_TABLES.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frame", default="outputs/sample_frame.csv.gz")
    parser.add_argument("--regression", default="outputs/p1_regression/per_cell.csv")
    parser.add_argument("--postinfo-outcome", default="outputs/p3_postinfo/per_outcome.csv")
    parser.add_argument("--postinfo-brhr", default="outputs/p3_postinfo/br_hr.csv")
    parser.add_argument("--output", default="outputs/mechanism_profiles")
    parser.add_argument("--min-type-support", type=int, default=5)
    args = parser.parse_args()
    if args.min_type_support <= 0:
        parser.error("--min-type-support must be positive")

    source_paths = {
        "sample_frame": ROOT / args.sample_frame,
        "regression": ROOT / args.regression,
        "postinfo_outcome": ROOT / args.postinfo_outcome,
        "postinfo_brhr": ROOT / args.postinfo_brhr,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    sample = pd.read_csv(source_paths["sample_frame"], low_memory=False)
    required = {
        "method", "setting", "seed", "split", "dataset_index", "label",
        "base_pred", "base_correct", "repaired", "regressed", "failure_type",
    }
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"sample frame missing columns: {sorted(missing)}")
    sample = sample[sample.method.isin(METHODS)].copy()
    sample["source_method"] = sample.method
    sample["method"] = sample.method.map(METHODS)
    for column in ("base_correct", "repaired", "regressed"):
        sample[column] = as_bool(sample[column])
    if not set(sample.split.unique()) <= {"clean", "held", "seen"}:
        raise AssertionError("unexpected sample-frame split")
    assert_paired_test_populations(sample)

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    per_type, ft_cell, ft_setting = failure_type_tables(sample, args.min_type_support)
    reg_cell, reg_setting = regression_tables(sample, source_paths["regression"])
    outcome_cell, brhr_cell, outcome_setting, brhr_setting = postinfo_tables(
        source_paths["postinfo_outcome"], source_paths["postinfo_brhr"]
    )
    ft_overall = setting_balanced_overall(ft_setting, ["method", "source_method"])
    reg_overall = setting_balanced_overall(reg_setting, ["method", "source_method"])
    brhr_overall = setting_balanced_overall(brhr_setting, ["arm", "r"])

    per_type = paper_order(per_type)
    ft_cell = paper_order(ft_cell)
    ft_setting = paper_order(ft_setting)
    ft_overall = paper_order(ft_overall)
    reg_cell = paper_order(reg_cell)
    reg_setting = paper_order(reg_setting)
    reg_overall = paper_order(reg_overall)
    outcome_cell = paper_order(outcome_cell)
    outcome_setting = paper_order(outcome_setting)
    brhr_cell = paper_order(brhr_cell)
    brhr_setting = paper_order(brhr_setting)
    brhr_overall = paper_order(brhr_overall)

    tables = {
        "failure_type_per_type.csv": per_type,
        "failure_type_summary_per_cell.csv": ft_cell,
        "failure_type_summary_per_setting.csv": ft_setting,
        "failure_type_summary_overall.csv": ft_overall,
        "regression_profile_per_cell.csv": reg_cell,
        "regression_profile_per_setting.csv": reg_setting,
        "regression_profile_overall.csv": reg_overall,
        "postinfo_outcome_per_cell.csv": outcome_cell,
        "postinfo_outcome_per_setting.csv": outcome_setting,
        "postinfo_br_hr_per_cell.csv": brhr_cell,
        "postinfo_br_hr_per_setting.csv": brhr_setting,
        "postinfo_br_hr_overall.csv": brhr_overall,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)

    write_markdown(output, ft_setting, reg_setting, brhr_setting,
                   args.min_type_support)
    provenance = {
        "analysis": "test-only mechanism profiles",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "reported_populations": {
            "repair": "base-wrong repair_holdout_unseen (bug_eval)",
            "regression": "base-correct clean_eval (clean_test)",
            "postinfo": "repair_holdout_unseen plus clean_eval, conditioned on outcome",
            "training_or_seen_reported": False,
        },
        "method_mapping": METHODS,
        "patchpro_included": False,
        "min_failure_type_support": args.min_type_support,
        "sources": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for key, path in source_paths.items()
        },
        "outputs": {filename: len(table) for filename, table in tables.items()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    print(f"[written] {output / 'RAW_TABLES.md'}")


if __name__ == "__main__":
    main()
