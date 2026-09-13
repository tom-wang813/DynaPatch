#!/usr/bin/env python3
"""Test whether a fixed patch works when it aligns with failure-specific directions.

The analysis is restricted to ``repair_holdout_unseen``.  Within each setting and seed it:

1. L2-normalises every input-specific DynaPatch residual direction.
2. Computes a normalised centroid for each ordered-pair failure type (label -> base prediction).
3. Measures cosine alignment between that centroid and the selected FixedPatch direction.
4. Pairs the alignment with FixedPatch and DynaPatch repair rates on exactly the same inputs.

Pooled failure types are not treated as independent replicates.  The primary association is a
distribution of within-cell Spearman correlations; pooled and per-setting summaries are exported
only as descriptive sensitivity analyses.  PCA is fitted on DynaPatch directions only, in one
predeclared illustrative cell (G-CN, seed 101), and is not used as quantitative evidence.

Run:
    .venv/bin/python scripts/analyze_fixed_patch_alignment.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynapatch-matplotlib-fixed-alignment")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import names as N  # noqa: E402


SEEDS = (101, 202, 303)
SUPPORT_THRESHOLDS = (2, 3, 5)
PRIMARY_SUPPORT = 3
PCA_SETTING = "gtsrb/convnext_tiny"
PCA_SEED = 101
PCA_TOP_TYPES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/analysis/fixed_patch_alignment.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise AssertionError("patch vectors must have finite, non-zero norms")
    return values / norms[:, None], norms


def load_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise AssertionError(f"checkpoint does not contain a state dict: {path}")
    return state


def selected_fixed_arms(sample_path: Path) -> dict[tuple[str, int], str]:
    frame = pd.read_csv(
        sample_path,
        usecols=["method", "setting", "seed", "split", "operating_point"],
    )
    fixed = frame[(frame.method == "FixedPatch") & (frame.split == "held")]
    unique = fixed[["setting", "seed", "operating_point"]].drop_duplicates()
    expected = len(N.SETTING_ORDER) * len(SEEDS)
    if len(unique) != expected:
        raise AssertionError(f"expected {expected} selected fixed cells, found {len(unique)}")
    if unique.duplicated(["setting", "seed"]).any():
        raise AssertionError("multiple FixedPatch arms selected in one setting/seed")
    return {
        (row.setting, int(row.seed)): str(row.operating_point)
        for row in unique.itertuples(index=False)
    }


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(x.to_numpy(), y.to_numpy())
    return float(result.statistic), float(result.pvalue)


def collect_types(
    sample_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    fixed_arms = selected_fixed_arms(sample_path)
    type_rows: list[dict[str, object]] = []
    pca_rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []

    for dataset, backbone, setting_label in N.SETTING_ORDER:
        setting = f"{dataset}/{backbone}"
        for seed in SEEDS:
            arm = fixed_arms[(setting, seed)]
            dp_dir = (
                ROOT
                / f"outputs/effect_dump_ep40ns_v8_s{seed}/{dataset}/{backbone}"
                / "deploy_direct/predictions"
            )
            dp_pred_path = dp_dir / "repair_holdout_unseen_predictions.csv"
            vector_path = dp_dir / "patch_vec_repair_holdout_unseen.npy"
            fixed_root = (
                ROOT
                / f"outputs/patch_ablation/{dataset}_{backbone}/{arm}_s{seed}_kfull"
            )
            fixed_pred_path = (
                fixed_root / "deploy/predictions/repair_holdout_unseen_predictions.csv"
            )
            checkpoint_path = fixed_root / "train/checkpoints/repair_last.pt"
            required = (dp_pred_path, vector_path, fixed_pred_path, checkpoint_path)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("missing frozen source(s): " + ", ".join(missing))

            dp = pd.read_csv(dp_pred_path)
            fixed = pd.read_csv(fixed_pred_path)
            vectors = np.load(vector_path).astype(np.float64)
            if len(dp) != len(vectors):
                raise AssertionError(f"prediction/vector row mismatch: {setting} s{seed}")
            if dp.dataset_index.duplicated().any() or fixed.dataset_index.duplicated().any():
                raise AssertionError(f"duplicate held dataset_index: {setting} s{seed}")
            if bool(dp.base_correct.any()) or bool(fixed.base_correct.any()):
                raise AssertionError(f"held repair population contains base-correct rows: {setting} s{seed}")

            fixed_cols = fixed[[
                "dataset_index", "label", "base_pred", "base_correct", "patched_correct"
            ]].rename(columns={"patched_correct": "fixed_correct"})
            paired = dp.merge(
                fixed_cols,
                on="dataset_index",
                suffixes=("_dp", "_fixed"),
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            if not paired._merge.eq("both").all() or len(paired) != len(dp):
                raise AssertionError(f"FixedPatch and DynaPatch held identities differ: {setting} s{seed}")
            for column in ("label", "base_pred", "base_correct"):
                if not paired[f"{column}_dp"].equals(paired[f"{column}_fixed"]):
                    raise AssertionError(f"base evidence mismatch in {column}: {setting} s{seed}")

            # Restore vector order after the identity-based merge rather than assuming CSV order.
            position = pd.Series(np.arange(len(dp)), index=dp.dataset_index)
            paired["vector_row"] = paired.dataset_index.map(position)
            if paired.vector_row.isna().any():
                raise AssertionError("failed to map DynaPatch vectors by dataset identity")
            ordered_vectors = vectors[paired.vector_row.astype(int).to_numpy()]
            unit_vectors, vector_norms = normalise_rows(ordered_vectors)

            state = load_state(checkpoint_path)
            if "hypernet.delta" not in state:
                raise AssertionError(f"hypernet.delta absent from {checkpoint_path}")
            fixed_delta = state["hypernet.delta"].detach().cpu().numpy().astype(np.float64)
            if fixed_delta.ndim != 1 or fixed_delta.shape[0] != unit_vectors.shape[1]:
                raise AssertionError(
                    f"fixed/DP patch dimension mismatch: {setting} s{seed}: "
                    f"{fixed_delta.shape} vs {unit_vectors.shape}"
                )
            fixed_norm = float(np.linalg.norm(fixed_delta))
            if not np.isfinite(fixed_norm) or fixed_norm <= 0:
                raise AssertionError(f"invalid fixed delta norm: {setting} s{seed}")
            fixed_unit = fixed_delta / fixed_norm

            paired["failure_type"] = (
                paired.label_dp.astype(str) + "->" + paired.base_pred_dp.astype(str)
            )
            paired["dp_correct"] = paired.patched_correct.astype(bool)
            paired["fixed_correct"] = paired.fixed_correct.astype(bool)

            for failure_type, indices in paired.groupby("failure_type", sort=True).groups.items():
                rows = np.asarray(list(indices), dtype=int)
                directions = unit_vectors[rows]
                raw_centroid = directions.mean(axis=0)
                concentration = float(np.linalg.norm(raw_centroid))
                if concentration <= 0:
                    alignment = np.nan
                else:
                    centroid = raw_centroid / concentration
                    alignment = float(fixed_unit @ centroid)
                group = paired.loc[rows]
                n_fail = int(len(group))
                fixed_repaired = int(group.fixed_correct.sum())
                dp_repaired = int(group.dp_correct.sum())
                type_rows.append({
                    "setting": setting,
                    "setting_label": setting_label,
                    "seed": seed,
                    "fixed_arm": arm,
                    "failure_type": failure_type,
                    "true_label": int(group.label_dp.iloc[0]),
                    "base_pred": int(group.base_pred_dp.iloc[0]),
                    "n_fail": n_fail,
                    "fixed_alignment": alignment,
                    "centroid_concentration": concentration,
                    "fixed_n_repaired": fixed_repaired,
                    "fixed_RR": fixed_repaired / n_fail,
                    "dp_n_repaired": dp_repaired,
                    "dp_RR": dp_repaired / n_fail,
                    "delta_RR_dp_minus_fixed": (dp_repaired - fixed_repaired) / n_fail,
                    "fixed_delta_norm": fixed_norm,
                    "dp_patch_norm_mean": float(vector_norms[rows].mean()),
                    "population": "base-wrong repair_holdout_unseen",
                })

            if setting == PCA_SETTING and seed == PCA_SEED:
                for row_index, row in paired.reset_index(drop=True).iterrows():
                    pca_rows.append({
                        "setting": setting,
                        "setting_label": setting_label,
                        "seed": seed,
                        "dataset_index": int(row.dataset_index),
                        "failure_type": row.failure_type,
                        "dp_correct": bool(row.dp_correct),
                        "vector_row": row_index,
                    })
                # Stored temporarily and consumed before returning; avoids a second source read.
                pca_payload = pd.DataFrame(pca_rows)
                pca_payload.attrs["unit_vectors"] = unit_vectors
                pca_payload.attrs["fixed_unit"] = fixed_unit

            for role, path in (
                ("dp_predictions", dp_pred_path),
                ("dp_patch_vectors", vector_path),
                ("fixed_predictions", fixed_pred_path),
                ("fixed_checkpoint", checkpoint_path),
            ):
                sources.append({
                    "setting": setting,
                    "seed": seed,
                    "role": role,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                })

    if "pca_payload" not in locals():
        raise AssertionError("predeclared PCA cell was not found")
    return pd.DataFrame(type_rows), pca_payload, sources


def aggregate_types(per_type: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["setting", "setting_label", "failure_type", "true_label", "base_pred"]
    for key, group in per_type.groupby(keys, sort=False):
        n_fail = int(group.n_fail.sum())
        rows.append({
            **dict(zip(keys, key)),
            "n_seeds": int(group.seed.nunique()),
            "n_fail": n_fail,
            "fixed_alignment_weighted_mean": float(
                np.average(group.fixed_alignment, weights=group.n_fail)
            ),
            "fixed_alignment_seed_mean": float(group.fixed_alignment.mean()),
            "centroid_concentration_seed_mean": float(group.centroid_concentration.mean()),
            "fixed_n_repaired": int(group.fixed_n_repaired.sum()),
            "fixed_RR": float(group.fixed_n_repaired.sum() / n_fail),
            "dp_n_repaired": int(group.dp_n_repaired.sum()),
            "dp_RR": float(group.dp_n_repaired.sum() / n_fail),
            "delta_RR_dp_minus_fixed": float(
                (group.dp_n_repaired.sum() - group.fixed_n_repaired.sum()) / n_fail
            ),
            "population": "base-wrong repair_holdout_unseen",
        })
    return pd.DataFrame(rows)


def correlation_tables(
    per_type: pd.DataFrame,
    per_setting_type: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_rows: list[dict[str, object]] = []
    setting_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    for minimum in SUPPORT_THRESHOLDS:
        eligible = per_type[per_type.n_fail >= minimum]
        for (setting, setting_label, seed), group in eligible.groupby(
            ["setting", "setting_label", "seed"], sort=False
        ):
            fixed_rho, fixed_p = safe_spearman(group.fixed_alignment, group.fixed_RR)
            gain_rho, gain_p = safe_spearman(
                group.fixed_alignment, group.delta_RR_dp_minus_fixed
            )
            cell_rows.append({
                "min_type_support": minimum,
                "setting": setting,
                "setting_label": setting_label,
                "seed": seed,
                "n_types": int(len(group)),
                "n_fail": int(group.n_fail.sum()),
                "rho_alignment_fixed_RR": fixed_rho,
                "p_alignment_fixed_RR": fixed_p,
                "rho_alignment_delta_RR": gain_rho,
                "p_alignment_delta_RR": gain_p,
            })

        eligible_setting = per_setting_type[per_setting_type.n_fail >= minimum]
        for (setting, setting_label), group in eligible_setting.groupby(
            ["setting", "setting_label"], sort=False
        ):
            fixed_rho, fixed_p = safe_spearman(
                group.fixed_alignment_weighted_mean, group.fixed_RR
            )
            gain_rho, gain_p = safe_spearman(
                group.fixed_alignment_weighted_mean, group.delta_RR_dp_minus_fixed
            )
            setting_rows.append({
                "min_type_support_pooled_seeds": minimum,
                "setting": setting,
                "setting_label": setting_label,
                "n_types": int(len(group)),
                "n_fail": int(group.n_fail.sum()),
                "rho_alignment_fixed_RR": fixed_rho,
                "p_alignment_fixed_RR": fixed_p,
                "rho_alignment_delta_RR": gain_rho,
                "p_alignment_delta_RR": gain_p,
                "interpretation": "descriptive; failure types pooled across seeds",
            })

        cell = pd.DataFrame(cell_rows)
        current = cell[cell.min_type_support == minimum]
        for metric in ("rho_alignment_fixed_RR", "rho_alignment_delta_RR"):
            valid = current[metric].dropna()
            sensitivity_rows.append({
                "min_type_support": minimum,
                "association": metric,
                "n_cells_valid": int(len(valid)),
                "n_cells_positive": int((valid > 0).sum()),
                "n_cells_negative": int((valid < 0).sum()),
                "median_within_cell_rho": float(valid.median()) if len(valid) else np.nan,
                "mean_within_cell_rho": float(valid.mean()) if len(valid) else np.nan,
                "primary_evidence": minimum == PRIMARY_SUPPORT,
            })

    return pd.DataFrame(cell_rows), pd.DataFrame(setting_rows), pd.DataFrame(sensitivity_rows)


def separation_vs_gain() -> tuple[pd.DataFrame, dict[str, float | int]]:
    geometry_path = ROOT / "outputs/p4_patchvec/per_cell.csv"
    rr_path = ROOT / "outputs/p2_inputspecific/per_cell.csv"
    geometry = pd.read_csv(geometry_path)
    rr = pd.read_csv(rr_path)
    geometry = geometry[
        (geometry.split == "held") & (geometry.grouping == "failure_type")
    ].copy()
    geometry["directional_separation"] = geometry.cos_between - geometry.cos_within
    sep = (
        geometry.groupby("setting", as_index=False)
        .agg(
            directional_separation=("directional_separation", "mean"),
            separation_n_seeds=("directional_separation", "count"),
        )
    )
    held = rr[rr.split == "held"]
    rates = held.pivot_table(
        index=["setting", "seed"], columns="method", values="rate", aggfunc="first"
    ).reset_index()
    required = {"FixedPatch", "DynaPatch (ungated)"}
    if not required.issubset(rates.columns):
        raise AssertionError("input-specific RR source is missing FixedPatch or DynaPatch")
    rates["rr_gain"] = rates["DynaPatch (ungated)"] - rates["FixedPatch"]
    gain = (
        rates.groupby("setting", as_index=False)
        .agg(rr_gain=("rr_gain", "mean"), rr_gain_n_seeds=("rr_gain", "count"))
    )
    full = pd.DataFrame({
        "setting": [f"{dataset}/{backbone}" for dataset, backbone, _ in N.SETTING_ORDER],
        "setting_label": [label for _, _, label in N.SETTING_ORDER],
    })
    table = full.merge(sep, on="setting", how="left", validate="one_to_one").merge(
        gain, on="setting", how="left", validate="one_to_one"
    )
    valid = table.dropna(subset=["directional_separation", "rr_gain"])
    rho, pvalue = safe_spearman(valid.directional_separation, valid.rr_gain)
    stats = {
        "n_settings_total": int(len(table)),
        "n_settings_valid": int(len(valid)),
        "spearman_rho": rho,
        "spearman_p": pvalue,
    }
    return table, stats


def leave_one_setting_out(separation: pd.DataFrame) -> pd.DataFrame:
    valid = separation.dropna(subset=["directional_separation", "rr_gain"])
    rows: list[dict[str, object]] = []
    for index, omitted in valid.iterrows():
        retained = valid.drop(index)
        rho, pvalue = safe_spearman(retained.directional_separation, retained.rr_gain)
        rows.append({
            "omitted_setting": omitted.setting,
            "omitted_setting_label": omitted.setting_label,
            "n_settings_retained": int(len(retained)),
            "spearman_rho": rho,
            "spearman_p": pvalue,
        })
    return pd.DataFrame(rows)


def make_pca_coordinates(payload: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    unit_vectors = payload.attrs["unit_vectors"]
    fixed_unit = payload.attrs["fixed_unit"]
    pca = PCA(n_components=2, svd_solver="full")
    coords = pca.fit_transform(unit_vectors)
    fixed_coords = pca.transform(fixed_unit[None, :])[0]
    out = payload.drop(columns="vector_row").copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    counts = out.failure_type.value_counts()
    top = list(counts.head(PCA_TOP_TYPES).index)
    out["display_group"] = out.failure_type.where(out.failure_type.isin(top), "Other")
    out["point_kind"] = "input-specific patch"
    fixed_row = pd.DataFrame([{
        "setting": PCA_SETTING,
        "setting_label": N.SETTING_LABEL[PCA_SETTING],
        "seed": PCA_SEED,
        "dataset_index": pd.NA,
        "failure_type": "FixedPatch",
        "dp_correct": pd.NA,
        "PC1": fixed_coords[0],
        "PC2": fixed_coords[1],
        "display_group": "FixedPatch",
        "point_kind": "fixed patch",
    }])
    out = pd.concat([out, fixed_row], ignore_index=True)
    meta = {
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "n_input_specific_patches": int(len(payload)),
        "n_failure_types": int(payload.failure_type.nunique()),
    }
    return out, meta


def plot_figure(
    pca_coordinates: pd.DataFrame,
    pca_meta: dict[str, float],
    separation: pd.DataFrame,
    separation_stats: dict[str, float | int],
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15), constrained_layout=True)

    ax = axes[0]
    points = pca_coordinates[pca_coordinates.point_kind == "input-specific patch"]
    group_order = [g for g in points.display_group.value_counts().index if g != "Other"]
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    if (points.display_group == "Other").any():
        other = points[points.display_group == "Other"]
        ax.scatter(other.PC1, other.PC2, s=12, alpha=0.24, c="#A9A9A9", linewidths=0,
                   label="Other failure types")
    for color, group_name in zip(palette, group_order):
        group = points[points.display_group == group_name]
        ax.scatter(group.PC1, group.PC2, s=18, alpha=0.68, c=color, linewidths=0,
                   label=f"{group_name} (n={len(group)})")
        ax.scatter(group.PC1.mean(), group.PC2.mean(), marker="D", s=54,
                   facecolors="none", edgecolors=color, linewidths=1.5)
    fixed = pca_coordinates[pca_coordinates.point_kind == "fixed patch"].iloc[0]
    ax.scatter(fixed.PC1, fixed.PC2, marker="*", s=190, c="#F0E442",
               edgecolors="black", linewidths=0.9, label="FixedPatch")
    ax.axhline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.set_xlabel(f"PC1 ({100*pca_meta['pc1_explained_variance_ratio']:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100*pca_meta['pc2_explained_variance_ratio']:.1f}% variance)")
    ax.set_title("(a) Normalised patch directions (G-CN, seed 101)", loc="left")
    ax.legend(loc="best", frameon=False, handletextpad=0.4)

    ax = axes[1]
    valid = separation.dropna(subset=["directional_separation", "rr_gain"])
    complete = valid.separation_n_seeds.eq(3)
    ax.scatter(valid.loc[complete, "directional_separation"], valid.loc[complete, "rr_gain"],
               s=39, c="#0072B2", edgecolors="white", linewidths=0.6,
               label="3 valid seeds")
    ax.scatter(valid.loc[~complete, "directional_separation"], valid.loc[~complete, "rr_gain"],
               s=45, facecolors="white", edgecolors="#0072B2", linewidths=1.3,
               label="2 valid seeds")
    if len(valid) >= 2:
        slope, intercept = np.polyfit(valid.directional_separation, valid.rr_gain, 1)
        xs = np.linspace(valid.directional_separation.min(), valid.directional_separation.max(), 100)
        ax.plot(xs, intercept + slope * xs, color="#666666", linewidth=1.0,
                linestyle="--", label="Linear visual guide")
    label_offsets = {
        "G-CN": (5, -10),
        "T-CN": (-25, 4),
        "L-VG": (2, 5),
    }
    for row in valid.itertuples(index=False):
        offset = label_offsets.get(row.setting_label, (3, 3))
        ax.annotate(row.setting_label, (row.directional_separation, row.rr_gain),
                    xytext=offset, textcoords="offset points", fontsize=7.5)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Directional separation\n(mean between-type − within-type cosine distance)")
    ax.set_ylabel("RR gain: input-specific − fixed")
    ax.set_title("(b) Direction heterogeneity vs. input-specific gain", loc="left")
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ={separation_stats['spearman_rho']:.3f}\n"
        f"p={separation_stats['spearman_p']:.3f}, n={separation_stats['n_settings_valid']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85,
              "edgecolor": "#CCCCCC"},
    )
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(render(row[column]) for column in columns) + " |"
            for _, row in frame.iterrows()]
    return "\n".join([header, rule, *rows])


def write_raw_tables(
    path: Path,
    cell_correlations: pd.DataFrame,
    setting_correlations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    separation: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    separation_stats: dict[str, float | int],
    timestamp: str,
) -> None:
    primary_cells = cell_correlations[cell_correlations.min_type_support == PRIMARY_SUPPORT]
    primary_settings = setting_correlations[
        setting_correlations.min_type_support_pooled_seeds == PRIMARY_SUPPORT
    ]
    parts = [
        "# FixedPatch alignment raw summary tables",
        "",
        f"Generated (UTC): `{timestamp}`",
        "",
        "Population: base-wrong `repair_holdout_unseen` inputs only. Full failure-type rows are "
        "in `per_failure_type_seed.csv` and `per_failure_type_setting.csv`.",
        "",
        "## Within-cell association (primary min support = 3)",
        "",
        markdown_table(primary_cells, [
            "setting_label", "seed", "n_types", "n_fail",
            "rho_alignment_fixed_RR", "p_alignment_fixed_RR",
            "rho_alignment_delta_RR", "p_alignment_delta_RR",
        ]),
        "",
        "## Per-setting association after pooling same failure types across seeds (descriptive)",
        "",
        markdown_table(primary_settings, [
            "setting_label", "n_types", "n_fail", "rho_alignment_fixed_RR",
            "p_alignment_fixed_RR", "rho_alignment_delta_RR", "p_alignment_delta_RR",
        ]),
        "",
        "## Support-threshold sensitivity",
        "",
        markdown_table(sensitivity, list(sensitivity.columns)),
        "",
        "## Directional separation versus input-specific RR gain (all 12 settings)",
        "",
        markdown_table(separation, [
            "setting_label", "directional_separation", "separation_n_seeds",
            "rr_gain", "rr_gain_n_seeds",
        ]),
        "",
        f"Spearman rho = {separation_stats['spearman_rho']:.6f}, "
        f"p = {separation_stats['spearman_p']:.6f}, "
        f"n = {separation_stats['n_settings_valid']}. L-DN remains NA because its held "
        "failure-type geometry lacks enough same-type pairs.",
        "",
        "## Leave-one-setting-out sensitivity for the cross-setting association",
        "",
        markdown_table(leave_one_out, [
            "omitted_setting_label", "n_settings_retained", "spearman_rho", "spearman_p",
        ]),
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    global SEEDS, SUPPORT_THRESHOLDS, PRIMARY_SUPPORT, PCA_SETTING, PCA_SEED, PCA_TOP_TYPES
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    SEEDS = tuple(int(seed) for seed in config["data"]["seeds"])
    SUPPORT_THRESHOLDS = tuple(
        int(value) for value in config["analysis"]["support_thresholds"]
    )
    PRIMARY_SUPPORT = int(config["analysis"]["primary_support_threshold"])
    if PRIMARY_SUPPORT not in SUPPORT_THRESHOLDS:
        raise AssertionError("primary support threshold must be in support_thresholds")
    PCA_SETTING = str(config["visualization"]["pca_setting"])
    PCA_SEED = int(config["visualization"]["pca_seed"])
    PCA_TOP_TYPES = int(config["visualization"]["top_failure_types"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (ROOT / config["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = ROOT / "outputs/sample_frame.csv.gz"
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)

    per_type, pca_payload, sources = collect_types(sample_path)
    expected_cells = len(N.SETTING_ORDER) * len(SEEDS)
    if per_type.groupby(["setting", "seed"]).ngroups != expected_cells:
        raise AssertionError("failure-type output does not cover all 36 setting/seed cells")
    if not np.allclose(per_type.fixed_RR, per_type.fixed_n_repaired / per_type.n_fail):
        raise AssertionError("FixedPatch type RR denominator mismatch")
    if not np.allclose(per_type.dp_RR, per_type.dp_n_repaired / per_type.n_fail):
        raise AssertionError("DynaPatch type RR denominator mismatch")
    if not per_type.fixed_alignment.dropna().between(-1 - 1e-9, 1 + 1e-9).all():
        raise AssertionError("cosine alignment outside [-1, 1]")

    per_setting_type = aggregate_types(per_type)
    cell_corr, setting_corr, sensitivity = correlation_tables(per_type, per_setting_type)
    separation, separation_stats = separation_vs_gain()
    leave_one_out = leave_one_setting_out(separation)
    pca_coordinates, pca_meta = make_pca_coordinates(pca_payload)

    per_type.to_csv(output_dir / "per_failure_type_seed.csv", index=False)
    per_setting_type.to_csv(output_dir / "per_failure_type_setting.csv", index=False)
    cell_corr.to_csv(output_dir / "per_cell_correlations.csv", index=False)
    setting_corr.to_csv(output_dir / "per_setting_correlations.csv", index=False)
    sensitivity.to_csv(output_dir / "support_sensitivity.csv", index=False)
    separation.to_csv(output_dir / "directional_separation_vs_rr_gain.csv", index=False)
    leave_one_out.to_csv(output_dir / "directional_separation_leave_one_out.csv", index=False)
    pca_coordinates.to_csv(output_dir / "pca_coordinates_g_cn_s101.csv", index=False)

    timestamp = datetime.now(timezone.utc).isoformat()
    exact_command = ".venv/bin/python " + " ".join(sys.argv)
    summary = {
        "generated_utc": timestamp,
        "command": exact_command,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "population": "base-wrong repair_holdout_unseen",
        "primary_min_failure_type_support": PRIMARY_SUPPORT,
        "n_cells": expected_cells,
        "n_failure_type_seed_rows": int(len(per_type)),
        "n_failure_type_setting_rows": int(len(per_setting_type)),
        "pca": {
            "role": "illustrative visualization only",
            "fit_population": "normalised input-specific directions only",
            "selection_rule": "predeclared G-CN seed 101 for sample size and legibility; not selected by effect size",
            "setting": PCA_SETTING,
            "seed": PCA_SEED,
            **pca_meta,
        },
        "directional_separation_vs_rr_gain": separation_stats,
        "directional_separation_leave_one_out": {
            "rho_min": float(leave_one_out.spearman_rho.min()),
            "rho_max": float(leave_one_out.spearman_rho.max()),
            "p_max": float(leave_one_out.spearman_p.max()),
        },
        "within_cell_support_sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_manifest = {
        "generated_utc": timestamp,
        "command": exact_command,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "sample_frame": {
            "path": str(sample_path.relative_to(ROOT)),
            "sha256": sha256(sample_path),
        },
        "derived_sources": [
            {
                "path": "outputs/p4_patchvec/per_cell.csv",
                "sha256": sha256(ROOT / "outputs/p4_patchvec/per_cell.csv"),
            },
            {
                "path": "outputs/p2_inputspecific/per_cell.csv",
                "sha256": sha256(ROOT / "outputs/p2_inputspecific/per_cell.csv"),
            },
        ],
        "cell_sources": sources,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    write_raw_tables(
        output_dir / "RAW_TABLES.md",
        cell_corr,
        setting_corr,
        sensitivity,
        separation,
        leave_one_out,
        separation_stats,
        timestamp,
    )
    plot_figure(
        pca_coordinates,
        pca_meta,
        separation,
        separation_stats,
        output_dir / "fig_fixed_alignment_pca_correlation.png",
        output_dir / "fig_fixed_alignment_pca_correlation.pdf",
    )

    # Record the exact repository state without requiring the worktree to be clean.
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unavailable"
    (output_dir / "git_revision.txt").write_text(revision + "\n", encoding="utf-8")

    print(f"[written] {output_dir.relative_to(ROOT)}")
    print(json.dumps(summary["directional_separation_vs_rr_gain"], indent=2))
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
