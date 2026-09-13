#!/usr/bin/env python3
"""Plot the setting-balanced RQ3 confidence-change distributions with Seaborn.

This script is plotting-only: it reads the compact CSV produced by
``build_results_draft.py`` and does not train a model or initialize a GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "rq_results_draft" / "rq3_confidence_distribution.csv"
FIG = ROOT / "note" / "figures" / "rq_results_draft"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "rq_results_draft" / ".mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


OUTCOME_ORDER = ("successful repair", "ineffective change", "regression")
DISPLAY_NAMES = {
    "successful repair": "Successful repair",
    "ineffective change": "No benefit",
    "regression": "Regression",
}
COLORS = {
    "successful repair": "#2F6F73",
    "ineffective change": "#78838E",
    "regression": "#C66A45",
}
LINESTYLES = {
    "successful repair": "-",
    "ineffective change": "--",
    "regression": ":",
}


def plot_confidence_distribution(
    data: pd.DataFrame,
    output_dir: Path = FIG,
) -> dict[str, float]:
    """Write the weighted ECDF as PNG/PDF and return P(delta confidence > 0)."""
    required = {"outcome", "confidence_change", "weight"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    unknown = set(data["outcome"]).difference(OUTCOME_ORDER)
    if unknown:
        raise ValueError(f"Unknown outcomes: {sorted(unknown)}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("Distribution data must be non-empty and contain no missing values")
    weight_sums = data.groupby("outcome", observed=True)["weight"].sum()
    if set(weight_sums.index) != set(OUTCOME_ORDER) or not np.allclose(weight_sums, 1.0):
        raise ValueError("Weights must sum to one independently for all three outcomes")

    positive_fraction = {
        outcome: float(group.loc[group["confidence_change"] > 0, "weight"].sum())
        for outcome, group in data.groupby("outcome", observed=True)
    }

    style = {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#2C3137",
        "axes.labelcolor": "#2C3137",
        "xtick.color": "#2C3137",
        "ytick.color": "#2C3137",
        "text.color": "#2C3137",
        "grid.color": "#E8EBEF",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
    with sns.axes_style("whitegrid", rc=style), sns.plotting_context("paper", font_scale=1.15):
        fig, ax = plt.subplots(figsize=(7.4, 4.7))
        for outcome in OUTCOME_ORDER:
            group = data[data["outcome"] == outcome]
            label = f"{DISPLAY_NAMES[outcome]} ({100 * positive_fraction[outcome]:.1f}% > 0)"
            sns.ecdfplot(
                data=group,
                x="confidence_change",
                weights="weight",
                stat="proportion",
                complementary=False,
                color=COLORS[outcome],
                linestyle=LINESTYLES[outcome],
                linewidth=2.2,
                label=label,
                ax=ax,
            )

        ax.axvline(0, color="#2C3137", linewidth=1.0, alpha=0.70, zorder=0)
        ax.set(
            xlim=(-1, 1),
            ylim=(0, 1),
            xlabel=r"Patch-induced confidence change ($\Delta p_{\max}$)",
            ylabel="Cumulative probability",
            title="Post-Patch Confidence Changes across Patch Outcomes",
        )
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.legend(frameon=False, loc="upper left", handlelength=2.6)
        sns.despine(ax=ax)
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / "fig_rq3_confidence_distribution"
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return positive_fraction


def main() -> None:
    data = pd.read_csv(DATA)
    fractions = plot_confidence_distribution(data)
    summary = ", ".join(
        f"{DISPLAY_NAMES[outcome]}={100 * fractions[outcome]:.1f}%"
        for outcome in OUTCOME_ORDER
    )
    print(f"[written] {FIG / 'fig_rq3_confidence_distribution.png'}")
    print(f"[written] {FIG / 'fig_rq3_confidence_distribution.pdf'}")
    print(f"[positive confidence change] {summary}")


if __name__ == "__main__":
    main()
