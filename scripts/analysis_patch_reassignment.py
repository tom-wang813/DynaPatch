#!/usr/bin/env python3
"""Counterfactual patch-effect reassignment for the DynaPatch mechanism claim.

The existing effect dumps contain base logits z_B and patched logits z_P for every
input.  Their difference dz = z_P - z_B is exactly the classifier-space effect of
that input's generated feature patch.  Reassigning dz between held-out failures
preserves the complete empirical effect distribution while destroying its match to
the input.  No model loading, training, or GPU computation is involved.

The experiment also swaps only directions or only magnitudes, and shuffles within
true-class or (true-class, base-prediction) groups to diagnose specialization grain.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def stable_seed(base: int, *parts: object) -> int:
    raw = ":".join(map(str, (base, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def class_metrics(pred: np.ndarray, labels: np.ndarray, minimum_n: int) -> tuple[float, float]:
    rates = []
    estimable = []
    for cls in np.unique(labels):
        mask = labels == cls
        rate = float(np.mean(pred[mask] == labels[mask]))
        rates.append(rate)
        if int(mask.sum()) >= minimum_n:
            estimable.append(rate)
    macro = float(np.mean(rates)) if rates else math.nan
    coverage = float(np.mean(np.asarray(estimable) >= 0.5)) if estimable else math.nan
    return macro, coverage


def grouped_permutation(groups: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Permute within each group of matching `groups` values. A group of size 1 has no other
    member to swap with, so its one item is left in place (not eligible -- shuffling a
    singleton is a no-op, not evidence the shuffle doesn't matter). Returns the permutation
    AND the per-item eligible mask (which item actually belongs to a group that was shuffled),
    not just its mean -- the mean alone cannot support restricting a metric to only the rows
    that were actually reassigned.
    """
    idx = np.arange(len(groups))
    eligible = np.zeros(len(groups), dtype=bool)
    buckets: dict[object, list[int]] = defaultdict(list)
    for i, group in enumerate(groups.tolist()):
        buckets[group].append(i)
    for members in buckets.values():
        if len(members) > 1:
            eligible[members] = True
            idx[members] = rng.permutation(members)
    return idx, eligible


def reassigned_effect(
    effect: np.ndarray,
    labels: np.ndarray,
    base_pred: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(effect)
    if condition == "original":
        return effect, np.ones(n, dtype=bool), 0.0
    if condition in {"global_effect_shuffle", "global_direction_shuffle", "global_magnitude_shuffle"}:
        perm = rng.permutation(n)
        eligible = np.full(n, n > 1, dtype=bool)
    elif condition == "within_true_class_shuffle":
        perm, eligible = grouped_permutation(labels, rng)
    elif condition == "within_failure_type_shuffle":
        groups = np.asarray([f"{y}->{b}" for y, b in zip(labels, base_pred)], dtype=object)
        perm, eligible = grouped_permutation(groups, rng)
    else:
        raise ValueError(f"unknown condition: {condition}")

    moved = float(np.mean(perm != np.arange(n)))
    if condition in {"global_effect_shuffle", "within_true_class_shuffle", "within_failure_type_shuffle"}:
        return effect[perm], eligible, moved

    norms = np.linalg.norm(effect, axis=1)
    unit = effect / np.maximum(norms[:, None], 1e-12)
    if condition == "global_direction_shuffle":
        return unit[perm] * norms[:, None], eligible, moved
    return unit * norms[perm, None], eligible, moved


def evaluate(base: np.ndarray, effect: np.ndarray, labels: np.ndarray,
             base_pred: np.ndarray, minimum_n: int,
             mask: np.ndarray | None = None) -> dict[str, float]:
    """`mask` restricts every metric to a subset of rows (e.g. only the rows a grouped shuffle
    actually reassigned) -- comparing original vs shuffled on the SAME restricted subset is
    what makes the restricted comparison fair; the two calls must pass the identical mask."""
    if mask is not None:
        base, effect, labels, base_pred = base[mask], effect[mask], labels[mask], base_pred[mask]
    logits = base + effect
    pred = logits.argmax(axis=1)
    rr = float(np.mean(pred == labels)) if len(labels) else math.nan
    macro, coverage = class_metrics(pred, labels, minimum_n)
    row = np.arange(len(labels))
    margin_gain = float(np.mean(effect[row, labels] - effect[row, base_pred])) if len(labels) else math.nan
    return {
        "RR_held": rr,
        "macro_RR": macro,
        "coverage_at_0.5": coverage,
        "target_margin_gain": margin_gain,
        "flip_rate": float(np.mean(pred != base_pred)) if len(labels) else math.nan,
    }


def summarize_draws(draws: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in draws[0]:
        values = [x[metric] for x in draws if not math.isnan(x[metric])]
        out[f"{metric}_mean"] = float(np.mean(values)) if values else math.nan
        out[f"{metric}_std"] = float(np.std(values)) if values else math.nan
        out[f"{metric}_p025"] = quantile(values, 0.025) if values else math.nan
        out[f"{metric}_p975"] = quantile(values, 0.975) if values else math.nan
    return out


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    out_dir = ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_dir / "config.json")

    split = config["data"]["split"]
    # split_file: the population name to use when building the on-disk filenames, when it
    # differs from `split`'s label in this experiment's rows/metrics (e.g. NN-Patching/
    # PatchNAS's per-sample dumps call the held-out population "bug_eval", not
    # "repair_holdout_unseen"). Defaults to `split` (DynaPatch's effect_dump naming).
    split_file = config["data"].get("split_file", split)
    # patched_pred_col: which CSV column holds the RAW (unconditional) patch application,
    # i.e. argmax(patched_logits). DynaPatch's effect_dump is ungated, so its own
    # "patched_pred" column already is that. NN-Patching/PatchNAS's per-sample dumps are
    # ROUTER methods -- their "patched_pred" column is the POST-ROUTING prediction (reverts
    # to base_pred when declined), not the raw patch output; "patch_pred_raw" is the column
    # that always is (see scripts/baseline_prior_patches.py:dump_per_sample).
    patched_pred_col = config["data"].get("patched_pred_col", "patched_pred")
    conditions = config["interventions"]
    n_perm = int(config["permutations"])
    base_seed = int(config["random_seed"])
    minimum_n = int(config["minimum_class_n"])
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for dataset in config["data"]["datasets"]:
        for backbone in config["data"]["backbones"]:
            setting = f"{dataset}/{backbone}"
            for seed in config["data"]["seeds"]:
                dump_pattern = config["data"].get("dump_pattern_overrides", {}).get(
                    backbone, config["data"]["dump_pattern"]
                )
                pred_dir = ROOT / dump_pattern.format(
                    seed=seed, dataset=dataset, backbone=backbone
                )
                required = [
                    pred_dir / f"base_logits_{split_file}.npy",
                    pred_dir / f"patched_logits_{split_file}.npy",
                    pred_dir / f"{split_file}_predictions.csv",
                ]
                if not all(p.is_file() for p in required):
                    missing.append(f"{setting}/s{seed}: " + ", ".join(str(p) for p in required if not p.is_file()))
                    continue

                base = np.load(required[0]).astype(np.float64)
                patched = np.load(required[1]).astype(np.float64)
                with required[2].open(newline="") as f:
                    table = list(csv.DictReader(f))
                labels = np.asarray([int(x["label"]) for x in table], dtype=np.int64)
                csv_base_pred = np.asarray([int(x["base_pred"]) for x in table], dtype=np.int64)
                csv_patched_pred = np.asarray([int(x[patched_pred_col]) for x in table], dtype=np.int64)

                if not (len(base) == len(patched) == len(labels)):
                    raise RuntimeError(f"row mismatch in {setting}/s{seed}")
                if not np.array_equal(base.argmax(1), csv_base_pred):
                    raise RuntimeError(f"base argmax mismatch in {setting}/s{seed}")
                if not np.array_equal(patched.argmax(1), csv_patched_pred):
                    raise RuntimeError(f"patched argmax mismatch in {setting}/s{seed}")

                mask = csv_base_pred != labels
                base = base[mask]
                patched = patched[mask]
                labels = labels[mask]
                base_pred = csv_base_pred[mask]
                effect = patched - base
                original = evaluate(base, effect, labels, base_pred, minimum_n)
                reconstructed = base + effect
                max_reconstruction_error = float(np.max(np.abs(reconstructed - patched)))

                for condition in conditions:
                    repeats = 1 if condition == "original" else n_perm
                    rng = np.random.default_rng(stable_seed(base_seed, setting, seed, condition))
                    draws = []
                    draws_eligible = []
                    eligible_masks = []
                    moved_values = []
                    for _ in range(repeats):
                        assigned, eligible_mask, moved = reassigned_effect(
                            effect, labels, base_pred, condition, rng
                        )
                        draws.append(evaluate(base, assigned, labels, base_pred, minimum_n))
                        # eligible_mask is deterministic per (setting, condition) -- it depends
                        # only on which failure-type/class groups have >1 member, not on the
                        # random permutation draw -- so restricting to it is a fixed, fair
                        # comparison against `original` restricted to the SAME rows, not an
                        # artifact of one draw's luck.
                        draws_eligible.append(evaluate(base, assigned, labels, base_pred,
                                                       minimum_n, mask=eligible_mask))
                        eligible_masks.append(eligible_mask)
                        moved_values.append(moved)
                    summary = summarize_draws(draws)
                    rr_values = [x["RR_held"] for x in draws]
                    eligible_mask = eligible_masks[0]
                    n_eligible = int(eligible_mask.sum())
                    if n_eligible:
                        summary_eligible = summarize_draws(draws_eligible)
                        original_eligible = evaluate(base, effect, labels, base_pred,
                                                     minimum_n, mask=eligible_mask)
                        rr_values_eligible = [x["RR_held"] for x in draws_eligible]
                        original_rr_eligible = original_eligible["RR_held"]
                        original_minus_mean_rr_eligible = (
                            original_rr_eligible - summary_eligible["RR_held_mean"])
                        p_null_ge_original_eligible = (
                            float((1 + sum(x >= original_rr_eligible for x in rr_values_eligible))
                                 / (1 + len(rr_values_eligible)))
                            if condition != "original" else math.nan)
                        rr_held_mean_eligible = summary_eligible["RR_held_mean"]
                        rr_held_std_eligible = summary_eligible["RR_held_std"]
                        rr_held_p025_eligible = summary_eligible["RR_held_p025"]
                        rr_held_p975_eligible = summary_eligible["RR_held_p975"]
                    else:
                        original_rr_eligible = math.nan
                        original_minus_mean_rr_eligible = math.nan
                        p_null_ge_original_eligible = math.nan
                        rr_held_mean_eligible = math.nan
                        rr_held_std_eligible = math.nan
                        rr_held_p025_eligible = math.nan
                        rr_held_p975_eligible = math.nan
                    rows.append({
                        "experiment_id": config["experiment_id"],
                        "setting": setting,
                        "dataset": dataset,
                        "backbone": backbone,
                        "seed": seed,
                        "condition": condition,
                        "n": len(labels),
                        "n_classes": len(np.unique(labels)),
                        "n_estimable_classes": sum(int(np.sum(labels == c)) >= minimum_n for c in np.unique(labels)),
                        "n_permutations": repeats,
                        "eligible_fraction": float(eligible_mask.mean()),
                        "n_eligible": n_eligible,
                        "moved_fraction": float(np.mean(moved_values)),
                        "original_RR_held": original["RR_held"],
                        "original_minus_mean_RR": original["RR_held"] - summary["RR_held_mean"],
                        "p_null_ge_original": float((1 + sum(x >= original["RR_held"] for x in rr_values)) / (1 + len(rr_values))) if condition != "original" else math.nan,
                        # _eligible: the SAME comparison, restricted to rows whose group had
                        # >1 member and so were actually reassigned to a DIFFERENT failure's
                        # patch -- singletons (shuffled to themselves, a no-op) diluted the
                        # unrestricted numbers above toward "no effect" regardless of whether
                        # direction matters within a group. See note/RQ1_INPUT_SPECIFICITY.md
                        # §3b.
                        "original_RR_held_eligible": original_rr_eligible,
                        "RR_held_mean_eligible": rr_held_mean_eligible,
                        "RR_held_std_eligible": rr_held_std_eligible,
                        "RR_held_p025_eligible": rr_held_p025_eligible,
                        "RR_held_p975_eligible": rr_held_p975_eligible,
                        "original_minus_mean_RR_eligible": original_minus_mean_rr_eligible,
                        "p_null_ge_original_eligible": p_null_ge_original_eligible,
                        "max_reconstruction_error": max_reconstruction_error,
                        **summary,
                    })

    if missing:
        raise RuntimeError("missing inputs:\n" + "\n".join(missing))
    if len(rows) != 36 * len(conditions):
        raise RuntimeError(f"expected {36 * len(conditions)} rows, got {len(rows)}")

    fieldnames = list(rows[0])
    with (out_dir / "per_cell.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, object]] = {}
    for condition in conditions:
        cell = [r for r in rows if r["condition"] == condition]
        deltas = [float(r["original_minus_mean_RR"]) for r in cell]
        # eligible-restricted: only cells where at least one row was actually reassigned
        # (n_eligible > 0) contribute -- a cell with zero eligible rows has NaN eligible
        # fields by construction and must not silently become 0 in a median.
        cell_e = [r for r in cell if int(r["n_eligible"]) > 0]
        deltas_e = [float(r["original_minus_mean_RR_eligible"]) for r in cell_e]
        aggregate[condition] = {
            "cells": len(cell),
            "median_RR_held": float(np.median([float(r["RR_held_mean"]) for r in cell])),
            "median_original_minus_mean_RR": float(np.median(deltas)),
            "W_T_L_original_vs_null_mean": [
                sum(x > 1e-12 for x in deltas),
                sum(abs(x) <= 1e-12 for x in deltas),
                sum(x < -1e-12 for x in deltas),
            ],
            "cells_original_above_null_p975": sum(
                float(r["original_RR_held"]) > float(r["RR_held_p975"]) + 1e-12 for r in cell
            ) if condition != "original" else None,
            "median_eligible_fraction": float(np.median([float(r["eligible_fraction"]) for r in cell])),
            "median_moved_fraction": float(np.median([float(r["moved_fraction"]) for r in cell])),
            "median_macro_RR": float(np.nanmedian([float(r["macro_RR_mean"]) for r in cell])),
            "median_coverage_at_0.5": float(np.nanmedian([float(r["coverage_at_0.5_mean"]) for r in cell])),
            "median_target_margin_gain": float(np.median([float(r["target_margin_gain_mean"]) for r in cell])),
            "median_flip_rate": float(np.median([float(r["flip_rate_mean"]) for r in cell])),
            "cells_with_eligible_rows": len(cell_e),
            "median_RR_held_eligible": (
                float(np.median([float(r["RR_held_mean_eligible"]) for r in cell_e]))
                if cell_e else math.nan),
            "median_original_minus_mean_RR_eligible": (
                float(np.median(deltas_e)) if deltas_e else math.nan),
            "W_T_L_original_vs_null_mean_eligible": [
                sum(x > 1e-12 for x in deltas_e),
                sum(abs(x) <= 1e-12 for x in deltas_e),
                sum(x < -1e-12 for x in deltas_e),
            ] if deltas_e else None,
            "cells_original_above_null_p975_eligible": sum(
                float(r["original_RR_held_eligible"]) > float(r["RR_held_p975_eligible"]) + 1e-12
                for r in cell_e
            ) if condition != "original" and cell_e else None,
        }

    global_result = aggregate["global_effect_shuffle"]
    advantage = float(global_result["median_original_minus_mean_RR"])
    above = int(global_result["cells_original_above_null_p975"])
    if advantage > 0.05 and above >= 30:
        decision = "SUPPORT"
    elif advantage <= 0.02 or above < 24:
        decision = "FALSIFY"
    else:
        decision = "INCONCLUSIVE"

    result = {
        "experiment_id": config["experiment_id"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "config": config,
        "sanity": {
            "cells": 36,
            "rows": len(rows),
            "maximum_reconstruction_error": max(float(r["max_reconstruction_error"]) for r in rows),
            "missing_inputs": missing,
        },
        "aggregate": aggregate,
        "decision": decision,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# {config['experiment_id']}",
        "",
        f"Decision: **{decision}**",
        "",
        "| condition | median RR | original − null | W/T/L | original > null 97.5% | eligible | moved | macro RR | cov@.5 | margin gain | flip rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        a = aggregate[condition]
        wtl = "/".join(map(str, a["W_T_L_original_vs_null_mean"]))
        above_text = "—" if a["cells_original_above_null_p975"] is None else str(a["cells_original_above_null_p975"])
        lines.append(
            f"| {condition} | {a['median_RR_held']:.4f} | {a['median_original_minus_mean_RR']:+.4f} | "
            f"{wtl} | {above_text} | {a['median_eligible_fraction']:.3f} | {a['median_moved_fraction']:.3f} | "
            f"{a['median_macro_RR']:.4f} | {a['median_coverage_at_0.5']:.4f} | "
            f"{a['median_target_margin_gain']:.4f} | {a['median_flip_rate']:.4f} |"
        )
    lines += [
        "",
        "The grouped shuffles retain singleton groups. Read them together with `eligible` and `moved`; they diagnose granularity rather than provide a fair global null.",
        "",
        "## Eligible-restricted (singleton groups excluded)",
        "",
        "Same comparison, but BOTH `original` and the shuffled draws are restricted to only the "
        "rows whose group had >1 member and so were actually reassigned to a DIFFERENT "
        "failure's patch -- a singleton group's one row is shuffled to itself (a no-op), which "
        "silently dilutes the table above toward \"no effect\" regardless of whether direction "
        "matters within a group. Answers: does direction matter WITHIN a failure type, "
        "specifically on the failures where that question was actually put to the test?",
        "",
        "| condition | cells w/ eligible rows | median RR (eligible) | original − null (eligible) | W/T/L (eligible) | original > null 97.5% (eligible) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        a = aggregate[condition]
        if not a["cells_with_eligible_rows"]:
            lines.append(f"| {condition} | 0 | — | — | — | — |")
            continue
        wtl_e = "/".join(map(str, a["W_T_L_original_vs_null_mean_eligible"]))
        above_e = a["cells_original_above_null_p975_eligible"]
        above_e_text = "—" if above_e is None else str(above_e)
        lines.append(
            f"| {condition} | {a['cells_with_eligible_rows']} | {a['median_RR_held_eligible']:.4f} | "
            f"{a['median_original_minus_mean_RR_eligible']:+.4f} | {wtl_e} | {above_e_text} |"
        )
    lines += [
        "",
        f"Maximum reconstruction error: `{result['sanity']['maximum_reconstruction_error']:.3e}`.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
