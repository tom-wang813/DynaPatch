"""Diagnostics for validating repair experiments before formal reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from src.experiment.stage3 import build_stage3_bundle, load_repair_checkpoint, load_resolved_config, resolve_runtime_device
from src.training.loops import collect_prediction_rows, evaluate_repair_model


def _artifact_paths(artifact_root: Path) -> tuple[Path, Path, Path]:
    config_path = artifact_root / "config_resolved.yaml"
    checkpoint_path = artifact_root / "checkpoints" / "repair_last.pt"
    diagnostics_dir = artifact_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return config_path, checkpoint_path, diagnostics_dir


def _prediction_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(rows)
    base_correct = sum(bool(row["base_correct"]) for row in rows)
    patched_correct = sum(bool(row["patched_correct"]) for row in rows)
    repaired = sum(bool(row["repaired"]) for row in rows)
    regressed = sum(bool(row["regressed"]) for row in rows)
    route_weights = [float(row["route_weight"]) for row in rows]
    patch_norms = [float(row["patch_norm"]) for row in rows]
    return {
        "count": total,
        "base_correct": base_correct,
        "patched_correct": patched_correct,
        "repaired": repaired,
        "regressed": regressed,
        "patched_accuracy": patched_correct / max(total, 1),
        "repaired_rate": repaired / max(total, 1),
        "regressed_rate": regressed / max(total, 1),
        "mean_route_weight": sum(route_weights) / max(total, 1),
        "mean_patch_norm": sum(patch_norms) / max(total, 1),
    }


@torch.no_grad()
def _compute_loss_profile(model, loader, loss_fn, device: torch.device) -> dict[str, float]:
    """Profile cross-entropy behavior to explain odd loss values."""
    total = 0
    loss_sum = 0.0
    true_prob_sum = 0.0
    pred_conf_sum = 0.0
    wrong_pred_conf_sum = 0.0
    wrong_count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)  # [batch, num_classes]
        pred = torch.argmax(logits, dim=1)  # [batch]
        batch_size = int(labels.size(0))
        loss = loss_fn(logits, labels)

        total += batch_size
        loss_sum += float(loss.item()) * batch_size
        true_prob_sum += float(probs.gather(1, labels.view(-1, 1)).sum().item())
        pred_conf_sum += float(probs.gather(1, pred.view(-1, 1)).sum().item())

        wrong_mask = pred != labels
        wrong_count += int(wrong_mask.sum().item())
        if bool(wrong_mask.any().item()):
            wrong_pred_conf_sum += float(probs.gather(1, pred.view(-1, 1))[wrong_mask].sum().item())

    return {
        "mean_cross_entropy": loss_sum / max(total, 1),
        "mean_true_class_probability": true_prob_sum / max(total, 1),
        "mean_predicted_class_confidence": pred_conf_sum / max(total, 1),
        "wrong_prediction_count": wrong_count,
        "mean_wrong_predicted_confidence": wrong_pred_conf_sum / max(wrong_count, 1),
    }


@torch.no_grad()
def _compute_split_diagnostics(
    model,
    base_model,
    loader,
    loss_fn,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    metric_eval = evaluate_repair_model(
        model=model,
        base_model=base_model,
        loader=loader,
        loss_fn=loss_fn,
        device=device,
        compute_rr=(split_name == "clean_eval"),
    )
    rows = collect_prediction_rows(
        model=model,
        base_model=base_model,
        loader=loader,
        device=device,
        split_name=split_name,
    )
    summary = _prediction_summary(rows)
    loss_profile = _compute_loss_profile(model, loader, loss_fn, device)
    return {
        "evaluate_repair_model": {
            "loss": metric_eval.loss,
            "accuracy": metric_eval.accuracy,
            "num_samples": metric_eval.num_samples,
            "rr": metric_eval.rr,
        },
        "prediction_summary": summary,
        "consistency": {
            "accuracy_minus_patched_accuracy": metric_eval.accuracy - float(summary["patched_accuracy"]),
            "count_matches": metric_eval.num_samples == int(summary["count"]),
        },
        "loss_profile": loss_profile,
    }


def run_consistency_check(artifact_root: Path, device_arg: str | None = None) -> Path:
    """Re-evaluate a saved checkpoint and compare metric pathways split by split."""
    config_path, checkpoint_path, diagnostics_dir = _artifact_paths(artifact_root)
    cfg = load_resolved_config(config_path)
    device = resolve_runtime_device(cfg, device_arg)
    backbone, model, dataloaders, loss_fn = build_stage3_bundle(cfg, device)
    load_repair_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    backbone.to(device).eval()

    report = {
        "artifact_root": str(artifact_root),
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "splits": {},
    }
    for split_name in ("bug_support", "bug_eval", "clean_eval"):
        loader = dataloaders["bug_train"] if split_name == "bug_support" else dataloaders[split_name]
        report["splits"][split_name] = _compute_split_diagnostics(
            model=model,
            base_model=backbone,
            loader=loader,
            loss_fn=loss_fn,
            device=device,
            split_name=split_name,
        )

    output_path = diagnostics_dir / "consistency_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


@torch.no_grad()
def _summarize_router_threshold(
    model,
    base_model,
    loader,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    base_model.eval()

    total = 0
    base_correct = 0
    patched_correct = 0
    repaired = 0
    regressed = 0
    route_hits = 0
    distance_sum = 0.0
    distance_min = float("inf")
    distance_max = 0.0
    patch_norm_sum = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model.forward_with_intermediates(inputs)
        base_logits = base_model(inputs)
        base_pred = torch.argmax(base_logits, dim=1)  # [batch]
        patched_pred = torch.argmax(outputs["logits"], dim=1)  # [batch]
        _, min_dist = model.router(outputs["route_feat"])
        route_weight = outputs["route_weight"].view(-1)  # [batch]
        patch_norm = outputs["gated_patch"].flatten(1).norm(dim=1)  # [batch]

        batch_size = int(labels.size(0))
        total += batch_size
        base_correct += int((base_pred == labels).sum().item())
        patched_correct += int((patched_pred == labels).sum().item())
        repaired += int(((base_pred != labels) & (patched_pred == labels)).sum().item())
        regressed += int(((base_pred == labels) & (patched_pred != labels)).sum().item())
        route_hits += int((route_weight > 0).sum().item())
        distance_sum += float(min_dist.sum().item())
        distance_min = min(distance_min, float(min_dist.min().item()))
        distance_max = max(distance_max, float(min_dist.max().item()))
        patch_norm_sum += float(patch_norm.sum().item())

    return {
        "count": total,
        "base_correct": base_correct,
        "patched_correct": patched_correct,
        "repaired": repaired,
        "regressed": regressed,
        "route_hits": route_hits,
        "route_hit_rate": route_hits / max(total, 1),
        "patched_accuracy": patched_correct / max(total, 1),
        "mean_min_distance": distance_sum / max(total, 1),
        "min_min_distance": 0.0 if total == 0 else distance_min,
        "max_min_distance": distance_max,
        "mean_patch_norm": patch_norm_sum / max(total, 1),
    }


def run_router_threshold_sweep(
    artifact_root: Path,
    thresholds: list[float],
    device_arg: str | None = None,
) -> Path:
    """Replay a trained routed checkpoint under multiple router thresholds."""
    config_path, checkpoint_path, diagnostics_dir = _artifact_paths(artifact_root)
    cfg = load_resolved_config(config_path)
    device = resolve_runtime_device(cfg, device_arg)
    backbone, model, dataloaders, _loss_fn = build_stage3_bundle(cfg, device)
    load_repair_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    backbone.to(device).eval()

    sweep_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        model.router.threshold = float(threshold)
        row = {
            "threshold": float(threshold),
            "bug_support": _summarize_router_threshold(model, backbone, dataloaders["bug_train"], device, "bug_support"),
            "bug_eval": _summarize_router_threshold(model, backbone, dataloaders["bug_eval"], device, "bug_eval"),
            "clean_eval": _summarize_router_threshold(model, backbone, dataloaders["clean_eval"], device, "clean_eval"),
        }
        sweep_rows.append(row)

    output_json = diagnostics_dir / "router_threshold_sweep.json"
    output_csv = diagnostics_dir / "router_threshold_sweep.csv"
    output_json.write_text(json.dumps(sweep_rows, indent=2, sort_keys=True), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "threshold",
            "split",
            "count",
            "base_correct",
            "patched_correct",
            "patched_accuracy",
            "repaired",
            "regressed",
            "route_hits",
            "route_hit_rate",
            "mean_min_distance",
            "min_min_distance",
            "max_min_distance",
            "mean_patch_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep_rows:
            threshold = row["threshold"]
            for split_name in ("bug_support", "bug_eval", "clean_eval"):
                writer.writerow({"threshold": threshold, "split": split_name, **row[split_name]})

    return output_json


def main() -> None:
    """Command-line entrypoint for diagnostics."""
    parser = argparse.ArgumentParser(description="Run DynaPatch diagnostics from saved artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    consistency_parser = subparsers.add_parser("consistency", help="Compare metric and prediction-summary pathways.")
    consistency_parser.add_argument("--artifact-root", required=True, type=Path)
    consistency_parser.add_argument("--device", default=None)

    sweep_parser = subparsers.add_parser("router-sweep", help="Sweep router thresholds on a saved routed checkpoint.")
    sweep_parser.add_argument("--artifact-root", required=True, type=Path)
    sweep_parser.add_argument("--device", default=None)
    sweep_parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        required=True,
        help="Threshold values to replay without retraining.",
    )

    args = parser.parse_args()
    if args.command == "consistency":
        output = run_consistency_check(args.artifact_root, args.device)
    else:
        output = run_router_threshold_sweep(args.artifact_root, args.thresholds, args.device)
    print(output)


if __name__ == "__main__":
    main()
