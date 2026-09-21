"""Unified deployment-time evaluation stage for trained repair models."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig, OmegaConf

from src.experiment.runner import ExperimentRunner
from src.experiment.stage3 import build_stage3_bundle, resolve_runtime_device
from src.models.dynapatch.deployment_policy import build_deployment_policy
from src.models.dynapatch.gate import FeatureGate, compute_gate_features
from src.models.dynapatch.repair_bank import load_repair_bank, save_repair_bank


def validate_deploy_eval_config(cfg: DictConfig) -> None:
    """Validate the minimum config contract for deployment evaluation runs."""
    if "experiment" not in cfg or cfg.experiment.stage != "deploy_eval":
        raise ValueError("Deployment runner requires `experiment.stage=deploy_eval`.")
    if cfg.model.get("checkpoint_path") is None:
        raise ValueError("Deployment runner requires `model.checkpoint_path`.")
    if cfg.get("deployment") is None:
        raise ValueError("Deployment runner requires a `deployment` section.")
    if cfg.deployment.get("checkpoint_path") is None:
        raise ValueError("Deployment runner requires `deployment.checkpoint_path`.")


def _build_eval_loader(loader: DataLoader, batch_size: int, num_workers: int) -> DataLoader:
    """Create a deterministic eval loader from an existing dataset."""
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _dataset_indices(loader: DataLoader) -> list[int]:
    """Return the sample indices represented by a loader's dataset."""
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        return list(dataset.indices)
    return list(range(len(dataset)))


def _load_checkpoint_flexibly(model, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a training checkpoint while tolerating router-size changes."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    target_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        if key not in target_state:
            skipped_keys.append(key)
            continue
        if target_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    # Shape-mismatched tensors are dropped on purpose (the router can change size between train
    # and deploy), but dropping the PATCH silently is not tolerable: on 2026-07-29 a dump whose
    # config said hidden=64 against a hidden=512 checkpoint discarded all 11 hypernet tensors and
    # evaluated a randomly initialised patch, reporting RR 0.009-0.039 against a true 0.367-0.800
    # with no visible error. Warn on stderr so a sweep log carries the evidence.
    dropped_patch = [k for k in skipped_keys if k.startswith("hypernet.")]
    if dropped_patch:
        warnings.warn(
            f"{len(dropped_patch)} hypernet tensor(s) were dropped while loading "
            f"{Path(checkpoint_path).as_posix()} (shape mismatch or absent in the model). The "
            f"patch being evaluated is NOT the trained one -- check that the deploy config's "
            f"model.* matches the checkpoint. First: {dropped_patch[:3]}",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "checkpoint": checkpoint,
        "skipped_keys": skipped_keys,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


@torch.no_grad()
@torch.no_grad()
def _build_support_patch_bank(
    model,
    backbone,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute support route features and direct patches for deployment use."""
    route_batches: list[torch.Tensor] = []
    patch_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    model.eval()
    backbone.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        shallow_feat = model.decomposition.extract_shallow(inputs)
        route_feat = model.decomposition.router_features(shallow_feat)
        if hasattr(model, "patch_from_shallow"):
            patch = model.patch_from_shallow(shallow_feat, route_feat)
        else:
            context, _prototype_ids = model._prototype_context(route_feat)
            patch = model.hypernet(shallow_feat, context)
        predictions = backbone(inputs).argmax(dim=1)
        route_batches.append(route_feat)
        patch_batches.append(patch)
        label_batches.append(labels)
        prediction_batches.append(predictions)

    if not route_batches:
        raise ValueError("Cannot build deployment support bank from an empty seen-repair loader.")
    return (
        torch.cat(route_batches, dim=0),
        torch.cat(patch_batches, dim=0),
        torch.cat(label_batches, dim=0),
        torch.cat(prediction_batches, dim=0),
    )


@torch.no_grad()
def _build_support_gate_bank(
    *,
    model,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    gate_source: str,
) -> torch.Tensor | None:
    """Precompute support features for an optional external deployment gate."""
    feature_batches: list[torch.Tensor] = []
    model.eval()
    for inputs, _labels in loader:
        inputs = inputs.to(device)
        shallow_feat = model.decomposition.extract_shallow(inputs)
        route_feat = model.decomposition.router_features(shallow_feat)
        gate_feat = _gate_features_from_inputs(
            inputs=inputs,
            route_feat=route_feat,
            cfg=cfg,
            gate_source=gate_source,
        )
        feature_batches.append(gate_feat)

    if not feature_batches:
        return None
    return torch.cat(feature_batches, dim=0)


def _resolve_policy_semantics(cfg: DictConfig) -> dict[str, object]:
    """Resolve the requested deployment policy into the effective official semantics."""
    requested_policy = str(cfg.deployment.policy)
    enforce_router_bypass = bool(cfg.deployment.get("enforce_router_bypass", True))
    effective_policy = requested_policy
    compatibility_note = None
    if enforce_router_bypass and requested_policy == "direct_generalization":
        effective_policy = "gated_direct"
        compatibility_note = (
            "Official deployment semantics require `reject -> bypass`, "
            "so `direct_generalization` is evaluated as `gated_direct`."
        )
    return {
        "requested_policy": requested_policy,
        "effective_policy": effective_policy,
        "enforce_router_bypass": enforce_router_bypass,
        "compatibility_note": compatibility_note,
    }


def _repair_bank_cfg(cfg: DictConfig) -> DictConfig:
    """Return the deployment repair-bank config block, defaulting to an empty node."""
    repair_bank_cfg = cfg.deployment.get("repair_bank")
    return repair_bank_cfg if repair_bank_cfg is not None else OmegaConf.create({})


def _normalize_repair_bank_selection(selection: str) -> str:
    """Normalize the repair-bank selection mode."""
    normalized = selection.strip().lower()
    allowed = {"support", "verified_repaired", "verified_correct"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported deployment.repair_bank.selection: {selection}")
    return normalized


def _load_or_build_repair_bank(
    *,
    model,
    backbone,
    seen_loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    gate_source: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Resolve deployment support banks from either a saved artifact or seen support."""
    repair_bank_cfg = _repair_bank_cfg(cfg)
    load_path = repair_bank_cfg.get("load_path")
    if load_path:
        payload = load_repair_bank(str(load_path))
        support_route_feat = payload["route_feat"].to(device)
        support_patches = payload["patches"].to(device)
        gate_support_feat = payload.get("gate_features")
        if gate_support_feat is not None:
            gate_support_feat = gate_support_feat.to(device)
        elif gate_source != "route_feat":
            raise ValueError(
                "Loaded repair bank is missing `gate_features`, but deployment.gate_source is not `route_feat`."
            )
        info = {
            "source": "loaded",
            "path": str(Path(load_path)),
            "selection": str(payload.get("selection", "loaded")),
            "count": int(support_route_feat.size(0)),
            "metadata": payload.get("metadata", {}),
        }
        return support_route_feat, support_patches, gate_support_feat, info

    support_route_feat, support_patches, support_labels, support_predictions = _build_support_patch_bank(
        model,
        backbone,
        seen_loader,
        device,
    )
    gate_support_feat = _build_support_gate_bank(
        model=model,
        loader=seen_loader,
        device=device,
        cfg=cfg,
        gate_source=gate_source,
    )
    info = {
        "source": "seen_support",
        "path": None,
        "selection": "support",
        "count": int(support_route_feat.size(0)),
        "labels_available": True,
        "predictions_available": True,
        "support_labels": support_labels,
        "support_predictions": support_predictions,
    }
    return support_route_feat, support_patches, gate_support_feat, info


def _weighted_patch_lookup(
    query_route_feat: torch.Tensor,
    support_route_feat: torch.Tensor,
    support_patches: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """Return a distance-weighted patch mixture from the seen-repair bank."""
    dist = torch.cdist(query_route_feat, support_route_feat)
    support_count = int(support_route_feat.size(0))
    if top_k is not None and 0 < top_k < support_count:
        neighbor_dist, neighbor_idx = torch.topk(dist, k=top_k, dim=1, largest=False)
        weights = torch.softmax(-neighbor_dist / max(temperature, 1.0e-8), dim=1)
        neighbor_patches = support_patches[neighbor_idx]
        return torch.sum(weights.unsqueeze(-1) * neighbor_patches, dim=1)
    weights = torch.softmax(-dist / max(temperature, 1.0e-8), dim=1)
    return weights @ support_patches


def _unnormalize_inputs(inputs: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Map normalized images back to raw pixel space for input-space gating."""
    mean = torch.tensor(cfg.dataset.mean, device=inputs.device, dtype=inputs.dtype).view(1, -1, 1, 1)
    std = torch.tensor(cfg.dataset.std, device=inputs.device, dtype=inputs.dtype).view(1, -1, 1, 1)
    return inputs * std + mean


def _gate_features_from_inputs(
    *,
    inputs: torch.Tensor,
    route_feat: torch.Tensor,
    cfg: DictConfig,
    gate_source: str,
) -> torch.Tensor:
    """Build gate features in either route-feature or input space."""
    if gate_source == "route_feat":
        return route_feat
    if gate_source == "input_raw":
        raw_inputs = _unnormalize_inputs(inputs, cfg)
        return torch.flatten(raw_inputs, 1)  # [batch, c*h*w]
    raise ValueError(f"Unsupported deployment.gate_source: {gate_source}")


def _pairwise_gate_distance(
    query: torch.Tensor,
    support: torch.Tensor,
    metric: str,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute pairwise distances for deployment-time external gating."""
    if chunk_size is not None and chunk_size > 0 and support.size(0) > chunk_size:
        chunks = []
        for start in range(0, support.size(0), chunk_size):
            support_chunk = support[start : start + chunk_size]
            chunks.append(_pairwise_gate_distance(query, support_chunk, metric, chunk_size=None))
        return torch.cat(chunks, dim=1)
    if metric == "l2":
        return torch.cdist(query, support)
    if metric == "linf":
        return torch.abs(query.unsqueeze(1) - support.unsqueeze(0)).amax(dim=-1)
    raise ValueError(f"Unsupported deployment.gate_metric: {metric}")


def _external_gate_forward(
    *,
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    threshold: float,
    metric: str,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return accept decisions and min distances using an external support bank."""
    dist = _pairwise_gate_distance(query_features, support_features, metric, chunk_size=chunk_size)
    min_dist = torch.min(dist, dim=1).values
    accept = (min_dist < threshold).float().view(-1, 1)
    return accept, min_dist


def _empty_summary() -> dict[str, float | int]:
    """Return a zero-filled summary payload."""
    return {
        "count": 0,
        "base_correct": 0,
        "patched_correct": 0,
        "repaired": 0,
        "regressed": 0,
        "accepted": 0,
        "rejected": 0,
        "accepted_correct": 0,
        "accepted_wrong": 0,
        "rejected_correct": 0,
        "rejected_wrong": 0,
        "base_accuracy": 0.0,
        "patched_accuracy": 0.0,
        "repaired_rate": 0.0,
        "regressed_rate": 0.0,
        "rr": 0.0,
        "router_accept_rate": 0.0,
        "fallback_usage_rate": 0.0,
        "reject_rate": 0.0,
        "mean_min_distance": 0.0,
        "mean_patch_norm": 0.0,
    }


def _empty_accumulator() -> dict[str, float | int]:
    """Return zero-filled running counters for split aggregation."""
    return {
        "count": 0,
        "base_correct": 0,
        "patched_correct": 0,
        "repaired": 0,
        "regressed": 0,
        "accepted": 0,
        "rejected": 0,
        "accepted_correct": 0,
        "accepted_wrong": 0,
        "rejected_correct": 0,
        "rejected_wrong": 0,
        "router_accept_sum": 0.0,
        "fallback_usage_sum": 0.0,
        "distance_sum": 0.0,
        "patch_norm_sum": 0.0,
    }


def _finalize_summary(acc: dict[str, float | int]) -> dict[str, float | int]:
    """Convert running counters into split-level rates."""
    total = int(acc["count"])
    base_correct = int(acc["base_correct"])
    patched_correct = int(acc["patched_correct"])
    repaired = int(acc["repaired"])
    regressed = int(acc["regressed"])
    accepted = int(acc["accepted"])
    rejected = int(acc["rejected"])
    accepted_correct = int(acc["accepted_correct"])
    accepted_wrong = int(acc["accepted_wrong"])
    rejected_correct = int(acc["rejected_correct"])
    rejected_wrong = int(acc["rejected_wrong"])
    if total <= 0:
        return _empty_summary()
    return {
        "count": total,
        "base_correct": base_correct,
        "patched_correct": patched_correct,
        "repaired": repaired,
        "regressed": regressed,
        "accepted": accepted,
        "rejected": rejected,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted_wrong,
        "rejected_correct": rejected_correct,
        "rejected_wrong": rejected_wrong,
        "base_accuracy": base_correct / total,
        "patched_accuracy": patched_correct / total,
        "repaired_rate": repaired / total,
        "regressed_rate": regressed / total,
        "rr": regressed / max(base_correct, 1),
        "router_accept_rate": float(acc["router_accept_sum"]) / total,
        "fallback_usage_rate": float(acc["fallback_usage_sum"]) / total,
        "reject_rate": rejected / total,
        "mean_min_distance": float(acc["distance_sum"]) / total,
        "mean_patch_norm": float(acc["patch_norm_sum"]) / total,
    }


def _annotate_split_gate_quality(
    split_name: str,
    summary: dict[str, float | int],
) -> dict[str, float | int]:
    """Attach split-specific gate-quality metrics to a finalized summary."""
    annotated = dict(summary)
    count = int(summary["count"])
    if split_name == "repair_support_seen":
        annotated["false_reject_count"] = int(summary["rejected"])
        annotated["false_reject_rate"] = float(summary["reject_rate"])
        annotated["false_accept_count"] = 0
        annotated["false_accept_rate"] = 0.0
    elif split_name == "clean_eval":
        annotated["false_reject_count"] = 0
        annotated["false_reject_rate"] = 0.0
        annotated["false_accept_count"] = int(summary["accepted"])
        annotated["false_accept_rate"] = float(summary["router_accept_rate"])
    else:
        annotated["false_reject_count"] = 0
        annotated["false_reject_rate"] = 0.0
        annotated["false_accept_count"] = 0
        annotated["false_accept_rate"] = 0.0
    annotated["accept_correct_rate"] = float(summary["accepted_correct"]) / max(count, 1)
    annotated["accept_wrong_rate"] = float(summary["accepted_wrong"]) / max(count, 1)
    annotated["reject_correct_rate"] = float(summary["rejected_correct"]) / max(count, 1)
    annotated["reject_wrong_rate"] = float(summary["rejected_wrong"]) / max(count, 1)
    return annotated


def _write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist per-sample deployment predictions for post-hoc analysis."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _deployed_forward(
    *,
    model,
    inputs: torch.Tensor,
    policy: str,
    support_route_feat: torch.Tensor,
    support_patches: torch.Tensor,
    weighted_temperature: float,
    weighted_top_k: int | None,
    gate_support_feat: torch.Tensor | None,
    gate_source: str,
    gate_metric: str,
    gate_threshold: float | None,
    gate_chunk_size: int | None,
    cfg: DictConfig,
) -> dict[str, torch.Tensor]:
    """Run one deployment-time forward pass under the configured policy."""
    shallow_feat = model.decomposition.extract_shallow(inputs)
    route_feat = model.decomposition.router_features(shallow_feat)
    _context, prototype_ids = model._prototype_context(route_feat)
    if hasattr(model, "patch_from_shallow"):
        direct_patch = model.patch_from_shallow(shallow_feat, route_feat)
    else:
        context, _prototype_ids = model._prototype_context(route_feat)
        direct_patch = model.hypernet(shallow_feat, context)
    gate_query_feat = route_feat
    if gate_support_feat is not None and gate_threshold is not None:
        gate_query_feat = _gate_features_from_inputs(
            inputs=inputs,
            route_feat=route_feat,
            cfg=cfg,
            gate_source=gate_source,
        )
        route_accept, min_dist = _external_gate_forward(
            query_features=gate_query_feat,
            support_features=gate_support_feat,
            threshold=gate_threshold,
            metric=gate_metric,
            chunk_size=gate_chunk_size,
        )
    else:
        # Every shipped config sets deployment.gate_threshold, so this branch is never taken in
        # practice; it used to fall back to the in-model DistanceRouter, which was removed after
        # confirming that (see the `dynapatch-deploy-policy-promotion-quirk` memory note).
        raise ValueError(
            "deployment.gate_source/gate_threshold must be set (external gate) -- the in-model "
            "router fallback was removed as unreachable dead code."
        )
    route_accept_flat = route_accept.view(-1)
    accept_mask = route_accept_flat > 0.0
    fallback_mask = ~accept_mask
    deployment_policy = build_deployment_policy(policy)
    deployed_patch = deployment_policy.compose(
        direct_patch=direct_patch,
        accept_mask=accept_mask,
        route_feat=route_feat,
        support_route_feat=support_route_feat,
        support_patches=support_patches,
        weighted_lookup=_weighted_patch_lookup,
        weighted_temperature=weighted_temperature,
        weighted_top_k=weighted_top_k,
    )

    gated_patch = model.patch_operator(
        deployed_patch,
        torch.ones(inputs.size(0), 1, device=deployed_patch.device),
    )
    if hasattr(model, "logits_from_patch"):
        patched_logits = model.logits_from_patch(shallow_feat, gated_patch)
    else:
        if getattr(model.decomposition, "insertion_point", "cross_layer") == "same_layer":
            patched_shallow = model.decomposition.apply_insertion_patch(shallow_feat, gated_patch)
            deep_feat = model.decomposition.extract_deep(patched_shallow)
            patched_logits = model.decomposition.classify(deep_feat, None)
        else:
            deep_feat = model.decomposition.extract_deep(shallow_feat)
            patched_logits = model.decomposition.classify(deep_feat, gated_patch)
    patch_norm = gated_patch.flatten(1).norm(dim=1)
    return {
        "patched_logits": patched_logits,
        "route_accept": route_accept_flat,
        "fallback_mask": fallback_mask,
        "min_dist": min_dist,
        "patch_norm": patch_norm,
        "prototype_ids": prototype_ids,
        "route_feat": route_feat,
        "direct_patch": direct_patch,
        "deployed_patch": deployed_patch,
        "gated_patch": gated_patch,
        "gate_query_feat": gate_query_feat,
    }


def _adversarial_settings(cfg: DictConfig) -> dict[str, object]:
    """Resolve deployment-time PGD settings."""
    adv_cfg = cfg.deployment.get("adversarial")
    default_epsilon = float(cfg.fault.epsilon)
    default_steps = int(cfg.fault.steps)
    default_splits = ["repair_support_seen", "repair_holdout_unseen", "clean_eval"]
    if adv_cfg is None:
        return {
            "enabled": False,
            "epsilon": default_epsilon,
            "epsilons": [default_epsilon],
            "steps": default_steps,
            "step_size": None,
            "splits": default_splits,
        }

    enabled = bool(adv_cfg.get("enabled", False))
    steps = int(adv_cfg.get("steps", default_steps))
    epsilon_list_cfg = adv_cfg.get("epsilons")
    if epsilon_list_cfg is None:
        epsilons = [float(adv_cfg.get("epsilon", default_epsilon))]
    else:
        epsilons = [float(value) for value in epsilon_list_cfg]
        if not epsilons:
            epsilons = [default_epsilon]
    epsilon = float(epsilons[-1])
    step_size_cfg = adv_cfg.get("step_size")
    step_size = None if step_size_cfg is None else float(step_size_cfg)
    split_names = adv_cfg.get("splits", default_splits)
    return {
        "enabled": enabled,
        "epsilon": epsilon,
        "epsilons": epsilons,
        "steps": steps,
        "step_size": step_size,
        "splits": list(split_names),
    }


def _should_include_in_repair_bank(
    *,
    selection: str,
    base_correct: bool,
    patched_correct: bool,
    router_accept: bool,
) -> bool:
    """Decide whether a sample should be exported into the managed repair bank."""
    if selection == "support":
        return True
    if selection == "verified_repaired":
        return router_accept and (not base_correct) and patched_correct
    if selection == "verified_correct":
        return router_accept and patched_correct
    raise ValueError(f"Unsupported repair-bank selection: {selection}")


def _collect_repair_bank_candidates(
    *,
    split_name: str,
    rows: list[dict[str, object]],
    route_feat: torch.Tensor,
    direct_patch: torch.Tensor,
    gate_query_feat: torch.Tensor,
    labels: torch.Tensor,
    base_pred: torch.Tensor,
    patched_pred: torch.Tensor,
    selection: str,
    bucket: dict[str, Any],
) -> None:
    """Accumulate verified deployment-time repair-bank members from one batch."""
    if split_name != "repair_support_seen":
        return
    for idx, row in enumerate(rows):
        if not _should_include_in_repair_bank(
            selection=selection,
            base_correct=bool(row["base_correct"]),
            patched_correct=bool(row["patched_correct"]),
            router_accept=bool(row["router_accept"]),
        ):
            continue
        bucket["route_feat"].append(route_feat[idx].detach().cpu())
        bucket["patches"].append(direct_patch[idx].detach().cpu())
        bucket["gate_features"].append(gate_query_feat[idx].detach().cpu())
        bucket["labels"].append(labels[idx].detach().cpu())
        bucket["predictions"].append(base_pred[idx].detach().cpu())
        bucket["patched_predictions"].append(patched_pred[idx].detach().cpu())
        bucket["metadata"].append(
            {
                "dataset_index": int(row["dataset_index"]),
                "label": int(row["label"]),
                "base_pred": int(row["base_pred"]),
                "patched_pred": int(row["patched_pred"]),
                "router_accept": bool(row["router_accept"]),
                "repaired": bool(row["repaired"]),
                "patch_norm": float(row["patch_norm"]),
                "min_distance": float(row["min_distance"]),
            }
        )


def _build_inference_checks(
    *,
    policy_info: dict[str, object],
    split_metrics: dict[str, dict[str, float | int]],
    weighted_temperature: float,
    weighted_top_k: int | None,
    adversarial_metrics: dict[str, dict[str, float | int]] | None,
    adversarial_settings: dict[str, object],
) -> dict[str, object]:
    """Collect the deployment-time checks we care about into one metrics block."""
    seen = split_metrics["repair_support_seen"]
    unseen = split_metrics["repair_holdout_unseen"]
    clean = split_metrics["clean_eval"]
    effective_policy = str(policy_info["effective_policy"])
    payload = {
        "contract": {
            "requested_policy": str(policy_info["requested_policy"]),
            "effective_policy": effective_policy,
            "enforce_router_bypass": bool(policy_info["enforce_router_bypass"]),
            "compatibility_note": policy_info["compatibility_note"],
            "policy_semantics": {
                "direct_generalization": "Legacy alias. Under official semantics this is promoted to `gated_direct` so reject bypasses patching.",
                "gated_direct": "Apply the direct patch only when the router accepts; otherwise bypass patching with a zero fallback.",
            }[effective_policy],
            "clean_behavior_note": "Official deployment semantics expose clean false accepts and seen false rejects under reject-bypass evaluation.",
        },
        "behavior": {
            "seen_repaired": int(seen["repaired"]),
            "seen_repair_rate": float(seen["repaired_rate"]),
            "unseen_repaired": int(unseen["repaired"]),
            "unseen_repair_rate": float(unseen["repaired_rate"]),
            "clean_acc": float(clean["patched_accuracy"]),
            "clean_rr": float(clean["rr"]),
        },
        "routing": {
            "seen_router_accept_rate": float(seen["router_accept_rate"]),
            "unseen_router_accept_rate": float(unseen["router_accept_rate"]),
            "clean_router_accept_rate": float(clean["router_accept_rate"]),
            "seen_fallback_usage_rate": float(seen["fallback_usage_rate"]),
            "unseen_fallback_usage_rate": float(unseen["fallback_usage_rate"]),
            "clean_fallback_usage_rate": float(clean["fallback_usage_rate"]),
            "seen_reject_rate": float(seen["reject_rate"]),
            "unseen_reject_rate": float(unseen["reject_rate"]),
            "clean_reject_rate": float(clean["reject_rate"]),
            "seen_false_reject_count": int(seen["false_reject_count"]),
            "seen_false_reject_rate": float(seen["false_reject_rate"]),
            "clean_false_accept_count": int(clean["false_accept_count"]),
            "clean_false_accept_rate": float(clean["false_accept_rate"]),
            "seen_accept_correct": int(seen["accepted_correct"]),
            "seen_accept_wrong": int(seen["accepted_wrong"]),
            "seen_reject_correct": int(seen["rejected_correct"]),
            "seen_reject_wrong": int(seen["rejected_wrong"]),
            "clean_accept_correct": int(clean["accepted_correct"]),
            "clean_accept_wrong": int(clean["accepted_wrong"]),
            "clean_reject_correct": int(clean["rejected_correct"]),
            "clean_reject_wrong": int(clean["rejected_wrong"]),
            "seen_mean_min_distance": float(seen["mean_min_distance"]),
            "unseen_mean_min_distance": float(unseen["mean_min_distance"]),
            "clean_mean_min_distance": float(clean["mean_min_distance"]),
        },
        "patch": {
            "seen_mean_patch_norm": float(seen["mean_patch_norm"]),
            "unseen_mean_patch_norm": float(unseen["mean_patch_norm"]),
            "clean_mean_patch_norm": float(clean["mean_patch_norm"]),
            "weighted_temperature": weighted_temperature,
            "weighted_top_k": weighted_top_k,
        },
    }
    if adversarial_metrics is not None and bool(adversarial_settings.get("enabled", False)):
        curve = adversarial_metrics.get("curve") if isinstance(adversarial_metrics, dict) else None
        if curve:
            eps_key = sorted(curve.keys(), key=float)[-1]
            selected_adv = curve[eps_key]
            selected_step_size = float(selected_adv.get("step_size", 0.0))
        else:
            eps_key = f"{float(adversarial_settings['epsilon']):.6f}"
            selected_adv = adversarial_metrics
            selected_step_size = float(adversarial_settings.get("step_size") or 0.0)
        adv_seen = selected_adv.get("repair_support_seen", _empty_summary())
        adv_unseen = selected_adv.get("repair_holdout_unseen", _empty_summary())
        adv_clean = selected_adv.get("clean_eval", _empty_summary())
        payload["robustness"] = {
            "attack": {
                "epsilon": float(eps_key),
                "epsilons": [float(v) for v in adversarial_settings.get("epsilons", [float(eps_key)])],
                "steps": int(adversarial_settings["steps"]),
                "step_size": selected_step_size,
                "splits": list(adversarial_settings["splits"]),
            },
            "behavior": {
                "seen_clean_repair_rate": float(seen["repaired_rate"]),
                "seen_pgd_repair_rate": float(adv_seen["patched_accuracy"]),
                "unseen_clean_repair_rate": float(unseen["repaired_rate"]),
                "unseen_pgd_repair_rate": float(adv_unseen["patched_accuracy"]),
                "clean_clean_acc": float(clean["patched_accuracy"]),
                "clean_pgd_acc": float(adv_clean["patched_accuracy"]),
            },
            "routing": {
                "seen_pgd_router_accept_rate": float(adv_seen["router_accept_rate"]),
                "unseen_pgd_router_accept_rate": float(adv_unseen["router_accept_rate"]),
                "clean_pgd_router_accept_rate": float(adv_clean["router_accept_rate"]),
            },
        }
    return payload


def _empty_repair_bank_bucket() -> dict[str, Any]:
    """Return an empty in-memory bucket for managed repair-bank export."""
    return {
        "route_feat": [],
        "patches": [],
        "gate_features": [],
        "labels": [],
        "predictions": [],
        "patched_predictions": [],
        "metadata": [],
    }


def _export_managed_repair_bank(
    *,
    record,
    cfg: DictConfig,
    selection: str,
    bucket: dict[str, Any],
    gate_source: str,
    source_info: dict[str, Any],
) -> dict[str, Any]:
    """Write a managed repair bank artifact when requested."""
    repair_bank_cfg = _repair_bank_cfg(cfg)
    export_enabled = bool(repair_bank_cfg.get("export", True))
    if not export_enabled:
        return {
            "exported": False,
            "selection": selection,
            "count": 0,
            "path": None,
            "summary_path": None,
            "source": source_info["source"],
        }

    count = len(bucket["metadata"])
    if count == 0:
        return {
            "exported": False,
            "selection": selection,
            "count": 0,
            "path": None,
            "summary_path": None,
            "source": source_info["source"],
        }

    payload = {
        "selection": selection,
        "gate_source": gate_source,
        "route_feat": torch.stack(bucket["route_feat"], dim=0),
        "patches": torch.stack(bucket["patches"], dim=0),
        "gate_features": torch.stack(bucket["gate_features"], dim=0),
        "labels": torch.stack(bucket["labels"], dim=0),
        "predictions": torch.stack(bucket["predictions"], dim=0),
        "patched_predictions": torch.stack(bucket["patched_predictions"], dim=0),
        "metadata": bucket["metadata"],
        "source": source_info["source"],
    }
    summary = {
        "experiment_id": cfg.experiment.id,
        "selection": selection,
        "gate_source": gate_source,
        "count": count,
        "source": source_info["source"],
        "compatibility_note": "Managed repair bank stores deployment-verified seen cases for future reuse.",
    }
    artifact_dir = Path(record.artifact_root) / "repair_bank"
    bank_path, summary_path = save_repair_bank(
        artifact_dir=artifact_dir,
        payload=payload,
        summary=summary,
    )
    return {
        "exported": True,
        "selection": selection,
        "count": count,
        "path": str(bank_path),
        "summary_path": str(summary_path),
        "source": source_info["source"],
    }


@torch.no_grad()
def _evaluate_split(
    *,
    model,
    backbone,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    policy: str,
    support_route_feat: torch.Tensor,
    support_patches: torch.Tensor,
    weighted_temperature: float,
    weighted_top_k: int | None,
    gate_support_feat: torch.Tensor | None,
    gate_source: str,
    gate_metric: str,
    gate_threshold: float | None,
    gate_chunk_size: int | None,
    cfg: DictConfig,
    repair_bank_selection: str | None = None,
    repair_bank_bucket: dict[str, Any] | None = None,
    feature_dir: Path | None = None,
    feature_gate: FeatureGate | None = None,
    feature_gate_lambda: float = 1.0,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    """Evaluate one split under the configured deployment fallback policy.

    `feature_gate`, when given, is the paper's actual 9-feature commit/rollback gate
    (`src/models/dynapatch/gate.py`) and takes over the accept/reject decision entirely --
    overriding whatever `policy`/`gate_threshold` already decided, since those are confirmed
    permanently-open no-ops for every shipped setting (see the
    `dynapatch-deploy-policy-promotion-quirk` memory note). `route_accept`/`fallback_mask` are
    reassigned to the feature-gate's decision so predictions.csv reflects what actually happened.
    """
    rows: list[dict[str, object]] = []
    indices = _dataset_indices(loader)
    cursor = 0
    acc = _empty_accumulator()
    route_feat_list: list[torch.Tensor] = []
    logit_list: list[torch.Tensor] = []
    patched_logit_list: list[torch.Tensor] = []
    patch_vec_list: list[torch.Tensor] = []
    index_list: list[int] = []

    model.eval()
    backbone.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = int(labels.size(0))

        deployed = _deployed_forward(
            model=model,
            inputs=inputs,
            policy=policy,
            support_route_feat=support_route_feat,
            support_patches=support_patches,
            weighted_temperature=weighted_temperature,
            weighted_top_k=weighted_top_k,
            gate_support_feat=gate_support_feat,
            gate_source=gate_source,
            gate_metric=gate_metric,
            gate_threshold=gate_threshold,
            gate_chunk_size=gate_chunk_size,
            cfg=cfg,
        )
        patched_logits = deployed["patched_logits"]
        route_accept_flat = deployed["route_accept"]
        fallback_mask = deployed["fallback_mask"]
        min_dist = deployed["min_dist"]
        patch_norm = deployed["patch_norm"]
        prototype_ids = deployed["prototype_ids"]
        route_feat = deployed["route_feat"]
        direct_patch = deployed["direct_patch"]
        gate_query_feat = deployed["gate_query_feat"]
        base_logits = backbone(inputs)

        if feature_gate is not None:
            gate_feats = compute_gate_features(base_logits, patched_logits)
            accept = feature_gate.decide(gate_feats, lam=feature_gate_lambda)
            patched_logits = torch.where(accept.view(-1, 1), patched_logits, base_logits)
            route_accept_flat = accept.float()
            fallback_mask = ~accept

        if feature_dir is not None:
            route_feat_list.append(route_feat.detach().cpu())
            logit_list.append(base_logits.detach().cpu())
            patched_logit_list.append(patched_logits.detach().cpu())
            # Delta-h = the residual representation shift actually applied at classify().
            patch_vec_list.append(deployed["gated_patch"].detach().cpu())
            index_list.extend(indices[cursor : cursor + batch_size])

        base_probs = torch.softmax(base_logits, dim=1)
        patched_probs = torch.softmax(patched_logits, dim=1)
        base_pred = torch.argmax(base_logits, dim=1)
        patched_pred = torch.argmax(patched_logits, dim=1)
        base_correct_mask = base_pred == labels
        patched_correct_mask = patched_pred == labels
        repaired_mask = (~base_correct_mask) & patched_correct_mask
        regressed_mask = base_correct_mask & (~patched_correct_mask)

        acc["count"] += batch_size
        acc["base_correct"] += int(base_correct_mask.sum().item())
        acc["patched_correct"] += int(patched_correct_mask.sum().item())
        acc["repaired"] += int(repaired_mask.sum().item())
        acc["regressed"] += int(regressed_mask.sum().item())
        accept_bool = route_accept_flat > 0.0
        reject_bool = ~accept_bool
        acc["accepted"] += int(accept_bool.sum().item())
        acc["rejected"] += int(reject_bool.sum().item())
        acc["accepted_correct"] += int((accept_bool & patched_correct_mask).sum().item())
        acc["accepted_wrong"] += int((accept_bool & (~patched_correct_mask)).sum().item())
        acc["rejected_correct"] += int((reject_bool & patched_correct_mask).sum().item())
        acc["rejected_wrong"] += int((reject_bool & (~patched_correct_mask)).sum().item())
        acc["router_accept_sum"] += float(route_accept_flat.sum().item())
        acc["fallback_usage_sum"] += float(fallback_mask.float().sum().item())
        acc["distance_sum"] += float(min_dist.sum().item())
        acc["patch_norm_sum"] += float(patch_norm.sum().item())

        prototype_ids_cpu = prototype_ids.detach().cpu().tolist()
        batch_rows: list[dict[str, object]] = []
        for offset in range(batch_size):
            batch_rows.append(
                {
                    "split": split_name,
                    "dataset_index": int(indices[cursor + offset]),
                    "label": int(labels[offset].item()),
                    "base_pred": int(base_pred[offset].item()),
                    "patched_pred": int(patched_pred[offset].item()),
                    "base_confidence": float(base_probs[offset, base_pred[offset]].item()),
                    "patched_confidence": float(patched_probs[offset, patched_pred[offset]].item()),
                    "base_correct": bool(base_correct_mask[offset].item()),
                    "patched_correct": bool(patched_correct_mask[offset].item()),
                    "repaired": bool(repaired_mask[offset].item()),
                    "regressed": bool(regressed_mask[offset].item()),
                    "router_accept": bool(route_accept_flat[offset].item() > 0.0),
                    "fallback_used": bool(fallback_mask[offset].item()),
                    "min_distance": float(min_dist[offset].item()),
                    "patch_norm": float(patch_norm[offset].item()),
                    "prototype_id": int(prototype_ids_cpu[offset]),
                }
            )
        rows.extend(batch_rows)
        if repair_bank_selection is not None and repair_bank_bucket is not None:
            _collect_repair_bank_candidates(
                split_name=split_name,
                rows=batch_rows,
                route_feat=route_feat,
                direct_patch=direct_patch,
                gate_query_feat=gate_query_feat,
                labels=labels,
                base_pred=base_pred,
                patched_pred=patched_pred,
                selection=repair_bank_selection,
                bucket=repair_bank_bucket,
            )
        cursor += batch_size

    if feature_dir is not None and route_feat_list:
        feature_dir.mkdir(parents=True, exist_ok=True)
        np.save(feature_dir / f"route_features_{split_name}.npy",
                torch.cat(route_feat_list, dim=0).numpy())
        np.save(feature_dir / f"base_logits_{split_name}.npy",
                torch.cat(logit_list, dim=0).numpy())
        np.save(feature_dir / f"patched_logits_{split_name}.npy",
                torch.cat(patched_logit_list, dim=0).numpy())
        if split_name == "clean_train":
            # Downstream only ever takes ||delta|| of this tensor (analyze_response_gate_lobo
            # .chunked_norm). On the train split the full tensor is up to 27k x 25088 floats
            # per cell, so store the norm and skip 2.7 GB of disk per VGG cell.
            np.save(feature_dir / f"patch_norm_{split_name}.npy",
                    torch.cat(patch_vec_list, dim=0).float().norm(dim=1).numpy())
        else:
            np.save(feature_dir / f"patch_vec_{split_name}.npy",
                    torch.cat(patch_vec_list, dim=0).numpy())
        np.save(feature_dir / f"dataset_indices_{split_name}.npy",
                np.array(index_list, dtype=np.int64))

    return _annotate_split_gate_quality(split_name, _finalize_summary(acc)), rows


def _pgd_attack_deployed(
    *,
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    policy: str,
    support_route_feat: torch.Tensor,
    support_patches: torch.Tensor,
    weighted_temperature: float,
    weighted_top_k: int | None,
    gate_support_feat: torch.Tensor | None,
    gate_source: str,
    gate_metric: str,
    gate_threshold: float | None,
    gate_chunk_size: int | None,
    cfg: DictConfig,
    epsilon: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    """Attack the deployed patched model with a simple PGD inner loop."""
    if steps <= 0 or epsilon <= 0.0:
        return inputs.detach()

    delta = torch.zeros_like(inputs, requires_grad=True)
    model.eval()
    for _ in range(steps):
        adv_inputs = inputs + delta
        deployed = _deployed_forward(
            model=model,
            inputs=adv_inputs,
            policy=policy,
            support_route_feat=support_route_feat,
            support_patches=support_patches,
            weighted_temperature=weighted_temperature,
            weighted_top_k=weighted_top_k,
            gate_support_feat=gate_support_feat,
            gate_source=gate_source,
            gate_metric=gate_metric,
            gate_threshold=gate_threshold,
            gate_chunk_size=gate_chunk_size,
            cfg=cfg,
        )
        loss = F.cross_entropy(deployed["patched_logits"], labels)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = (delta + step_size * grad.sign()).clamp(min=-epsilon, max=epsilon).detach()
        delta.requires_grad_(True)
    return (inputs + delta).detach()


@torch.no_grad()
def _evaluate_split_under_attack(
    *,
    model,
    backbone,
    loader: DataLoader,
    device: torch.device,
    policy: str,
    support_route_feat: torch.Tensor,
    support_patches: torch.Tensor,
    weighted_temperature: float,
    weighted_top_k: int | None,
    gate_support_feat: torch.Tensor | None,
    gate_source: str,
    gate_metric: str,
    gate_threshold: float | None,
    gate_chunk_size: int | None,
    cfg: DictConfig,
    split_name: str,
    epsilon: float,
    steps: int,
    step_size: float,
) -> dict[str, float | int]:
    """Evaluate one split after attacking the deployed patched model with PGD."""
    acc = _empty_accumulator()

    model.eval()
    backbone.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = int(labels.size(0))
        base_logits = backbone(inputs)
        base_pred = torch.argmax(base_logits, dim=1)
        base_correct_mask = base_pred == labels

        with torch.enable_grad():
            adv_inputs = _pgd_attack_deployed(
                model=model,
                inputs=inputs,
                labels=labels,
                policy=policy,
                support_route_feat=support_route_feat,
                support_patches=support_patches,
                weighted_temperature=weighted_temperature,
                weighted_top_k=weighted_top_k,
                gate_support_feat=gate_support_feat,
                gate_source=gate_source,
                gate_metric=gate_metric,
                gate_threshold=gate_threshold,
                gate_chunk_size=gate_chunk_size,
                cfg=cfg,
                epsilon=epsilon,
                steps=steps,
                step_size=step_size,
            )

        deployed = _deployed_forward(
            model=model,
            inputs=adv_inputs,
            policy=policy,
            support_route_feat=support_route_feat,
            support_patches=support_patches,
            weighted_temperature=weighted_temperature,
            weighted_top_k=weighted_top_k,
            gate_support_feat=gate_support_feat,
            gate_source=gate_source,
            gate_metric=gate_metric,
            gate_threshold=gate_threshold,
            gate_chunk_size=gate_chunk_size,
            cfg=cfg,
        )
        patched_logits = deployed["patched_logits"]
        route_accept_flat = deployed["route_accept"]
        fallback_mask = deployed["fallback_mask"]
        min_dist = deployed["min_dist"]
        patch_norm = deployed["patch_norm"]

        patched_pred = torch.argmax(patched_logits, dim=1)
        patched_correct_mask = patched_pred == labels
        repaired_mask = (~base_correct_mask) & patched_correct_mask
        regressed_mask = base_correct_mask & (~patched_correct_mask)

        acc["count"] += batch_size
        acc["base_correct"] += int(base_correct_mask.sum().item())
        acc["patched_correct"] += int(patched_correct_mask.sum().item())
        acc["repaired"] += int(repaired_mask.sum().item())
        acc["regressed"] += int(regressed_mask.sum().item())
        accept_bool = route_accept_flat > 0.0
        reject_bool = ~accept_bool
        acc["accepted"] += int(accept_bool.sum().item())
        acc["rejected"] += int(reject_bool.sum().item())
        acc["accepted_correct"] += int((accept_bool & patched_correct_mask).sum().item())
        acc["accepted_wrong"] += int((accept_bool & (~patched_correct_mask)).sum().item())
        acc["rejected_correct"] += int((reject_bool & patched_correct_mask).sum().item())
        acc["rejected_wrong"] += int((reject_bool & (~patched_correct_mask)).sum().item())
        acc["router_accept_sum"] += float(route_accept_flat.sum().item())
        acc["fallback_usage_sum"] += float(fallback_mask.float().sum().item())
        acc["distance_sum"] += float(min_dist.sum().item())
        acc["patch_norm_sum"] += float(patch_norm.sum().item())

    return _annotate_split_gate_quality(split_name, _finalize_summary(acc))


def run_deploy_eval(cfg: DictConfig) -> str:
    """Run deployment-time evaluation from a resolved Hydra config."""
    validate_deploy_eval_config(cfg)
    runner = ExperimentRunner(cfg)
    record = runner.bootstrap()
    print(f"[Start] experiment={cfg.experiment.id} stage={cfg.experiment.stage}", flush=True)

    device = resolve_runtime_device(cfg)
    print(f"[Start] resolved_device={device}", flush=True)
    backbone, model, dataloaders, _loss_fn = build_stage3_bundle(cfg, device)

    load_info = _load_checkpoint_flexibly(model, str(cfg.deployment.checkpoint_path))
    model.to(device).eval()
    backbone.to(device).eval()

    eval_batch_size = int(cfg.evaluation.get("batch_size", cfg.train_loop.batch_size))
    num_workers = int(cfg.runtime.get("num_workers", 0))
    seen_loader = _build_eval_loader(dataloaders["bug_train"], eval_batch_size, num_workers)
    unseen_loader = _build_eval_loader(dataloaders["bug_eval"], eval_batch_size, num_workers)
    clean_loader = _build_eval_loader(dataloaders["clean_eval"], eval_batch_size, num_workers)

    gate_source = str(cfg.deployment.get("gate_source", "route_feat"))
    gate_metric = str(cfg.deployment.get("gate_metric", "l2"))
    gate_threshold_cfg = cfg.deployment.get("gate_threshold")
    gate_threshold = None if gate_threshold_cfg is None else float(gate_threshold_cfg)
    gate_chunk_size_cfg = cfg.deployment.get("gate_chunk_size")
    gate_chunk_size = None if gate_chunk_size_cfg is None else int(gate_chunk_size_cfg)
    if gate_source == "input_raw" and gate_threshold is None:
        gate_threshold = float(cfg.fault.epsilon)
    support_route_feat, support_patches, gate_support_feat, repair_bank_source = _load_or_build_repair_bank(
        model=model,
        backbone=backbone,
        seen_loader=seen_loader,
        device=device,
        cfg=cfg,
        gate_source=gate_source,
    )

    policy_info = _resolve_policy_semantics(cfg)
    policy = str(policy_info["effective_policy"])
    weighted_temperature = float(cfg.deployment.get("weighted_temperature", 0.25))
    weighted_top_k_cfg = cfg.deployment.get("weighted_top_k")
    weighted_top_k = None if weighted_top_k_cfg is None else int(weighted_top_k_cfg)
    adversarial_settings = _adversarial_settings(cfg)
    repair_bank_selection = _normalize_repair_bank_selection(
        str(_repair_bank_cfg(cfg).get("selection", "verified_repaired"))
    )
    repair_bank_bucket = _empty_repair_bank_bucket()

    split_specs = {
        "repair_support_seen": seen_loader,
        "repair_holdout_unseen": unseen_loader,
        "clean_eval": clean_loader,
    }
    # OPTIONAL fourth split, off by default. `clean_train` is the dataset's TRAIN split: it is
    # disjoint from every reported population (bug_eval and clean_test both come from the test
    # split), so patch behaviour dumped here can train a deployment gate without touching
    # anything the paper reports on. It exists because the gate's negative class -- clean inputs
    # the patch breaks -- numbers only 1-79 per cell on clean_calib, which is too few to
    # estimate the 7 post-repair feature dimensions from.
    if bool(cfg.deployment.get("dump_clean_train", False)):
        split_specs["clean_train"] = _build_eval_loader(
            dataloaders["clean_train"], eval_batch_size, num_workers)
    split_metrics: dict[str, dict[str, float | int]] = {}
    adversarial_metrics: dict[str, dict[str, float | int]] = {}
    prediction_dir = Path(record.artifact_root) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    save_features = bool(cfg.deployment.get("save_route_features", False))
    feature_dir = prediction_dir if save_features else None

    feature_gate_path = cfg.deployment.get("feature_gate_path")
    feature_gate = FeatureGate.load(str(feature_gate_path)) if feature_gate_path is not None else None
    feature_gate_lambda = float(cfg.deployment.get("feature_gate_lambda", 1.0))
    if feature_gate is not None:
        print(f"[Start] feature_gate loaded from {feature_gate_path} (lambda={feature_gate_lambda})", flush=True)

    for split_name, loader in split_specs.items():
        summary, rows = _evaluate_split(
            model=model,
            backbone=backbone,
            loader=loader,
            device=device,
            split_name=split_name,
            policy=policy,
            support_route_feat=support_route_feat,
            support_patches=support_patches,
            weighted_temperature=weighted_temperature,
            weighted_top_k=weighted_top_k,
            gate_support_feat=gate_support_feat,
            gate_source=gate_source,
            gate_metric=gate_metric,
            gate_threshold=gate_threshold,
            gate_chunk_size=gate_chunk_size,
            cfg=cfg,
            repair_bank_selection=repair_bank_selection,
            repair_bank_bucket=repair_bank_bucket,
            feature_dir=feature_dir,
            feature_gate=feature_gate,
            feature_gate_lambda=feature_gate_lambda,
        )
        if bool(cfg.artifacts.get("save_predictions", True)):
            _write_prediction_csv(prediction_dir / f"{split_name}_predictions.csv", rows)
        if split_name == "clean_train":
            # A dump-only split. Its CSV and feature tensors are written above -- that is the
            # whole point of the split -- but it must not reach split_metrics, which the report
            # builders index by the three shipped names.
            continue
        split_metrics[split_name] = summary

        if bool(adversarial_settings["enabled"]) and split_name in adversarial_settings["splits"]:
            for epsilon in adversarial_settings["epsilons"]:
                epsilon_key = f"{float(epsilon):.6f}"
                if "curve" not in adversarial_metrics:
                    adversarial_metrics["curve"] = {}
                if epsilon_key not in adversarial_metrics["curve"]:
                    adversarial_metrics["curve"][epsilon_key] = {}
                step_size = adversarial_settings["step_size"]
                if step_size is None:
                    step_size = float(epsilon) / max(int(adversarial_settings["steps"]) // 2, 1)
                print(f"[Adversarial] eps={epsilon_key} split={split_name} start", flush=True)
                adversarial_metrics["curve"][epsilon_key][split_name] = _evaluate_split_under_attack(
                    model=model,
                    backbone=backbone,
                    loader=loader,
                    device=device,
                    policy=policy,
                    support_route_feat=support_route_feat,
                    support_patches=support_patches,
                    weighted_temperature=weighted_temperature,
                    weighted_top_k=weighted_top_k,
                    gate_support_feat=gate_support_feat,
                    gate_source=gate_source,
                    gate_metric=gate_metric,
                    gate_threshold=gate_threshold,
                    gate_chunk_size=gate_chunk_size,
                    cfg=cfg,
                    split_name=split_name,
                    epsilon=float(epsilon),
                    steps=int(adversarial_settings["steps"]),
                    step_size=float(step_size),
                )
                adversarial_metrics["curve"][epsilon_key]["step_size"] = float(step_size)
                print(f"[Adversarial] eps={epsilon_key} split={split_name} done", flush=True)

    exported_repair_bank = _export_managed_repair_bank(
        record=record,
        cfg=cfg,
        selection=repair_bank_selection,
        bucket=repair_bank_bucket,
        gate_source=gate_source,
        source_info=repair_bank_source,
    )
    inference_checks = _build_inference_checks(
        policy_info=policy_info,
        split_metrics=split_metrics,
        weighted_temperature=weighted_temperature,
        weighted_top_k=weighted_top_k,
        adversarial_metrics=adversarial_metrics if adversarial_metrics else None,
        adversarial_settings=adversarial_settings,
    )
    metrics = {
        "status": "completed",
        "resolved_config_preview": OmegaConf.to_container(cfg.experiment, resolve=True),
        "policy": policy,
        "policy_request": str(policy_info["requested_policy"]),
        "policy_enforcement": {
            "effective_policy": str(policy_info["effective_policy"]),
            "enforce_router_bypass": bool(policy_info["enforce_router_bypass"]),
            "compatibility_note": policy_info["compatibility_note"],
        },
        "checkpoint_source": str(cfg.deployment.checkpoint_path),
        "checkpoint_experiment_id": load_info["checkpoint"].get("experiment_id"),
        "checkpoint_epoch": load_info["checkpoint"].get("epoch"),
        "checkpoint_load_info": {
            "skipped_keys": load_info["skipped_keys"],
            "missing_keys": load_info["missing_keys"],
            "unexpected_keys": load_info["unexpected_keys"],
        },
        "router": {
            "threshold": float(cfg.repair.tau_dist),
            "num_support": int(support_route_feat.size(0)),
            "gate_source": gate_source,
            "gate_metric": gate_metric,
            "gate_threshold": gate_threshold,
        },
        "weighted_fallback": {
            "temperature": weighted_temperature,
            "top_k": weighted_top_k,
        },
        "gate_quality": {
            "repair_support_seen": {
                "false_reject_count": int(split_metrics["repair_support_seen"]["false_reject_count"]),
                "false_reject_rate": float(split_metrics["repair_support_seen"]["false_reject_rate"]),
                "accept_correct": int(split_metrics["repair_support_seen"]["accepted_correct"]),
                "accept_wrong": int(split_metrics["repair_support_seen"]["accepted_wrong"]),
                "reject_correct": int(split_metrics["repair_support_seen"]["rejected_correct"]),
                "reject_wrong": int(split_metrics["repair_support_seen"]["rejected_wrong"]),
            },
            "repair_holdout_unseen": {
                "accept_correct": int(split_metrics["repair_holdout_unseen"]["accepted_correct"]),
                "accept_wrong": int(split_metrics["repair_holdout_unseen"]["accepted_wrong"]),
                "reject_correct": int(split_metrics["repair_holdout_unseen"]["rejected_correct"]),
                "reject_wrong": int(split_metrics["repair_holdout_unseen"]["rejected_wrong"]),
            },
            "clean_eval": {
                "false_accept_count": int(split_metrics["clean_eval"]["false_accept_count"]),
                "false_accept_rate": float(split_metrics["clean_eval"]["false_accept_rate"]),
                "accept_correct": int(split_metrics["clean_eval"]["accepted_correct"]),
                "accept_wrong": int(split_metrics["clean_eval"]["accepted_wrong"]),
                "reject_correct": int(split_metrics["clean_eval"]["rejected_correct"]),
                "reject_wrong": int(split_metrics["clean_eval"]["rejected_wrong"]),
            },
        },
        "repair_bank": {
            "source": repair_bank_source["source"],
            "loaded_path": repair_bank_source["path"],
            "source_count": int(repair_bank_source["count"]),
            "selection": repair_bank_selection,
            "exported": exported_repair_bank,
        },
        "adversarial": {
            "enabled": bool(adversarial_settings["enabled"]),
            "epsilon": float(adversarial_settings["epsilon"]),
            "epsilons": [float(v) for v in adversarial_settings["epsilons"]],
            "steps": int(adversarial_settings["steps"]),
            "step_size": adversarial_settings["step_size"],
            "splits": list(adversarial_settings["splits"]),
        },
        "splits": split_metrics,
        "adversarial_splits": adversarial_metrics,
        "inference_checks": inference_checks,
        "summary": {
            "seen_repaired": int(split_metrics["repair_support_seen"]["repaired"]),
            "seen_repair_rate": float(split_metrics["repair_support_seen"]["repaired_rate"]),
            "seen_router_accept_rate": float(split_metrics["repair_support_seen"]["router_accept_rate"]),
            "seen_false_reject_rate": float(split_metrics["repair_support_seen"]["false_reject_rate"]),
            "seen_false_reject_count": int(split_metrics["repair_support_seen"]["false_reject_count"]),
            "seen_robust_repair_rate": float((adversarial_metrics.get("curve", {}).get(f"{float(adversarial_settings['epsilon']):.6f}", {}).get("repair_support_seen", _empty_summary()))["patched_accuracy"]),
            "unseen_repaired": int(split_metrics["repair_holdout_unseen"]["repaired"]),
            "unseen_repair_rate": float(split_metrics["repair_holdout_unseen"]["repaired_rate"]),
            "unseen_router_accept_rate": float(split_metrics["repair_holdout_unseen"]["router_accept_rate"]),
            "unseen_robust_repair_rate": float((adversarial_metrics.get("curve", {}).get(f"{float(adversarial_settings['epsilon']):.6f}", {}).get("repair_holdout_unseen", _empty_summary()))["patched_accuracy"]),
            "clean_acc": float(split_metrics["clean_eval"]["patched_accuracy"]),
            "clean_rr": float(split_metrics["clean_eval"]["rr"]),
            "clean_router_accept_rate": float(split_metrics["clean_eval"]["router_accept_rate"]),
            "clean_false_accept_rate": float(split_metrics["clean_eval"]["false_accept_rate"]),
            "clean_false_accept_count": int(split_metrics["clean_eval"]["false_accept_count"]),
            "clean_fallback_usage_rate": float(split_metrics["clean_eval"]["fallback_usage_rate"]),
            "clean_robust_acc": float((adversarial_metrics.get("curve", {}).get(f"{float(adversarial_settings['epsilon']):.6f}", {}).get("clean_eval", _empty_summary()))["patched_accuracy"]),
        },
    }
    runner.save_metrics(metrics)

    print(f"Completed deploy eval run at: {record.artifact_root}", flush=True)
    print(f"Resolved config: {record.config_path}", flush=True)
    print(f"Metrics: {record.metrics_path}", flush=True)
    return record.artifact_root
