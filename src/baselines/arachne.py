"""Arachne-style sparse last-layer search baseline.

This is a no-gate, search-based repair baseline over the final linear layer.
It differs from gradient head-only fine-tuning by:

- selecting a sparse subset of influential parameters
- updating them through greedy coordinate search
- evaluating candidates on a bug/clean mixed objective
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ParameterRef:
    name: str
    flat_index: int
    tensor_index: tuple[int, ...]


@dataclass
class SearchConfig:
    top_k: int = 64
    rounds: int = 3
    step_scale: float = 0.5
    clean_tradeoff: float = 0.25
    max_clean_batches: int = 8


@dataclass
class FeatureBank:
    features: torch.Tensor
    labels: torch.Tensor
    base_pred: torch.Tensor


def _flatten_with_refs(linear: nn.Linear) -> tuple[torch.Tensor, list[ParameterRef]]:
    refs: list[ParameterRef] = []
    values: list[torch.Tensor] = []
    offset = 0
    for name, param in (("weight", linear.weight), ("bias", linear.bias)):
        flat = param.detach().reshape(-1)
        values.append(flat)
        for idx in range(flat.numel()):
            if name == "weight":
                row = idx // linear.in_features
                col = idx % linear.in_features
                tensor_index = (row, col)
            else:
                tensor_index = (idx,)
            refs.append(ParameterRef(name=name, flat_index=offset + idx, tensor_index=tensor_index))
        offset += flat.numel()
    return torch.cat(values, dim=0), refs


def evaluate_sparse_objective_from_features(
    *,
    classifier: nn.Linear,
    bug_bank: FeatureBank,
    clean_bank: FeatureBank | None,
    clean_tradeoff: float,
) -> dict[str, float]:
    with torch.no_grad():
        base_weight = classifier.weight.detach()
        base_bias = classifier.bias.detach()
        bug_logits = F.linear(bug_bank.features, base_weight, base_bias)
        bug_pred = bug_logits.argmax(dim=1)
        repaired = int(((bug_bank.base_pred != bug_bank.labels) & (bug_pred == bug_bank.labels)).sum().item())
        harmful = int(((bug_bank.base_pred == bug_bank.labels) & (bug_pred != bug_bank.labels)).sum().item())
        bug_total = int(bug_bank.labels.numel())

        clean_correct = 0
        clean_base_correct = 0
        clean_total = 0
        if clean_bank is not None:
            clean_logits = F.linear(clean_bank.features, base_weight, base_bias)
            clean_pred = clean_logits.argmax(dim=1)
            clean_base_correct = int((clean_bank.base_pred == clean_bank.labels).sum().item())
            clean_correct = int((clean_pred == clean_bank.labels).sum().item())
            clean_total = int(clean_bank.labels.numel())

    repaired_rate = repaired / max(bug_total, 1)
    harmful_rate = harmful / max(bug_total, 1)
    clean_acc = clean_correct / max(clean_total, 1)
    score = repaired_rate - harmful_rate + (clean_tradeoff * clean_acc)
    return {
        "score": float(score),
        "repaired_rate": float(repaired_rate),
        "harmful_rate": float(harmful_rate),
        "clean_acc": float(clean_acc),
        "clean_base_correct": float(clean_base_correct),
    }


def select_topk_parameters(
    model: nn.Module,
    architecture: str,
    classifier: nn.Linear,
    loader,
    device: torch.device,
    top_k: int,
) -> list[ParameterRef]:
    del architecture
    original_requires_grad = [param.requires_grad for param in classifier.parameters()]
    for param in classifier.parameters():
        param.requires_grad = True
    classifier.zero_grad(set_to_none=True)
    model.train()
    batches = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        batches += 1
    if batches == 0:
        for param, flag in zip(classifier.parameters(), original_requires_grad):
            param.requires_grad = flag
        return []

    weight_grad = classifier.weight.grad.detach().abs().reshape(-1)
    bias_grad = classifier.bias.grad.detach().abs().reshape(-1)
    grad_scores = torch.cat([weight_grad, bias_grad], dim=0)
    _flat_values, refs = _flatten_with_refs(classifier)
    top_k = min(int(top_k), grad_scores.numel())
    indices = torch.topk(grad_scores, k=top_k).indices.tolist()
    for param, flag in zip(classifier.parameters(), original_requires_grad):
        param.requires_grad = flag
    return [refs[idx] for idx in indices]


def _get_param_value(linear: nn.Linear, ref: ParameterRef) -> float:
    tensor = getattr(linear, ref.name)
    return float(tensor[ref.tensor_index].item())


def _set_param_value(linear: nn.Linear, ref: ParameterRef, value: float) -> None:
    tensor = getattr(linear, ref.name)
    tensor.data[ref.tensor_index] = value


def greedy_coordinate_search(
    classifier: nn.Linear,
    bug_bank: FeatureBank,
    clean_bank: FeatureBank | None,
    selected_refs: list[ParameterRef],
    config: SearchConfig,
) -> dict[str, object]:
    weight_scale = float(classifier.weight.detach().std().item()) if classifier.weight.numel() > 1 else 1.0
    bias_scale = float(classifier.bias.detach().std().item()) if classifier.bias.numel() > 1 else 1.0
    base_metrics = evaluate_sparse_objective_from_features(
        classifier=classifier,
        bug_bank=bug_bank,
        clean_bank=clean_bank,
        clean_tradeoff=config.clean_tradeoff,
    )
    best_score = float(base_metrics["score"])
    accepted_steps: list[dict[str, object]] = []

    for round_idx in range(int(config.rounds)):
        round_improved = False
        for ref in selected_refs:
            current = _get_param_value(classifier, ref)
            scale = weight_scale if ref.name == "weight" else bias_scale
            candidate_values = [
                current,
                current + (config.step_scale * scale),
                current - (config.step_scale * scale),
                current + (2.0 * config.step_scale * scale),
                current - (2.0 * config.step_scale * scale),
            ]
            local_best_value = current
            local_best_metrics = base_metrics
            local_best_score = best_score
            for candidate in candidate_values:
                _set_param_value(classifier, ref, candidate)
                metrics = evaluate_sparse_objective_from_features(
                    classifier=classifier,
                    bug_bank=bug_bank,
                    clean_bank=clean_bank,
                    clean_tradeoff=config.clean_tradeoff,
                )
                score = float(metrics["score"])
                if score > local_best_score + 1.0e-8:
                    local_best_score = score
                    local_best_value = candidate
                    local_best_metrics = metrics
            _set_param_value(classifier, ref, local_best_value)
            if local_best_score > best_score + 1.0e-8:
                best_score = local_best_score
                base_metrics = local_best_metrics
                round_improved = True
                accepted_steps.append(
                    {
                        "round": round_idx + 1,
                        "parameter": ref.name,
                        "tensor_index": list(ref.tensor_index),
                        "value": float(local_best_value),
                        "score": float(local_best_score),
                        "repaired_rate": float(local_best_metrics["repaired_rate"]),
                        "harmful_rate": float(local_best_metrics["harmful_rate"]),
                        "clean_acc": float(local_best_metrics["clean_acc"]),
                    }
                )
        if not round_improved:
            break

    return {
        "best_score": float(best_score),
        "selected_top_k": len(selected_refs),
        "steps": accepted_steps,
        "final_search_metrics": base_metrics,
    }
