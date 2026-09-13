"""Metrics for repair-only experiments."""

from __future__ import annotations

import torch


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute loss-agnostic classification metrics from logits and labels."""
    predictions = torch.argmax(logits, dim=1)  # [batch]
    correct = (predictions == labels).float()  # [batch]
    return {
        "accuracy": float(correct.mean().item()),
        "num_samples": float(labels.numel()),
    }


def compute_binary_regression_rate(
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute regression rate relative to base predictions on clean data."""
    base_pred = torch.argmax(base_logits, dim=1)  # [batch]
    patched_pred = torch.argmax(patched_logits, dim=1)  # [batch]
    base_correct = base_pred == labels  # [batch]
    regressed = base_correct & (patched_pred != labels)  # [batch]
    normalizer = torch.clamp(base_correct.float().sum(), min=1.0)
    return float(regressed.float().sum().item() / normalizer.item())
