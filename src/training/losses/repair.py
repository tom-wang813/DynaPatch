"""Repair-only training loss."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepairClassificationLoss(nn.Module):
    """Cross-entropy repair loss on the defect set."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the repair objective on bug examples."""
        return F.cross_entropy(logits, labels)


class FocalRepairClassificationLoss(nn.Module):
    """Focal-style hard-example reweighting on the defect set."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Emphasize bug examples that remain hard for the current repair model."""
        per_example_ce = F.cross_entropy(logits, labels, reduction="none")
        true_probs = torch.softmax(logits, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
        weights = torch.pow(1.0 - true_probs.clamp(min=1.0e-8, max=1.0), self.gamma)
        return (weights * per_example_ce).mean()


class SafetyAwareRepairClassificationLoss(nn.Module):
    """Bug-side repair loss with explicit safety/risk semantics.

    This keeps the standard CE objective, but augments it with:
    - critical-class upweighting on the true label
    - differentiable expected-risk under a provided risk matrix
    """

    def __init__(
        self,
        risk_matrix_path: str,
        lambda_safety_risk: float = 0.0,
        critical_class_weight: float = 1.0,
        critical_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        risk_path = Path(risk_matrix_path)
        payload = json.loads(risk_path.read_text(encoding="utf-8"))
        matrix_key = "matrix" if "matrix" in payload else "risk_matrix"
        if matrix_key not in payload:
            raise KeyError("Safety risk payload must contain `matrix` or `risk_matrix`.")
        matrix = torch.tensor(payload[matrix_key], dtype=torch.float32)
        if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
            raise ValueError("Risk matrix must be square [num_classes, num_classes].")
        self.register_buffer("risk_matrix", matrix)
        self.lambda_safety_risk = float(lambda_safety_risk)
        self.critical_class_weight = float(critical_class_weight)

        critical_mask = torch.zeros(matrix.size(0), dtype=torch.bool)
        if critical_indices is not None:
            for idx in critical_indices:
                if 0 <= int(idx) < critical_mask.numel():
                    critical_mask[int(idx)] = True
        self.register_buffer("critical_mask", critical_mask)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Combine CE with expected safety risk on bug samples."""
        per_example_ce = F.cross_entropy(logits, labels, reduction="none")
        critical_mask = self.critical_mask.to(labels.device)
        if self.critical_class_weight > 1.0:
            weights = torch.ones_like(per_example_ce)
            critical = critical_mask[labels]
            weights = torch.where(
                critical,
                torch.full_like(weights, self.critical_class_weight),
                weights,
            )
            ce_term = (weights * per_example_ce).mean()
        else:
            ce_term = per_example_ce.mean()

        if self.lambda_safety_risk <= 0.0:
            return ce_term

        probs = torch.softmax(logits, dim=1)  # [batch, C]
        risk_matrix = self.risk_matrix.to(logits.device)
        per_label_risk = risk_matrix[labels]  # [batch, C]
        expected_risk = (probs * per_label_risk).sum(dim=1).mean()
        return ce_term + self.lambda_safety_risk * expected_risk
