"""Routing logic for verified bug detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class DistanceRouter(nn.Module):
    """Distance-based router over a verified support set."""

    def __init__(
        self,
        num_experts: int,
        shallow_dim: int,
        threshold: float = 0.5,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.expert_keys = nn.Parameter(torch.randn(num_experts, shallow_dim))

    def forward(self, router_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary routing weights and minimum distances."""
        dist = torch.cdist(router_features, self.expert_keys)  # [batch, num_experts]
        min_dist = torch.min(dist, dim=1).values  # [batch]
        is_bug = (min_dist < self.threshold).float().view(-1, 1)  # [batch, 1]
        return is_bug, min_dist

    def soft_forward(self, router_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sigmoid routing weights and minimum distances."""
        dist = torch.cdist(router_features, self.expert_keys)  # [batch, num_experts]
        min_dist = torch.min(dist, dim=1).values  # [batch]
        logits = (self.threshold - min_dist) / max(self.temperature, 1.0e-8)  # [batch]
        weight = torch.sigmoid(logits).view(-1, 1)  # [batch, 1]
        return weight, min_dist

    def seed_support(self, router_features: torch.Tensor) -> None:
        """Seed support keys from observed shallow features."""
        num_to_fill = min(self.expert_keys.size(0), router_features.size(0))
        if num_to_fill <= 0:
            return
        with torch.no_grad():
            self.expert_keys[:num_to_fill] = router_features[:num_to_fill].clone()
