"""Fallback strategies for unseen bug states."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class FallbackPolicy(nn.Module):
    """Fallback patch strategy used when a bug is outside the verified support set."""

    def __init__(self, strategy: Literal["zero", "direct"] = "zero") -> None:
        super().__init__()
        self.strategy = strategy

    def forward(self, patch: torch.Tensor, route_weight: torch.Tensor) -> torch.Tensor:
        """Apply the configured fallback policy."""
        if self.strategy == "direct":
            return patch
        return torch.zeros_like(patch) * route_weight
