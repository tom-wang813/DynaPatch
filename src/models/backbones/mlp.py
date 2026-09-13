"""Simple MLP backbone for tabular safety benchmarks such as ACAS Xu."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    """Feed-forward MLP with explicit feature extractor and linear classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 64, 64),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(v) for v in hidden_dims]]
        if len(dims) < 2:
            raise ValueError("MLPBackbone requires at least one hidden layer.")

        normalized_activation = activation.lower()
        if normalized_activation == "relu":
            activation_ctor = nn.ReLU
        elif normalized_activation == "tanh":
            activation_ctor = nn.Tanh
        else:
            raise ValueError(f"Unsupported MLP activation: {activation}")

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_ctor())
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], int(num_classes))

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(v) for v in hidden_dims]
        self.out_dim = dims[-1]

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > 2:
            inputs = torch.flatten(inputs, 1)
        return self.feature_layers(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))
