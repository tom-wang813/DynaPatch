"""Hypernetwork that synthesizes residual patches."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block for smoother continuous patch generation."""

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden_dim = int(max(dim * expansion, dim))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class HyperNetworkPatchGenerator(nn.Module):
    """Generate a residual patch from shallow features."""

    def __init__(
        self,
        model_type: str,
        shallow_dim: int,
        out_dim: int,
        context_dim: int = 0,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 1,
        hypernet_style: str = "plain",
        pool_size: int = 1,
        condition_source: str = "shallow",
        num_basis: int = 0,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.context_dim = context_dim
        self.num_hidden_layers = max(int(num_hidden_layers), 1)
        self.hypernet_style = str(hypernet_style).lower()
        # pool_size=1 is plain global average pooling: the conditioning signal collapses to one
        # value per channel and all spatial layout is discarded. Larger grids keep a coarse
        # spatial map, which costs input width (shallow_dim * pool_size^2) but no extra depth,
        # so the hypernet still runs on its own CUDA stream alongside the deep path.
        self.pool_size = max(int(pool_size), 1)
        # "shallow" conditions on the early feature map, which lets the patch be computed in
        # parallel with the deep path. "deep" conditions on the final representation instead --
        # the one the patch is actually added to, and the one the misclassification lives in --
        # but the caller must then run the deep path first, serialising the two streams.
        self.condition_source = str(condition_source).lower()

        cells = self.pool_size * self.pool_size
        if self.condition_source == "deep":
            # deep_feat arrives already pooled and flattened as [B, out_dim].
            input_dim = out_dim + context_dim
        elif model_type == "ViT":
            input_dim = shallow_dim * 197 + context_dim
        else:
            input_dim = shallow_dim * cells + context_dim
        default_hidden_dim = 128 if model_type == "ViT" else shallow_dim * 2
        self.hidden_dim = int(default_hidden_dim if hidden_dim is None else hidden_dim)
        self.pool = (
            nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
            if model_type != "ViT" and self.condition_source != "deep"
            else nn.Identity()
        )
        self.flatten = nn.Flatten()

        if self.hypernet_style in ("residual_ln", "bank_coef"):
            self.input_proj = nn.Linear(input_dim, self.hidden_dim)
            self.blocks = nn.ModuleList(
                [ResidualMLPBlock(self.hidden_dim) for _ in range(self.num_hidden_layers)]
            )
            self.output_norm = nn.LayerNorm(self.hidden_dim)
            if self.hypernet_style == "bank_coef":
                # The delta's DIRECTION is not learned: it is spanned by the classifier's own
                # logit basis (d logits / d deep_feat), which is fixed and supplied by
                # set_basis(). Only the per-input MIXTURE over that basis is learned, so the
                # output layer shrinks from hidden x out_dim to hidden x num_basis --
                # 18x smaller on ConvNeXt, ~580x on VGG-16. Tests whether the trained hypernet
                # is learning anything beyond selectivity (see PITFALLS 2026-07-28).
                if num_basis <= 0:
                    raise ValueError("hypernet_style='bank_coef' requires num_basis > 0")
                self.num_basis = int(num_basis)
                self.coef_proj = nn.Linear(self.hidden_dim, self.num_basis)
                self.register_buffer("basis", torch.zeros(self.num_basis, out_dim))
                self.output_proj = None
            else:
                self.output_proj = nn.Linear(self.hidden_dim, out_dim)
            self.network = None
        else:
            layers: list[nn.Module] = []
            in_dim = input_dim
            for _ in range(self.num_hidden_layers):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                in_dim = self.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            self.network = nn.Sequential(*layers)
            self.input_proj = None
            self.blocks = nn.ModuleList()
            self.output_norm = None
            self.output_proj = None

    def set_basis(self, basis: torch.Tensor) -> None:
        """Install the fixed direction basis for hypernet_style='bank_coef'.

        `basis` is [num_basis, out_dim], normally d logits / d deep_feat of the frozen
        classifier. It is a buffer, not a parameter -- it is never updated by training.
        """
        if self.hypernet_style != "bank_coef":
            raise RuntimeError("set_basis() is only meaningful for hypernet_style='bank_coef'")
        if tuple(basis.shape) != tuple(self.basis.shape):
            raise ValueError(f"basis shape {tuple(basis.shape)} != expected {tuple(self.basis.shape)}")
        self.basis.copy_(basis.detach().to(self.basis.dtype))

    def forward(self, shallow_feat: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Generate the last-layer residual patch."""
        pooled = self.pool(shallow_feat)          # [B, C, pool_size, pool_size]
        flattened = self.flatten(pooled)          # [B, C * pool_size^2]
        if self.context_dim > 0:
            if context is None:
                context = torch.zeros(flattened.size(0), self.context_dim, device=flattened.device, dtype=flattened.dtype)
            flattened = torch.cat([flattened, context], dim=1)

        if self.hypernet_style in ("residual_ln", "bank_coef"):
            hidden = self.input_proj(flattened)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = self.output_norm(hidden)
            if self.hypernet_style == "bank_coef":
                if not bool(torch.any(self.basis != 0)):
                    raise RuntimeError("bank_coef hypernet used before set_basis() was called")
                coef = self.coef_proj(hidden)          # [batch, num_basis]
                return coef @ self.basis               # [batch, out_dim]
            return self.output_proj(hidden)

        hidden = flattened
        offset = 0
        for _ in range(self.num_hidden_layers):
            hidden = self.network[offset](hidden)
            hidden = self.network[offset + 1](hidden)
            offset += 2
        return self.network[offset](hidden)


class ConstantPatchGenerator(nn.Module):
    """Single shared residual patch — no input conditioning.

    Drop-in replacement for HyperNetworkPatchGenerator.  The same delta
    vector is broadcast to every sample in the batch; the hypernet is
    eliminated entirely.  All downstream components (router, patch_operator,
    deployment policy) work unchanged.
    """

    def __init__(self, patch_dim: int) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.delta = nn.Parameter(torch.zeros(patch_dim))

    def forward(self, shallow_feat: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Return delta expanded to match batch size."""
        batch = shallow_feat.size(0)
        return self.delta.unsqueeze(0).expand(batch, -1).contiguous()
