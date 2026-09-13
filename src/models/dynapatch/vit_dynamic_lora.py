"""Dynamic LoRA components for ViT-oriented repair experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicLoRAHeadGenerator(nn.Module):
    """Generate per-sample low-rank updates for a ViT classification head.

    This first version targets the final classifier head rather than internal
    attention blocks. It is intentionally lightweight so that we can compare it
    fairly against head-only tuning and sparse last-layer repair on small or
    medium-scale safety benchmarks.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        rank: int = 8,
        attention_rank: int = 0,
        mlp_hidden_dim: int = 256,
        num_hidden_layers: int = 1,
        pool: str = "cls",
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.attention_rank = int(max(attention_rank, 0))
        self.pool = str(pool).lower()
        self.context_dim = int(max(context_dim, 0))

        if self.rank <= 0:
            raise ValueError("DynamicLoRAHeadGenerator requires rank > 0.")

        layers: list[nn.Module] = []
        in_dim = self.hidden_dim + self.context_dim
        hidden = int(max(mlp_hidden_dim, self.hidden_dim))
        for _ in range(int(max(num_hidden_layers, 1))):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.proj_a = nn.Linear(in_dim, self.rank * self.hidden_dim)
        self.proj_b = nn.Linear(in_dim, self.out_dim * self.rank)
        if self.attention_rank > 0:
            self.attn_proj_a = nn.Linear(in_dim, self.attention_rank * self.hidden_dim)
            self.attn_proj_b = nn.Linear(in_dim, self.hidden_dim * self.attention_rank)
        else:
            self.attn_proj_a = None
            self.attn_proj_b = None

    def _pool_tokens(self, shallow_feat: torch.Tensor) -> torch.Tensor:
        if shallow_feat.dim() != 3:
            raise ValueError(
                f"DynamicLoRAHeadGenerator expects ViT tokens with shape [batch, tokens, dim], got {tuple(shallow_feat.shape)}."
            )
        if self.pool == "cls":
            return shallow_feat[:, 0, :]  # [batch, dim]
        if self.pool == "mean":
            return shallow_feat.mean(dim=1)  # [batch, dim]
        raise ValueError(f"Unsupported ViT dynamic LoRA pooling mode: {self.pool}")

    def forward(
        self,
        shallow_feat: torch.Tensor,
        prototype_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self._pool_tokens(shallow_feat)  # [batch, dim]
        if self.context_dim > 0:
            if prototype_context is None:
                prototype_context = torch.zeros(
                    pooled.size(0),
                    self.context_dim,
                    device=pooled.device,
                    dtype=pooled.dtype,
                )  # [batch, context_dim]
            hidden_inputs = torch.cat([pooled, prototype_context], dim=1)  # [batch, dim + context_dim]
        else:
            hidden_inputs = pooled  # [batch, dim]
        hidden = self.backbone(hidden_inputs) if len(self.backbone) > 0 else hidden_inputs
        matrix_a = self.proj_a(hidden).view(-1, self.rank, self.hidden_dim)  # [batch, rank, d]
        matrix_b = self.proj_b(hidden).view(-1, self.out_dim, self.rank)  # [batch, c, rank]
        outputs = {"lora_a": matrix_a, "lora_b": matrix_b}
        if self.attention_rank > 0 and self.attn_proj_a is not None and self.attn_proj_b is not None:
            attn_a = self.attn_proj_a(hidden).view(-1, self.attention_rank, self.hidden_dim)  # [batch, r_a, d]
            attn_b = self.attn_proj_b(hidden).view(-1, self.hidden_dim, self.attention_rank)  # [batch, d, r_a]
            outputs["attn_lora_a"] = attn_a
            outputs["attn_lora_b"] = attn_b
        return outputs


class StaticLoRAHeadGenerator(nn.Module):
    """Learn a shared LoRA adapter for ViT repair.

    This baseline removes all input-conditioned dynamics. It keeps the same
    downstream deployment and gate protocol as the dynamic variant so we can
    isolate whether the gains or failures come from LoRA itself versus the
    per-sample conditioning mechanism.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        rank: int = 8,
        attention_rank: int = 0,
        init_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.attention_rank = int(max(attention_rank, 0))
        if self.rank <= 0:
            raise ValueError("StaticLoRAHeadGenerator requires rank > 0.")

        scale = float(init_scale)
        self.lora_a = nn.Parameter(torch.randn(self.rank, self.hidden_dim) * scale)  # [rank, d]
        self.lora_b = nn.Parameter(torch.randn(self.out_dim, self.rank) * scale)  # [c, rank]
        if self.attention_rank > 0:
            self.attn_lora_a = nn.Parameter(torch.randn(self.attention_rank, self.hidden_dim) * scale)  # [r_a, d]
            self.attn_lora_b = nn.Parameter(torch.randn(self.hidden_dim, self.attention_rank) * scale)  # [d, r_a]
        else:
            self.attn_lora_a = None
            self.attn_lora_b = None

    def forward(
        self,
        shallow_feat: torch.Tensor,
        prototype_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if shallow_feat.dim() != 3:
            raise ValueError(
                f"StaticLoRAHeadGenerator expects ViT tokens with shape [batch, tokens, dim], got {tuple(shallow_feat.shape)}."
            )
        batch = int(shallow_feat.size(0))
        lora_a = self.lora_a.unsqueeze(0).expand(batch, -1, -1)  # [batch, rank, d]
        lora_b = self.lora_b.unsqueeze(0).expand(batch, -1, -1)  # [batch, c, rank]
        outputs = {"lora_a": lora_a, "lora_b": lora_b}
        if self.attn_lora_a is not None and self.attn_lora_b is not None:
            outputs["attn_lora_a"] = self.attn_lora_a.unsqueeze(0).expand(batch, -1, -1)  # [batch, r_a, d]
            outputs["attn_lora_b"] = self.attn_lora_b.unsqueeze(0).expand(batch, -1, -1)  # [batch, d, r_a]
        return outputs


def apply_dynamic_lora_to_linear(
    *,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor | None,
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    """Apply a per-sample LoRA update to a linear head.

    Args:
        base_weight: `[out_dim, hidden_dim]`
        base_bias: `[out_dim]` or `None`
        inputs: `[batch, hidden_dim]`
        lora_a: `[batch, rank, hidden_dim]`
        lora_b: `[batch, out_dim, rank]`
    """
    base_logits = torch.nn.functional.linear(inputs, base_weight, base_bias)  # [batch, out_dim]
    delta_weight = torch.bmm(lora_b, lora_a)  # [batch, out_dim, hidden_dim]
    delta_logits = torch.bmm(delta_weight, inputs.unsqueeze(-1)).squeeze(-1)  # [batch, out_dim]
    return base_logits + delta_logits


def apply_dynamic_lora_to_tokens(
    *,
    tokens: torch.Tensor,
    lora_a: torch.Tensor | None,
    lora_b: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a per-sample low-rank adapter to token features.

    Args:
        tokens: `[batch, tokens, hidden_dim]`
        lora_a: `[batch, rank, hidden_dim]` or `None`
        lora_b: `[batch, hidden_dim, rank]` or `None`
    """
    if lora_a is None or lora_b is None:
        return tokens
    delta_weight = torch.bmm(lora_b, lora_a)  # [batch, hidden_dim, hidden_dim]
    delta_tokens = torch.matmul(tokens, delta_weight.transpose(1, 2))  # [batch, tokens, hidden_dim]
    return tokens + delta_tokens
