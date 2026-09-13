"""ViT repair model using a dynamic LoRA update on the classifier head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.dynapatch.prototype_bank import PrototypeBank
from src.models.dynapatch.confusion_pair_bank import ConfusionPairBank
from src.models.dynapatch.router import DistanceRouter
from src.models.dynapatch.vit_dynamic_lora import apply_dynamic_lora_to_linear, apply_dynamic_lora_to_tokens


def _resolve_linear_head(classifier: nn.Module) -> tuple[nn.Module, nn.Linear]:
    """Return a feature preprocessor and final linear head for ViT classifiers."""
    if isinstance(classifier, nn.Linear):
        return nn.Identity(), classifier
    if isinstance(classifier, nn.Sequential) and len(classifier) >= 1 and isinstance(classifier[-1], nn.Linear):
        pre = nn.Sequential(*list(classifier.children())[:-1])
        head = classifier[-1]
        return pre, head
    if hasattr(classifier, "head") and isinstance(classifier.head, nn.Linear):
        pre = nn.Sequential(*[m for name, m in classifier.named_children() if name != "head"])
        return pre if len(pre) > 0 else nn.Identity(), classifier.head
    raise NotImplementedError(f"Unsupported ViT classifier structure: {type(classifier).__name__}")


class ViTDynamicLoRAModel(nn.Module):
    """DynaPatch-style wrapper using dynamic LoRA for ViT classifier-head repair."""

    def __init__(
        self,
        decomposition,
        router: DistanceRouter,
        lora_generator: nn.Module,
        prototype_bank: PrototypeBank | ConfusionPairBank | None = None,
        route_mode: str = "distance",
        class_group_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.decomposition = decomposition
        self.router = router
        self.lora_generator = lora_generator
        self.prototype_bank = prototype_bank
        self.route_mode = str(route_mode)
        if class_group_matrix is None:
            self.register_buffer("class_group_matrix", torch.empty(0, 0), persistent=False)
        else:
            self.register_buffer("class_group_matrix", class_group_matrix.float(), persistent=False)

        self.feature_pre, self.linear_head = _resolve_linear_head(self.decomposition.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(x)["logits"]

    def seed_support(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            shallow_feat = self.decomposition.extract_shallow(x)
            route_feat = self.decomposition.router_features(shallow_feat)  # [batch, shallow_dim]
            self.router.seed_support(route_feat)
            if isinstance(self.prototype_bank, PrototypeBank):
                self.prototype_bank.seed_from_features(route_feat)

    def trainable_parameters(self):
        yield from self.router.parameters()
        yield from self.lora_generator.parameters()

    def _prototype_context(self, route_feat: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.prototype_bank is None:
            prototype_ids = torch.full((route_feat.size(0),), -1, device=route_feat.device, dtype=torch.long)  # [batch]
            return None, prototype_ids
        context, prototype_ids = self.prototype_bank.lookup(route_feat)
        return context, prototype_ids

    def _group_context(self, deep_feat: torch.Tensor) -> torch.Tensor | None:
        if self.class_group_matrix.numel() == 0:
            return None
        deep_pre = self.feature_pre(deep_feat)  # [batch, dim]
        base_logits = torch.nn.functional.linear(
            deep_pre,
            self.linear_head.weight,
            self.linear_head.bias,
        )  # [batch, classes]
        base_probs = torch.softmax(base_logits, dim=1)  # [batch, classes]
        group_matrix = self.class_group_matrix.to(device=base_probs.device, dtype=base_probs.dtype)  # [classes, groups]
        return base_probs @ group_matrix  # [batch, groups]

    def _conditioning_context(
        self,
        *,
        route_feat: torch.Tensor,
        deep_feat: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        prototype_context, prototype_ids = self._prototype_context(route_feat)
        group_context = self._group_context(deep_feat)
        if prototype_context is None:
            return group_context, prototype_ids
        if group_context is None:
            return prototype_context, prototype_ids
        return torch.cat([prototype_context, group_context], dim=1), prototype_ids  # [batch, proto+groups]

    def _route(self, route_feat: torch.Tensor, patch_summary: torch.Tensor) -> torch.Tensor:
        if self.route_mode == "always_on":
            return torch.ones(patch_summary.size(0), 1, device=patch_summary.device)  # [batch, 1]
        if self.route_mode == "soft_distance":
            route_weight, _ = self.router.soft_forward(route_feat)
            return route_weight
        route_weight, _ = self.router(route_feat)
        return route_weight

    def patch_from_shallow(
        self,
        shallow_feat: torch.Tensor,
        route_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a flattened dynamic-LoRA patch summary for deployment use."""
        if route_feat is None:
            route_feat = self.decomposition.router_features(shallow_feat)  # [batch, dim]
        deep_feat = self.decomposition.extract_deep(shallow_feat)  # [batch, dim]
        context, _prototype_ids = self._conditioning_context(route_feat=route_feat, deep_feat=deep_feat)
        lora = self.lora_generator(shallow_feat, prototype_context=context)
        lora_a = lora["lora_a"]  # [batch, rank, dim]
        lora_b = lora["lora_b"]  # [batch, classes, rank]
        parts = [lora_a.flatten(1), lora_b.flatten(1)]
        if "attn_lora_a" in lora and "attn_lora_b" in lora:
            parts.append(lora["attn_lora_a"].flatten(1))
            parts.append(lora["attn_lora_b"].flatten(1))
        return torch.cat(parts, dim=1)  # [batch, flat_patch]

    def patch_operator(self, patch: torch.Tensor, route_weight: torch.Tensor) -> torch.Tensor:
        """Scale a flattened LoRA patch summary by the route weight."""
        return patch * route_weight  # [batch, flat_patch]

    def _decode_patch(
        self,
        patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Decode a flattened patch summary back into LoRA matrices."""
        batch = patch.size(0)
        rank = int(self.lora_generator.rank)
        hidden_dim = int(self.lora_generator.hidden_dim)
        out_dim = int(self.lora_generator.out_dim)
        offset = 0
        head_a_size = rank * hidden_dim
        head_b_size = out_dim * rank
        lora_a = patch[:, offset : offset + head_a_size].view(batch, rank, hidden_dim)  # [batch, rank, dim]
        offset += head_a_size
        lora_b = patch[:, offset : offset + head_b_size].view(batch, out_dim, rank)  # [batch, classes, rank]
        offset += head_b_size

        attn_a = None
        attn_b = None
        attn_rank = int(getattr(self.lora_generator, "attention_rank", 0))
        if attn_rank > 0:
            attn_a_size = attn_rank * hidden_dim
            attn_b_size = hidden_dim * attn_rank
            attn_a = patch[:, offset : offset + attn_a_size].view(batch, attn_rank, hidden_dim)  # [batch, r_a, dim]
            offset += attn_a_size
            attn_b = patch[:, offset : offset + attn_b_size].view(batch, hidden_dim, attn_rank)  # [batch, dim, r_a]
        return lora_a, lora_b, attn_a, attn_b

    def logits_from_patch(self, shallow_feat: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """Apply a flattened dynamic-LoRA patch summary at deployment time."""
        lora_a, lora_b, attn_a, attn_b = self._decode_patch(patch)
        adapted_shallow = apply_dynamic_lora_to_tokens(
            tokens=shallow_feat,
            lora_a=attn_a,
            lora_b=attn_b,
        )  # [batch, tokens, dim]
        deep_feat = self.decomposition.extract_deep(adapted_shallow)  # [batch, dim]
        deep_pre = self.feature_pre(deep_feat)  # [batch, dim]
        return apply_dynamic_lora_to_linear(
            base_weight=self.linear_head.weight,
            base_bias=self.linear_head.bias,
            inputs=deep_pre,
            lora_a=lora_a,
            lora_b=lora_b,
        )  # [batch, classes]

    def forward_with_intermediates(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shallow_feat = self.decomposition.extract_shallow(x)  # [batch, tokens, dim]
        route_feat = self.decomposition.router_features(shallow_feat)  # [batch, dim]
        deep_feat = self.decomposition.extract_deep(shallow_feat)  # [batch, dim]
        context, prototype_ids = self._conditioning_context(route_feat=route_feat, deep_feat=deep_feat)
        lora = self.lora_generator(shallow_feat, prototype_context=context)
        lora_a = lora["lora_a"]  # [batch, rank, dim]
        lora_b = lora["lora_b"]  # [batch, classes, rank]
        attn_lora_a = lora.get("attn_lora_a")
        attn_lora_b = lora.get("attn_lora_b")

        parts = [lora_a.flatten(1), lora_b.flatten(1)]
        if attn_lora_a is not None and attn_lora_b is not None:
            parts.append(attn_lora_a.flatten(1))
            parts.append(attn_lora_b.flatten(1))
        patch = torch.cat(parts, dim=1)  # [batch, flat_patch]
        route_weight = self._route(route_feat, patch)  # [batch, 1]
        gated_lora_a = lora_a * route_weight.unsqueeze(-1)  # [batch, rank, dim]
        gated_lora_b = lora_b * route_weight.unsqueeze(-1)  # [batch, classes, rank]
        gated_attn_lora_a = None if attn_lora_a is None else attn_lora_a * route_weight.unsqueeze(-1)  # [batch, r_a, dim]
        gated_attn_lora_b = None if attn_lora_b is None else attn_lora_b * route_weight.unsqueeze(-1)  # [batch, dim, r_a]
        gated_parts = [gated_lora_a.flatten(1), gated_lora_b.flatten(1)]
        if gated_attn_lora_a is not None and gated_attn_lora_b is not None:
            gated_parts.append(gated_attn_lora_a.flatten(1))
            gated_parts.append(gated_attn_lora_b.flatten(1))
        gated_patch = torch.cat(gated_parts, dim=1)  # [batch, flat_patch]
        adapted_shallow = apply_dynamic_lora_to_tokens(
            tokens=shallow_feat,
            lora_a=gated_attn_lora_a,
            lora_b=gated_attn_lora_b,
        )  # [batch, tokens, dim]
        deep_feat = self.decomposition.extract_deep(adapted_shallow)  # [batch, dim]
        deep_pre = self.feature_pre(deep_feat)  # [batch, dim]

        logits = apply_dynamic_lora_to_linear(
            base_weight=self.linear_head.weight,
            base_bias=self.linear_head.bias,
            inputs=deep_pre,
            lora_a=gated_lora_a,
            lora_b=gated_lora_b,
        )  # [batch, classes]

        return {
            "shallow_feat": shallow_feat,
            "route_feat": route_feat,
            "patch": patch,
            "route_weight": route_weight,
            "prototype_ids": prototype_ids,
            "deep_feat": deep_feat,
            "gated_patch": gated_patch,
            "logits": logits,
        }
