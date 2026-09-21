"""Factory helpers for pluggable DynaPatch components."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.dynapatch.decomposition import build_decomposition
from src.models.dynapatch.patch_operator import ResidualPatchOperator


def build_dynapatch_decomposition(base_model: nn.Module, cfg: DictConfig) -> Any:
    """Build the requested backbone decomposition / insertion view."""
    decomposition_name = str(cfg.repair.get("decomposition", "backbone_split"))
    if decomposition_name != "backbone_split":
        raise ValueError(f"Unsupported decomposition type: {decomposition_name}")
    return build_decomposition(
        base_model=base_model,
        model_type=str(cfg.model.type),
        split_depth=int(cfg.repair.split_depth) if cfg.repair.get("split_depth") is not None else None,
        split_layer=cfg.repair.get("split_layer"),
        insertion_point=str(cfg.repair.get("insertion_point", "cross_layer")),
        # Defaults to "deep_feat", i.e. every pre-2026-07-31 run reproduces unchanged.
        patch_site=str(cfg.repair.get("patch_site", "deep_feat")),
    )


def build_memory_bank(cfg: DictConfig, device: torch.device) -> nn.Module | None:
    """Build the configured conditioning / repair memory bank.

    Always `None` for every shipped config: `conditioning.enabled` is never set. The
    `prototype_bank`/`confusion_pair_bank` backends this used to dispatch to were removed after
    confirming that (see the `dynapatch-deploy-policy-promotion-quirk` memory note) -- this still
    raises loudly rather than silently no-op-ing if `conditioning.enabled: true` is ever set again.
    """
    conditioning_cfg = cfg.get("conditioning")
    if conditioning_cfg is None or not bool(conditioning_cfg.get("enabled", False)):
        return None
    raise ValueError(
        "conditioning.enabled=true is no longer supported -- the prototype_bank/"
        "confusion_pair_bank backends were removed as unreachable dead code."
    )


def context_dim_from_memory_bank(cfg: DictConfig, memory_bank: nn.Module | None) -> int:
    """Return the extra context width injected into the hypernetwork."""
    if memory_bank is None:
        return 0
    return int(cfg.model.shallow_dim)


def patch_dim_from_decomposition(cfg: DictConfig, decomposition: Any) -> int:
    """Return the patch width required by the chosen insertion point."""
    patch_dim = getattr(decomposition, "patch_dim", None)
    if patch_dim is not None:
        return int(patch_dim)
    return int(cfg.model.out_dim)


def build_patch_operator(cfg: DictConfig, device: torch.device) -> nn.Module:
    """Build the configured patch application operator."""
    operator_name = str(cfg.method.get("patch_operator", "residual"))
    if operator_name != "residual":
        raise ValueError(f"Unsupported patch operator: {operator_name}")
    epsilon_max_cfg = cfg.repair.get("epsilon_max")
    epsilon_max = None if epsilon_max_cfg is None else float(epsilon_max_cfg)
    # `repair.budget` defaults to l_inf, so omitting it reproduces every pre-2026-07-30 run exactly.
    budget = str(cfg.repair.get("budget", "l_inf"))
    l2_max_cfg = cfg.repair.get("l2_max")
    l2_max = None if l2_max_cfg is None else float(l2_max_cfg)
    return ResidualPatchOperator(
        epsilon_max=epsilon_max, budget=budget, l2_max=l2_max,
        contract_tau=float(cfg.repair.get("contract_tau", 1e-3)),
        contract_scope=str(cfg.repair.get("contract_scope", "all")),
    ).to(device)


@torch.no_grad()
def seed_memory_bank(
    memory_bank: nn.Module | None,
    *,
    route_feat: torch.Tensor,
    labels: torch.Tensor | None = None,
    predictions: torch.Tensor | None = None,
) -> None:
    """Seed a memory bank if it exposes a recognized support-seeding API."""
    if memory_bank is None:
        return
    if hasattr(memory_bank, "seed_from_features"):
        memory_bank.seed_from_features(route_feat)
        return
    if hasattr(memory_bank, "seed_from_support"):
        if labels is None or predictions is None:
            raise ValueError("Support-aware memory banks require labels and predictions for seeding.")
        memory_bank.seed_from_support(route_feat, labels, predictions)
        return
    raise ValueError(f"Unsupported memory bank type: {type(memory_bank).__name__}")
