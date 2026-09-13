"""Deployment-time patch composition policies."""

from __future__ import annotations

from collections.abc import Callable

import torch

WeightedLookup = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float, int | None], torch.Tensor]


class DeploymentPolicy:
    """Base class for deployment-time patch composition."""

    name = "base"

    def compose(
        self,
        *,
        direct_patch: torch.Tensor,
        accept_mask: torch.Tensor,
        route_feat: torch.Tensor,
        support_route_feat: torch.Tensor,
        support_patches: torch.Tensor,
        weighted_lookup: WeightedLookup,
        weighted_temperature: float,
        weighted_top_k: int | None,
    ) -> torch.Tensor:
        raise NotImplementedError


class DirectGeneralizationPolicy(DeploymentPolicy):
    """Always trust the direct hypernet patch."""

    name = "direct_generalization"

    def compose(self, **kwargs) -> torch.Tensor:
        return kwargs["direct_patch"]


class DistanceWeightedPolicy(DeploymentPolicy):
    """Use direct patches on accepted states and retrieval-weighted patches otherwise."""

    name = "distance_weighted"

    def compose(self, **kwargs) -> torch.Tensor:
        direct_patch = kwargs["direct_patch"]
        accept_mask = kwargs["accept_mask"]
        weighted_patch = kwargs["weighted_lookup"](
            kwargs["route_feat"],
            kwargs["support_route_feat"],
            kwargs["support_patches"],
            kwargs["weighted_temperature"],
            kwargs["weighted_top_k"],
        )
        fallback_mask = ~accept_mask
        return torch.where(fallback_mask.view(-1, 1), weighted_patch, direct_patch)


class GatedDirectPolicy(DeploymentPolicy):
    """Zero out direct patches that fail router acceptance."""

    name = "gated_direct"

    def compose(self, **kwargs) -> torch.Tensor:
        direct_patch = kwargs["direct_patch"]
        accept_mask = kwargs["accept_mask"]
        return torch.where(accept_mask.view(-1, 1), direct_patch, torch.zeros_like(direct_patch))


class GatedWeightedPolicy(DeploymentPolicy):
    """Use direct patches when accepted and retrieval-weighted patches otherwise."""

    name = "gated_weighted"

    def compose(self, **kwargs) -> torch.Tensor:
        direct_patch = kwargs["direct_patch"]
        accept_mask = kwargs["accept_mask"]
        weighted_patch = kwargs["weighted_lookup"](
            kwargs["route_feat"],
            kwargs["support_route_feat"],
            kwargs["support_patches"],
            kwargs["weighted_temperature"],
            kwargs["weighted_top_k"],
        )
        return torch.where(accept_mask.view(-1, 1), direct_patch, weighted_patch)


class BackboneOnlyPolicy(DeploymentPolicy):
    """Disable patching entirely and return the frozen backbone prediction."""

    name = "backbone_only"

    def compose(self, **kwargs) -> torch.Tensor:
        direct_patch = kwargs["direct_patch"]
        return torch.zeros_like(direct_patch)


POLICIES: dict[str, type[DeploymentPolicy]] = {
    policy.name: policy
    for policy in (
        DirectGeneralizationPolicy,
        DistanceWeightedPolicy,
        GatedDirectPolicy,
        GatedWeightedPolicy,
        BackboneOnlyPolicy,
    )
}


def build_deployment_policy(name: str) -> DeploymentPolicy:
    """Instantiate a deployment policy by name."""
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported deployment policy: {name}") from exc
