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
    """Always trust the direct hypernet patch.

    Requested directly by every shipped `deployment.policy: direct_generalization`, but never
    actually reached in practice: `_resolve_policy_semantics` in deploy_eval.py promotes this
    request to `gated_direct` whenever `enforce_router_bypass` is true (the unconfigured
    default everywhere). Kept only because `build_deployment_policy` still needs a class to
    resolve the literal `direct_generalization` name if `enforce_router_bypass=false` were ever
    set explicitly.
    """

    name = "direct_generalization"

    def compose(self, **kwargs) -> torch.Tensor:
        return kwargs["direct_patch"]


class GatedDirectPolicy(DeploymentPolicy):
    """Zero out direct patches that fail router acceptance.

    This is the class every shipped deploy run actually instantiates (via the
    `direct_generalization` -> `gated_direct` promotion above). With every setting's
    `deployment.gate_threshold: 9999.0`, `accept_mask` is empirically always True (verified:
    0 rejections across gtsrb/resnet50's clean_eval/bug_eval/bug_train populations), so this
    reduces to the same "always apply the patch" behavior as `DirectGeneralizationPolicy` for
    every shipped setting -- the online router never actually gates anything. The real
    accept/reject decision behind the paper's "DynaPatch (gated)" vs "-NoGate" numbers is the
    offline 9-feature logistic-regression gate (`scripts/gate_protocol_b.py` and friends),
    applied post-hoc to these always-ungated predictions -- not this class.
    """

    name = "gated_direct"

    def compose(self, **kwargs) -> torch.Tensor:
        direct_patch = kwargs["direct_patch"]
        accept_mask = kwargs["accept_mask"]
        return torch.where(accept_mask.view(-1, 1), direct_patch, torch.zeros_like(direct_patch))


POLICIES: dict[str, type[DeploymentPolicy]] = {
    policy.name: policy
    for policy in (
        DirectGeneralizationPolicy,
        GatedDirectPolicy,
    )
}


def build_deployment_policy(name: str) -> DeploymentPolicy:
    """Instantiate a deployment policy by name."""
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported deployment policy: {name}") from exc
