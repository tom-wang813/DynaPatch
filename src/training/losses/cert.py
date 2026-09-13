"""Certification-style guard losses for clean samples."""

from __future__ import annotations

import torch
import torch.nn as nn


def _linear_interval_margin(
    base_logits: torch.Tensor,
    logit_radius: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return lower/upper logits and certified margin for a linear classifier."""
    lower_logits = base_logits - logit_radius  # [batch, num_classes]
    upper_logits = base_logits + logit_radius  # [batch, num_classes]

    true_lower = lower_logits.gather(1, labels.view(-1, 1)).squeeze(1)  # [batch]
    masked_upper = upper_logits.clone()
    masked_upper.scatter_(1, labels.view(-1, 1), float("-inf"))
    other_upper = masked_upper.max(dim=1).values  # [batch]
    cert_margin = true_lower - other_upper  # [batch]
    return lower_logits, upper_logits, cert_margin


def _resolve_classifier_for_interval(
    classifier: nn.Module,
) -> tuple[callable, nn.Linear]:
    """Return a feature preprocessor and the final linear head used for interval bounds.

    Supported forms:
    - ``nn.Linear``
    - ``nn.Sequential(..., nn.Linear)`` where preceding modules are deterministic feature
      preprocessors such as ``LayerNorm2d`` and ``Flatten`` used by ConvNeXt.
    """
    if isinstance(classifier, nn.Linear):
        return (lambda x: x), classifier
    if isinstance(classifier, nn.Sequential) and len(classifier) >= 1 and isinstance(classifier[-1], nn.Linear):
        feature_pre = nn.Sequential(*list(classifier.children())[:-1])
        linear = classifier[-1]
        return feature_pre, linear
    raise NotImplementedError(
        "Certification losses currently support `nn.Linear` or "
        "`nn.Sequential(..., nn.Linear)` final classifiers."
    )


class CertificationGuardLoss(nn.Module):
    """Legacy static guard loss using only the configured patch radius."""

    def __init__(self, epsilon_max: float, margin: float = 0.0) -> None:
        super().__init__()
        self.epsilon_max = float(epsilon_max)
        self.margin = float(margin)

    def forward(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute clean-data certification guard loss with a static last-layer radius."""
        outputs = model.forward_with_intermediates(inputs)
        deep_feat = outputs["deep_feat"]  # [batch, d]
        route_weight = outputs["route_weight"]  # [batch, 1]
        classifier = model.decomposition.classifier
        feature_pre, linear = _resolve_classifier_for_interval(classifier)

        deep_pre = feature_pre(deep_feat)
        deep_flat = torch.flatten(deep_pre, 1)  # [batch, d]
        base_logits = linear(deep_flat)  # [batch, num_classes]

        patch_radius = route_weight * self.epsilon_max  # [batch, 1]
        abs_weight = linear.weight.abs()  # [num_classes, d]
        logit_radius = patch_radius * abs_weight.sum(dim=1).unsqueeze(0)  # [batch, num_classes]

        _lower_logits, _upper_logits, cert_margin = _linear_interval_margin(
            base_logits=base_logits,
            logit_radius=logit_radius,
            labels=labels,
        )
        cert_loss = torch.clamp(self.margin - cert_margin, min=0.0).mean()

        stats = {
            "clean_route_hit": float(route_weight.mean().item()),
            "cert_margin": float(cert_margin.mean().item()),
        }
        return cert_loss, stats


class DynamicIBPCertificationLoss(nn.Module):
    """Paper-aligned dynamic cert loss using clean-triggered patch intervals."""

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Use the clean-triggered routed patch to build dynamic linear interval bounds."""
        outputs = model.forward_with_intermediates(inputs)
        deep_feat = outputs["deep_feat"]  # [batch, d]
        gated_patch = outputs["gated_patch"]  # [batch, d]
        route_weight = outputs["route_weight"]  # [batch, 1]
        classifier = model.decomposition.classifier
        feature_pre, linear = _resolve_classifier_for_interval(classifier)

        deep_pre = feature_pre(deep_feat)
        if deep_feat.dim() == 4:
            patch_view = gated_patch.view(gated_patch.size(0), gated_patch.size(1), 1, 1)
            patched_pre = feature_pre(deep_feat + patch_view)
        else:
            patched_pre = feature_pre(deep_feat + gated_patch)
        deep_flat = torch.flatten(deep_pre, 1)  # [batch, d]
        patch_flat = torch.flatten(patched_pre - deep_pre, 1)  # [batch, d]
        base_logits = linear(deep_flat)  # [batch, num_classes]

        # Dynamic IBP-style interval: each clean-triggered patch component defines its own
        # uncertainty radius instead of collapsing everything to a single epsilon_max scalar.
        abs_weight = linear.weight.abs()  # [num_classes, d]
        patch_radius = patch_flat.abs()  # [batch, d]
        logit_radius = patch_radius @ abs_weight.t()  # [batch, num_classes]

        _lower_logits, _upper_logits, cert_margin = _linear_interval_margin(
            base_logits=base_logits,
            logit_radius=logit_radius,
            labels=labels,
        )
        cert_loss = torch.clamp(self.margin - cert_margin, min=0.0).mean()

        stats = {
            "clean_route_hit": float(route_weight.mean().item()),
            "cert_margin": float(cert_margin.mean().item()),
            "dynamic_patch_l1": float(patch_radius.mean().item()),
        }
        return cert_loss, stats


class BugSideDynamicIBPCertificationLoss(nn.Module):
    """Bug-side last-layer dynamic interval loss around patched logits.

    This is the most formal variant currently supported by the frozen-backbone setup:
    it certifies a dynamic interval around the patched last-layer representation for bug
    samples, rather than a full input-space certificate through the entire backbone.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = model.forward_with_intermediates(inputs)
        deep_feat = outputs["deep_feat"]  # [batch, d]
        gated_patch = outputs["gated_patch"]  # [batch, d]
        route_weight = outputs["route_weight"]  # [batch, 1]
        patched_logits = outputs["logits"]  # [batch, num_classes]
        classifier = model.decomposition.classifier
        feature_pre, linear = _resolve_classifier_for_interval(classifier)

        deep_pre = feature_pre(deep_feat)
        if deep_feat.dim() == 4:
            patch_view = gated_patch.view(gated_patch.size(0), gated_patch.size(1), 1, 1)
            patched_pre = feature_pre(deep_feat + patch_view)
        else:
            patched_pre = feature_pre(deep_feat + gated_patch)
        patch_flat = torch.flatten(patched_pre - deep_pre, 1)  # [batch, d]
        abs_weight = linear.weight.abs()  # [num_classes, d]
        patch_radius = patch_flat.abs()  # [batch, d]
        logit_radius = patch_radius @ abs_weight.t()  # [batch, num_classes]

        _lower_logits, _upper_logits, cert_margin = _linear_interval_margin(
            base_logits=patched_logits,
            logit_radius=logit_radius,
            labels=labels,
        )
        cert_loss = torch.clamp(self.margin - cert_margin, min=0.0).mean()

        stats = {
            "clean_route_hit": float(route_weight.mean().item()),
            "cert_margin": float(cert_margin.mean().item()),
            "dynamic_patch_l1": float(patch_radius.mean().item()),
        }
        return cert_loss, stats
