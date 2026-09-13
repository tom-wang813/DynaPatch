"""Last-layer repair baselines used for safety comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ResidualLinear(nn.Module):
    """Frozen base linear layer plus a zero-initialized trainable residual."""

    def __init__(self, base_linear: nn.Linear) -> None:
        super().__init__()
        self.base = base_linear
        self.delta = nn.Linear(base_linear.in_features, base_linear.out_features, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta(x)


@dataclass
class BaselineModelBundle:
    model: nn.Module
    trainable_parameters: list[nn.Parameter]


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _resnet_classifier(model: nn.Module) -> nn.Module:
    return model.fc


def _set_resnet_classifier(model: nn.Module, classifier: nn.Module) -> None:
    model.fc = classifier


def _densenet_classifier(model: nn.Module) -> nn.Module:
    return model.classifier


def _set_densenet_classifier(model: nn.Module, classifier: nn.Module) -> None:
    model.classifier = classifier


def _vgg_classifier(model: nn.Module) -> nn.Module:
    return model.classifier[-1]


def _set_vgg_classifier(model: nn.Module, classifier: nn.Module) -> None:
    layers = list(model.classifier.children())
    layers[-1] = classifier
    model.classifier = nn.Sequential(*layers)


def _convnext_classifier(model: nn.Module) -> nn.Module:
    return model.classifier[-1]


def _set_convnext_classifier(model: nn.Module, classifier: nn.Module) -> None:
    layers = list(model.classifier.children())
    layers[-1] = classifier
    model.classifier = nn.Sequential(*layers)


def _mlp_classifier(model: nn.Module) -> nn.Module:
    return model.classifier


def _set_mlp_classifier(model: nn.Module, classifier: nn.Module) -> None:
    model.classifier = classifier


def _classifier_getter_setter(architecture: str) -> tuple[callable, callable]:
    normalized = architecture.lower()
    if normalized == "resnet50":
        return _resnet_classifier, _set_resnet_classifier
    if normalized == "densenet121":
        return _densenet_classifier, _set_densenet_classifier
    if normalized == "vgg16":
        return _vgg_classifier, _set_vgg_classifier
    if normalized == "convnext":
        return _convnext_classifier, _set_convnext_classifier
    if normalized == "mlp":
        return _mlp_classifier, _set_mlp_classifier
    raise ValueError(f"Unsupported architecture for last-layer baseline: {architecture}")


def get_classifier_module(model: nn.Module, architecture: str) -> nn.Module:
    getter, _ = _classifier_getter_setter(architecture)
    return getter(model)


def set_classifier_module(model: nn.Module, architecture: str, classifier: nn.Module) -> None:
    _, setter = _classifier_getter_setter(architecture)
    setter(model, classifier)


def configure_baseline_model(model: nn.Module, architecture: str, mode: str) -> BaselineModelBundle:
    """Freeze the backbone and expose the requested last-layer baseline variant."""
    if mode in {"full_finetune", "full_finetune_safety", "full_finetune_distr"}:
        for param in model.parameters():
            param.requires_grad = True
        return BaselineModelBundle(model=model, trainable_parameters=list(model.parameters()))

    _freeze_all(model)
    get_classifier, set_classifier = _classifier_getter_setter(architecture)
    classifier = get_classifier(model)

    if mode in {"head_only", "head_only_safety"}:
        for param in classifier.parameters():
            param.requires_grad = True
        return BaselineModelBundle(model=model, trainable_parameters=list(classifier.parameters()))

    if mode == "last_layer_delta":
        if not isinstance(classifier, nn.Linear):
            raise TypeError(f"Expected a linear classifier for {architecture}, got: {type(classifier)!r}")
        residual_classifier = ResidualLinear(classifier)
        set_classifier(model, residual_classifier)
        return BaselineModelBundle(model=model, trainable_parameters=list(residual_classifier.delta.parameters()))

    raise ValueError(f"Unsupported head-repair baseline mode: {mode}")


class HeadOnlyFineTuneBaseline(nn.Module):
    """Semantic alias for the standard head-only fine-tuning baseline."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class HeadOnlySafetyBaseline(nn.Module):
    """Semantic alias for the safety-aware head-only fine-tuning baseline."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class LastLayerDeltaBaseline(nn.Module):
    """Semantic alias for a frozen backbone with residual last-layer delta."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
