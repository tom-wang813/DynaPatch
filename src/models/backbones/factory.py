"""Backbone model construction utilities."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from src.models.backbones.mlp import MLPBackbone


def _resolve_torchvision_weights(architecture: str, pretrained_weights: str | None):
    """Resolve a torchvision weights enum from a config string."""
    if pretrained_weights is None:
        return None

    try:
        import torchvision.models as tv_models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required to resolve pretrained backbone weights."
        ) from exc

    weights_key = pretrained_weights.upper()
    weight_enums = {
        "resnet50": tv_models.ResNet50_Weights,
        "densenet121": tv_models.DenseNet121_Weights,
        "vgg16": tv_models.VGG16_Weights,
        "vit": tv_models.ViT_B_16_Weights,
        "swin": tv_models.Swin_T_Weights,
        "convnext": tv_models.ConvNeXt_Tiny_Weights,
    }
    if architecture not in weight_enums:
        raise ValueError(f"Unsupported pretrained backbone architecture: {architecture}")

    try:
        return getattr(weight_enums[architecture], weights_key)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported pretrained weight spec `{pretrained_weights}` for `{architecture}`."
        ) from exc


def build_backbone(
    architecture: str,
    num_classes: int | None = None,
    pretrained_weights: str | None = None,
    input_dim: int | None = None,
    hidden_dims: list[int] | tuple[int, ...] | None = None,
    activation: str = "relu",
) -> nn.Module:
    """Instantiate a torchvision backbone by architecture name."""
    if architecture == "mlp":
        if num_classes is None:
            raise ValueError("MLP backbone requires `num_classes`.")
        if input_dim is None:
            raise ValueError("MLP backbone requires `input_dim`.")
        return MLPBackbone(
            input_dim=int(input_dim),
            num_classes=int(num_classes),
            hidden_dims=(64, 64, 64) if hidden_dims is None else hidden_dims,
            activation=activation,
        )

    if pretrained_weights is not None:
        cache_root = Path(os.environ.get("TORCH_HOME", Path.cwd() / "artifacts" / "torch_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_root)

    try:
        import torchvision.models as tv_models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required to build research backbones. "
            "Install project dependencies before running stage training scripts."
        ) from exc

    weights = _resolve_torchvision_weights(architecture, pretrained_weights)
    factories = {
        "resnet50": lambda: tv_models.resnet50(weights=weights),
        "densenet121": lambda: tv_models.densenet121(weights=weights),
        "vgg16": lambda: tv_models.vgg16(weights=weights),
        "vit": lambda: tv_models.vit_b_16(weights=weights),
        "swin": lambda: tv_models.swin_t(weights=weights),
        "convnext": lambda: tv_models.convnext_tiny(weights=weights),
    }
    if architecture not in factories:
        raise ValueError(f"Unsupported backbone architecture: {architecture}")
    model = factories[architecture]()
    if num_classes is None:
        return model

    if architecture == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if architecture == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    if architecture == "vgg16":
        classifier_layers = list(model.classifier.children())
        in_features = model.classifier[-1].in_features
        classifier_layers[-1] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*classifier_layers)
        return model

    if architecture == "vit":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if architecture == "swin":
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model

    if architecture == "convnext":
        classifier_layers = list(model.classifier.children())
        in_features = model.classifier[-1].in_features
        classifier_layers[-1] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*classifier_layers)
        return model

    return model


def load_backbone_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load a backbone checkpoint from disk."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model
