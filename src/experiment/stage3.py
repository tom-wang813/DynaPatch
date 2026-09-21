"""Stage-3 experiment assembly helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data import build_repair_dataloaders
from src.models.backbones.factory import build_backbone, load_backbone_checkpoint
from src.models.dynapatch.factory import (
    build_dynapatch_decomposition,
    build_memory_bank,
    build_patch_operator,
    patch_dim_from_decomposition,
    context_dim_from_memory_bank,
)
from src.models.dynapatch.hypernet import ConstantPatchGenerator, HyperNetworkPatchGenerator
from src.models.dynapatch.model import DynaPatchModel
from src.training.losses import RepairClassificationLoss


def load_resolved_config(path: str | Path) -> DictConfig:
    """Load a resolved Hydra config from disk."""
    return OmegaConf.load(Path(path))


def resolve_runtime_device(cfg: DictConfig, device: str | None = None) -> torch.device:
    """Resolve the runtime device while respecting CUDA availability."""
    if device is not None:
        return torch.device(device)
    requested = str(cfg.runtime.device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("cuda") and bool(cfg.runtime.get("fail_on_cpu_fallback", False)):
        raise RuntimeError(
            f"Requested runtime.device={requested}, but CUDA is unavailable; refusing silent CPU fallback."
        )
    return torch.device("cpu")



def build_stage3_bundle(
    cfg: DictConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, torch.utils.data.DataLoader], RepairClassificationLoss]:
    """Build the frozen backbone, repair model, dataloaders, and loss."""
    backbone = build_backbone(
        architecture=str(cfg.model.architecture),
        num_classes=int(cfg.dataset.num_classes),
        pretrained_weights=cfg.model.get("pretrained_weights"),
        input_dim=cfg.dataset.get("input_dim"),
        hidden_dims=cfg.model.get("hidden_dims"),
        activation=str(cfg.model.get("activation", "relu")),
    ).to(device)
    backbone = load_backbone_checkpoint(backbone, str(cfg.model.checkpoint_path)).to(device)

    dataloaders = build_repair_dataloaders(cfg)
    method_name = str(cfg.method.get("name", "hypernet_only"))
    decomposition = build_dynapatch_decomposition(backbone, cfg)
    prototype_bank = build_memory_bank(cfg, device)

    _patch_dim = patch_dim_from_decomposition(cfg, decomposition)
    if method_name == "fixed_patch":
        hypernet = ConstantPatchGenerator(patch_dim=_patch_dim).to(device)
    else:
        hypernet = HyperNetworkPatchGenerator(
            model_type=str(cfg.model.type),
            shallow_dim=int(cfg.model.shallow_dim),
            out_dim=_patch_dim,
            context_dim=context_dim_from_memory_bank(cfg, prototype_bank),
            hidden_dim=int(cfg.model.get("hypernet_hidden_dim", 128)),
            num_hidden_layers=int(cfg.model.get("hypernet_num_hidden_layers", 1)),
            hypernet_style=str(cfg.model.get("hypernet_style", "plain")),
            pool_size=int(cfg.model.get("hypernet_pool_size", 1)),
            condition_source=str(cfg.model.get("hypernet_condition_source", "shallow")),
            num_basis=int(cfg.dataset.num_classes),
        ).to(device)
    patch_operator = build_patch_operator(cfg, device)
    model = DynaPatchModel(
        decomposition=decomposition,
        hypernet=hypernet,
        patch_operator=patch_operator,
        async_mode=str(cfg.method.async_mode),
    ).to(device)

    loss_fn = RepairClassificationLoss()
    return backbone, model, dataloaders, loss_fn


def load_repair_checkpoint(model: DynaPatchModel, checkpoint_path: str | Path) -> dict:
    """Load a stage-3 repair checkpoint into the assembled model."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint
