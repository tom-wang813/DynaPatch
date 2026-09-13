"""Stage-3 experiment assembly helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data import build_repair_dataloaders
from src.experiment.fixed_adv_cache import build_fixed_adv_loader
from src.models.backbones.factory import build_backbone, load_backbone_checkpoint
from src.models.dynapatch.factory import (
    build_dynapatch_decomposition,
    build_memory_bank,
    build_patch_operator,
    patch_dim_from_decomposition,
    build_router,
    context_dim_from_memory_bank,
)
from src.models.dynapatch.hypernet import ConstantPatchGenerator, HyperNetworkPatchGenerator
from src.models.dynapatch.vit_dynamic_lora import DynamicLoRAHeadGenerator, StaticLoRAHeadGenerator
from src.models.dynapatch.vit_dynamic_lora_model import ViTDynamicLoRAModel
from src.models.dynapatch.model import DynaPatchModel
from src.models.sync import PropertyPatchModel
from src.training.losses import RepairClassificationLoss
from src.utils.label_groups import build_class_group_matrix_for_dataset


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


def _fixed_adv_cache_path(
    cfg: DictConfig,
    fixed_adv_cfg: DictConfig,
    split_name: str,
    loader: torch.utils.data.DataLoader,
) -> Path | None:
    cache_dir = fixed_adv_cfg.get("cache_dir")
    if cache_dir is None:
        cache_dir = "outputs/cache/fixed_adv"

    experiment_id = str(cfg.experiment.id if "experiment" in cfg and "id" in cfg.experiment else cfg.get("name", "experiment"))
    cache_namespace = str(fixed_adv_cfg.get("cache_namespace", experiment_id))
    checkpoint_name = Path(str(cfg.model.checkpoint_path)).stem
    dataset_name = str(cfg.dataset.name)
    signature = "|".join(
        [
            cache_namespace,
            dataset_name,
            checkpoint_name,
            split_name,
            str(len(loader.dataset)),
            str(float(fixed_adv_cfg.get("epsilon", cfg.fault.epsilon))),
            str(int(fixed_adv_cfg.get("pgd_rounds", 0))),
            str(int(fixed_adv_cfg.get("pgd_steps", cfg.fault.steps))),
            str(float(fixed_adv_cfg.get("pgd_step_size", 0.0))),
            str(int(fixed_adv_cfg.get("fgsm_samples", 0))),
            str(float(fixed_adv_cfg.get("fgsm_step_size", 0.0))),
            str(bool(fixed_adv_cfg.get(f"include_clean_{'train' if split_name == 'bug_train' else 'eval'}", False))),
        ]
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    filename = f"{split_name}_{digest}.pt"
    return Path(str(cache_dir)) / cache_namespace / filename


def _maybe_wrap_fixed_adv_cache(
    cfg: DictConfig,
    backbone: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, torch.utils.data.DataLoader]:
    """Optionally replace bug loaders with fixed adversarial caches."""
    fixed_adv_cfg = cfg.data.get("fixed_adv_cache")
    if fixed_adv_cfg is None or not bool(fixed_adv_cfg.get("enabled", False)):
        return dataloaders

    wrapped = dict(dataloaders)
    wrapped["bug_train_clean"] = dataloaders["bug_train"]
    wrapped["bug_eval_clean"] = dataloaders["bug_eval"]

    epsilon = float(fixed_adv_cfg.get("epsilon", cfg.fault.epsilon))
    pgd_rounds = int(fixed_adv_cfg.get("pgd_rounds", 0))
    pgd_steps = int(fixed_adv_cfg.get("pgd_steps", cfg.fault.steps))
    pgd_step_size = float(fixed_adv_cfg.get("pgd_step_size", epsilon / max(pgd_steps, 1)))
    fgsm_samples = int(fixed_adv_cfg.get("fgsm_samples", 0))
    fgsm_step_size = float(fixed_adv_cfg.get("fgsm_step_size", epsilon))
    include_clean_train = bool(fixed_adv_cfg.get("include_clean_train", True))
    include_clean_eval = bool(fixed_adv_cfg.get("include_clean_eval", False))

    wrapped["bug_train"] = build_fixed_adv_loader(
        loader=wrapped["bug_train_clean"],
        model=backbone,
        device=device,
        batch_size=int(cfg.train_loop.batch_size),
        shuffle=True,
        epsilon=epsilon,
        pgd_rounds=pgd_rounds,
        pgd_steps=pgd_steps,
        pgd_step_size=pgd_step_size,
        fgsm_samples=fgsm_samples,
        fgsm_step_size=fgsm_step_size,
        include_clean=include_clean_train,
        cache_path=_fixed_adv_cache_path(cfg, fixed_adv_cfg, "bug_train", wrapped["bug_train_clean"]),
    )
    wrapped["bug_eval"] = build_fixed_adv_loader(
        loader=wrapped["bug_eval_clean"],
        model=backbone,
        device=device,
        batch_size=int(cfg.evaluation.get("batch_size", cfg.train_loop.batch_size)),
        shuffle=False,
        epsilon=epsilon,
        pgd_rounds=pgd_rounds,
        pgd_steps=pgd_steps,
        pgd_step_size=pgd_step_size,
        fgsm_samples=fgsm_samples,
        fgsm_step_size=fgsm_step_size,
        include_clean=include_clean_eval,
        cache_path=_fixed_adv_cache_path(cfg, fixed_adv_cfg, "bug_eval", wrapped["bug_eval_clean"]),
    )
    return wrapped


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
    dataloaders = _maybe_wrap_fixed_adv_cache(cfg, backbone, dataloaders, device)
    method_name = str(cfg.method.get("name", "hypernet_only"))
    if method_name == "property_patch":
        model = PropertyPatchModel(
            base_model=backbone,
            model_type=str(cfg.model.type),
            feature_dim=int(cfg.model.out_dim),
            num_experts=int(cfg.repair.support_set_k),
            patch_hidden_dim=int(cfg.model.get("property_patch_hidden_dim", 0)),
            patch_num_hidden_layers=int(cfg.model.get("property_patch_num_hidden_layers", 1)),
            indicator_radius=float(cfg.repair.get("property_patch_radius", cfg.repair.tau_dist)),
            exact_indicator=bool(cfg.method.get("property_patch_exact_indicator", False)),
            input_shape=(
                (
                    int(cfg.dataset.get("input_dim"))
                    if str(cfg.model.type) == "MLP"
                    else int(cfg.dataset.input_channels)
                ),
                1 if str(cfg.model.type) == "MLP" else int(cfg.dataset.input_resolution),
                1 if str(cfg.model.type) == "MLP" else int(cfg.dataset.input_resolution),
            ),
        ).to(device)
        support_inputs = []
        support_loader = dataloaders.get("bug_train_clean", dataloaders["bug_train"])
        for inputs, _labels in support_loader:
            support_inputs.append(inputs.to(device))
        if support_inputs:
            model.seed_support(torch.cat(support_inputs, dim=0))
        loss_fn = RepairClassificationLoss()
        return backbone, model, dataloaders, loss_fn

    if method_name in {"vit_dynamic_lora", "vit_static_lora", "vit_prototype_dynamic_lora", "vit_group_dynamic_lora"}:
        if str(cfg.model.type) != "ViT":
            raise ValueError(f"`{method_name}` is only supported when model.type == 'ViT'.")
        decomposition = build_dynapatch_decomposition(backbone, cfg)
        router = build_router(cfg, device)
        prototype_bank = build_memory_bank(cfg, device)
        classifier = decomposition.classifier
        if hasattr(classifier, "head") and hasattr(classifier.head, "out_features"):
            num_classes = int(classifier.head.out_features)
        elif hasattr(classifier, "__getitem__") and hasattr(classifier[-1], "out_features"):
            num_classes = int(classifier[-1].out_features)
        else:
            raise ValueError("Unable to infer ViT classifier output dimension for dynamic LoRA.")
        class_group_matrix = None
        if method_name == "vit_group_dynamic_lora":
            class_group_matrix, _group_names, _class_to_group = build_class_group_matrix_for_dataset(str(cfg.dataset.name))
        if method_name in {"vit_dynamic_lora", "vit_prototype_dynamic_lora", "vit_group_dynamic_lora"}:
            context_dim = 0
            if method_name == "vit_prototype_dynamic_lora":
                context_dim += int(cfg.model.shallow_dim)
            if method_name == "vit_group_dynamic_lora":
                context_dim += int(class_group_matrix.size(1))
            if method_name == "vit_prototype_dynamic_lora" and prototype_bank is None:
                raise ValueError("`vit_prototype_dynamic_lora` requires `conditioning=prototype_bank`.")
            lora_generator = DynamicLoRAHeadGenerator(
                hidden_dim=int(cfg.model.shallow_dim),
                out_dim=num_classes,
                rank=int(cfg.model.get("lora_rank", 8)),
                attention_rank=int(cfg.model.get("lora_attention_rank", 0)),
                mlp_hidden_dim=int(cfg.model.get("lora_hidden_dim", 256)),
                num_hidden_layers=int(cfg.model.get("lora_num_hidden_layers", 1)),
                pool=str(cfg.model.get("lora_pool", "cls")),
                context_dim=context_dim,
            ).to(device)
        else:
            lora_generator = StaticLoRAHeadGenerator(
                hidden_dim=int(cfg.model.shallow_dim),
                out_dim=num_classes,
                rank=int(cfg.model.get("lora_rank", 8)),
                attention_rank=int(cfg.model.get("lora_attention_rank", 0)),
                init_scale=float(cfg.model.get("lora_init_scale", 1.0e-3)),
            ).to(device)
        model = ViTDynamicLoRAModel(
            decomposition=decomposition,
            router=router,
            lora_generator=lora_generator,
            prototype_bank=prototype_bank,
            route_mode=str(cfg.method.get("route_mode", "distance")),
            class_group_matrix=class_group_matrix,
        ).to(device)
        loss_fn = RepairClassificationLoss()
        return backbone, model, dataloaders, loss_fn

    decomposition = build_dynapatch_decomposition(backbone, cfg)
    router = build_router(cfg, device)
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
        router=router,
        hypernet=hypernet,
        patch_operator=patch_operator,
        prototype_bank=prototype_bank,
        async_mode=str(cfg.method.async_mode),
        route_mode=str(cfg.method.get("route_mode", "distance")),
    ).to(device)

    loss_fn = RepairClassificationLoss()
    return backbone, model, dataloaders, loss_fn


def load_repair_checkpoint(model: DynaPatchModel, checkpoint_path: str | Path) -> dict:
    """Load a stage-3 repair checkpoint into the assembled model."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint
