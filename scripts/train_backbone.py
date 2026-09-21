"""Train a backbone classifier before repair experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datasets import build_classification_dataloaders
from src.experiment.runner import ExperimentRunner
from src.models.backbones.factory import build_backbone


def _validate_backbone_config(cfg: DictConfig) -> None:
    """Validate the backbone training config."""
    if "experiment" not in cfg or cfg.experiment.stage != "backbone_finetune":
        raise ValueError("`scripts/train_backbone.py` requires `experiment.stage=backbone_finetune`.")


def _move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one classification batch onto the requested device."""
    inputs, labels = batch
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def _run_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    """Run one train or eval epoch."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)  # [batch]
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += float((predictions == labels).float().sum().item())
        total_samples += batch_size

    normalizer = max(total_samples, 1)
    return {
        "loss": total_loss / normalizer,
        "accuracy": total_correct / normalizer,
        "num_samples": float(total_samples),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a plain classifier backbone for downstream bug-set construction."""
    _validate_backbone_config(cfg)
    # The `train=` Hydra group (configs/train/*.yaml) composes into cfg.train, but
    # build_classification_dataloaders and the optimizer setup below share code with the
    # DynaPatch/baseline pipelines, which read hyperparameters from a `train_loop` section
    # (a distinct key in their resolved configs/shuffled_split_source/*.yaml files). Alias it here so both
    # invocation styles resolve to the same hyperparameters without duplicating them.
    # Same reason: build_classification_dataloaders also expects an `evaluation` section
    # (present in the resolved DynaPatch/baseline configs); backbone training has no separate
    # eval-time hyperparameters, so default it to empty and let it fall back to train batch size.
    with open_dict(cfg):
        cfg.train_loop = cfg.train
        if "evaluation" not in cfg:
            cfg.evaluation = {}
    runner = ExperimentRunner(cfg)
    record = runner.bootstrap()

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_backbone(
        architecture=str(cfg.model.architecture),
        num_classes=int(cfg.dataset.num_classes),
        pretrained_weights=cfg.model.get("pretrained_weights"),
        input_dim=cfg.dataset.get("input_dim"),
        hidden_dims=cfg.model.get("hidden_dims"),
        activation=str(cfg.model.get("activation", "relu")),
    ).to(device)

    dataloaders = build_classification_dataloaders(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train_loop.lr),
        weight_decay=float(cfg.train_loop.weight_decay),
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(int(cfg.train_loop.epochs)):
        train_metrics = _run_epoch(model, dataloaders["train"], optimizer, device)
        eval_metrics = _run_epoch(model, dataloaders["eval"], None, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "eval_loss": eval_metrics["loss"],
                "eval_acc": eval_metrics["accuracy"],
            }
        )
        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f}"
        )

    checkpoint_dir = Path(record.artifact_root) / "checkpoints"
    checkpoint_path = checkpoint_dir / "backbone_last.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    metrics = {
        "status": "completed",
        "experiment_id": cfg.experiment.id,
        "stage": cfg.experiment.stage,
        "device": str(device),
        "artifact_root": record.artifact_root,
        "checkpoint_path": str(checkpoint_path),
        "resolved_config_preview": OmegaConf.to_container(cfg.experiment, resolve=True),
        "history": history,
        "final": history[-1] if history else {},
    }
    runner.save_metrics(metrics)

    print(f"Completed backbone finetune run at: {record.artifact_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {record.metrics_path}")


if __name__ == "__main__":
    main()
