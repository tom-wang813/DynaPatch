#!/usr/bin/env python3
"""Train last-layer repair baselines on existing bug/support manifests."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import configure_baseline_model  # noqa: E402
from src.data.factory import build_dataset, build_repair_dataloaders  # noqa: E402
from src.experiment.runner import ExperimentRunner  # noqa: E402
from src.models.backbones.factory import build_backbone, load_backbone_checkpoint  # noqa: E402
from src.training.losses.repair import RepairClassificationLoss, SafetyAwareRepairClassificationLoss  # noqa: E402


def _load_manifest(path_value: str) -> list[int]:
    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("indices", [])
    return [int(v) for v in payload]


def _resolve_clean_indices(cfg, *, support_indices: list[int], unseen_indices: list[int]) -> list[int]:
    """Resolve clean-eval indices, defaulting to the complement of bug splits."""
    clean_path = cfg.data.get("clean_eval_indices_path")
    if clean_path is not None:
        return _load_manifest(str(clean_path))

    test_dataset = build_dataset(cfg, train=False)
    excluded = set(int(v) for v in support_indices) | set(int(v) for v in unseen_indices)
    return [idx for idx in range(len(test_dataset)) if idx not in excluded]


def _move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = batch
    non_blocking = device.type == "cuda"
    return inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)


def _build_split_loader(cfg, indices: list[int], batch_size: int, *, shuffle: bool) -> DataLoader:
    test_dataset = build_dataset(cfg, train=False)
    workers = int(cfg.runtime.get("num_workers", 0))
    kwargs = {
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(Subset(test_dataset, indices), batch_size=batch_size, **kwargs)


def _clean_replay_consistency_loss(
    model: torch.nn.Module,
    base_reference: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        base_logits = base_reference(inputs)  # [batch, classes]
        base_pred = torch.argmax(base_logits, dim=1)  # [batch]
        keep = base_pred == labels  # [batch]
    if not bool(keep.any().item()):
        return torch.zeros((), device=inputs.device)
    patched_logits = model(inputs)  # [batch, classes]
    return F.kl_div(
        F.log_softmax(patched_logits[keep], dim=1),
        F.softmax(base_logits[keep], dim=1),
        reduction="batchmean",
    )


def _clean_replay_ce_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs)  # [batch, classes]
    return F.cross_entropy(logits, labels)


def _run_epoch(
    model: torch.nn.Module,
    base_reference: torch.nn.Module,
    bug_loader: DataLoader,
    clean_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_fn: torch.nn.Module,
    lambda_clean_replay: float,
    clean_replay_batch_limit: int | None,
    clean_replay_objective: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    base_reference.eval()

    clean_iter = cycle(clean_loader) if clean_loader is not None else None
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    clean_replay_total = 0.0
    batches = 0

    for batch in bug_loader:
        inputs, labels = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)  # [batch, classes]
        repair_loss = loss_fn(logits, labels)
        clean_replay_loss = torch.zeros((), device=device)
        if lambda_clean_replay > 0.0:
            if clean_iter is None:
                raise ValueError("clean_loader is required when lambda_clean_replay > 0.")
            clean_inputs, clean_labels = _move_batch(next(clean_iter), device)
            if clean_replay_batch_limit is not None and clean_replay_batch_limit > 0:
                clean_inputs = clean_inputs[:clean_replay_batch_limit]
                clean_labels = clean_labels[:clean_replay_batch_limit]
            if clean_replay_objective == "kl":
                clean_replay_loss = _clean_replay_consistency_loss(model, base_reference, clean_inputs, clean_labels)
            elif clean_replay_objective == "ce":
                clean_replay_loss = _clean_replay_ce_loss(model, clean_inputs, clean_labels)
            else:
                raise ValueError(f"Unsupported clean replay objective: {clean_replay_objective}")

        loss = repair_loss + float(lambda_clean_replay) * clean_replay_loss
        if is_train:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)  # [batch]
        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += float((predictions == labels).float().sum().item())
        total_samples += batch_size
        clean_replay_total += float(clean_replay_loss.item())
        batches += 1

    normalizer = max(total_samples, 1)
    return {
        "loss": total_loss / normalizer,
        "accuracy": total_correct / normalizer,
        "num_samples": float(total_samples),
        "clean_replay_loss": clean_replay_total / max(batches, 1),
    }


@torch.no_grad()
def _collect_prediction_rows(
    model: torch.nn.Module,
    base_reference: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> list[dict[str, object]]:
    model.eval()
    base_reference.eval()

    dataset = loader.dataset
    dataset_indices = list(dataset.indices) if isinstance(dataset, Subset) else list(range(len(dataset)))
    rows: list[dict[str, object]] = []
    cursor = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        base_logits = base_reference(inputs)  # [batch, classes]
        patched_logits = model(inputs)  # [batch, classes]
        base_probs = torch.softmax(base_logits, dim=1)  # [batch, classes]
        patched_probs = torch.softmax(patched_logits, dim=1)  # [batch, classes]
        base_pred = torch.argmax(base_logits, dim=1)  # [batch]
        patched_pred = torch.argmax(patched_logits, dim=1)  # [batch]

        batch_size = int(labels.size(0))
        batch_indices = dataset_indices[cursor : cursor + batch_size]
        cursor += batch_size
        arange = torch.arange(batch_size, device=device)
        base_conf = base_probs[arange, base_pred]  # [batch]
        patched_conf = patched_probs[arange, patched_pred]  # [batch]

        for idx in range(batch_size):
            label = int(labels[idx].item())
            base_prediction = int(base_pred[idx].item())
            patched_prediction = int(patched_pred[idx].item())
            rows.append(
                {
                    "split": split_name,
                    "dataset_index": int(batch_indices[idx]),
                    "label": label,
                    "base_pred": base_prediction,
                    "patched_pred": patched_prediction,
                    "base_conf": float(base_conf[idx].item()),
                    "patched_conf": float(patched_conf[idx].item()),
                    "patch_norm": 0.0,
                    "base_correct": base_prediction == label,
                    "patched_correct": patched_prediction == label,
                    "repaired": base_prediction != label and patched_prediction == label,
                    "regressed": base_prediction == label and patched_prediction != label,
                }
            )
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    total = len(rows)
    base_correct = sum(1 for row in rows if bool(row["base_correct"]))
    patched_correct = sum(1 for row in rows if bool(row["patched_correct"]))
    regressed = sum(1 for row in rows if bool(row["regressed"]))
    repaired = sum(1 for row in rows if bool(row["repaired"]))
    return {
        "count": total,
        "base_correct": base_correct,
        "patched_correct": patched_correct,
        "repaired": repaired,
        "regressed": regressed,
        "base_accuracy": base_correct / max(total, 1),
        "patched_accuracy": patched_correct / max(total, 1),
        "repaired_rate": repaired / max(total, 1),
        "regressed_rate": regressed / max(total, 1),
        "rr": regressed / max(base_correct, 1),
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Resolved stage-3 config or compatible Hydra config.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "head_only",
            "head_only_safety",
            "full_finetune",
            "full_finetune_safety",
        ),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--clean-replay-weight", type=float, default=0.0)
    parser.add_argument("--clean-replay-objective", choices=("kl", "ce"), default="kl")
    parser.add_argument("--clean-replay-subset-size", type=int, default=2048)
    parser.add_argument("--clean-replay-batch-size", type=int, default=None)
    parser.add_argument("--clean-replay-batch-limit", type=int, default=None)
    parser.add_argument("--lambda-safety-risk", type=float, default=0.0)
    parser.add_argument("--critical-class-weight", type=float, default=1.0)
    parser.add_argument("--safety-risk-matrix-path", default=None)
    parser.add_argument("--critical-indices", default=None, help="Comma-separated critical class ids.")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Dot-path overrides applied after loading config, e.g. data.bug_indices_path=... "
        "Needed to point a baseline at a different split; the resolved config carries its own "
        "bug_bank paths, so overriding only some of them silently mixes splits.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        for override in args.overrides:
            if "=" not in override:
                raise ValueError(f"Override must be key=value, got: {override!r}")
            key, value = override.split("=", 1)
            OmegaConf.update(cfg, key, value, merge=True)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    cfg.artifacts.root = str(args.output_root)
    cfg.experiment.stage = "baseline_head_repair"
    cfg.experiment.id = f"{cfg.experiment.id}_{args.mode}"
    if args.train_batch_size is not None:
        cfg.train_loop.batch_size = int(args.train_batch_size)
    if args.eval_batch_size is not None:
        cfg.evaluation.batch_size = int(args.eval_batch_size)
    if args.num_workers is not None:
        cfg.runtime.num_workers = int(args.num_workers)

    runner = ExperimentRunner(cfg)
    record = runner.bootstrap()

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    architecture = str(cfg.model.architecture)
    num_classes = int(cfg.dataset.num_classes)
    checkpoint_path = str(cfg.model.checkpoint_path)

    base_reference = build_backbone(
        architecture=architecture,
        num_classes=num_classes,
        pretrained_weights=cfg.model.get("pretrained_weights"),
        input_dim=cfg.dataset.get("input_dim"),
        hidden_dims=cfg.model.get("hidden_dims"),
        activation=str(cfg.model.get("activation", "relu")),
    )
    base_reference = load_backbone_checkpoint(base_reference, checkpoint_path).to(device)
    model = copy.deepcopy(base_reference).to(device)

    bundle = configure_baseline_model(model, architecture=architecture, mode=args.mode)
    model = bundle.model.to(device)
    trainable_parameters = bundle.trainable_parameters
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))

    dataloaders = build_repair_dataloaders(cfg)
    clean_train_loader = dataloaders["clean_train"]
    if args.clean_replay_subset_size is not None and args.clean_replay_subset_size > 0:
        generator = torch.Generator()
        generator.manual_seed(int(cfg.seed))
        perm = torch.randperm(len(clean_train_loader.dataset), generator=generator).tolist()
        subset = perm[: min(len(perm), int(args.clean_replay_subset_size))]
        workers = max(int(getattr(clean_train_loader, "num_workers", 0)), 0)
        loader_kwargs = {
            "shuffle": True,
            "num_workers": workers,
            "pin_memory": True,
        }
        if workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        clean_train_loader = DataLoader(
            Subset(clean_train_loader.dataset, subset),
            batch_size=int(args.clean_replay_batch_size or clean_train_loader.batch_size),
            **loader_kwargs,
        )

    critical_indices = []
    if args.critical_indices:
        critical_indices = [int(v) for v in str(args.critical_indices).split(",") if str(v).strip()]

    if args.mode in {"head_only_safety", "full_finetune_safety"}:
        if args.safety_risk_matrix_path is None:
            raise ValueError("--safety-risk-matrix-path is required for safety-aware baseline modes.")
        loss_fn = SafetyAwareRepairClassificationLoss(
            risk_matrix_path=str(args.safety_risk_matrix_path),
            lambda_safety_risk=float(args.lambda_safety_risk),
            critical_class_weight=float(args.critical_class_weight),
            critical_indices=critical_indices,
        )
    else:
        loss_fn = RepairClassificationLoss()

    history: list[dict[str, float | int]] = []
    for epoch in range(int(args.epochs)):
        train_metrics = _run_epoch(
            model=model,
            base_reference=base_reference,
            bug_loader=dataloaders["bug_train"],
            clean_loader=clean_train_loader if float(args.clean_replay_weight) > 0.0 else None,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            lambda_clean_replay=float(args.clean_replay_weight),
            clean_replay_batch_limit=args.clean_replay_batch_limit,
            clean_replay_objective=str(args.clean_replay_objective),
        )
        eval_metrics = _run_epoch(
            model=model,
            base_reference=base_reference,
            bug_loader=dataloaders["bug_eval"],
            clean_loader=None,
            optimizer=None,
            device=device,
            loss_fn=RepairClassificationLoss(),
            lambda_clean_replay=0.0,
            clean_replay_batch_limit=None,
            clean_replay_objective=str(args.clean_replay_objective),
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_clean_replay_loss": train_metrics["clean_replay_loss"],
                "eval_loss": eval_metrics["loss"],
                "eval_acc": eval_metrics["accuracy"],
            }
        )
        print(
            f"[Epoch {epoch + 1}] train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"clean_replay={train_metrics['clean_replay_loss']:.4f} eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f}",
            flush=True,
        )

    checkpoint_dir = Path(record.artifact_root) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path_out = checkpoint_dir / "baseline_last.pt"
    torch.save({"state_dict": model.state_dict(), "mode": args.mode}, checkpoint_path_out)

    support_indices = _load_manifest(str(cfg.data.bug_indices_path))
    # Publication-facing comparison should use the canonical unseen holdout
    # (`bug_eval_indices_path`) rather than the smaller bug-validation split.
    unseen_path = cfg.data.get("bug_eval_indices_path") or cfg.data.get("bug_val_indices_path")
    unseen_indices = _load_manifest(str(unseen_path))
    clean_indices = _resolve_clean_indices(cfg, support_indices=support_indices, unseen_indices=unseen_indices)
    eval_batch_size = int(cfg.evaluation.get("batch_size", cfg.train_loop.batch_size))
    support_loader = _build_split_loader(cfg, support_indices, eval_batch_size, shuffle=False)
    unseen_loader = _build_split_loader(cfg, unseen_indices, eval_batch_size, shuffle=False)
    clean_loader = _build_split_loader(cfg, clean_indices, eval_batch_size, shuffle=False)

    all_rows = (
        _collect_prediction_rows(model, base_reference, support_loader, device, "repair_support_seen")
        + _collect_prediction_rows(model, base_reference, unseen_loader, device, "repair_holdout_unseen")
        + _collect_prediction_rows(model, base_reference, clean_loader, device, "clean_eval")
    )
    predictions_dir = Path(record.artifact_root) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    split_to_filename = {
        "repair_support_seen": "repair_support_seen_predictions.csv",
        "repair_holdout_unseen": "repair_holdout_unseen_predictions.csv",
        "clean_eval": "clean_eval_predictions.csv",
    }
    for split_name, filename in split_to_filename.items():
        rows = [row for row in all_rows if row["split"] == split_name]
        _write_rows(predictions_dir / filename, rows)

    support_summary = _summarize([row for row in all_rows if row["split"] == "repair_support_seen"])
    unseen_summary = _summarize([row for row in all_rows if row["split"] == "repair_holdout_unseen"])
    clean_summary = _summarize([row for row in all_rows if row["split"] == "clean_eval"])
    system_metrics = {
        "seen_total": support_summary["count"],
        "seen_correct": support_summary["patched_correct"],
        "seen_accuracy": support_summary["patched_accuracy"],
        "unseen_total": unseen_summary["count"],
        "unseen_correct": unseen_summary["patched_correct"],
        "unseen_accuracy": unseen_summary["patched_accuracy"],
        "clean_total": clean_summary["count"],
        "clean_correct": clean_summary["patched_correct"],
        "clean_accuracy": clean_summary["patched_accuracy"],
        "rr": clean_summary["rr"],
    }
    metrics = {
        "status": "completed",
        "mode": args.mode,
        "artifact_root": record.artifact_root,
        "checkpoint_path": str(checkpoint_path_out),
        "history": history,
        "system_metrics": system_metrics,
        "support_summary": support_summary,
        "unseen_summary": unseen_summary,
        "clean_summary": clean_summary,
    }
    runner.save_metrics(metrics)
    (Path(record.artifact_root) / "system_metrics.json").write_text(json.dumps(system_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"artifact_root": record.artifact_root, "system_metrics": system_metrics}, indent=2))


if __name__ == "__main__":
    main()
