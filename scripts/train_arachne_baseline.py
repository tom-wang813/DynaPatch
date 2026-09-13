#!/usr/bin/env python3
"""Train an Arachne-style sparse last-layer search baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import FeatureBank, SearchConfig, get_classifier_module, greedy_coordinate_search, select_topk_parameters  # noqa: E402
from src.data.factory import build_dataset, build_repair_dataloaders  # noqa: E402
from src.experiment.runner import ExperimentRunner  # noqa: E402
from src.models.backbones.factory import build_backbone, load_backbone_checkpoint  # noqa: E402


def _load_manifest(path_value: str) -> list[int]:
    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("indices", [])
    return [int(v) for v in payload]


def _resolve_clean_indices(cfg, *, support_indices: list[int], unseen_indices: list[int]) -> list[int]:
    clean_path = cfg.data.get("clean_eval_indices_path")
    if clean_path is not None:
        return _load_manifest(str(clean_path))
    test_dataset = build_dataset(cfg, train=False)
    excluded = set(int(v) for v in support_indices) | set(int(v) for v in unseen_indices)
    return [idx for idx in range(len(test_dataset)) if idx not in excluded]


def _move_batch(batch, device: torch.device):
    inputs, labels = batch
    non_blocking = device.type == "cuda"
    return inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)


def _build_split_loader(cfg, indices: list[int], batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = build_dataset(cfg, train=False)
    workers = int(cfg.runtime.get("num_workers", 0))
    kwargs = {
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, **kwargs)


@torch.no_grad()
def _collect_feature_bank(model, classifier, loader, device: torch.device) -> FeatureBank:
    model.eval()
    features_parts = []
    labels_parts = []
    base_pred_parts = []
    captured = {}

    def _hook(_module, inputs, _output):
        captured["features"] = inputs[0].detach()

    hook = classifier.register_forward_hook(_hook)
    try:
        for batch in loader:
            inputs, labels = _move_batch(batch, device)
            logits = model(inputs)
            features_parts.append(captured["features"].detach().cpu())
            labels_parts.append(labels.detach().cpu())
            base_pred_parts.append(logits.argmax(dim=1).detach().cpu())
    finally:
        hook.remove()

    return FeatureBank(
        features=torch.cat(features_parts, dim=0).to(device),
        labels=torch.cat(labels_parts, dim=0).to(device),
        base_pred=torch.cat(base_pred_parts, dim=0).to(device),
    )


@torch.no_grad()
def _collect_prediction_rows(model, base_reference, loader, device: torch.device, split_name: str):
    model.eval()
    base_reference.eval()
    dataset = loader.dataset
    dataset_indices = list(dataset.indices) if isinstance(dataset, Subset) else list(range(len(dataset)))
    rows = []
    cursor = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        base_logits = base_reference(inputs)
        patched_logits = model(inputs)
        base_probs = torch.softmax(base_logits, dim=1)
        patched_probs = torch.softmax(patched_logits, dim=1)
        base_pred = base_logits.argmax(dim=1)
        patched_pred = patched_logits.argmax(dim=1)
        batch_size = int(labels.size(0))
        batch_indices = dataset_indices[cursor : cursor + batch_size]
        cursor += batch_size
        arange = torch.arange(batch_size, device=device)
        base_conf = base_probs[arange, base_pred]
        patched_conf = patched_probs[arange, patched_pred]
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


def _write_rows(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--step-scale", type=float, default=0.5)
    parser.add_argument("--clean-tradeoff", type=float, default=0.25)
    parser.add_argument("--clean-replay-subset-size", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Dot-path overrides applied after loading config, e.g. data.bug_indices_path=... "
        "Required to point this baseline at a different split: the resolved config carries its "
        "own bug_bank paths, and overriding only some of them silently mixes splits.",
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
    cfg.experiment.stage = "baseline_arachne_sparse_search"
    cfg.experiment.id = f"{cfg.experiment.id}_arachne_style"
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

    classifier = get_classifier_module(model, architecture)
    if not isinstance(classifier, torch.nn.Linear):
        raise TypeError(f"Arachne-style baseline currently expects a linear classifier, got: {type(classifier)!r}")
    for param in model.parameters():
        param.requires_grad = False

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
            batch_size=int(args.train_batch_size or cfg.train_loop.batch_size),
            **loader_kwargs,
        )

    bug_bank = _collect_feature_bank(model, classifier, dataloaders["bug_train"], device)
    clean_bank = _collect_feature_bank(model, classifier, clean_train_loader, device) if clean_train_loader is not None else None

    selected_refs = select_topk_parameters(
        model=model,
        architecture=architecture,
        classifier=classifier,
        loader=dataloaders["bug_train"],
        device=device,
        top_k=int(args.top_k),
    )
    search_result = greedy_coordinate_search(
        classifier=classifier,
        bug_bank=bug_bank,
        clean_bank=clean_bank,
        selected_refs=selected_refs,
        config=SearchConfig(
            top_k=int(args.top_k),
            rounds=int(args.rounds),
            step_scale=float(args.step_scale),
            clean_tradeoff=float(args.clean_tradeoff),
            max_clean_batches=8,
        ),
    )

    checkpoint_dir = Path(record.artifact_root) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path_out = checkpoint_dir / "arachne_style_last.pt"
    torch.save({"state_dict": model.state_dict(), "mode": "arachne_style", "search_result": search_result}, checkpoint_path_out)

    support_indices = _load_manifest(str(cfg.data.bug_indices_path))
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
    for split_name in ["repair_support_seen", "repair_holdout_unseen", "clean_eval"]:
        rows = [row for row in all_rows if row["split"] == split_name]
        _write_rows(predictions_dir / f"{split_name}_predictions.csv", rows)

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
        "mode": "arachne_style",
        "artifact_root": record.artifact_root,
        "checkpoint_path": str(checkpoint_path_out),
        "search_result": search_result,
        "selected_parameter_count": len(selected_refs),
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
