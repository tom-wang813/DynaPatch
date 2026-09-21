"""Repair-only training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


@dataclass
class RepairEpochResult:
    """Aggregate metrics for one train or eval pass."""

    loss: float
    accuracy: float
    num_samples: int
    rr: float | None = None
    robust_loss: float | None = None
    cert_loss: float | None = None
    field_loss: float | None = None
    decision_ball_loss: float | None = None
    patch_reg_loss: float | None = None
    clean_replay_loss: float | None = None
    clean_route_hit: float | None = None
    cert_margin: float | None = None


def _move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = batch
    non_blocking = device.type == "cuda"
    return inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)


def _freeze_backbone_modules(model: nn.Module) -> None:
    """Freeze the backbone decomposition while keeping repair modules trainable."""
    if not hasattr(model, "decomposition"):
        return
    for module_name in ("shallow", "deep", "classifier", "base_model", "features", "norm", "permute", "flatten", "avgpool"):
        module = getattr(model.decomposition, module_name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        for param in module.parameters():
            param.requires_grad = False


def _clean_replay_consistency_loss(
    model: nn.Module,
    base_model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    objective: str = "kl",
) -> torch.Tensor:
    """Clean-data term for the patch.

    "kl" (default) keeps patched predictions close to the frozen backbone on base-correct clean
    inputs: a pure preservation objective that, by masking on `base_correct`, discards every clean
    sample the backbone already gets wrong. Those discarded samples are exactly the failure modes
    absent from the repair evidence, so under "kl" the patch is structurally unable to learn from
    them -- while the weight-retraining rival trains on clean data with cross-entropy and can.

    "ce" removes both restrictions: cross-entropy against the true labels over the whole clean
    batch, base-correct or not. It gives the patch the same opportunity at the cost of a weaker
    preservation guarantee, so Reg/CReg must be read alongside RR whenever it is enabled.
    """
    objective = str(objective).lower()
    if objective == "ce":
        patched_logits = model(inputs)  # [batch, num_classes]
        return F.cross_entropy(patched_logits, labels)
    if objective != "kl":
        raise ValueError(f"unsupported clean_replay_objective: {objective!r} (use 'kl' or 'ce')")

    with torch.no_grad():
        base_logits = base_model(inputs)  # [batch, num_classes]
        base_pred = torch.argmax(base_logits, dim=1)  # [batch]
        base_correct_mask = base_pred == labels  # [batch]

    if not bool(base_correct_mask.any().item()):
        return torch.zeros((), device=inputs.device)

    patched_logits = model(inputs)  # [batch, num_classes]
    masked_base_logits = base_logits[base_correct_mask]  # [kept, num_classes]
    masked_patched_logits = patched_logits[base_correct_mask]  # [kept, num_classes]
    return F.kl_div(
        F.log_softmax(masked_patched_logits, dim=1),
        F.softmax(masked_base_logits, dim=1),
        reduction="batchmean",
    )


def train_repair_only(
    model: nn.Module,
    base_model: nn.Module,
    bug_loader: DataLoader,
    clean_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    lambda_patch_l2: float = 0.0,
    lambda_clean_replay: float = 0.0,
    clean_replay_batch_limit: int | None = None,
    clean_replay_objective: str = "kl",
    grad_clip_norm: float | None = None,
    log_every_n_batches: int | None = None,
    bug_augment_views: int = 0,
    lambda_delta_consistency: float = 0.0,
) -> RepairEpochResult:
    """Train the repair modules on the defect set only.

    `bug_augment_views` adds semantically-neutral views of each failure to the repair loss, so
    the patch is asked to fix a neighbourhood rather than the exact observed pixels.
    `lambda_delta_consistency` additionally penalises the patch for changing across those views,
    which constrains it toward corrections that transfer instead of per-sample memorisation.
    Both target the seen->unseen repair gap; neither touches the frozen backbone.
    """
    _freeze_backbone_modules(model)
    model.train()
    base_model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    patch_reg_total = 0.0
    clean_replay_total = 0.0
    batch_count = 0
    clean_iter = cycle(clean_loader) if clean_loader is not None else None

    n_views = max(int(bug_augment_views), 0)
    if lambda_delta_consistency > 0.0 and n_views == 0:
        n_views = 1  # the consistency term needs at least one alternative view to compare against

    for batch_index, batch in enumerate(bug_loader):
        inputs, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        view_inputs = _sample_mt_views(inputs, n_views) if n_views > 0 else []
        if view_inputs:
            # One forward over [original, view_1, ... view_n] keeps the patch, the logits and the
            # BN/eval state consistent across views.
            stacked = torch.cat([inputs, *view_inputs], dim=0)  # [(1+n)*batch, C, H, W]
            stacked_labels = labels.repeat(1 + len(view_inputs))
            outputs = model.forward_with_intermediates(stacked)
            logits = outputs["logits"][: labels.size(0)]  # report metrics on the originals only
            repair_loss = loss_fn(outputs["logits"], stacked_labels)
        else:
            outputs = model.forward_with_intermediates(inputs)
            logits = outputs["logits"]
            repair_loss = loss_fn(logits, labels)
        patch_reg_loss = torch.zeros((), device=device)
        clean_replay_loss = torch.zeros((), device=device)
        delta_consistency_loss = torch.zeros((), device=device)
        if lambda_delta_consistency > 0.0 and view_inputs:
            patch_all = outputs["gated_patch"].flatten(1)  # [(1+n)*batch, patch_dim]
            base_patch = patch_all[: labels.size(0)]
            view_patches = patch_all[labels.size(0):].view(len(view_inputs), labels.size(0), -1)
            delta_consistency_loss = (view_patches - base_patch.unsqueeze(0)).norm(dim=2).mean()
        if lambda_patch_l2 > 0.0:
            patch_reg_loss = outputs["gated_patch"].flatten(1).norm(dim=1).mean()  # [batch]
        if lambda_clean_replay > 0.0:
            if clean_iter is None:
                raise ValueError("clean_loader is required when lambda_clean_replay > 0.")
            clean_inputs, clean_labels = _move_batch(next(clean_iter), device)
            if clean_replay_batch_limit is not None and clean_replay_batch_limit > 0:
                clean_inputs = clean_inputs[:clean_replay_batch_limit]
                clean_labels = clean_labels[:clean_replay_batch_limit]
            clean_replay_loss = _clean_replay_consistency_loss(
                model=model,
                base_model=base_model,
                inputs=clean_inputs,
                labels=clean_labels,
                objective=clean_replay_objective,
            )
        loss = (
            repair_loss
            + float(lambda_patch_l2) * patch_reg_loss
            + float(lambda_clean_replay) * clean_replay_loss
            + float(lambda_delta_consistency) * delta_consistency_loss
        )
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)  # [batch]
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += float((predictions == labels).float().sum().item())
        total_samples += int(labels.size(0))
        patch_reg_total += float(patch_reg_loss.item())
        clean_replay_total += float(clean_replay_loss.item())
        batch_count += 1
        if log_every_n_batches is not None and log_every_n_batches > 0 and ((batch_index + 1) % log_every_n_batches == 0):
            print(
                f"[TrainStep] batch={batch_index + 1} "
                f"batch_size={labels.size(0)} "
                f"repair={float(repair_loss.item()):.4f} "
                f"patch_reg={float(patch_reg_loss.item()):.4f} "
                f"clean_replay={float(clean_replay_loss.item()):.4f} "
                f"loss={float(loss.item()):.4f}",
                flush=True,
            )

    normalizer = max(total_samples, 1)
    batch_normalizer = max(batch_count, 1)
    return RepairEpochResult(
        loss=total_loss / normalizer,
        accuracy=total_correct / normalizer,
        num_samples=total_samples,
        patch_reg_loss=patch_reg_total / batch_normalizer if lambda_patch_l2 > 0.0 else None,
        clean_replay_loss=clean_replay_total / batch_normalizer if lambda_clean_replay > 0.0 else None,
    )


def train_repair_with_anchor_cache_ce(
    model: nn.Module,
    base_model: nn.Module,
    anchor_loader: DataLoader,
    cache_loader: DataLoader,
    clean_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    lambda_cache_repair: float,
    cert_loss_fn: nn.Module | None,
    lambda_cert: float,
    lambda_clean_replay: float,
    cert_on_bug: bool,
    cert_every_n_steps: int,
    cert_clean_batch_limit: int | None,
    clean_replay_batch_limit: int | None,
    device: torch.device,
    grad_clip_norm: float | None = None,
    log_every_n_batches: int | None = None,
) -> RepairEpochResult:
    """Train on anchor bugs and fixed-cache adversarial bugs with optional clean-side cert."""
    _freeze_backbone_modules(model)
    model.train()
    base_model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    cert_total = 0.0
    clean_replay_total = 0.0
    clean_route_hit_total = 0.0
    cert_margin_total = 0.0
    batch_count = 0
    anchor_iter = cycle(anchor_loader)
    clean_iter = cycle(clean_loader) if clean_loader is not None else None
    cert_every_n_steps = max(int(cert_every_n_steps), 1)

    for batch_index, cache_batch in enumerate(cache_loader):
        if batch_index < 3:
            print(f"[TrainStepBegin] batch={batch_index + 1}", flush=True)
        anchor_inputs, anchor_labels = _move_batch(next(anchor_iter), device)
        cache_inputs, cache_labels = _move_batch(cache_batch, device)

        optimizer.zero_grad(set_to_none=True)
        anchor_logits = model(anchor_inputs)
        cache_logits = model(cache_inputs)
        anchor_loss = loss_fn(anchor_logits, anchor_labels)
        cache_loss = loss_fn(cache_logits, cache_labels)
        cert_loss = torch.zeros((), device=device)
        clean_replay_loss = torch.zeros((), device=device)
        clean_stats = {"clean_route_hit": 0.0, "cert_margin": 0.0}
        cert_active_this_step = (
            cert_loss_fn is not None
            and lambda_cert > 0.0
            and (batch_index % cert_every_n_steps == 0)
        )
        if cert_active_this_step:
            if cert_on_bug:
                cert_inputs = cache_inputs
                cert_labels = cache_labels
            else:
                if clean_iter is None:
                    raise ValueError("clean_loader is required when cert is active on clean samples.")
                clean_inputs, clean_labels = _move_batch(next(clean_iter), device)
                if cert_clean_batch_limit is not None and cert_clean_batch_limit > 0:
                    clean_inputs = clean_inputs[:cert_clean_batch_limit]
                    clean_labels = clean_labels[:cert_clean_batch_limit]
                cert_inputs = clean_inputs
                cert_labels = clean_labels
            cert_loss, clean_stats = cert_loss_fn(model=model, inputs=cert_inputs, labels=cert_labels)
        if lambda_clean_replay > 0.0:
            if clean_iter is None:
                raise ValueError("clean_loader is required when lambda_clean_replay > 0.")
            clean_inputs, clean_labels = _move_batch(next(clean_iter), device)
            if clean_replay_batch_limit is not None and clean_replay_batch_limit > 0:
                clean_inputs = clean_inputs[:clean_replay_batch_limit]
                clean_labels = clean_labels[:clean_replay_batch_limit]
            clean_replay_loss = _clean_replay_consistency_loss(
                model=model,
                base_model=base_model,
                inputs=clean_inputs,
                labels=clean_labels,
            )
        total_objective = (
            anchor_loss
            + float(lambda_cache_repair) * cache_loss
            + float(lambda_cert) * cert_loss
            + float(lambda_clean_replay) * clean_replay_loss
        )
        total_objective.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        cache_predictions = torch.argmax(cache_logits, dim=1)  # [batch]
        batch_size = int(cache_labels.size(0))
        total_loss += float(total_objective.item()) * batch_size
        total_correct += float((cache_predictions == cache_labels).float().sum().item())
        total_samples += batch_size
        cert_total += float(cert_loss.item())
        clean_replay_total += float(clean_replay_loss.item())
        clean_route_hit_total += float(clean_stats["clean_route_hit"])
        cert_margin_total += float(clean_stats["cert_margin"])
        batch_count += 1
        if batch_index < 3:
            print(
                f"[TrainStepEnd] batch={batch_index + 1} "
                f"anchor_bs={anchor_labels.size(0)} "
                f"cache_bs={cache_labels.size(0)} "
                f"anchor_loss={float(anchor_loss.item()):.4f} "
                f"cache_loss={float(cache_loss.item()):.4f} "
                f"cert_loss={float(cert_loss.item()):.4f} "
                f"clean_replay={float(clean_replay_loss.item()):.4f}",
                flush=True,
            )
        if log_every_n_batches is not None and log_every_n_batches > 0 and ((batch_index + 1) % log_every_n_batches == 0):
            print(
                f"[TrainStep] batch={batch_index + 1} "
                f"anchor_bs={anchor_labels.size(0)} "
                f"cache_bs={cache_labels.size(0)} "
                f"anchor_loss={float(anchor_loss.item()):.4f} "
                f"cache_loss={float(cache_loss.item()):.4f} "
                f"cert_loss={float(cert_loss.item()):.4f} "
                f"clean_replay={float(clean_replay_loss.item()):.4f}",
                flush=True,
            )

    normalizer = max(total_samples, 1)
    batch_normalizer = max(batch_count, 1)
    return RepairEpochResult(
        loss=total_loss / normalizer,
        accuracy=total_correct / normalizer,
        num_samples=total_samples,
        cert_loss=cert_total / batch_normalizer if cert_loss_fn is not None else None,
        clean_replay_loss=clean_replay_total / batch_normalizer if lambda_clean_replay > 0.0 else None,
        clean_route_hit=clean_route_hit_total / batch_normalizer if cert_loss_fn is not None else None,
        cert_margin=cert_margin_total / batch_normalizer if cert_loss_fn is not None else None,
    )


@torch.no_grad()
def evaluate_repair_model(
    model: nn.Module,
    base_model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    compute_rr: bool = False,
) -> RepairEpochResult:
    """Evaluate the repaired model on a bug or clean split."""
    model.eval()
    base_model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    rr_regressed = 0
    rr_base_correct = 0

    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        outputs = model.forward_with_intermediates(inputs)
        patched_logits = outputs["logits"]
        loss = loss_fn(patched_logits, labels)

        predictions = torch.argmax(patched_logits, dim=1)  # [batch]
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += float((predictions == labels).float().sum().item())
        total_samples += int(labels.size(0))

        if compute_rr:
            base_logits = base_model(inputs)
            base_pred = torch.argmax(base_logits, dim=1)  # [batch]
            base_correct = base_pred == labels  # [batch]
            regressed = base_correct & (predictions != labels)  # [batch]
            # Count regressions globally over all base-correct clean examples instead of
            # averaging per-batch RR, which can drift from row-level summaries.
            rr_regressed += int(regressed.sum().item())
            rr_base_correct += int(base_correct.sum().item())

    normalizer = max(total_samples, 1)
    rr = rr_regressed / max(rr_base_correct, 1) if compute_rr else None
    return RepairEpochResult(
        loss=total_loss / normalizer,
        accuracy=total_correct / normalizer,
        num_samples=total_samples,
        rr=rr,
    )


@torch.no_grad()
def collect_prediction_rows(
    model: nn.Module,
    base_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> list[dict[str, Any]]:
    """Collect per-sample base and patched predictions for downstream failure analysis."""
    model.eval()
    base_model.eval()

    dataset = loader.dataset
    if isinstance(dataset, Subset):
        dataset_indices = list(dataset.indices)
    else:
        dataset_indices = list(range(len(dataset)))

    rows: list[dict[str, Any]] = []
    cursor = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        outputs = model.forward_with_intermediates(inputs)
        patched_logits = outputs["logits"]
        base_logits = base_model(inputs)

        base_probs = torch.softmax(base_logits, dim=1)  # [batch, num_classes]
        patched_probs = torch.softmax(patched_logits, dim=1)  # [batch, num_classes]
        base_pred = torch.argmax(base_logits, dim=1)  # [batch]
        patched_pred = torch.argmax(patched_logits, dim=1)  # [batch]
        route_weight = outputs["route_weight"].flatten(1).mean(dim=1)  # [batch]
        patch_norm = outputs["gated_patch"].flatten(1).norm(dim=1)  # [batch]
        prototype_ids = outputs["prototype_ids"]  # [batch]

        batch_size = int(labels.size(0))
        batch_indices = dataset_indices[cursor: cursor + batch_size]
        cursor += batch_size

        # Gather per-sample confidence in one vectorised op, then move everything
        # to CPU in a single transfer per tensor — avoids one CUDA sync per .item() call.
        arange = torch.arange(batch_size, device=device)
        base_confs = base_probs[arange, base_pred]      # [batch]
        patched_confs = patched_probs[arange, patched_pred]  # [batch]

        labels_list = labels.tolist()
        base_pred_list = base_pred.tolist()
        patched_pred_list = patched_pred.tolist()
        base_confs_list = base_confs.tolist()
        patched_confs_list = patched_confs.tolist()
        route_weight_list = route_weight.tolist()
        patch_norm_list = patch_norm.tolist()
        prototype_ids_list = prototype_ids.tolist()

        for idx in range(batch_size):
            label = labels_list[idx]
            base_prediction = base_pred_list[idx]
            patched_prediction = patched_pred_list[idx]
            rows.append(
                {
                    "split": split_name,
                    "dataset_index": batch_indices[idx],
                    "label": label,
                    "base_pred": base_prediction,
                    "patched_pred": patched_prediction,
                    "base_conf": base_confs_list[idx],
                    "patched_conf": patched_confs_list[idx],
                    "route_weight": route_weight_list[idx],
                    "patch_norm": patch_norm_list[idx],
                    "prototype_id": prototype_ids_list[idx],
                    "base_correct": base_prediction == label,
                    "patched_correct": patched_prediction == label,
                    "repaired": base_prediction != label and patched_prediction == label,
                    "regressed": base_prediction == label and patched_prediction != label,
                }
            )
    return rows


@dataclass
class MTConsistencyResult:
    """Per-MR and aggregate metamorphic correctness results."""

    # per-MR accuracy: fraction where predict(T(x)) == y (ground-truth label)
    mr_pass_rates: dict[str, float] = field(default_factory=dict)
    all_pass_rate: float = 0.0
    # gate consistency: fraction where gate routing decision agrees after transform
    gate_mr_pass_rates: dict[str, float] = field(default_factory=dict)
    gate_all_pass_rate: float = 0.0
    # original accuracy on x (baseline to compare against)
    base_accuracy: float = 0.0
    num_samples: int = 0


def _sample_mt_views(inputs: torch.Tensor, num_views: int) -> list[torch.Tensor]:
    """Draw `num_views` distinct semantically-neutral views of a batch.

    Uses the same transform family as the MT consistency evaluation, so training-time
    augmentation and the metamorphic metric stay defined against one another.
    """
    if num_views <= 0:
        return []
    views = _mt_transforms(inputs)
    names = list(views)
    picked = torch.randperm(len(names))[: min(num_views, len(names))]
    return [views[names[int(i)]] for i in picked]


def _mt_transforms(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Semantically-neutral transforms for image MT evaluation.

    Only transforms that preserve class semantics are included.  Horizontal flip
    is intentionally excluded because it changes the meaning of directional signs.
    All transforms operate on normalized float tensors of shape [B, C, H, W].
    """
    try:
        import torchvision.transforms.functional as TF
    except ImportError as exc:
        raise ImportError("torchvision is required for MT evaluation.") from exc

    return {
        "rotate_cw5":   TF.rotate(inputs, angle=5),
        "rotate_ccw5":  TF.rotate(inputs, angle=-5),
        "noise":        inputs + torch.randn_like(inputs) * 0.1,
        "brightness":   inputs + 0.15,
        "contrast":     (inputs - inputs.mean(dim=(2, 3), keepdim=True)) * 1.2 + inputs.mean(dim=(2, 3), keepdim=True),
    }


@torch.no_grad()
def evaluate_mt_consistency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    gate_threshold: float = 0.5,
) -> MTConsistencyResult:
    """Evaluate metamorphic consistency of both predictions and gate routing.

    For each sample, the reference is the model's output on the original input.
    A sample passes MR_i (prediction) if T(x) gives the same predicted class.
    A sample passes MR_i (gate) if the gate's binary routing decision agrees
    (route_weight > gate_threshold on both or neither).

    Semantically-altering transforms (e.g. hflip for traffic signs) are excluded.
    """
    model.eval()

    mr_names: list[str] | None = None
    mr_correct: dict[str, int] = {}
    gate_consistent: dict[str, int] = {}
    all_correct = 0
    gate_all_consistent = 0
    base_correct_count = 0
    total = 0

    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        outputs = model.forward_with_intermediates(inputs)
        ref_pred = torch.argmax(outputs["logits"], dim=1)                          # [B]
        ref_gate = (outputs["route_weight"].flatten(1).mean(dim=1) > gate_threshold)  # [B] bool
        base_correct_count += int((ref_pred == labels).sum().item())

        transforms = _mt_transforms(inputs)
        if mr_names is None:
            mr_names = list(transforms.keys())
            mr_correct = {name: 0 for name in mr_names}
            gate_consistent = {name: 0 for name in mr_names}

        per_all = torch.ones(inputs.size(0), dtype=torch.bool, device=device)
        gate_all = torch.ones(inputs.size(0), dtype=torch.bool, device=device)

        for name, t_inputs in transforms.items():
            t_out = model.forward_with_intermediates(t_inputs)
            t_pred = torch.argmax(t_out["logits"], dim=1)
            t_gate = (t_out["route_weight"].flatten(1).mean(dim=1) > gate_threshold)

            correct = t_pred == labels        # predict(T(x)) == y
            gate_match = t_gate == ref_gate

            mr_correct[name] += int(correct.sum().item())
            gate_consistent[name] += int(gate_match.sum().item())
            per_all &= correct
            gate_all &= gate_match

        all_correct += int(per_all.sum().item())
        gate_all_consistent += int(gate_all.sum().item())
        total += int(inputs.size(0))

    if total == 0 or mr_names is None:
        return MTConsistencyResult(num_samples=0)

    return MTConsistencyResult(
        mr_pass_rates={name: mr_correct[name] / total for name in mr_names},
        all_pass_rate=all_correct / total,
        gate_mr_pass_rates={name: gate_consistent[name] / total for name in mr_names},
        gate_all_pass_rate=gate_all_consistent / total,
        base_accuracy=base_correct_count / total,
        num_samples=total,
    )
