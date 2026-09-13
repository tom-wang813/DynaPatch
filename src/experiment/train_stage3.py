"""Stage-3 training entrypoint implemented in src for shared dispatch."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig, OmegaConf

from src.experiment.runner import ExperimentRunner
from src.experiment.stage3 import build_stage3_bundle, resolve_runtime_device
from src.models.dynapatch.model import DynaPatchModel
from src.models.dynapatch.factory import seed_memory_bank
from src.training.losses import (
    BugSideDynamicIBPCertificationLoss,
    CertificationGuardLoss,
    DynamicIBPCertificationLoss,
    FocalRepairClassificationLoss,
    FunctionalRepairBallLoss,
    PatchConsistencyFieldLoss,
    PatchedDecisionBallLoss,
    RepairClassificationLoss,
    RobustRepairLoss,
    SafetyAwareRepairClassificationLoss,
)
from src.training.loops import (
    collect_prediction_rows,
    evaluate_mt_consistency,
    evaluate_repair_model,
    evaluate_repair_model_under_attack,
    MTConsistencyResult,
    RepairEpochResult,
    train_repair_with_anchor_cache_ce,
    train_repair_only,
    train_repair_with_dual_objectives,
)


def validate_stage3_config(cfg: DictConfig) -> None:
    """Validate the minimum config contract for stage-3 repair runs."""
    if "experiment" not in cfg or cfg.experiment.stage != "stage3_repair":
        raise ValueError("Stage-3 runner requires `experiment.stage=stage3_repair`.")
    if cfg.model.get("checkpoint_path") is None:
        raise ValueError("Stage-3 runner requires `model.checkpoint_path`.")


def _write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist per-sample predictions for post-run analysis."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_predictions(rows: list[dict[str, object]]) -> dict[str, object]:
    """Compute split-level repair and regression counts from prediction rows."""
    total = len(rows)
    repaired = sum(1 for row in rows if bool(row["repaired"]))
    regressed = sum(1 for row in rows if bool(row["regressed"]))
    patched_correct = sum(1 for row in rows if bool(row["patched_correct"]))
    base_correct = sum(1 for row in rows if bool(row["base_correct"]))
    mean_route = sum(float(row["route_weight"]) for row in rows) / max(total, 1)
    mean_patch_norm = sum(float(row["patch_norm"]) for row in rows) / max(total, 1)
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
        "mean_route_weight": mean_route,
        "mean_patch_norm": mean_patch_norm,
    }


def _summarize_clean_eval(
    clean_eval: object,
    bug_eval_summary: dict[str, object],
) -> dict[str, object]:
    """Build a lightweight clean summary without row-level export on every epoch."""
    return {
        "patched_accuracy": float(clean_eval.accuracy),
        "rr": 0.0 if clean_eval.rr is None else float(clean_eval.rr),
        "regressed": 0,
        "repaired": 0,
        "repaired_rate": 0.0,
        "patched_correct": 0,
        "base_correct": 0,
        "count": int(clean_eval.num_samples),
        "base_accuracy": 0.0,
        "regressed_rate": 0.0,
        "mean_route_weight": 0.0,
        "mean_patch_norm": 0.0,
        "bug_eval_reference": float(bug_eval_summary["patched_accuracy"]),
    }


def _build_epoch_prediction_summary(
    model: DynaPatchModel,
    backbone,
    bug_eval_loader,
    clean_eval: object,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    """Collect lightweight epoch summaries for early stopping."""
    bug_eval_rows = collect_prediction_rows(
        model=model,
        base_model=backbone,
        loader=bug_eval_loader,
        device=device,
        split_name="bug_eval",
    )
    bug_eval_summary = _summarize_predictions(bug_eval_rows)
    return {
        "bug_eval": bug_eval_summary,
        "clean_eval": _summarize_clean_eval(clean_eval, bug_eval_summary),
    }


def _optional_float(value: object) -> float | None:
    """Return ``None`` for unset scalars and ``float(value)`` otherwise."""
    if value is None:
        return None
    return float(value)


def _resolve_critical_indices(cfg: DictConfig) -> list[int]:
    """Resolve critical class ids for safety-aware repair losses."""
    raw = cfg.loss.get("critical_indices")
    if raw is not None:
        return [int(v) for v in raw]

    risk_cfg_path = cfg.loss.get("safety_risk_cfg_path")
    if risk_cfg_path is None:
        return []
    risk_cfg = OmegaConf.load(str(risk_cfg_path))
    critical: list[int] = []
    for ids in risk_cfg.get("critical_signs", {}).values():
        if ids is None:
            continue
        if isinstance(ids, (list, tuple)):
            for v in ids:
                try:
                    critical.append(int(v))
                except (TypeError, ValueError):
                    continue
            continue
        if isinstance(ids, DictConfig):
            nested_values = []
            for nested in ids.values():
                if isinstance(nested, (list, tuple)):
                    nested_values.extend(nested)
                else:
                    nested_values.append(nested)
            for v in nested_values:
                try:
                    critical.append(int(v))
                except (TypeError, ValueError):
                    continue
            continue
        try:
            critical.append(int(ids))
        except (TypeError, ValueError):
            continue
    return sorted(set(critical))


def _loader_num_samples(loader: object) -> int:
    """Best-effort sample count without forcing eager materialization of lazy loaders."""
    if hasattr(loader, "num_samples"):
        return int(getattr(loader, "num_samples"))
    return int(len(loader.dataset))


def _build_bug_eval_monitor_loader(
    bug_eval_loader: DataLoader,
    monitor_size: int | None,
    seed: int,
) -> DataLoader:
    """Build a deterministic bug-eval monitor loader for lightweight per-epoch evaluation."""
    if monitor_size is None or monitor_size <= 0:
        return bug_eval_loader

    dataset = bug_eval_loader.dataset
    dataset_size = len(dataset)
    if monitor_size >= dataset_size:
        return bug_eval_loader

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=generator).tolist()
    monitor_indices = perm[:monitor_size]
    monitor_dataset = Subset(dataset, monitor_indices)
    source_workers = max(int(getattr(bug_eval_loader, "num_workers", 0)), 0)
    kwargs = {
        "shuffle": False,
        "num_workers": source_workers,
        "pin_memory": True,
    }
    if source_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        monitor_dataset,
        batch_size=bug_eval_loader.batch_size,
        **kwargs,
    )


def _should_run_eval(epoch_index: int, eval_every_n_epochs: int) -> bool:
    """Return whether this epoch should run monitor evaluation."""
    if eval_every_n_epochs <= 1:
        return True
    return ((epoch_index + 1) % eval_every_n_epochs) == 0


def _should_run_full_eval(
    epoch_index: int,
    total_epochs: int,
    full_eval_every_n_epochs: int,
    full_eval_after_epoch: int,
) -> bool:
    """Return whether this epoch should run full bug_eval instead of monitor eval."""
    current_epoch = epoch_index + 1
    if current_epoch == total_epochs:
        return True
    if full_eval_after_epoch > 0 and current_epoch >= full_eval_after_epoch:
        return True
    if full_eval_every_n_epochs > 0 and (current_epoch % full_eval_every_n_epochs) == 0:
        return True
    return False


def _early_stop_key(metric_name: str, prediction_summary: dict[str, dict[str, object]]) -> tuple[float, ...]:
    """Return a lexicographic score where larger is always better."""
    bug_eval = prediction_summary["bug_eval"]
    clean_eval = prediction_summary["clean_eval"]

    repaired = float(bug_eval["repaired"])
    repaired_rate = float(bug_eval["repaired_rate"])
    bug_acc = float(bug_eval["patched_accuracy"])
    clean_acc = float(clean_eval["patched_accuracy"])
    rr = float(clean_eval["rr"])
    regressed = float(clean_eval["regressed"])

    if metric_name == "heldout_repaired":
        return (repaired, -rr, clean_acc, bug_acc, -regressed)
    if metric_name == "heldout_adv_repaired":
        adv_bug_acc = float(prediction_summary.get("bug_eval_adv", {}).get("patched_accuracy", 0.0))
        adv_repaired = float(prediction_summary.get("bug_eval_adv", {}).get("repaired", 0.0))
        return (adv_bug_acc, adv_repaired, repaired, -rr, clean_acc, bug_acc)
    if metric_name == "repaired_rate":
        return (repaired_rate, repaired, -rr, clean_acc, bug_acc)
    if metric_name == "bug_acc" or metric_name == "rsr":
        return (bug_acc, repaired, -rr, clean_acc)
    if metric_name == "clean_acc":
        return (clean_acc, -rr, repaired, bug_acc)
    if metric_name == "rr":
        return (-rr, repaired, clean_acc, bug_acc)
    if metric_name == "safety_balanced":
        safety_score = clean_acc - rr  # prioritize clean retention while still penalizing regressions
        return (safety_score, bug_acc, repaired, -regressed)
    raise ValueError(f"Unsupported early stopping metric: {metric_name}")


def _ramp_weight(epoch_index: int, start_epoch: int, ramp_epochs: int, target: float) -> float:
    """Linearly ramp a loss weight from zero to its target."""
    if target <= 0.0:
        return 0.0
    if epoch_index < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return target
    progress = min(epoch_index - start_epoch + 1, ramp_epochs)
    return target * float(progress / ramp_epochs)


def _scheduled_loss_weights(cfg: DictConfig, epoch_index: int) -> tuple[float, float]:
    """Return epoch-specific robust and cert weights under the configured schedule."""
    target_robust = float(cfg.loss.get("lambda_robust", 0.0))
    target_cert = float(cfg.loss.get("lambda_cert", 0.0))
    schedule = cfg.loss.get("schedule")
    if schedule is None:
        return target_robust, target_cert

    repair_only_epochs = int(schedule.get("repair_only_epochs", 0))
    robust_start_epoch = int(schedule.get("robust_start_epoch", repair_only_epochs))
    cert_start_epoch = int(schedule.get("cert_start_epoch", robust_start_epoch))
    robust_ramp_epochs = int(schedule.get("robust_ramp_epochs", 0))
    cert_ramp_epochs = int(schedule.get("cert_ramp_epochs", 0))

    robust_weight = _ramp_weight(
        epoch_index=epoch_index,
        start_epoch=robust_start_epoch,
        ramp_epochs=robust_ramp_epochs,
        target=target_robust,
    )
    cert_weight = _ramp_weight(
        epoch_index=epoch_index,
        start_epoch=cert_start_epoch,
        ramp_epochs=cert_ramp_epochs,
        target=target_cert,
    )
    return robust_weight, cert_weight


def _scheduled_aux_loss_weights(cfg: DictConfig, epoch_index: int) -> dict[str, float]:
    """Return epoch-specific auxiliary loss weights for two-stage repair schedules."""
    active = {
        "lambda_field": float(cfg.loss.get("lambda_field", 0.0)),
        "lambda_decision_ball": float(cfg.loss.get("lambda_decision_ball", 0.0)),
        "lambda_patch_l2": float(cfg.loss.get("lambda_patch_l2", 0.0)),
        "lambda_clean_replay": float(cfg.loss.get("lambda_clean_replay", 0.0)),
        "lambda_safety_risk": float(cfg.loss.get("lambda_safety_risk", 0.0)),
    }

    two_stage = cfg.loss.get("two_stage")
    if two_stage is None or not bool(two_stage.get("enabled", False)):
        return active

    current_epoch = epoch_index + 1
    transition_epoch = int(two_stage.get("transition_epoch", 1))
    stage_prefix = "stage1" if current_epoch < transition_epoch else "stage2"
    for key, target in list(active.items()):
        active[key] = float(two_stage.get(f"{stage_prefix}_{key}", target))
    return active


def _phase_label(in_warmup: bool, robust_weight: float, cert_weight: float) -> str:
    """Return a readable phase label for logs and metrics."""
    if in_warmup:
        return "warmup"
    if robust_weight <= 0.0 and cert_weight <= 0.0:
        return "repair_only"
    if robust_weight > 0.0 and cert_weight <= 0.0:
        return "repair_plus_robust"
    if robust_weight <= 0.0 and cert_weight > 0.0:
        return "repair_plus_cert"
    return "dual_objective"


def _two_stage_early_stop_state(cfg: DictConfig, epoch_index: int) -> tuple[bool, str, int, int]:
    """Return whether early stopping is active and which metric/patience to use."""
    metric = str(cfg.train_loop.get("early_stop_metric", "heldout_repaired"))
    patience = int(cfg.train_loop.get("early_stop_patience", cfg.train_loop.epochs))
    min_epochs = int(cfg.train_loop.get("early_stop_min_epochs", 1))

    two_stage = cfg.loss.get("two_stage")
    if two_stage is None or not bool(two_stage.get("enabled", False)):
        return True, metric, patience, min_epochs

    current_epoch = epoch_index + 1
    transition_epoch = int(two_stage.get("transition_epoch", 1))
    if current_epoch < transition_epoch:
        metric = str(two_stage.get("stage1_early_stop_metric", "heldout_repaired"))
        return False, metric, patience, min_epochs

    metric = str(two_stage.get("stage2_early_stop_metric", metric))
    patience = int(two_stage.get("stage2_early_stop_patience", patience))
    min_epochs = int(two_stage.get("stage2_early_stop_min_epochs", min_epochs))
    return True, metric, patience, min_epochs


@torch.no_grad()
def _seed_router_support(model: DynaPatchModel, backbone, loader, device: torch.device) -> None:
    """Seed router support keys and optional conditioning state from the bug support set."""
    router_budget = int(model.router.expert_keys.size(0))
    use_full_support = bool(getattr(model, "prototype_bank", None) is not None)
    if router_budget <= 0 and not use_full_support:
        return

    support_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for inputs, _labels in loader:
        if not use_full_support and router_budget <= 0:
            break
        if use_full_support:
            batch_inputs = inputs.to(device)
            batch_labels = _labels.to(device)
        else:
            batch_inputs = inputs[:router_budget].to(device)
            batch_labels = _labels[:router_budget].to(device)
        support_batches.append(batch_inputs)
        label_batches.append(batch_labels)
        router_budget -= int(batch_inputs.size(0))

    if not support_batches:
        return

    support_inputs = torch.cat(support_batches, dim=0)  # [support_k, c, h, w]
    support_labels = torch.cat(label_batches, dim=0)  # [support_k]
    shallow_feat = model.decomposition.extract_shallow(support_inputs)
    route_feat = model.decomposition.router_features(shallow_feat)  # [support_k, shallow_dim]
    model.router.seed_support(route_feat)
    predictions = backbone(support_inputs).argmax(dim=1)  # [support_k]
    seed_memory_bank(
        model.prototype_bank,
        route_feat=route_feat,
        labels=support_labels,
        predictions=predictions,
    )


def _install_random_basis(model: DynaPatchModel, seed: int) -> None:
    """Install a random orthonormal basis of the same rank -- the necessary control for the
    logit-gradient basis.

    `bank_coef` restricts the patch to `num_basis` directions and reports no loss of repair rate.
    Two explanations are indistinguishable from that result alone: (a) the classifier's logit
    gradients are the RIGHT directions, or (b) any low-rank subspace of that size is enough and
    the win is dimensionality reduction. This installs (b): same rank, same parameter count, same
    training, directions carrying no information about the classifier. Scaled to the logit basis's
    mean row norm so the two differ in DIRECTION only, not in magnitude -- an unscaled random
    basis would change the effective learning rate on the coefficients and confound the test.

    If (b) matches (a), the claim must be weakened to "a low-rank patch suffices".
    """
    basis = model.hypernet.basis                      # [num_basis, patch_dim], already holds the
                                                      # logit basis: the caller installs it first
                                                      # so its row norm can be matched here.
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    m = torch.randn(basis.size(1), basis.size(0), generator=g)  # [patch_dim, num_basis]
    q, _ = torch.linalg.qr(m)                         # [patch_dim, num_basis], orthonormal cols
    rand = q.T.to(basis.device, basis.dtype)          # [num_basis, patch_dim]
    scale = float(basis.norm(dim=1).mean())
    if scale <= 0:
        raise RuntimeError("random basis must be scaled to the logit basis; install it first")
    model.hypernet.set_basis(rand * scale)
    print(f"[bank_coef] RANDOM orthonormal basis installed: {tuple(rand.shape)} "
          f"(scaled to row norm {float(scale):.4f}, seed {seed})", flush=True)


def _install_logit_basis(model: DynaPatchModel, loader, device: torch.device) -> None:
    """Fix the patch's direction basis to the frozen classifier's own logit gradients.

    For hypernet_style='bank_coef' the delta is `coef(x) @ basis`, so the directions the patch
    can move in are decided here, once, and never trained. `basis[c]` is d logit_c / d patch
    evaluated at patch=0 and averaged over a batch -- the exact linearisation that made the
    closed-form correction work (PITFALLS 2026-07-28: the delta-to-logit map is linear to
    R^2=0.991, so the direction carries no information that needs learning). Differentiating
    w.r.t. the patch rather than the feature keeps this correct for every backbone, including
    VGG-16 whose classifier head is a non-linear MLP over an unpooled 25088-d input.
    """
    dec = model.decomposition
    batch = next(iter(loader))
    inputs = (batch[0] if isinstance(batch, (list, tuple)) else batch["input"]).to(device)
    with torch.no_grad():
        shallow = dec.extract_shallow(inputs)
        deep = dec.extract_deep(shallow)
    patch_dim = model.hypernet.basis.size(1)
    probe = torch.zeros(deep.size(0), patch_dim, device=device, requires_grad=True)
    logits = dec.classify(deep, probe)  # [batch, num_classes]
    rows = []
    for c in range(logits.size(1)):
        (grad,) = torch.autograd.grad(logits[:, c].sum(), probe, retain_graph=(c + 1 < logits.size(1)))
        rows.append(grad.mean(0))  # [patch_dim]
    basis = torch.stack(rows)  # [num_classes, patch_dim]
    model.hypernet.set_basis(basis)
    print(f"[bank_coef] basis installed: {tuple(basis.shape)} "
          f"(row norm mean {basis.norm(dim=1).mean().item():.4f})", flush=True)


def _build_clean_monitor_loader(
    clean_eval_loader: DataLoader,
    monitor_size: int | None,
    seed: int,
) -> DataLoader:
    """Build a deterministic clean-monitor loader for per-epoch evaluation."""
    if monitor_size is None or monitor_size <= 0:
        return clean_eval_loader

    dataset = clean_eval_loader.dataset
    dataset_size = len(dataset)
    if monitor_size >= dataset_size:
        return clean_eval_loader

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=generator).tolist()
    monitor_indices = perm[:monitor_size]
    monitor_dataset = Subset(dataset, monitor_indices)
    source_workers = max(int(getattr(clean_eval_loader, "num_workers", 0)), 0)
    kwargs = {
        "shuffle": False,
        "num_workers": source_workers,
        "pin_memory": True,
    }
    if source_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        monitor_dataset,
        batch_size=clean_eval_loader.batch_size,
        **kwargs,
    )


def _build_cert_train_loader(
    clean_train_loader: DataLoader,
    subset_size: int | None,
    batch_size: int | None,
    seed: int,
) -> DataLoader:
    """Build a deterministic lightweight clean loader used only for cert steps."""
    dataset = clean_train_loader.dataset
    dataset_size = len(dataset)
    if subset_size is not None and subset_size > 0 and subset_size < dataset_size:
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(dataset_size, generator=generator).tolist()
        dataset = Subset(dataset, perm[:subset_size])

    resolved_batch_size = batch_size if batch_size is not None and batch_size > 0 else clean_train_loader.batch_size
    source_workers = max(int(getattr(clean_train_loader, "num_workers", 0)), 0)
    kwargs = {
        "shuffle": True,
        "num_workers": source_workers,
        "pin_memory": True,
    }
    if source_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        **kwargs,
    )


def _uncertified_indices(
    model: DynaPatchModel,
    loader: DataLoader,
    device: torch.device,
    radius: float,
) -> list[int]:
    """Indices of `loader.dataset` whose prediction the bound does NOT prove un-flippable.

    All regression lives here, by the certificate's own definition: a certified input cannot
    change class under any patch obeying the budget, so it cannot regress either. Uniform replay
    therefore spends most of its budget on inputs that provably cannot move. The mask depends only
    on the FROZEN classifier and the base logits -- never on the patch -- and is fixed before the
    first optimizer step, so using it is not tuning on an evaluation signal.

    Criterion is the ALL scope of scripts/analyze_certified.py: certified iff
    `z_a - z_b > radius * ||w_b - w_a||_2` for every b != a.

    `W` is taken from `hypernet.basis`, whose rows are `d logit_c / d patch` of the frozen head
    (see _install_logit_basis). That is deliberately the SAME matrix the certificate is computed
    from, so the split here and the numbers in section 5.1 cannot drift apart.
    """
    basis = getattr(model.hypernet, "basis", None)
    if basis is None or not bool(torch.any(basis != 0)):
        raise RuntimeError(
            "clean_replay_uncertified_only requires hypernet_style='bank_coef' with the logit "
            "basis already installed -- it reuses that basis as the classifier matrix."
        )
    W = basis.detach().to(device).float()                              # [C, patch_dim]
    D = torch.cdist(W.unsqueeze(0), W.unsqueeze(0), p=2)[0]            # [C, C]
    D.fill_diagonal_(0.0)

    dec = model.decomposition
    patch_dim = W.size(1)
    keep: list[int] = []
    offset = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = (batch[0] if isinstance(batch, (list, tuple)) else batch["input"]).to(device)
            deep = dec.extract_deep(dec.extract_shallow(inputs))
            zero = torch.zeros(deep.size(0), patch_dim, device=device)
            z = dec.classify(deep, zero)                               # [batch, C] base logits
            pred = z.argmax(1)                                         # [batch]
            za = z.gather(1, pred[:, None])                            # [batch, 1]
            slack = (za - z) - radius * D[pred]                        # [batch, C]
            slack.scatter_(1, pred[:, None], 1.0)                      # own class never binds
            certified = (slack > 0).all(1)                             # [batch]
            keep.extend((offset + torch.nonzero(~certified).flatten().cpu()).tolist())
            offset += inputs.size(0)
    if was_training:
        model.train()
    return keep


def _build_clean_replay_loader(
    clean_train_loader: DataLoader,
    subset_size: int | None,
    batch_size: int | None,
    seed: int,
    restrict_indices: list[int] | None = None,
) -> DataLoader:
    """Build a lightweight clean replay loader for regularizing patch behavior on clean samples.

    `restrict_indices` narrows the sampling pool before the subset draw. Passing the uncertified
    indices concentrates the same replay budget on the only inputs that can regress; passing a
    random subset of equal size is the control arm that separates "the certificate picked the
    right inputs" from "the pool merely got smaller" (note/PREREG_20260801.md).
    """
    dataset = clean_train_loader.dataset
    if restrict_indices is not None:
        dataset = Subset(dataset, restrict_indices)
    dataset_size = len(dataset)
    if subset_size is not None and subset_size > 0 and subset_size < dataset_size:
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(dataset_size, generator=generator).tolist()
        dataset = Subset(dataset, perm[:subset_size])

    resolved_batch_size = batch_size if batch_size is not None and batch_size > 0 else clean_train_loader.batch_size
    source_workers = max(int(getattr(clean_train_loader, "num_workers", 0)), 0)
    kwargs = {
        "shuffle": True,
        "num_workers": source_workers,
        "pin_memory": True,
    }
    if source_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        **kwargs,
    )


def run_stage3_experiment(cfg: DictConfig) -> str:
    """Run a complete stage-3 repair experiment from a resolved Hydra config."""
    validate_stage3_config(cfg)
    runner = ExperimentRunner(cfg)
    record = runner.bootstrap()
    print(f"[Start] experiment={cfg.experiment.id} stage={cfg.experiment.stage}", flush=True)

    device = resolve_runtime_device(cfg)
    print(f"[Start] resolved_device={device}", flush=True)
    backbone, model, dataloaders, _unused_loss_fn = build_stage3_bundle(cfg, device)
    fixed_adv_cfg = cfg.data.get("fixed_adv_cache")
    fixed_adv_cache_enabled = fixed_adv_cfg is not None and bool(fixed_adv_cfg.get("enabled", False))
    method_name = str(cfg.method.get("name", "hypernet_only"))
    use_prototype_bank = bool(getattr(model, "prototype_bank", None) is not None)
    clean_monitor_size_cfg = cfg.evaluation.get("clean_monitor_size")
    clean_monitor_size = None if clean_monitor_size_cfg is None else int(clean_monitor_size_cfg)
    clean_monitor_loader = _build_clean_monitor_loader(
        clean_eval_loader=dataloaders["clean_eval"],
        monitor_size=clean_monitor_size,
        seed=int(cfg.seed),
    )
    bug_eval_monitor_size_cfg = cfg.evaluation.get("bug_eval_monitor_size")
    bug_eval_monitor_size = None if bug_eval_monitor_size_cfg is None else int(bug_eval_monitor_size_cfg)
    bug_eval_monitor_loader = _build_bug_eval_monitor_loader(
        bug_eval_loader=dataloaders["bug_eval"],
        monitor_size=bug_eval_monitor_size,
        seed=int(cfg.seed),
    )
    cert_clean_subset_size_cfg = cfg.loss.get("cert_clean_subset_size")
    cert_clean_subset_size = None if cert_clean_subset_size_cfg is None else int(cert_clean_subset_size_cfg)
    cert_clean_batch_size_cfg = cfg.loss.get("cert_clean_batch_size")
    cert_clean_batch_size = None if cert_clean_batch_size_cfg is None else int(cert_clean_batch_size_cfg)
    cert_clean_loader = _build_cert_train_loader(
        clean_train_loader=dataloaders["clean_train"],
        subset_size=cert_clean_subset_size,
        batch_size=cert_clean_batch_size,
        seed=int(cfg.seed),
    )
    clean_replay_subset_size_cfg = cfg.loss.get("clean_replay_subset_size")
    clean_replay_subset_size = None if clean_replay_subset_size_cfg is None else int(clean_replay_subset_size_cfg)
    clean_replay_batch_size_cfg = cfg.loss.get("clean_replay_batch_size")
    clean_replay_batch_size = None if clean_replay_batch_size_cfg is None else int(clean_replay_batch_size_cfg)
    clean_replay_loader = _build_clean_replay_loader(
        clean_train_loader=dataloaders["clean_train"],
        subset_size=clean_replay_subset_size,
        batch_size=clean_replay_batch_size,
        seed=int(cfg.seed) + 17,
    )
    print(
        "[Start] dataloaders ready "
        f"bug_train={_loader_num_samples(dataloaders['bug_train'])} "
        f"bug_eval={_loader_num_samples(dataloaders['bug_eval'])} "
        f"bug_eval_monitor={_loader_num_samples(bug_eval_monitor_loader)} "
        f"clean_eval={_loader_num_samples(dataloaders['clean_eval'])} "
        f"clean_monitor={_loader_num_samples(clean_monitor_loader)} "
        f"cert_clean={_loader_num_samples(cert_clean_loader)} "
        f"clean_replay={_loader_num_samples(clean_replay_loader)}",
        flush=True,
    )
    if method_name != "property_patch" and (str(cfg.method.get("route_mode", "distance")) != "always_on" or use_prototype_bank):
        _seed_router_support(model, backbone, dataloaders.get("bug_train_clean", dataloaders["bug_train"]), device)

    if str(cfg.model.get("hypernet_style", "plain")) == "bank_coef":
        # basis_mode=random_orthonormal is the control for the logit basis, not an alternative
        # method: it holds rank, parameter count and magnitude fixed and removes only the
        # classifier information. See _install_random_basis.
        _install_logit_basis(model, dataloaders["bug_train"], device)
        if str(cfg.model.get("basis_mode", "logit_grad")) == "random_orthonormal":
            # overwrite AFTER the logit basis so the random one inherits its magnitude
            _install_random_basis(model, int(cfg.get("seed", 0)))

    # Certificate-guided replay (note/PREREG_20260801.md, experiment A). Rebuilt HERE rather than
    # at the original construction site because it needs the logit basis, which is installed just
    # above. The default path is untouched, so a run with the flag off is byte-identical to before.
    _replay_pool = str(cfg.loss.get("clean_replay_pool", "uniform"))
    if _replay_pool != "uniform":
        _radius = float(cfg.repair.get("l2_max"))
        _unc = _uncertified_indices(model, dataloaders["clean_train"], device, _radius)
        _n_total = len(dataloaders["clean_train"].dataset)
        if _replay_pool == "uncertified":
            _pool = _unc
        elif _replay_pool == "random_matched":
            # Control arm: same pool SIZE, membership chosen at random. Separates "the certificate
            # picked the right inputs" from "the pool merely got smaller". Without this arm the
            # pre-registered branch 1 cannot be claimed.
            _g = torch.Generator(); _g.manual_seed(int(cfg.seed) + 991)
            _pool = torch.randperm(_n_total, generator=_g)[:len(_unc)].tolist()
        else:
            raise ValueError(f"unsupported loss.clean_replay_pool: {_replay_pool!r}")
        if not _pool:
            raise RuntimeError(f"clean_replay_pool={_replay_pool!r} selected 0 of {_n_total} "
                               "clean-train samples; refusing to train on an empty replay pool")
        clean_replay_loader = _build_clean_replay_loader(
            clean_train_loader=dataloaders["clean_train"],
            subset_size=clean_replay_subset_size,
            batch_size=clean_replay_batch_size,
            seed=int(cfg.seed) + 17,
            restrict_indices=_pool,
        )
        print(f"[clean_replay] pool={_replay_pool} radius={_radius} "
              f"uncertified={len(_unc)}/{_n_total} ({len(_unc)/max(_n_total,1):.1%}) "
              f"-> replay pool {len(_pool)}, loader {_loader_num_samples(clean_replay_loader)}",
              flush=True)

    optimizer = torch.optim.AdamW(
        params=list(model.trainable_parameters()),
        lr=float(cfg.train_loop.lr),
        weight_decay=float(cfg.train_loop.weight_decay),
    )
    repair_loss_name = str(cfg.loss.get("repair_loss", "ce"))
    objective_name = str(cfg.loss.get("objective", "repair_only"))
    if repair_loss_name == "ce":
        if float(cfg.loss.get("lambda_safety_risk", 0.0)) > 0.0:
            risk_matrix_path = cfg.loss.get("safety_risk_matrix_path")
            if risk_matrix_path is None:
                raise ValueError("Safety-aware repair loss requires `loss.safety_risk_matrix_path`.")
            repair_loss_fn = SafetyAwareRepairClassificationLoss(
                risk_matrix_path=str(risk_matrix_path),
                lambda_safety_risk=float(cfg.loss.get("lambda_safety_risk", 0.0)),
                critical_class_weight=float(cfg.loss.get("critical_class_weight", 1.0)),
                critical_indices=_resolve_critical_indices(cfg),
            )
        else:
            repair_loss_fn = RepairClassificationLoss()
    elif repair_loss_name == "focal":
        if float(cfg.loss.get("lambda_safety_risk", 0.0)) > 0.0:
            raise ValueError("Safety-aware repair loss currently supports `repair_loss=ce` only.")
        repair_loss_fn = FocalRepairClassificationLoss(gamma=float(cfg.loss.get("focal_gamma", 2.0)))
    else:
        raise ValueError(f"Unsupported repair loss: {repair_loss_name}")
    robust_loss_fn = None
    cert_loss_fn = None
    cert_scope = str(cfg.loss.get("cert_scope", "clean"))
    field_loss_fn = None
    decision_ball_loss_fn = None
    if objective_name not in {"repair_only", "anchor_cache_ce"}:
        robust_loss_fn = RobustRepairLoss(
            epsilon=float(cfg.fault.epsilon),
            steps=int(cfg.fault.steps),
            step_size=float(cfg.loss.get("pgd_step_size", cfg.fault.epsilon / max(cfg.fault.steps // 2, 1))),
        )
    if float(cfg.loss.get("lambda_field", 0.0)) > 0.0:
        field_loss_fn = PatchConsistencyFieldLoss(
            epsilon=float(cfg.loss.get("field_epsilon", cfg.fault.epsilon)),
            steps=int(cfg.loss.get("field_steps", cfg.fault.steps)),
            step_size=_optional_float(cfg.loss.get("field_step_size")),
            compare=str(cfg.loss.get("field_compare", "gated_patch")),
        )
    if float(cfg.loss.get("lambda_decision_ball", 0.0)) > 0.0:
        decision_ball_variant = str(cfg.loss.get("decision_ball_variant", "margin_kl"))
        if decision_ball_variant == "functional_margin":
            decision_ball_loss_fn = FunctionalRepairBallLoss(
                epsilon=float(cfg.loss.get("decision_ball_epsilon", cfg.fault.epsilon)),
                steps=int(cfg.loss.get("decision_ball_steps", cfg.fault.steps)),
                step_size=_optional_float(cfg.loss.get("decision_ball_step_size")),
                target_margin=float(cfg.loss.get("decision_ball_target_margin", 0.0)),
                attack_objective=str(cfg.loss.get("decision_ball_attack_objective", "margin")),
                num_random_samples=int(cfg.loss.get("decision_ball_random_samples", 4)),
                clean_weight=float(cfg.loss.get("decision_ball_clean_weight", 1.0)),
                worst_case_weight=float(cfg.loss.get("decision_ball_worst_case_weight", 1.0)),
                random_weight=float(cfg.loss.get("decision_ball_random_weight", 1.0)),
            )
        else:
            decision_ball_loss_fn = PatchedDecisionBallLoss(
                epsilon=float(cfg.loss.get("decision_ball_epsilon", cfg.fault.epsilon)),
                steps=int(cfg.loss.get("decision_ball_steps", cfg.fault.steps)),
                step_size=_optional_float(cfg.loss.get("decision_ball_step_size")),
                consistency_weight=float(cfg.loss.get("decision_ball_consistency_weight", 1.0)),
                detach_target=bool(cfg.loss.get("decision_ball_detach_target", True)),
                target_margin=float(cfg.loss.get("decision_ball_target_margin", 0.0)),
                attack_objective=str(cfg.loss.get("decision_ball_attack_objective", "margin")),
                variant=decision_ball_variant,
            )
    if float(cfg.loss.get("lambda_cert", 0.0)) > 0.0:
        cert_variant = str(cfg.loss.get("cert_variant", "static_patch_guard"))
        cert_margin = float(cfg.loss.get("cert_margin", 0.0))
        if cert_variant == "static_patch_guard":
            if cert_scope != "clean":
                raise ValueError("`static_patch_guard` only supports `loss.cert_scope=clean`.")
            if epsilon_max is None:
                raise ValueError("Certification loss requires a finite `repair.epsilon_max` bound.")
            cert_loss_fn = CertificationGuardLoss(
                epsilon_max=epsilon_max,
                margin=cert_margin,
            )
        elif cert_variant == "dynamic_ibp":
            if cert_scope == "clean":
                cert_loss_fn = DynamicIBPCertificationLoss(margin=cert_margin)
            elif cert_scope == "bug":
                cert_loss_fn = BugSideDynamicIBPCertificationLoss(margin=cert_margin)
            else:
                raise ValueError(f"Unsupported cert scope: {cert_scope}")
        else:
            raise ValueError(f"Unsupported cert variant: {cert_variant}")

    history: list[dict[str, float | int | None]] = []
    warmup_epochs = int(cfg.train_loop.get("warmup_epochs", 0))
    loss_normalization = str(cfg.loss.get("normalization", "none"))
    early_stop_metric = str(cfg.train_loop.get("early_stop_metric", "heldout_repaired"))
    early_stop_patience = int(cfg.train_loop.get("early_stop_patience", cfg.train_loop.epochs))
    early_stop_min_epochs = int(cfg.train_loop.get("early_stop_min_epochs", 1))
    best_epoch = 0
    best_score: tuple[float, ...] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_prediction_summary: dict[str, dict[str, object]] | None = None
    best_epoch_record: dict[str, float | int | None] | None = None
    best_post_robust_epoch = 0
    best_post_robust_score: tuple[float, ...] | None = None
    best_post_robust_state_dict: dict[str, torch.Tensor] | None = None
    best_post_robust_prediction_summary: dict[str, dict[str, object]] | None = None
    best_post_robust_epoch_record: dict[str, float | int | None] | None = None
    best_post_cert_epoch = 0
    best_post_cert_score: tuple[float, ...] | None = None
    best_post_cert_state_dict: dict[str, torch.Tensor] | None = None
    best_post_cert_prediction_summary: dict[str, dict[str, object]] | None = None
    best_post_cert_epoch_record: dict[str, float | int | None] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    two_stage_cfg = cfg.loss.get("two_stage")
    two_stage_enabled = two_stage_cfg is not None and bool(two_stage_cfg.get("enabled", False))
    two_stage_transition_epoch = int(two_stage_cfg.get("transition_epoch", 1)) if two_stage_enabled else 0
    two_stage_reset_on_transition = bool(two_stage_cfg.get("reset_early_stop_on_transition", True)) if two_stage_enabled else False
    log_every_n_batches_cfg = cfg.train_loop.get("log_every_n_batches")
    log_every_n_batches = None if log_every_n_batches_cfg is None else int(log_every_n_batches_cfg)
    eval_every_n_epochs = int(cfg.evaluation.get("eval_every_n_epochs", 1))
    full_eval_every_n_epochs = int(cfg.evaluation.get("full_eval_every_n_epochs", 0))
    full_eval_after_epoch = int(cfg.evaluation.get("full_eval_after_epoch", 0))

    for epoch in range(int(cfg.train_loop.epochs)):
        in_warmup = epoch < warmup_epochs
        scheduled_lambda_robust, scheduled_lambda_cert = _scheduled_loss_weights(cfg, epoch)
        scheduled_aux_weights = _scheduled_aux_loss_weights(cfg, epoch)
        active_lambda_robust = 0.0 if in_warmup else scheduled_lambda_robust
        active_lambda_cert = 0.0 if in_warmup else scheduled_lambda_cert
        active_lambda_field = 0.0 if in_warmup else float(scheduled_aux_weights["lambda_field"])
        active_lambda_decision_ball = 0.0 if in_warmup else float(scheduled_aux_weights["lambda_decision_ball"])
        active_lambda_patch_l2 = 0.0 if in_warmup else float(scheduled_aux_weights["lambda_patch_l2"])
        active_lambda_clean_replay = 0.0 if in_warmup else float(scheduled_aux_weights["lambda_clean_replay"])
        active_lambda_safety_risk = 0.0 if in_warmup else float(scheduled_aux_weights["lambda_safety_risk"])
        if isinstance(repair_loss_fn, SafetyAwareRepairClassificationLoss):
            repair_loss_fn.lambda_safety_risk = active_lambda_safety_risk
        early_stop_active, current_early_stop_metric, current_early_stop_patience, current_early_stop_min_epochs = (
            _two_stage_early_stop_state(cfg, epoch)
        )
        phase_label = _phase_label(in_warmup, active_lambda_robust, active_lambda_cert)
        if two_stage_enabled and not in_warmup:
            current_epoch = epoch + 1
            if current_epoch < two_stage_transition_epoch:
                phase_label = "repair_first"
            else:
                phase_label = "safety_tightening"
        print(f"[Epoch {epoch + 1}] train_start phase={phase_label}", flush=True)
        if objective_name == "anchor_cache_ce":
            train_result = train_repair_with_anchor_cache_ce(
                model=model,
                base_model=backbone,
                anchor_loader=dataloaders.get("bug_train_clean", dataloaders["bug_train"]),
                cache_loader=dataloaders["bug_train"],
                clean_loader=cert_clean_loader if active_lambda_cert > 0.0 and cert_scope != "bug" else None,
                optimizer=optimizer,
                loss_fn=repair_loss_fn,
                lambda_cache_repair=float(cfg.loss.get("lambda_cache_repair", 1.0)),
                cert_loss_fn=cert_loss_fn if active_lambda_cert > 0.0 else None,
                lambda_cert=active_lambda_cert,
                lambda_clean_replay=float(cfg.loss.get("lambda_clean_replay", 0.0)),
                cert_on_bug=(cert_scope == "bug" and active_lambda_cert > 0.0),
                cert_every_n_steps=int(cfg.loss.get("cert_every_n_steps", 1)),
                cert_clean_batch_limit=(
                    None
                    if cfg.loss.get("cert_clean_batch_limit") is None
                    else int(cfg.loss.get("cert_clean_batch_limit"))
                ),
                clean_replay_batch_limit=(
                    None
                    if cfg.loss.get("clean_replay_batch_limit") is None
                    else int(cfg.loss.get("clean_replay_batch_limit"))
                ),
                device=device,
                grad_clip_norm=float(cfg.train_loop.grad_clip_norm),
                log_every_n_batches=log_every_n_batches,
            )
        elif (
            (robust_loss_fn is None and cert_loss_fn is None and field_loss_fn is None and decision_ball_loss_fn is None)
            or in_warmup
            or (active_lambda_robust <= 0.0 and active_lambda_cert <= 0.0 and active_lambda_field <= 0.0 and active_lambda_decision_ball <= 0.0)
        ):
            train_result = train_repair_only(
                model=model,
                base_model=backbone,
                bug_loader=dataloaders["bug_train"],
                clean_loader=clean_replay_loader if float(cfg.loss.get("lambda_clean_replay", 0.0)) > 0.0 else None,
                optimizer=optimizer,
                loss_fn=repair_loss_fn,
                device=device,
                lambda_patch_l2=active_lambda_patch_l2,
                lambda_clean_replay=active_lambda_clean_replay,
                clean_replay_batch_limit=(
                    None
                    if cfg.loss.get("clean_replay_batch_limit") is None
                    else int(cfg.loss.get("clean_replay_batch_limit"))
                ),
                clean_replay_objective=str(cfg.loss.get("clean_replay_objective", "kl")),
                grad_clip_norm=float(cfg.train_loop.grad_clip_norm),
                log_every_n_batches=log_every_n_batches,
                bug_augment_views=int(cfg.loss.get("bug_augment_views", 0)),
                lambda_delta_consistency=float(cfg.loss.get("lambda_delta_consistency", 0.0)),
            )
        else:
            train_result = train_repair_with_dual_objectives(
                model=model,
                base_model=backbone,
                bug_loader=dataloaders["bug_train"],
                clean_loader=clean_replay_loader,
                optimizer=optimizer,
                repair_loss_fn=repair_loss_fn,
                robust_loss_fn=robust_loss_fn if active_lambda_robust > 0.0 else None,
                cert_loss_fn=cert_loss_fn if active_lambda_cert > 0.0 else None,
                field_loss_fn=field_loss_fn if active_lambda_field > 0.0 else None,
                decision_ball_loss_fn=decision_ball_loss_fn if active_lambda_decision_ball > 0.0 else None,
                lambda_repair=float(cfg.loss.get("lambda_repair", 1.0)),
                lambda_robust=active_lambda_robust,
                lambda_cert=active_lambda_cert,
                lambda_field=active_lambda_field,
                lambda_decision_ball=active_lambda_decision_ball,
                lambda_patch_l2=active_lambda_patch_l2,
                lambda_clean_replay=active_lambda_clean_replay,
                cert_on_bug=(cert_scope == "bug" and active_lambda_cert > 0.0),
                cert_every_n_steps=int(cfg.loss.get("cert_every_n_steps", 1)),
                cert_clean_batch_limit=(
                    None
                    if cfg.loss.get("cert_clean_batch_limit") is None
                    else int(cfg.loss.get("cert_clean_batch_limit"))
                ),
                clean_replay_batch_limit=(
                    None
                    if cfg.loss.get("clean_replay_batch_limit") is None
                    else int(cfg.loss.get("clean_replay_batch_limit"))
                ),
                loss_normalization=loss_normalization,
                device=device,
                grad_clip_norm=float(cfg.train_loop.grad_clip_norm),
                log_every_n_batches=log_every_n_batches,
            )
        print(f"[Epoch {epoch + 1}] train_done", flush=True)
        run_eval_this_epoch = _should_run_eval(epoch, eval_every_n_epochs)
        run_full_eval_this_epoch = _should_run_full_eval(
            epoch_index=epoch,
            total_epochs=int(cfg.train_loop.epochs),
            full_eval_every_n_epochs=full_eval_every_n_epochs,
            full_eval_after_epoch=full_eval_after_epoch,
        )
        eval_bug_loader = dataloaders["bug_eval"] if run_full_eval_this_epoch else bug_eval_monitor_loader
        eval_scope = "full" if run_full_eval_this_epoch else "monitor"

        if run_eval_this_epoch:
            print(f"[Epoch {epoch + 1}] bug_eval_start scope={eval_scope}", flush=True)
            bug_eval = evaluate_repair_model(
                model=model,
                base_model=backbone,
                loader=eval_bug_loader,
                loss_fn=repair_loss_fn,
                device=device,
                compute_rr=False,
            )
            print(f"[Epoch {epoch + 1}] bug_eval_done scope={eval_scope}", flush=True)
            print(f"[Epoch {epoch + 1}] clean_eval_start scope=monitor", flush=True)
            clean_eval = evaluate_repair_model(
                model=model,
                base_model=backbone,
                loader=clean_monitor_loader,
                loss_fn=repair_loss_fn,
                device=device,
                compute_rr=True,
            )
            print(f"[Epoch {epoch + 1}] clean_eval_done scope=monitor", flush=True)
            epoch_prediction_summary = _build_epoch_prediction_summary(
                model=model,
                backbone=backbone,
                bug_eval_loader=eval_bug_loader,
                clean_eval=clean_eval,
                device=device,
            )
        else:
            epoch_prediction_summary = best_prediction_summary if best_prediction_summary is not None else {
                "bug_eval": {
                    "patched_accuracy": 0.0,
                    "repaired": 0,
                    "repaired_rate": 0.0,
                    "regressed": 0,
                    "rr": 0.0,
                    "count": 0,
                    "base_accuracy": 0.0,
                    "regressed_rate": 0.0,
                    "mean_route_weight": 0.0,
                    "mean_patch_norm": 0.0,
                    "base_correct": 0,
                    "patched_correct": 0,
                },
                "clean_eval": {
                    "patched_accuracy": 0.0,
                    "rr": 0.0,
                    "regressed": 0,
                    "repaired": 0,
                    "repaired_rate": 0.0,
                    "patched_correct": 0,
                    "base_correct": 0,
                    "count": 0,
                    "base_accuracy": 0.0,
                    "regressed_rate": 0.0,
                    "mean_route_weight": 0.0,
                    "mean_patch_norm": 0.0,
                    "bug_eval_reference": 0.0,
                },
            }
            bug_eval = RepairEpochResult(loss=0.0, accuracy=0.0, num_samples=0, rr=None)
            clean_eval = RepairEpochResult(loss=0.0, accuracy=0.0, num_samples=0, rr=0.0)
        if decision_ball_loss_fn is not None and not fixed_adv_cache_enabled:
            adv_bug_eval = evaluate_repair_model_under_attack(
                model=model,
                base_model=backbone,
                loader=eval_bug_loader,
                device=device,
                epsilon=float(cfg.loss.get("decision_ball_epsilon", cfg.fault.epsilon)),
                steps=int(cfg.loss.get("decision_ball_steps", cfg.fault.steps)),
                step_size=_optional_float(cfg.loss.get("decision_ball_step_size")),
            )
            epoch_prediction_summary["bug_eval_adv"] = {
                "patched_accuracy": float(adv_bug_eval.accuracy),
                "repaired": int(round(float(adv_bug_eval.accuracy) * float(adv_bug_eval.num_samples))),
                "count": int(adv_bug_eval.num_samples),
            }
        epoch_bug_summary = epoch_prediction_summary["bug_eval"]
        epoch_clean_summary = epoch_prediction_summary["clean_eval"]
        epoch_adv_bug_summary = epoch_prediction_summary.get("bug_eval_adv")
        history.append(
            {
                "epoch": epoch + 1,
                "train_bug_loss": train_result.loss,
                "train_bug_acc": train_result.accuracy,
                "eval_bug_loss": bug_eval.loss,
                "rsr": float(epoch_bug_summary["patched_accuracy"]),
                "bug_acc": float(epoch_bug_summary["patched_accuracy"]),
                "clean_acc": float(epoch_clean_summary["patched_accuracy"]),
                "rr": float(epoch_clean_summary["rr"]),
                "repaired": int(epoch_bug_summary["repaired"]),
                "repaired_rate": float(epoch_bug_summary["repaired_rate"]),
                "heldout_adv_repaired": (
                    None if epoch_adv_bug_summary is None else int(epoch_adv_bug_summary["repaired"])
                ),
                "heldout_adv_repaired_rate": (
                    None if epoch_adv_bug_summary is None else float(epoch_adv_bug_summary["patched_accuracy"])
                ),
                "regressed": int(epoch_clean_summary["regressed"]),
                "clean_eval_scope": "monitor",
                "bug_eval_scope": eval_scope if run_eval_this_epoch else "skipped",
                "lambda_robust_active": active_lambda_robust,
                "lambda_cert_active": active_lambda_cert,
                "lambda_field_active": active_lambda_field,
                "lambda_decision_ball_active": active_lambda_decision_ball,
                "lambda_patch_l2_active": active_lambda_patch_l2,
                "lambda_clean_replay_active": active_lambda_clean_replay,
                "lambda_safety_risk_active": active_lambda_safety_risk,
                "train_robust_loss": train_result.robust_loss,
                "train_cert_loss": train_result.cert_loss,
                "train_field_loss": train_result.field_loss,
                "train_decision_ball_loss": train_result.decision_ball_loss,
                "train_patch_reg_loss": train_result.patch_reg_loss,
                "train_clean_replay_loss": train_result.clean_replay_loss,
                "clean_route_hit": train_result.clean_route_hit,
                "cert_margin": train_result.cert_margin,
                "phase": phase_label,
                "cert_scope": cert_scope,
            }
        )
        print(
            f"[Epoch {epoch + 1}] "
            f"phase={phase_label} "
            f"train_bug_loss={train_result.loss:.4f} "
            f"train_bug_acc={train_result.accuracy:.4f} "
            f"bug_acc={float(epoch_bug_summary['patched_accuracy']):.4f} "
            f"repaired={int(epoch_bug_summary['repaired'])} "
            f"adv_bug_acc={0.0 if epoch_adv_bug_summary is None else float(epoch_adv_bug_summary['patched_accuracy']):.4f} "
            f"clean_acc={float(epoch_clean_summary['patched_accuracy']):.4f} "
            f"rr={float(epoch_clean_summary['rr']):.4f} "
            f"lambda_robust={active_lambda_robust:.4f} "
            f"lambda_cert={active_lambda_cert:.4f} "
            f"lambda_patch_l2={active_lambda_patch_l2:.4f} "
            f"lambda_clean_replay={active_lambda_clean_replay:.4f} "
            f"lambda_safety_risk={active_lambda_safety_risk:.4f} "
            f"robust={0.0 if train_result.robust_loss is None else train_result.robust_loss:.4f} "
            f"cert={0.0 if train_result.cert_loss is None else train_result.cert_loss:.4f} "
            f"field={0.0 if train_result.field_loss is None else train_result.field_loss:.4f} "
            f"decision_ball={0.0 if train_result.decision_ball_loss is None else train_result.decision_ball_loss:.4f} "
            f"patch_reg={0.0 if train_result.patch_reg_loss is None else train_result.patch_reg_loss:.4f} "
            f"clean_replay={0.0 if train_result.clean_replay_loss is None else train_result.clean_replay_loss:.4f}"
            ,
            flush=True,
        )

        if two_stage_enabled and two_stage_reset_on_transition and (epoch + 1) == two_stage_transition_epoch:
            best_score = None
            best_epoch = 0
            best_state_dict = None
            best_prediction_summary = None
            best_epoch_record = None
            epochs_without_improvement = 0
            print(
                f"[TwoStage] transition_epoch={two_stage_transition_epoch} reset_early_stop_state=true",
                flush=True,
            )

        if run_eval_this_epoch and early_stop_active:
            current_score = _early_stop_key(current_early_stop_metric, epoch_prediction_summary)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(model.state_dict())
                best_prediction_summary = copy.deepcopy(epoch_prediction_summary)
                best_epoch_record = dict(history[-1])
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if run_eval_this_epoch and early_stop_active and active_lambda_robust > 0.0:
            if best_post_robust_score is None or current_score > best_post_robust_score:
                best_post_robust_score = current_score
                best_post_robust_epoch = epoch + 1
                best_post_robust_state_dict = copy.deepcopy(model.state_dict())
                best_post_robust_prediction_summary = copy.deepcopy(epoch_prediction_summary)
                best_post_robust_epoch_record = dict(history[-1])

        if run_eval_this_epoch and early_stop_active and active_lambda_cert > 0.0:
            if best_post_cert_score is None or current_score > best_post_cert_score:
                best_post_cert_score = current_score
                best_post_cert_epoch = epoch + 1
                best_post_cert_state_dict = copy.deepcopy(model.state_dict())
                best_post_cert_prediction_summary = copy.deepcopy(epoch_prediction_summary)
                best_post_cert_epoch_record = dict(history[-1])

        if (
            run_eval_this_epoch
            and early_stop_active
            and epoch + 1 >= current_early_stop_min_epochs
            and epochs_without_improvement >= current_early_stop_patience
        ):
            stopped_early = True
            print(
                f"[EarlyStop] metric={current_early_stop_metric} best_epoch={best_epoch} "
                f"patience={current_early_stop_patience} stopping_at_epoch={epoch + 1}"
                ,
                flush=True,
            )
            break

        runner.save_metrics(
            {
                "status": "running",
                "experiment_id": cfg.experiment.id,
                "stage": cfg.experiment.stage,
                "device": str(device),
                "artifact_root": record.artifact_root,
                "objective": str(cfg.loss.get("objective", "repair_only")),
                "history": history,
                "epoch_final": dict(history[-1]),
                "final": dict(history[-1]),
                "early_stopping": {
                    "metric": current_early_stop_metric,
                    "patience": current_early_stop_patience,
                    "min_epochs": current_early_stop_min_epochs,
                    "stopped_early": False,
                    "best_epoch": best_epoch,
                    "epochs_completed": len(history),
                    "two_stage_enabled": two_stage_enabled,
                    "two_stage_transition_epoch": two_stage_transition_epoch if two_stage_enabled else None,
                },
            }
        )

    epoch_final = history[-1] if history else {}
    metrics = {
        "status": "completed",
        "experiment_id": cfg.experiment.id,
        "stage": cfg.experiment.stage,
        "device": str(device),
        "artifact_root": record.artifact_root,
        "resolved_config_preview": OmegaConf.to_container(cfg.experiment, resolve=True),
        "objective": str(cfg.loss.get("objective", "repair_only")),
        "history": history,
        "epoch_final": epoch_final,
        "final": dict(epoch_final),
        "early_stopping": {
            "metric": current_early_stop_metric,
            "patience": current_early_stop_patience,
            "min_epochs": current_early_stop_min_epochs,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "two_stage_enabled": two_stage_enabled,
            "two_stage_transition_epoch": two_stage_transition_epoch if two_stage_enabled else None,
        },
    }
    runner.save_metrics(metrics)

    checkpoint_dir = Path(record.artifact_root) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "repair_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone_checkpoint_path": str(cfg.model.checkpoint_path),
            "experiment_id": str(cfg.experiment.id),
            "epoch": len(history),
        },
        checkpoint_path,
    )
    best_checkpoint_path = checkpoint_dir / "repair_best.pt"
    if best_state_dict is not None:
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "backbone_checkpoint_path": str(cfg.model.checkpoint_path),
                "experiment_id": str(cfg.experiment.id),
                "epoch": best_epoch,
                "early_stop_metric": early_stop_metric,
                "selection_scope": "global",
            },
            best_checkpoint_path,
        )

    best_post_robust_checkpoint_path = checkpoint_dir / "repair_best_post_robust.pt"
    if best_post_robust_state_dict is not None:
        torch.save(
            {
                "model_state_dict": best_post_robust_state_dict,
                "backbone_checkpoint_path": str(cfg.model.checkpoint_path),
                "experiment_id": str(cfg.experiment.id),
                "epoch": best_post_robust_epoch,
                "early_stop_metric": early_stop_metric,
                "selection_scope": "post_robust",
            },
            best_post_robust_checkpoint_path,
        )

    best_post_cert_checkpoint_path = checkpoint_dir / "repair_best_post_cert.pt"
    if best_post_cert_state_dict is not None:
        torch.save(
            {
                "model_state_dict": best_post_cert_state_dict,
                "backbone_checkpoint_path": str(cfg.model.checkpoint_path),
                "experiment_id": str(cfg.experiment.id),
                "epoch": best_post_cert_epoch,
                "early_stop_metric": early_stop_metric,
                "selection_scope": "post_cert",
            },
            best_post_cert_checkpoint_path,
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    prediction_dir = Path(record.artifact_root) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    bug_support_loader = torch.utils.data.DataLoader(
        dataloaders.get("bug_train_clean", dataloaders["bug_train"]).dataset,
        batch_size=int(cfg.evaluation.get("batch_size", cfg.train_loop.batch_size)),
        shuffle=False,
        num_workers=int(cfg.runtime.get("num_workers", 0)),
    )
    _final_clean_loader = dataloaders["clean_eval"]
    _final_clean_size_cfg = cfg.evaluation.get("final_clean_eval_size", None)
    if _final_clean_size_cfg is not None:
        _fcs = int(_final_clean_size_cfg)
        _ds = dataloaders["clean_eval"].dataset
        if _fcs < len(_ds):
            _idx = torch.randperm(len(_ds), generator=torch.Generator().manual_seed(0))[:_fcs].tolist()
            _final_clean_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(_ds, _idx),
                batch_size=dataloaders["clean_eval"].batch_size,
                shuffle=False,
                num_workers=int(cfg.runtime.get("num_workers", 0)),
            )
    prediction_splits = {
        "bug_support": bug_support_loader,
        "bug_eval": dataloaders["bug_eval"],
        "clean_eval": _final_clean_loader,
    }
    prediction_summary: dict[str, object] = {}
    for split_name, loader in prediction_splits.items():
        rows = collect_prediction_rows(
            model=model,
            base_model=backbone,
            loader=loader,
            device=device,
            split_name=split_name,
        )
        _write_prediction_csv(prediction_dir / f"{split_name}_predictions.csv", rows)
        prediction_summary[split_name] = _summarize_predictions(rows)

    print("[MT] running metamorphic consistency eval on bug_support (training bugs) ...", flush=True)
    train_mt_result = evaluate_mt_consistency(
        model=model,
        loader=bug_support_loader,
        device=device,
    )
    print(
        f"[MT] train_mt_all_pass={train_mt_result.all_pass_rate:.4f} "
        + " ".join(f"{k}={v:.4f}" for k, v in train_mt_result.mr_pass_rates.items()),
        flush=True,
    )

    final_prediction_metrics = {
        "bug_acc": float(prediction_summary["bug_eval"]["patched_accuracy"]),
        "rsr": float(prediction_summary["bug_eval"]["patched_accuracy"]),
        "repaired": int(prediction_summary["bug_eval"]["repaired"]),
        "repaired_rate": float(prediction_summary["bug_eval"]["repaired_rate"]),
        "clean_acc": float(prediction_summary["clean_eval"]["patched_accuracy"]),
        "rr": float(prediction_summary["clean_eval"]["rr"]),
        "regressed": int(prediction_summary["clean_eval"]["regressed"]),
        "train_bug_acc": float(prediction_summary["bug_support"]["patched_accuracy"]),
        "train_repaired": int(prediction_summary["bug_support"]["repaired"]),
        "train_repaired_rate": float(prediction_summary["bug_support"]["repaired_rate"]),
        "train_mt_all_pass_rate": train_mt_result.all_pass_rate,
        **{f"train_mt_{k}": v for k, v in train_mt_result.mr_pass_rates.items()},
    }

    if history:
        history[-1]["bug_acc"] = final_prediction_metrics["bug_acc"]
        history[-1]["rsr"] = final_prediction_metrics["rsr"]
        history[-1]["repaired"] = final_prediction_metrics["repaired"]
        history[-1]["repaired_rate"] = final_prediction_metrics["repaired_rate"]
        history[-1]["clean_acc"] = final_prediction_metrics["clean_acc"]
        history[-1]["rr"] = final_prediction_metrics["rr"]
        history[-1]["regressed"] = final_prediction_metrics["regressed"]
        history[-1]["train_bug_acc"] = final_prediction_metrics["train_bug_acc"]
        history[-1]["train_repaired"] = final_prediction_metrics["train_repaired"]
        history[-1]["train_repaired_rate"] = final_prediction_metrics["train_repaired_rate"]
        epoch_final = dict(history[-1])

    consistency_warnings: list[str] = []
    if history:
        if abs(float(epoch_final.get("bug_acc", 0.0)) - final_prediction_metrics["bug_acc"]) > 1.0e-9:
            consistency_warnings.append(
                "epoch_final.bug_acc disagrees with prediction_summary bug_eval patched_accuracy"
            )
        if abs(float(epoch_final.get("clean_acc", 0.0)) - final_prediction_metrics["clean_acc"]) > 1.0e-9:
            consistency_warnings.append(
                "epoch_final.clean_acc disagrees with prediction_summary clean_eval patched_accuracy"
            )
        if abs(float(epoch_final.get("rr", 0.0)) - final_prediction_metrics["rr"]) > 1.0e-9:
            consistency_warnings.append(
                "epoch_final.rr disagrees with prediction_summary clean_eval rr"
            )

    metrics["final"].update(final_prediction_metrics)

    runner.save_metrics(
        {
            **metrics,
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_post_robust_checkpoint_path": str(best_post_robust_checkpoint_path) if best_post_robust_state_dict is not None else None,
            "best_post_cert_checkpoint_path": str(best_post_cert_checkpoint_path) if best_post_cert_state_dict is not None else None,
            "best_epoch": best_epoch,
            "best_epoch_record": best_epoch_record,
            "best_prediction_summary": best_prediction_summary,
            "best_post_robust_epoch": best_post_robust_epoch if best_post_robust_state_dict is not None else None,
            "best_post_robust_epoch_record": best_post_robust_epoch_record,
            "best_post_robust_prediction_summary": best_post_robust_prediction_summary,
            "best_post_cert_epoch": best_post_cert_epoch if best_post_cert_state_dict is not None else None,
            "best_post_cert_epoch_record": best_post_cert_epoch_record,
            "best_post_cert_prediction_summary": best_post_cert_prediction_summary,
            "final": metrics["final"],
            "final_from_predictions": final_prediction_metrics,
            "prediction_summary": prediction_summary,
            "consistency_warnings": consistency_warnings,
        }
    )

    print(
        f"[Final] "
        f"train_bug_acc={final_prediction_metrics['train_bug_acc']:.4f} "
        f"train_mt_all={final_prediction_metrics['train_mt_all_pass_rate']:.4f} "
        f"bug_acc={final_prediction_metrics['bug_acc']:.4f} "
        f"repaired={final_prediction_metrics['repaired']} "
        f"clean_acc={final_prediction_metrics['clean_acc']:.4f} "
        f"rr={final_prediction_metrics['rr']:.4f}",
        flush=True,
    )
    print(f"Completed repair-only stage-3 run at: {record.artifact_root}", flush=True)
    print(f"Resolved config: {record.config_path}", flush=True)
    print(f"Metrics: {record.metrics_path}", flush=True)
    return record.artifact_root
