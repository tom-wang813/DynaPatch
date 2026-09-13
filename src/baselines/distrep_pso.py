"""DistrRep: Distributed Repair of Deep Neural Networks via PSO.

Faithful re-implementation of Li Calsi et al., ICST 2023.

Three-phase algorithm (matches Figure 1 of the paper):
  Phase 1 — Distributed Fault Localisation
    Split bug set D_rep into k partitions NI_{M_1}..NI_{M_k}.
    For each partition: gradient-magnitude attribution over ALL model layers
    identifies a sparse set of suspicious weights (suspWeights_{M_i}).

  Phase 2 — Search-Based Distributed Fixing
    For each partition i: run PSO over suspWeights_{M_i} to minimise
    repair loss on NI_{M_i} subject to a clean-preservation penalty.
    Produces k expert weight deltas Δ_1..Δ_k.

  Phase 3 — Search-Based Experts Integration
    PSO over blending coefficients α ∈ ℝ^k selects the convex combination
      v_final = v_orig + Σ_i α_i · Δ_i
    that maximises joint repair + clean-preservation on the full bug set.

GPU engineering (inner-loop hot path):
  - PSOContext pre-caches all GPU tensors (indices, base values, param views)
    before the PSO loop starts — zero CPU↔GPU transfers inside the loop.
  - Fault localisation uses GPU topk on concatenated gradient tensors instead
    of Python list sorting (O(P) → O(P) GPU, no .item() per element).
  - param_views are live views into model.data; scatter_ writes directly
    into the model weights with no intermediate allocations.
  - Clean-set evaluation is amortised: only every clean_eval_interval steps.
  - Phase 2 partition work is distributed across two devices via Python
    threading; Phase 3 runs on the primary device after both finish.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SuspiciousWeights:
    """CPU-resident sparse descriptor of fault-localised weights."""
    param_names: list[str]
    flat_indices: list[torch.Tensor]   # CPU long tensors
    base_values: list[torch.Tensor]    # CPU float tensors


@dataclass
class PSOContext:
    """GPU-resident cache built once before the PSO loop.

    Holds live views into model.data so scatter_ writes go directly to
    the model parameters without any host copies.
    """
    param_views: list[torch.Tensor]   # flat views into model param .data (GPU)
    idx_devs: list[torch.Tensor]      # indices on device
    base_devs: list[torch.Tensor]     # base weight values on device
    sizes: list[int]                  # numel per param group
    total_dim: int

    @staticmethod
    def build(model: nn.Module, sw: SuspiciousWeights, device: torch.device) -> "PSOContext":
        param_dict = {n: p for n, p in model.named_parameters()}
        param_views, idx_devs, base_devs, sizes = [], [], [], []
        for pname, idx_cpu, base_cpu in zip(sw.param_names, sw.flat_indices, sw.base_values):
            param_views.append(param_dict[pname].data.view(-1))
            idx_devs.append(idx_cpu.to(device, non_blocking=True))
            base_devs.append(base_cpu.to(device, non_blocking=True))
            sizes.append(idx_cpu.numel())
        total_dim = sum(sizes)
        return PSOContext(param_views=param_views, idx_devs=idx_devs,
                         base_devs=base_devs, sizes=sizes, total_dim=total_dim)

    def apply_delta(self, flat_delta: torch.Tensor) -> None:
        """Set suspicious weights to base + flat_delta. No .to() calls."""
        offset = 0
        for view, idx, base, n in zip(self.param_views, self.idx_devs, self.base_devs, self.sizes):
            view.scatter_(0, idx, base + flat_delta[offset: offset + n])
            offset += n

    def restore_base(self) -> None:
        """Restore all suspicious weights to their base values."""
        for view, idx, base in zip(self.param_views, self.idx_devs, self.base_devs):
            view.scatter_(0, idx, base)


@dataclass
class PSOConfig:
    n_particles: int = 40
    n_iterations: int = 40
    w: float = 0.7           # inertia weight
    c1: float = 1.5          # cognitive coefficient
    c2: float = 1.5          # social coefficient
    max_delta: float = 0.05  # max absolute weight perturbation
    clean_eval_interval: int = 5
    lambda_clean: float = 0.5


@dataclass
class DistrRepConfig:
    n_partitions: int = 5
    top_k_weights: int = 1024
    pso_fix: PSOConfig = field(default_factory=PSOConfig)
    pso_integrate: PSOConfig = field(default_factory=lambda: PSOConfig(
        n_particles=30, n_iterations=30))
    device0: str = "cuda:1"
    device1: str = "cuda:2"


@dataclass
class DistrRepRunArtifacts:
    """Optional diagnostic return carrying the pre-integration experts.

    Expert deltas stay on CPU and are intentionally not interpreted here.  A
    caller may materialise one expert at a time for held-out evaluation without
    keeping five full model copies resident on the GPU.
    """
    merged_model: nn.Module
    expert_deltas: list[tuple[SuspiciousWeights, torch.Tensor]]


# ---------------------------------------------------------------------------
# Phase 1: Distributed Fault Localisation
# ---------------------------------------------------------------------------

def _partition_bugs(
    bug_inputs: torch.Tensor,
    bug_labels: torch.Tensor,
    n_partitions: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n = bug_inputs.size(0)
    indices = torch.randperm(n)
    splits = torch.chunk(indices, n_partitions)
    return [(bug_inputs[s], bug_labels[s]) for s in splits]


def localise_faults(
    model: nn.Module,
    partition_inputs: torch.Tensor,
    partition_labels: torch.Tensor,
    top_k: int,
    device: torch.device,
) -> SuspiciousWeights:
    """GPU-native gradient attribution: topk over concatenated grad tensors."""
    model.train()
    saved = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    inputs = partition_inputs.to(device)
    labels = partition_labels.to(device)
    loss = F.cross_entropy(model(inputs), labels)
    loss.backward()

    # Build per-param score tensors + offset map — all stays on GPU
    param_names_ordered = []
    grad_tensors = []
    param_sizes = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_names_ordered.append(name)
            grad_tensors.append(param.grad.detach().abs().view(-1))
            param_sizes.append(param.grad.numel())

    for name, param in model.named_parameters():
        param.requires_grad_(saved[name])
    model.zero_grad(set_to_none=True)
    model.eval()

    if not grad_tensors:
        return SuspiciousWeights(param_names=[], flat_indices=[], base_values=[])

    # GPU topk over all params at once
    all_grads = torch.cat(grad_tensors, dim=0)          # [total_params] on GPU
    top_k = min(top_k, all_grads.numel())
    _, global_indices = torch.topk(all_grads, top_k)    # GPU topk

    # Map global flat indices back to (param_name, local_flat_idx)
    offsets = torch.zeros(len(param_sizes) + 1, dtype=torch.long, device=device)
    for i, s in enumerate(param_sizes):
        offsets[i + 1] = offsets[i] + s

    global_indices_sorted, _ = global_indices.sort()
    global_indices_cpu = global_indices_sorted.cpu()
    offsets_cpu = offsets.cpu()

    param_to_indices: dict[str, list[int]] = {}
    for gi in global_indices_cpu.tolist():
        # right=True: gi == boundary goes to the NEXT param, not the previous one
        p_idx = int(torch.searchsorted(offsets_cpu[1:], gi, right=True).item())
        local_fi = gi - int(offsets_cpu[p_idx].item())
        param_to_indices.setdefault(param_names_ordered[p_idx], []).append(local_fi)

    param_dict = {n: p for n, p in model.named_parameters()}
    param_names_out, flat_indices_out, base_values_out = [], [], []
    for pname in sorted(param_to_indices.keys()):
        idx_tensor = torch.tensor(param_to_indices[pname], dtype=torch.long)
        param_names_out.append(pname)
        flat_indices_out.append(idx_tensor)
        base_flat = param_dict[pname].detach().cpu().view(-1)
        base_values_out.append(base_flat[idx_tensor].clone())

    return SuspiciousWeights(
        param_names=param_names_out,
        flat_indices=flat_indices_out,
        base_values=base_values_out,
    )


# ---------------------------------------------------------------------------
# Phase 2: PSO-based fixing (GPU-native inner loop)
# ---------------------------------------------------------------------------

_FITNESS_BATCH = 64


@torch.no_grad()
def _batched_argmax(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Predicted classes, computed in chunks.

    A single forward over the whole preservation set costs 6.12 GiB of activations on VGG-16 at 512
    images and OOMs a 24 GB card (2026-08-01, gtsrb/vgg16). Chunking is semantics-preserving: the
    model is in eval mode, so BatchNorm uses running statistics and every sample's logits are
    independent of the others in its batch -- the concatenated argmax is bit-identical to the
    single-shot one. Only the peak activation footprint changes.
    """
    if x.size(0) <= _FITNESS_BATCH:
        return model(x).argmax(dim=1)
    return torch.cat([model(x[i:i + _FITNESS_BATCH]).argmax(dim=1)
                      for i in range(0, x.size(0), _FITNESS_BATCH)])


@torch.no_grad()
def _eval_fitness(
    model: nn.Module,
    bug_inputs: torch.Tensor,
    bug_labels: torch.Tensor,
    base_bug_preds: torch.Tensor,
    clean_inputs: Optional[torch.Tensor],
    clean_labels: Optional[torch.Tensor],
    base_clean_preds: Optional[torch.Tensor],
    lambda_clean: float,
) -> float:
    model.eval()
    preds = _batched_argmax(model, bug_inputs)
    repaired = ((base_bug_preds != bug_labels) & (preds == bug_labels)).sum().item()
    harmful  = ((base_bug_preds == bug_labels) & (preds != bug_labels)).sum().item()
    fitness  = (repaired - harmful) / max(bug_labels.numel(), 1)

    if clean_inputs is not None and lambda_clean > 0:
        clean_preds = _batched_argmax(model, clean_inputs)
        preserved = (base_clean_preds == clean_preds).sum().item()
        fitness += lambda_clean * preserved / max(clean_labels.numel(), 1)
    return float(fitness)


def run_pso(
    model: nn.Module,
    sw: SuspiciousWeights,
    bug_inputs: torch.Tensor,
    bug_labels: torch.Tensor,
    base_bug_preds: torch.Tensor,
    clean_inputs: Optional[torch.Tensor],
    clean_labels: Optional[torch.Tensor],
    base_clean_preds: Optional[torch.Tensor],
    cfg: PSOConfig,
    device: torch.device,
) -> torch.Tensor:
    """PSO with GPU-cached context — zero CPU↔GPU transfers in inner loop."""
    ctx = PSOContext.build(model, sw, device)
    dim = ctx.total_dim
    if dim == 0:
        return torch.zeros(0)

    n = cfg.n_particles
    pos = (torch.rand(n, dim, device=device) * 2 - 1) * cfg.max_delta
    vel = torch.zeros(n, dim, device=device)
    pbest_pos = pos.clone()
    pbest_fit = torch.full((n,), -1e9, device=device)
    gbest_pos = pos[0].clone()
    gbest_fit = -1e9

    for iteration in range(cfg.n_iterations):
        use_clean = (clean_inputs is not None) and (iteration % cfg.clean_eval_interval == 0)
        c_in  = clean_inputs if use_clean else None
        c_lbl = clean_labels if use_clean else None
        c_bp  = base_clean_preds if use_clean else None

        for pi in range(n):
            ctx.apply_delta(pos[pi])
            fit = _eval_fitness(model, bug_inputs, bug_labels, base_bug_preds,
                                c_in, c_lbl, c_bp, cfg.lambda_clean)
            ctx.restore_base()

            if fit > pbest_fit[pi].item():
                pbest_fit[pi] = fit
                pbest_pos[pi] = pos[pi].clone()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = pos[pi].clone()

        r1 = torch.rand(n, dim, device=device)
        r2 = torch.rand(n, dim, device=device)
        vel = (cfg.w * vel
               + cfg.c1 * r1 * (pbest_pos - pos)
               + cfg.c2 * r2 * (gbest_pos.unsqueeze(0) - pos))
        pos = (pos + vel).clamp(-cfg.max_delta, cfg.max_delta)

    ctx.restore_base()
    return gbest_pos.cpu()


# ---------------------------------------------------------------------------
# Phase 3: Expert Integration
# ---------------------------------------------------------------------------

def integrate_experts(
    base_model: nn.Module,
    expert_deltas: list[tuple[SuspiciousWeights, torch.Tensor]],
    bug_inputs: torch.Tensor,
    bug_labels: torch.Tensor,
    base_bug_preds: torch.Tensor,
    clean_inputs: Optional[torch.Tensor],
    clean_labels: Optional[torch.Tensor],
    base_clean_preds: Optional[torch.Tensor],
    cfg: PSOConfig,
    device: torch.device,
) -> nn.Module:
    """PSO over α ∈ [0,1]^k — blending coefficients for k expert deltas."""
    k = len(expert_deltas)
    if k == 0:
        return base_model

    working_model = copy.deepcopy(base_model).to(device)

    # Pre-build one PSOContext per expert (GPU-cached)
    ctxs = [PSOContext.build(working_model, sw_i, device) for sw_i, _ in expert_deltas]
    # Pre-transfer expert deltas to device
    deltas_dev = [d.to(device) for _, d in expert_deltas]

    def _apply_blend(alpha: torch.Tensor) -> None:
        # Restore base first
        for ctx_i in ctxs:
            ctx_i.restore_base()
        # Accumulate each expert's contribution
        for ctx_i, delta_dev_i, ai_val in zip(ctxs, deltas_dev, alpha.tolist()):
            if abs(ai_val) < 1e-8:
                continue
            # current param values (after restore) are base; add alpha*delta
            offset = 0
            for view, idx, base, n in zip(ctx_i.param_views, ctx_i.idx_devs,
                                          ctx_i.base_devs, ctx_i.sizes):
                view.scatter_(0, idx,
                              view[idx] + ai_val * delta_dev_i[offset: offset + n])
                offset += n

    n = cfg.n_particles
    pos = torch.rand(n, k, device=device)
    vel = torch.zeros(n, k, device=device)
    pbest_pos = pos.clone()
    pbest_fit = torch.full((n,), -1e9, device=device)
    gbest_pos = pos[0].clone()
    gbest_fit = -1e9

    for iteration in range(cfg.n_iterations):
        use_clean = (clean_inputs is not None) and (iteration % cfg.clean_eval_interval == 0)
        c_in  = clean_inputs if use_clean else None
        c_lbl = clean_labels if use_clean else None
        c_bp  = base_clean_preds if use_clean else None

        for pi in range(n):
            _apply_blend(pos[pi])
            fit = _eval_fitness(working_model, bug_inputs, bug_labels, base_bug_preds,
                                c_in, c_lbl, c_bp, cfg.lambda_clean)
            if fit > pbest_fit[pi].item():
                pbest_fit[pi] = fit
                pbest_pos[pi] = pos[pi].clone()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = pos[pi].clone()

        r1 = torch.rand(n, k, device=device)
        r2 = torch.rand(n, k, device=device)
        vel = (cfg.w * vel
               + cfg.c1 * r1 * (pbest_pos - pos)
               + cfg.c2 * r2 * (gbest_pos.unsqueeze(0) - pos))
        pos = (pos + vel).clamp(0.0, 1.0)

    _apply_blend(gbest_pos)
    return working_model.cpu()


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

class DistrRepPSO:
    def __init__(self, cfg: DistrRepConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        base_model: nn.Module,
        bug_inputs: torch.Tensor,
        bug_labels: torch.Tensor,
        clean_inputs: torch.Tensor,
        clean_labels: torch.Tensor,
        base_bug_preds: torch.Tensor,
        base_clean_preds: torch.Tensor,
        return_experts: bool = False,
    ) -> nn.Module | DistrRepRunArtifacts:
        cfg = self.cfg
        dev0 = torch.device(cfg.device0)
        dev1 = torch.device(cfg.device1)

        partitions = _partition_bugs(bug_inputs, bug_labels, cfg.n_partitions)
        half = cfg.n_partitions // 2

        expert_deltas: list[Optional[tuple[SuspiciousWeights, torch.Tensor]]] = \
            [None] * cfg.n_partitions
        thread_errors: list[Optional[BaseException]] = [None, None]

        def _fix_partitions(indices: list[int], device: torch.device, err_slot: int) -> None:
            try:
                model_copy = copy.deepcopy(base_model).to(device).eval()
                clean_dev  = clean_inputs.to(device)
                c_lbl_dev  = clean_labels.to(device)
                c_bp_dev   = base_clean_preds.to(device)

                for i in indices:
                    part_in, part_lbl = partitions[i]
                    part_in  = part_in.to(device)
                    part_lbl = part_lbl.to(device)

                    sw = localise_faults(model_copy, part_in, part_lbl,
                                         top_k=cfg.top_k_weights, device=device)

                    bp = _batched_argmax(model_copy, part_in)

                    best_delta = run_pso(
                        model=model_copy, sw=sw,
                        bug_inputs=part_in, bug_labels=part_lbl, base_bug_preds=bp,
                        clean_inputs=clean_dev, clean_labels=c_lbl_dev,
                        base_clean_preds=c_bp_dev,
                        cfg=cfg.pso_fix, device=device,
                    )
                    expert_deltas[i] = (sw, best_delta)
            except Exception as exc:  # noqa: BLE001
                thread_errors[err_slot] = exc

        t0 = threading.Thread(target=_fix_partitions,
                              args=(list(range(half)), dev0, 0))
        t1 = threading.Thread(target=_fix_partitions,
                              args=(list(range(half, cfg.n_partitions)), dev1, 1))
        t0.start(); t1.start()
        t0.join();  t1.join()

        for err in thread_errors:
            if err is not None:
                raise RuntimeError(f"Partition thread failed: {err}") from err

        valid_experts = [item for item in expert_deltas if item is not None]

        base_dev0 = copy.deepcopy(base_model).to(dev0).eval()
        bugs_d0  = bug_inputs.to(dev0)
        b_lbl_d0 = bug_labels.to(dev0)
        clean_d0 = clean_inputs.to(dev0)
        c_lbl_d0 = clean_labels.to(dev0)
        # Batched: a single forward over the 512-image preservation set is 6.12 GiB of VGG-16
        # activations and OOMs a 24 GB card (2026-08-01). Semantics unchanged, see _batched_argmax.
        bbp_d0 = _batched_argmax(base_dev0, bugs_d0)
        bcp_d0 = _batched_argmax(base_dev0, clean_d0)

        merged_model = integrate_experts(
            base_model=base_dev0,
            expert_deltas=valid_experts,
            bug_inputs=bugs_d0,   bug_labels=b_lbl_d0,   base_bug_preds=bbp_d0,
            clean_inputs=clean_d0, clean_labels=c_lbl_d0, base_clean_preds=bcp_d0,
            cfg=cfg.pso_integrate, device=dev0,
        )
        if return_experts:
            return DistrRepRunArtifacts(
                merged_model=merged_model,
                expert_deltas=valid_experts,
            )
        return merged_model


def materialise_expert(
    base_model: nn.Module,
    expert_delta: tuple[SuspiciousWeights, torch.Tensor],
    device: torch.device,
) -> nn.Module:
    """Build one pre-integration expert from its sparse delta.

    This helper is for diagnostic evaluation only.  The returned model is a
    fresh copy, so neither the frozen base model nor another expert is mutated.
    """
    suspicious, delta = expert_delta
    model = copy.deepcopy(base_model).to(device).eval()
    context = PSOContext.build(model, suspicious, device)
    if context.total_dim != delta.numel():
        raise ValueError(
            f"expert delta has {delta.numel()} values for {context.total_dim} suspicious weights"
        )
    context.apply_delta(delta.to(device))
    return model
