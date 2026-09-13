"""Arachne (Sohn, Kang & Yoo, TOSEM 2022) re-implemented for PyTorch classifiers.

The official artefact (https://github.com/coinse/arachne) is TensorFlow/Keras and is written
around its own Keras models and preprocessed index files, so it cannot load this project's
PyTorch checkpoints. This module re-implements the algorithm as specified in
`arachne/run_localise.py` and `arachne/search/de_vk.py`, keeping both of its defining parts --
the ones the previously-shipped `arachne_style` baseline dropped:

  bidirectional localisation  cost per weight is 2-D: forward impact (how much the weight
                              contributes to the output) and gradient loss (|dL/dw|). Each is
                              computed on the failing inputs and on the correct ones and combined
                              as `cost_failing / (1 + cost_correct)`; the selected weights are the
                              **Pareto front** of that 2-D cost, not a top-k of one scalar.
  differential evolution      population search over the selected weight VALUES (a candidate sets
                              them, it does not add a delta), mutation drawn from (0.5, 1),
                              recombination 0.7, fitness = loss_correct + patch_aggr *
                              loss_failing, early stop once the best fitness stops moving.

Everything runs on cached deep features: the backbone is frozen, so a candidate evaluation is one
matmul over the cache rather than a forward pass, which is what makes a population search
affordable over 12 settings.

This is still a re-implementation, not the authors' code -- label results "Arachne (our PyTorch
re-implementation)" and say so in any write-up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ArachneConfig:
    num_places: int = 64          # cap on the Pareto front (Arachne also caps the localised set)
    localise: str = "pareto"      # "pareto" = TOSEM'22 as published. "topn" = CAPACITY CONTROL,
                                  # NOT Arachne: the strict Pareto front is replaced by the top
                                  # `num_places` positions under the same bi-objective score. The
                                  # front is a subset of that ranking, so topn with num_places set
                                  # to the front's own size reproduces `pareto` exactly. Exists
                                  # only to separate "our method wins" from "our method may move
                                  # more parameters" -- any row produced with this MUST be
                                  # reported as a relaxation, never as Arachne.
    pop_size: int = 100
    max_iter: int = 100
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    patch_aggr: float = 10.0      # weight on the repair term; the paper sweeps this
    max_unchanged: int = 10       # early stop after this many iterations without improvement
    bound_scale: float = 2.0      # candidate range is init_value +/- bound_scale * |init_value|
    seed: int = 0


def _flat_index(shape: tuple[int, ...], flat: int) -> tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(flat, shape))


def _pareto_front(costs: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of a 2-D cost array (higher is better on both axes).

    Same sweep as run_localise.py: a point survives while some other point does not beat it on
    every axis.
    """
    idx = np.arange(costs.shape[0])
    c = costs.copy()
    i = 0
    while i < len(c):
        keep = np.any(c > c[i], axis=1)
        keep[i] = True
        idx = idx[keep]
        c = c[keep]
        i = np.sum(keep[:i]) + 1
    return idx


def bidirectional_localise(
    classifier: nn.Linear,
    feats_fail: torch.Tensor,
    labels_fail: torch.Tensor,
    feats_ok: torch.Tensor,
    labels_ok: torch.Tensor,
    num_places: int,
    mode: str = "pareto",
) -> list[tuple[int, ...]]:
    """Return weight positions on the Pareto front of (forward impact, gradient loss).

    With `mode="topn"` the Pareto step is skipped and the top `num_places` positions under the
    same bi-objective sum are returned instead. See ArachneConfig.localise: this is a capacity
    control, not the published algorithm.
    """
    W = classifier.weight  # [C, D]

    def costs_for(feats: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # forward impact: |w_cd| * mean_n |feat_nd| -- the weight's contribution to its logit
        fi = W.detach().abs() * feats.abs().mean(0, keepdim=True)  # [C, D]
        w = W.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(feats @ w.t() + classifier.bias.detach(), labels)
        (gl,) = torch.autograd.grad(loss, w)
        return fi, gl.abs()  # [C, D] each

    fi_f, gl_f = costs_for(feats_fail, labels_fail)
    if len(feats_ok) > 0:
        fi_o, gl_o = costs_for(feats_ok, labels_ok)
    else:
        fi_o = torch.zeros_like(fi_f)
        gl_o = torch.zeros_like(gl_f)

    # cost_failing / (1 + cost_correct): reward weights that matter for the failures without
    # mattering for what already works.
    combined = torch.stack([fi_f / (1.0 + fi_o), gl_f / (1.0 + gl_o)], dim=-1)  # [C, D, 2]
    flat = combined.reshape(-1, 2).cpu().numpy()

    # The front can be large; the sweep is O(n * front). Prescreen by the sum of both axes so the
    # Pareto step runs on a tractable candidate set, then keep the strongest survivors.
    order = np.argsort(-(flat[:, 0] + flat[:, 1]))
    if mode == "topn":
        # No Pareto step. The front is always a prefix-subset of this ranking, so this only ever
        # ADDS positions; num_places stops being a cap and becomes the actual capacity.
        front = order[:min(num_places, len(order))]
    else:
        prescreen = min(len(flat), max(4000, num_places * 50))
        cand = order[:prescreen]
        front = cand[_pareto_front(flat[cand])]
        front = front[np.argsort(-(flat[front, 0] + flat[front, 1]))][:num_places]
    return [_flat_index(tuple(W.shape), int(i)) for i in front]


def _fitness(
    W: torch.Tensor,
    bias: torch.Tensor,
    feats_ok: torch.Tensor,
    labels_ok: torch.Tensor,
    feats_fail: torch.Tensor,
    labels_fail: torch.Tensor,
    patch_aggr: float,
) -> float:
    """loss_correct + patch_aggr * loss_failing (de_vk.py::eval). Lower is better."""
    with torch.no_grad():
        l_ok = (F.cross_entropy(feats_ok @ W.t() + bias, labels_ok).item()
                if len(feats_ok) > 0 else 0.0)
        l_fail = F.cross_entropy(feats_fail @ W.t() + bias, labels_fail).item()
    return l_ok + patch_aggr * l_fail


def differential_evolution(
    classifier: nn.Linear,
    places: list[tuple[int, ...]],
    feats_ok: torch.Tensor,
    labels_ok: torch.Tensor,
    feats_fail: torch.Tensor,
    labels_fail: torch.Tensor,
    cfg: ArachneConfig,
) -> tuple[torch.Tensor, dict]:
    """Search the values of `places` in the classifier weight. Returns the patched weight."""
    rng = np.random.RandomState(cfg.seed)
    W0 = classifier.weight.detach().clone()
    bias = classifier.bias.detach()
    init = np.array([float(W0[p]) for p in places])
    span = cfg.bound_scale * np.maximum(np.abs(init), 1e-3)
    lo, hi = init - span, init + span

    def evaluate(vec: np.ndarray) -> float:
        W = W0.clone()
        for i, p in enumerate(places):
            W[p] = float(vec[i])
        return _fitness(W, bias, feats_ok, labels_ok, feats_fail, labels_fail, cfg.patch_aggr)

    pop = rng.uniform(lo, hi, size=(cfg.pop_size, len(places)))
    pop[0] = init  # keep the unpatched model in the population, as the reference implementation does
    fit = np.array([evaluate(ind) for ind in pop])
    best = int(np.argmin(fit))
    best_fit, unchanged, iters = fit[best], 0, 0

    for it in range(cfg.max_iter):
        iters = it + 1
        for j in range(cfg.pop_size):
            a, b, c = rng.choice([i for i in range(cfg.pop_size) if i != j], 3, replace=False)
            mu = rng.uniform(*cfg.mutation)
            donor = np.clip(pop[a] + mu * (pop[b] - pop[c]), lo, hi)
            forced = rng.randint(len(places))
            cross = (rng.random_sample(len(places)) < cfg.recombination)
            cross[forced] = True
            trial = np.where(cross, donor, pop[j])
            f = evaluate(trial)
            if f < fit[j]:
                pop[j], fit[j] = trial, f
        cur = float(fit.min())
        if cur < best_fit - 1e-9:
            best_fit, unchanged = cur, 0
        else:
            unchanged += 1
            if unchanged >= cfg.max_unchanged:
                break

    W = W0.clone()
    for i, p in enumerate(places):
        W[p] = float(pop[int(np.argmin(fit))][i])
    return W, {"iterations": iters, "best_fitness": best_fit, "num_places": len(places)}


def repair(
    classifier: nn.Linear,
    feats_fail: torch.Tensor,
    labels_fail: torch.Tensor,
    feats_ok: torch.Tensor,
    labels_ok: torch.Tensor,
    cfg: ArachneConfig,
) -> tuple[torch.Tensor, dict]:
    """Full Arachne pass: bidirectional localisation, then DE over the located weights."""
    places = bidirectional_localise(
        classifier, feats_fail, labels_fail, feats_ok, labels_ok, cfg.num_places, cfg.localise
    )
    if not places:
        return classifier.weight.detach().clone(), {"iterations": 0, "num_places": 0}
    return differential_evolution(
        classifier, places, feats_ok, labels_ok, feats_fail, labels_fail, cfg
    )
