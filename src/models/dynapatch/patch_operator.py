"""Patch application operators."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualPatchOperator(nn.Module):
    """Apply a routed residual patch to the final representation.

    Two budgets are available, and which one is correct is an empirical finding, not a preference:

    `l_inf` (default, and what every run before 2026-07-30 used): `||d||_inf <= epsilon_max`.
    `l2`: `||d||_2 <= l2_max`, applied per sample by radial projection.

    WHY `l2` WAS ADDED. Measured across all 12 settings at the shipped eps=0.4, the *useful*
    perturbation magnitude is nearly constant -- 10 of 12 settings land in ||d||_2 in [6.46, 12.06],
    across 3 datasets, 4 architectures, and feature dims from 768 to 25088. The two exceptions,
    gtsrb/vgg16 (30.62) and tt100k/vgg16 (49.16), are *exactly* the two regression anomalies
    (Reg 0.0461 and 0.4033). The cause is that an L-inf ball of radius eps has L2 radius
    `eps*sqrt(d)`: 11-18 for d<=2048, which sits right against the useful band, but **63 for
    d=25088**, i.e. 3-6x too loose. The two settings whose training used that extra room were
    damaged by it. An eps sweep confirms the direction and brackets the band on BOTH sides:
    tightening eps moves those two monotonically better at repair (tt100k/vgg16 RR_seen
    0.205 -> 0.389 as eps goes 0.4 -> 0.05), while lisa/vgg16 -- same d, but already inside the band
    at 10.34 -- is monotonically *harmed* (0.929 -> 0.583, its norm falling to 5.28, below the band).
    So the band is a real operating range, and constraining the L2 norm directly makes the budget
    dimension-invariant by construction. See note/PAPER.md section C''.

    It also tightens the certificate. Under an L2 budget the dual-norm step becomes Cauchy-Schwarz
    rather than Hoelder: `max_{||d||_2 <= B} (w_b - w_a).d = B*||w_b - w_a||_2`, and since
    `||.||_2 <= ||.||_1`, the non-regression criterion is strictly tighter than
    `eps*||w_b - w_a||_1` and no longer grows with dimension. `scripts/analyze_certified.py` must be
    switched to the L2 form before it is applied to `l2`-budget runs -- the L-inf criterion is NOT
    valid for them.

    ADDITIVE BY DESIGN: with no new arguments the behaviour is bit-identical to the previous
    implementation, so the ~300 existing result cells stay comparable.

    CONTRACT NOTE (`budget='contract'`, added 2026-08-16).
    The two budgets above are direction-blind: they cap the magnitude of `d` without asking which
    way it points. The postcondition they are there to protect is not magnitude-shaped. For a bare
    affine tail `z = W h + b`, the patch preserves the base prediction `a` on this input exactly
    when

        (w_a - w_b) . d  >  -(z_a - z_b)        for every  b != a,                    (*)

    an open polyhedral cone shifted by the per-input margins -- the *verification-derived*
    admissible region. The L2 ball of radius `B` is its largest input-blind inscribed ball, so an
    L2 budget is a conservative inner approximation of (*), tight only in the worst direction.

    We project onto (*) along the ray, exactly as the `l2` branch shrinks along the ray:

        t* = min(1, min_{b : g_b . d < 0} (m_b - tau) / (-g_b . d)),   d <- t* d

    with `g_b = w_a - w_b`, `m_b = z_a - z_b > 0`, and `tau > 0` a strictness margin. This is
    closed form, needs no QP or iteration, only ever shrinks, is differentiable almost everywhere,
    and is SOUND by construction: `t=0` is always feasible because every `m_b > 0` at the base
    prediction, and `t*` is the exact largest feasible step along that ray.

    Two things it is at once: the constraint RQ2 compares against a generic norm bound, and the
    runtime authorization guard `g_allow(x, d)` of the gate/guard ablation -- the same closed form
    serves both, since it can be evaluated on any input at deploy time.

    ADDITIVE: `budget` still defaults to `l_inf` and `project()` still accepts a lone tensor, so
    every pre-existing call site and result cell is bit-identical.
    """

    def __init__(
        self,
        epsilon_max: float | None = None,
        budget: str = "l_inf",
        l2_max: float | None = None,
        contract_tau: float = 1e-3,
        contract_scope: str = "all",
    ) -> None:
        super().__init__()
        if budget not in {"l_inf", "l2", "contract"}:
            raise ValueError(f"Unsupported patch budget: {budget!r} "
                             "(use 'l_inf', 'l2' or 'contract')")
        if budget == "l2" and l2_max is None:
            raise ValueError("budget='l2' requires repair.l2_max to be set")
        if contract_scope not in {"all", "critical"}:
            raise ValueError(f"Unsupported contract scope: {contract_scope!r}")
        self.epsilon_max = epsilon_max
        self.budget = budget
        self.l2_max = l2_max
        self.contract_tau = float(contract_tau)
        self.contract_scope = contract_scope

    @staticmethod
    def contract_step(
        patch: torch.Tensor,
        base_logits: torch.Tensor,
        weight: torch.Tensor,
        tau: float = 1e-3,
        scope_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Largest feasible ray step `t*` in [0, 1] for the per-input contract (*). Shape [batch, 1].

        `base_logits` [batch, C] are the UNPATCHED logits, `weight` [C, d] the frozen final Linear.
        `scope_mask` [batch] selects the inputs the contract is asserted on; unselected rows get
        `t = 1` (unconstrained). Returned as a scalar per row so the caller shrinks along the ray.
        """
        flat = patch.flatten(1)                                        # [batch, d]
        a = base_logits.argmax(1)                                      # [batch]
        wa = weight.index_select(0, a)                                 # [batch, d]
        # g_b . d for every b, computed as (w_a - w_b).d = (w_a.d) - (W d)_b
        wd = flat @ weight.t()                                         # [batch, C]
        g_dot = (flat * wa).sum(1, keepdim=True) - wd                  # [batch, C]
        m = base_logits.gather(1, a[:, None]) - base_logits            # [batch, C] margins, m_a = 0
        idx = torch.arange(flat.shape[0], device=flat.device)
        # The own-class row is the identity 0 > 0 and must not bind.
        g_dot = g_dot.clone(); g_dot[idx, a] = 0.0
        m = m.clone(); m[idx, a] = float("inf")

        # Only constraints the step moves TOWARDS violating can bind.
        binding = g_dot < 0
        ratio = torch.where(binding, (m - tau).clamp_min(0.0) / (-g_dot).clamp_min(1e-12),
                            torch.full_like(g_dot, float("inf")))
        t = ratio.min(1, keepdim=True).values.clamp(max=1.0)           # [batch, 1]; only shrink
        if scope_mask is not None:
            t = torch.where(scope_mask.view(-1, 1), t, torch.ones_like(t))
        return t

    def project(
        self,
        patch: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        scope_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bound the patch if a limit is configured."""
        if self.budget == "contract":
            if base_logits is None or weight is None:
                raise ValueError("budget='contract' needs base_logits and the frozen head weight; "
                                 "the caller must pass them (see DynaPatchModel.forward).")
            t = self.contract_step(patch, base_logits, weight, self.contract_tau, scope_mask)
            return (patch.flatten(1) * t).view_as(patch)
        if self.budget == "l2":
            flat = patch.flatten(1)                                   # [batch, d]
            norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)     # [batch, 1]
            scale = (self.l2_max / norm).clamp(max=1.0)                # [batch, 1]; only shrink
            return (flat * scale).view_as(patch)
        if self.epsilon_max is None:
            return patch
        return torch.clamp(patch, min=-self.epsilon_max, max=self.epsilon_max)

    def forward(
        self,
        patch: torch.Tensor,
        route_weight: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        scope_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the projected routed patch."""
        projected = self.project(patch, base_logits, weight, scope_mask)
        return projected * route_weight
