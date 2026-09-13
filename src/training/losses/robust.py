"""Robust repair loss based on PGD-style adversarial examples."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_pgd_adversarial_examples(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float | None = None,
) -> torch.Tensor:
    """Run a simple untargeted PGD inner loop around one bug batch."""
    epsilon = float(epsilon)
    steps = int(max(steps, 1))
    default_step = epsilon / max(steps // 2, 1)
    step = float(default_step if step_size is None else step_size)
    delta = torch.zeros_like(inputs, requires_grad=True)  # [batch, c, h, w]
    was_training = model.training
    model.eval()
    for _ in range(steps):
        logits = model(inputs + delta)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = delta + step * grad.sign()
        delta = torch.clamp(delta, min=-epsilon, max=epsilon).detach()
        delta.requires_grad_(True)
    if was_training:
        model.train()
    return (inputs + delta).detach()




def repaired_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """True-class logit margin against the strongest competing class."""
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    mask = F.one_hot(labels, num_classes=logits.size(1)).bool()
    competitor_logits = logits.masked_fill(mask, float('-inf')).max(dim=1).values
    return true_logits - competitor_logits


def repaired_margin_loss(logits: torch.Tensor, labels: torch.Tensor, target_margin: float = 0.0) -> torch.Tensor:
    """Penalize examples whose repaired margin falls below the target."""
    margin = repaired_margin(logits, labels)
    return F.softplus(target_margin - margin).mean()

class RobustRepairLoss(nn.Module):
    """Approximate the paper's robust repair objective with PGD."""

    def __init__(self, epsilon: float, steps: int, step_size: float | None = None) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.steps = int(max(steps, 1))
        default_step = self.epsilon / max(self.steps // 2, 1)
        self.step_size = float(default_step if step_size is None else step_size)

    def generate_adversarial_examples(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Run a simple untargeted PGD inner loop around one bug batch."""
        return generate_pgd_adversarial_examples(
            model=model,
            inputs=inputs,
            labels=labels,
            epsilon=self.epsilon,
            steps=self.steps,
            step_size=self.step_size,
        )

    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return robust loss and adversarial logits on one bug batch."""
        adv_inputs = self.generate_adversarial_examples(model=model, inputs=inputs, labels=labels)
        adv_logits = model(adv_inputs)
        return F.cross_entropy(adv_logits, labels), adv_logits


class PatchConsistencyFieldLoss(nn.Module):
    """Encourage the learned patch field to vary smoothly in a local PGD neighborhood."""

    def __init__(self, epsilon: float, steps: int, step_size: float | None = None, compare: str = "gated_patch") -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.steps = int(max(steps, 1))
        default_step = self.epsilon / max(self.steps // 2, 1)
        self.step_size = float(default_step if step_size is None else step_size)
        self.compare = str(compare)

    def forward(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adv_inputs = generate_pgd_adversarial_examples(
            model=model,
            inputs=inputs,
            labels=labels,
            epsilon=self.epsilon,
            steps=self.steps,
            step_size=self.step_size,
        )
        base_outputs = model.forward_with_intermediates(inputs)
        adv_outputs = model.forward_with_intermediates(adv_inputs)
        base_patch = base_outputs[self.compare]
        adv_patch = adv_outputs[self.compare]
        field_loss = F.mse_loss(adv_patch, base_patch)
        return field_loss, {
            "adv_inputs": adv_inputs,
            "base_patch": base_patch,
            "adv_patch": adv_patch,
        }


class PatchedDecisionBallLoss(nn.Module):
    """Harder seen-ball objective based on worst-case repaired margin."""

    def __init__(
        self,
        epsilon: float,
        steps: int,
        step_size: float | None = None,
        consistency_weight: float = 1.0,
        detach_target: bool = True,
        target_margin: float = 0.0,
        attack_objective: str = "margin",
        variant: str = "margin_kl",
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.steps = int(max(steps, 1))
        default_step = self.epsilon / max(self.steps // 2, 1)
        self.step_size = float(default_step if step_size is None else step_size)
        self.consistency_weight = float(consistency_weight)
        self.detach_target = bool(detach_target)
        self.target_margin = float(target_margin)
        self.attack_objective = str(attack_objective)
        self.variant = str(variant)

    def _generate_ball_adversarial_examples(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(inputs, requires_grad=True)
        was_training = model.training
        model.eval()
        for _ in range(self.steps):
            logits = model(inputs + delta)
            if self.attack_objective == "margin":
                loss = -repaired_margin(logits, labels).mean()
            else:
                loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            delta = delta + self.step_size * grad.sign()
            delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon).detach()
            delta.requires_grad_(True)
        if was_training:
            model.train()
        return (inputs + delta).detach()

    def forward(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adv_inputs = self._generate_ball_adversarial_examples(model=model, inputs=inputs, labels=labels)
        clean_outputs = model.forward_with_intermediates(inputs)
        adv_outputs = model.forward_with_intermediates(adv_inputs)
        clean_logits = clean_outputs["logits"]
        adv_logits = adv_outputs["logits"]
        clean_repair = repaired_margin_loss(clean_logits, labels, target_margin=self.target_margin)
        robust_repair = repaired_margin_loss(adv_logits, labels, target_margin=self.target_margin)
        target_logits = clean_logits.detach() if self.detach_target else clean_logits
        consistency = F.kl_div(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(target_logits, dim=1),
            reduction="batchmean",
        )

        if self.variant == "margin_kl":
            total = robust_repair + self.consistency_weight * consistency
        elif self.variant == "dual_margin":
            total = clean_repair + robust_repair
        elif self.variant == "dual_margin_kl":
            total = clean_repair + robust_repair + self.consistency_weight * consistency
        else:
            raise ValueError(f"Unsupported decision ball variant: {self.variant}")
        return total, {
            "adv_inputs": adv_inputs,
            "clean_logits": clean_logits,
            "adv_logits": adv_logits,
            "clean_repair": clean_repair.detach(),
            "robust_repair": robust_repair.detach(),
            "consistency": consistency.detach(),
            "clean_mean_margin": repaired_margin(clean_logits, labels).detach().mean(),
            "mean_margin": repaired_margin(adv_logits, labels).detach().mean(),
        }


class FunctionalRepairBallLoss(nn.Module):
    """Ball-level repair objective over clean, worst-case, and sampled in-ball points."""

    def __init__(
        self,
        epsilon: float,
        steps: int,
        step_size: float | None = None,
        target_margin: float = 0.0,
        attack_objective: str = "margin",
        num_random_samples: int = 4,
        clean_weight: float = 1.0,
        worst_case_weight: float = 1.0,
        random_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.steps = int(max(steps, 1))
        default_step = self.epsilon / max(self.steps // 2, 1)
        self.step_size = float(default_step if step_size is None else step_size)
        self.target_margin = float(target_margin)
        self.attack_objective = str(attack_objective)
        self.num_random_samples = int(max(num_random_samples, 0))
        self.clean_weight = float(clean_weight)
        self.worst_case_weight = float(worst_case_weight)
        self.random_weight = float(random_weight)

    def _generate_ball_adversarial_examples(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        delta = torch.zeros_like(inputs, requires_grad=True)  # [batch, c, h, w]
        was_training = model.training
        model.eval()
        for _ in range(self.steps):
            logits = model(inputs + delta)
            if self.attack_objective == "margin":
                loss = -repaired_margin(logits, labels).mean()
            else:
                loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            delta = delta + self.step_size * grad.sign()
            delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon).detach()
            delta.requires_grad_(True)
        if was_training:
            model.train()
        return (inputs + delta).detach()

    def _sample_ball_inputs(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if self.num_random_samples <= 0:
            return None
        random_delta = torch.empty(
            inputs.size(0),
            self.num_random_samples,
            *inputs.shape[1:],
            device=inputs.device,
            dtype=inputs.dtype,
        ).uniform_(-self.epsilon, self.epsilon)  # [batch, samples, c, h, w]
        return inputs.unsqueeze(1) + random_delta  # [batch, samples, c, h, w]

    def forward(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adv_inputs = self._generate_ball_adversarial_examples(model=model, inputs=inputs, labels=labels)
        clean_logits = model(inputs)
        adv_logits = model(adv_inputs)
        clean_repair = repaired_margin_loss(clean_logits, labels, target_margin=self.target_margin)
        worst_case_repair = repaired_margin_loss(adv_logits, labels, target_margin=self.target_margin)

        random_inputs = self._sample_ball_inputs(inputs)
        random_repair = torch.zeros((), device=inputs.device)
        random_mean_margin = torch.zeros((), device=inputs.device)
        if random_inputs is not None:
            flat_random_inputs = random_inputs.reshape(-1, *inputs.shape[1:])  # [batch*samples, c, h, w]
            flat_random_logits = model(flat_random_inputs)
            tiled_labels = labels.unsqueeze(1).expand(-1, self.num_random_samples).reshape(-1)  # [batch*samples]
            random_repair = repaired_margin_loss(
                flat_random_logits,
                tiled_labels,
                target_margin=self.target_margin,
            )
            random_mean_margin = repaired_margin(flat_random_logits, tiled_labels).detach().mean()

        total = (
            self.clean_weight * clean_repair
            + self.worst_case_weight * worst_case_repair
            + self.random_weight * random_repair
        )
        return total, {
            "adv_inputs": adv_inputs,
            "clean_logits": clean_logits,
            "adv_logits": adv_logits,
            "clean_repair": clean_repair.detach(),
            "worst_case_repair": worst_case_repair.detach(),
            "random_repair": random_repair.detach(),
            "clean_mean_margin": repaired_margin(clean_logits, labels).detach().mean(),
            "worst_case_mean_margin": repaired_margin(adv_logits, labels).detach().mean(),
            "random_mean_margin": random_mean_margin,
        }
