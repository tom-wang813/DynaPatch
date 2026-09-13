"""Fixed adversarial cache builders for repair-oriented training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    random_start: bool,
) -> torch.Tensor:
    if random_start:
        delta = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(inputs)
    delta.requires_grad_(True)
    was_training = model.training
    model.eval()
    logits = model(inputs + delta)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
    delta = delta + step_size * grad.sign()
    delta = torch.clamp(delta, min=-epsilon, max=epsilon).detach()
    if was_training:
        model.train()
    return (inputs + delta).detach()


def _pgd_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
    random_start: bool,
) -> torch.Tensor:
    if random_start:
        delta = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(inputs)
    delta.requires_grad_(True)
    was_training = model.training
    model.eval()
    for _ in range(int(max(steps, 1))):
        logits = model(inputs + delta)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = delta + step_size * grad.sign()
        delta = torch.clamp(delta, min=-epsilon, max=epsilon).detach()
        delta.requires_grad_(True)
    if was_training:
        model.train()
    return (inputs + delta).detach()


class LazyFixedAdvLoader:
    """Load a fixed adversarial TensorDataset only when iteration actually begins."""

    def __init__(
        self,
        *,
        cache_path: Path,
        batch_size: int,
        shuffle: bool,
        num_samples: int,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_samples = int(num_samples)
        self._dataset: TensorDataset | None = None

    def _ensure_dataset(self) -> TensorDataset:
        if self._dataset is None:
            payload = torch.load(self.cache_path, map_location="cpu")
            self._dataset = TensorDataset(payload["inputs"], payload["labels"])
        return self._dataset

    @property
    def dataset(self) -> TensorDataset:
        return self._ensure_dataset()

    def __iter__(self):
        dataset = self._ensure_dataset()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        return iter(loader)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // max(self.batch_size, 1)


def build_fixed_adv_loader(
    *,
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    epsilon: float,
    pgd_rounds: int,
    pgd_steps: int,
    pgd_step_size: float,
    fgsm_samples: int,
    fgsm_step_size: float,
    include_clean: bool,
    cache_path: str | Path | None = None,
) -> DataLoader:
    expected_multiplier = int(bool(include_clean)) + int(max(pgd_rounds, 0)) + int(max(fgsm_samples, 0))
    expected_num_samples = len(loader.dataset) * max(expected_multiplier, 1)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return LazyFixedAdvLoader(
                cache_path=cache_path,
                batch_size=batch_size,
                shuffle=shuffle,
                num_samples=expected_num_samples,
            )

    cached_inputs: list[torch.Tensor] = []
    cached_labels: list[torch.Tensor] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if include_clean:
            cached_inputs.append(inputs.detach().cpu())
            cached_labels.append(labels.detach().cpu())

        for _ in range(int(max(pgd_rounds, 0))):
            adv_inputs = _pgd_attack(
                model=model,
                inputs=inputs,
                labels=labels,
                epsilon=float(epsilon),
                steps=int(max(pgd_steps, 1)),
                step_size=float(pgd_step_size),
                random_start=True,
            )
            cached_inputs.append(adv_inputs.detach().cpu())
            cached_labels.append(labels.detach().cpu())

        for _ in range(int(max(fgsm_samples, 0))):
            adv_inputs = _fgsm_attack(
                model=model,
                inputs=inputs,
                labels=labels,
                epsilon=float(epsilon),
                step_size=float(fgsm_step_size),
                random_start=True,
            )
            cached_inputs.append(adv_inputs.detach().cpu())
            cached_labels.append(labels.detach().cpu())

    if not cached_inputs:
        raise ValueError('Fixed adversarial cache generation produced no samples.')

    cached_inputs_tensor = torch.cat(cached_inputs, dim=0)
    cached_labels_tensor = torch.cat(cached_labels, dim=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "inputs": cached_inputs_tensor,
                "labels": cached_labels_tensor,
                "num_samples": int(cached_labels_tensor.size(0)),
            },
            cache_path,
        )
    dataset = TensorDataset(cached_inputs_tensor, cached_labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
