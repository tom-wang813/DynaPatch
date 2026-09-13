"""Prototype-based conditioning for a single shared hypernetwork."""

from __future__ import annotations

import torch
import torch.nn as nn


class PrototypeBank(nn.Module):
    """Store support-derived feature prototypes and return nearest contexts."""

    def __init__(self, feature_dim: int, num_prototypes: int, num_iters: int = 25, random_seed: int = 42) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.num_iters = num_iters
        self.random_seed = random_seed
        self.register_buffer("prototypes", torch.zeros(num_prototypes, feature_dim))
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def seed_from_features(self, features: torch.Tensor) -> None:
        """Fit support prototypes with a simple k-means pass."""
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected support features of shape [n, {self.feature_dim}], got {tuple(features.shape)}."
            )
        if features.size(0) == 0:
            return

        device = features.device
        if features.size(0) <= self.num_prototypes:
            num_to_fill = features.size(0)
            self.prototypes.zero_()
            self.prototypes[:num_to_fill] = features[:num_to_fill]
            if num_to_fill < self.num_prototypes:
                # Avoid repeated identical prototypes when support coverage is sparse.
                num_missing = self.num_prototypes - num_to_fill
                base = features[:1].expand(num_missing, -1)
                noise_scale = max(float(features.std(unbiased=False).item()), 1.0e-6) * 1.0e-3
                noise = torch.randn_like(base) * noise_scale
                self.prototypes[num_to_fill:] = base + noise
            self.is_initialized.fill_(True)
            return

        generator = torch.Generator(device=device)
        generator.manual_seed(self.random_seed)
        perm = torch.randperm(features.size(0), generator=generator, device=device)
        centroids = features[perm[: self.num_prototypes]].clone()  # [k, d]

        for _ in range(self.num_iters):
            dist = torch.cdist(features, centroids)  # [n, k]
            assignments = torch.argmin(dist, dim=1)  # [n]
            updated = []
            for cluster_id in range(self.num_prototypes):
                mask = assignments == cluster_id
                if int(mask.sum().item()) == 0:
                    replacement_idx = int(torch.randint(features.size(0), (1,), generator=generator, device=device).item())
                    updated.append(features[replacement_idx])
                else:
                    updated.append(features[mask].mean(dim=0))
            centroids = torch.stack(updated, dim=0)  # [k, d]

        self.prototypes.copy_(centroids)
        self.is_initialized.fill_(True)

    def lookup(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest prototype context and prototype ids."""
        if not bool(self.is_initialized.item()):
            context = torch.zeros(features.size(0), self.feature_dim, device=features.device, dtype=features.dtype)
            prototype_ids = torch.full((features.size(0),), -1, device=features.device, dtype=torch.long)
            return context, prototype_ids

        prototypes = self.prototypes.to(device=features.device, dtype=features.dtype)  # [k, d]
        dist = torch.cdist(features, prototypes)  # [batch, k]
        prototype_ids = torch.argmin(dist, dim=1)  # [batch]
        context = prototypes[prototype_ids]  # [batch, d]
        return context, prototype_ids
