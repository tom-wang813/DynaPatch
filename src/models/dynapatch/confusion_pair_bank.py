"""Confusion-pair locality conditioning for a shared hypernetwork."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn


class ConfusionPairBank(nn.Module):
    """Build coarse locality groups from support-set confusion pairs."""

    def __init__(
        self,
        feature_dim: int,
        num_groups: int,
        num_classes: int,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        if num_groups <= 0:
            raise ValueError("ConfusionPairBank requires `num_groups > 0`.")

        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.random_seed = random_seed

        self.register_buffer("centroids", torch.zeros(num_groups, feature_dim))
        self.register_buffer("pair_table", torch.full((num_groups, 2), -1, dtype=torch.long))
        self.is_seeded = False

    @torch.no_grad()
    def seed_from_support(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        """Fit locality centroids from support features grouped by confusion pair."""
        if features.ndim != 2:
            raise ValueError("ConfusionPairBank expects flattened route features with shape [n, d].")
        if features.size(0) == 0:
            raise ValueError("ConfusionPairBank requires at least one support feature.")

        labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
        predictions_cpu = predictions.detach().to(device="cpu", dtype=torch.long)
        pair_sequence = [(int(y), int(y_hat)) for y, y_hat in zip(labels_cpu.tolist(), predictions_cpu.tolist())]
        pair_counts = Counter(pair_sequence)

        num_named_groups = min(max(self.num_groups - 1, 0), len(pair_counts))
        ranked_pairs = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        selected_pairs = [pair for pair, _count in ranked_pairs[:num_named_groups]]
        pair_to_group = {pair: group_id for group_id, pair in enumerate(selected_pairs)}
        fallback_group = self.num_groups - 1

        group_assignments = []
        for pair in pair_sequence:
            if pair in pair_to_group:
                group_assignments.append(pair_to_group[pair])
            elif self.num_groups == 1:
                group_assignments.append(0)
            else:
                group_assignments.append(fallback_group)
        assignment_tensor = torch.tensor(group_assignments, device=features.device, dtype=torch.long)

        global_mean = features.mean(dim=0)  # [d]
        feature_scale = float(features.std(unbiased=False).clamp_min(1e-6).item())
        generator = torch.Generator(device=features.device)
        generator.manual_seed(self.random_seed)

        centroids = torch.empty_like(self.centroids, device=features.device)
        pair_rows = torch.full_like(self.pair_table, -1, device=features.device)

        for group_id in range(self.num_groups):
            member_mask = assignment_tensor == group_id
            if bool(member_mask.any()):
                centroids[group_id] = features[member_mask].mean(dim=0)
            else:
                noise = torch.randn(self.feature_dim, generator=generator, device=features.device, dtype=features.dtype)
                centroids[group_id] = global_mean + 1e-3 * feature_scale * noise

        for group_id, pair in enumerate(selected_pairs):
            pair_rows[group_id, 0] = pair[0]
            pair_rows[group_id, 1] = pair[1]
        if self.num_groups > 1:
            pair_rows[fallback_group, 0] = -2
            pair_rows[fallback_group, 1] = -2

        self.centroids.copy_(centroids)
        self.pair_table.copy_(pair_rows)
        self.is_seeded = True

    @torch.no_grad()
    def lookup(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the nearest locality centroid and the assigned group ids."""
        if not self.is_seeded:
            context = torch.zeros(features.size(0), self.feature_dim, device=features.device, dtype=features.dtype)
            group_ids = torch.full((features.size(0),), -1, device=features.device, dtype=torch.long)
            return context, group_ids

        centroids = self.centroids.to(device=features.device, dtype=features.dtype)  # [k, d]
        dist = torch.cdist(features, centroids)  # [batch, k]
        group_ids = torch.argmin(dist, dim=1)  # [batch]
        context = centroids[group_ids]  # [batch, d]
        return context, group_ids
