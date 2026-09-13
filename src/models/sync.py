import torch
import torch.nn as nn


class PropertyPatchModel(nn.Module):
    """PatchPro-style indicator + patch-library baseline.

    Each support property gets its own dedicated patch module. At deployment, the
    indicator performs either an exact input-space membership test or a feature-
    space membership test, and the allocated patch set is summed as in PatchPro.
    """

    def __init__(
        self,
        base_model,
        model_type,
        feature_dim,
        num_experts,
        patch_hidden_dim: int = 0,
        patch_num_hidden_layers: int = 1,
        indicator_radius: float = 0.1,
        exact_indicator: bool = False,
        input_shape: tuple[int, int, int] = (3, 224, 224),
    ):
        super().__init__()
        self.model_type = model_type
        self.num_experts = int(num_experts)
        self.feature_dim = int(feature_dim)
        self.patch_hidden_dim = int(max(patch_hidden_dim, 0))
        self.patch_num_hidden_layers = int(max(patch_num_hidden_layers, 1))
        self.indicator_radius = float(indicator_radius)
        self.exact_indicator = bool(exact_indicator)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.input_dim = int(self.input_shape[0] * self.input_shape[1] * self.input_shape[2])

        if model_type == 'ResNet':
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.classifier = base_model.fc
        elif model_type == 'VGG':
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.classifier = base_model.classifier
        elif model_type == 'ViT':
            self.base_model = base_model
            self.classifier = base_model.heads
        elif model_type == 'Swin':
            self.features = base_model.features
            self.norm = base_model.norm
            self.permute = base_model.permute
            self.avgpool = base_model.avgpool
            self.flatten = base_model.flatten
            self.classifier = base_model.head
        elif model_type == 'ConvNeXt':
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.classifier = base_model.classifier
        elif model_type == 'MLP':
            self.feature_layers = base_model.feature_layers
            self.classifier = base_model.classifier
        else:
            raise NotImplementedError(f'Unsupported model_type: {model_type}')

        self._freeze_backbone()

        center_dim = self.input_dim if self.exact_indicator else self.feature_dim
        self.register_buffer('centers', torch.zeros(self.num_experts, center_dim))
        self.register_buffer('radii', torch.full((self.num_experts,), self.indicator_radius))
        self.register_buffer('active_mask', torch.zeros(self.num_experts, dtype=torch.bool))
        self.register_buffer('property_predictions', torch.full((self.num_experts,), -1, dtype=torch.long))

        self.patch_library = nn.ModuleList([self._build_patch_module() for _ in range(self.num_experts)])
        self._packed_patch_weights: list[torch.Tensor] = []
        self._packed_patch_biases: list[torch.Tensor] = []
        self._active_indices_cache: torch.Tensor | None = None
        self._active_predictions_cache: torch.Tensor | None = None

    def _freeze_backbone(self):
        modules = []
        for name in ('features', 'feature_layers', 'avgpool', 'classifier', 'base_model', 'norm', 'permute', 'flatten'):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, 'parameters'):
                modules.append(module)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _build_patch_module(self) -> nn.Module:
        if self.patch_num_hidden_layers <= 1 or self.patch_hidden_dim <= 0:
            return nn.Linear(self.feature_dim, self.feature_dim)

        dims = [self.feature_dim]
        dims.extend([self.patch_hidden_dim] * self.patch_num_hidden_layers)
        dims.append(self.feature_dim)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != self.feature_dim:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _get_features(self, x):
        if self.model_type == 'ResNet':
            x = self.features(x)
            return torch.flatten(x, 1)
        if self.model_type == 'VGG':
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)
        if self.model_type == 'ViT':
            x = self.base_model._process_input(x)
            n = x.shape[0]
            batch_class_token = self.base_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.base_model.encoder(x)
            return x[:, 0]
        if self.model_type == 'Swin':
            x = self.features(x)
            x = self.norm(x)
            x = self.permute(x)
            x = self.avgpool(x)
            return self.flatten(x)
        if self.model_type == 'ConvNeXt':
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)
        if self.model_type == 'MLP':
            if x.dim() > 2:
                x = torch.flatten(x, 1)
            return self.feature_layers(x)
        raise NotImplementedError(f'Unsupported model_type: {self.model_type}')

    def _classify(self, patched_feat: torch.Tensor):
        if self.model_type == 'ConvNeXt':
            return self.classifier(patched_feat.view(-1, patched_feat.size(1), 1, 1))
        if self.model_type == 'MLP':
            return self.classifier(patched_feat)
        return self.classifier(patched_feat)

    def _indicator_representation(self, x: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
        if self.exact_indicator:
            return torch.flatten(x, 1)  # [batch, input_dim]
        if feat is None:
            feat = self._get_features(x)
        return feat

    def seed_support(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            feat = self._get_features(x)
            indicator_repr = self._indicator_representation(x, feat)
            logits = self._classify(feat)
            predictions = torch.argmax(logits, dim=1)
            num_to_fill = min(self.num_experts, feat.size(0))
            if num_to_fill <= 0:
                return
            self.centers[:num_to_fill].copy_(indicator_repr[:num_to_fill])
            self.radii.fill_(self.indicator_radius)
            self.active_mask.zero_()
            self.active_mask[:num_to_fill] = True
            self.property_predictions.fill_(-1)
            self.property_predictions[:num_to_fill].copy_(predictions[:num_to_fill])
            self._refresh_active_patch_cache()

    def seed_properties(self, x: torch.Tensor) -> None:
        self.seed_support(x)

    def trainable_parameters(self):
        yield from self.patch_library.parameters()

    def _active_patch_metadata(self):
        if self._active_indices_cache is None or self._active_predictions_cache is None:
            active_indices = torch.nonzero(self.active_mask, as_tuple=False).view(-1)  # [E_active]
            active_predictions = self.property_predictions[self.active_mask]  # [E_active]
            self._active_indices_cache = active_indices
            self._active_predictions_cache = active_predictions
        active_indices = self._active_indices_cache
        active_predictions = self._active_predictions_cache
        return active_indices, active_predictions

    def _refresh_active_patch_cache(self) -> None:
        self._active_indices_cache = None
        self._active_predictions_cache = None
        self._packed_patch_weights = []
        self._packed_patch_biases = []

        active_indices = torch.nonzero(self.active_mask, as_tuple=False).view(-1)  # [E_active]
        if active_indices.numel() == 0:
            return

        active_modules = [self.patch_library[int(idx.item())] for idx in active_indices]
        if self.patch_num_hidden_layers <= 1 or self.patch_hidden_dim <= 0:
            weight = torch.stack([module.weight for module in active_modules], dim=0)  # [E_active, D_out, D_in]
            bias = torch.stack([module.bias for module in active_modules], dim=0)  # [E_active, D_out]
            self._packed_patch_weights = [weight]
            self._packed_patch_biases = [bias]
            return

        linear_groups: list[list[nn.Linear]] = []
        for module in active_modules:
            layers = [layer for layer in module if isinstance(layer, nn.Linear)]
            linear_groups.append(layers)

        num_linear_layers = len(linear_groups[0])
        for layer_idx in range(num_linear_layers):
            weight = torch.stack([layers[layer_idx].weight for layers in linear_groups], dim=0)  # [E_active, D_out, D_in]
            bias = torch.stack([layers[layer_idx].bias for layers in linear_groups], dim=0)  # [E_active, D_out]
            self._packed_patch_weights.append(weight)
            self._packed_patch_biases.append(bias)

    def _compute_indicator_and_allocation(
        self,
        indicator_repr: torch.Tensor,
        base_predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.active_mask.any():
            batch = indicator_repr.size(0)
            empty_mask = torch.zeros(batch, 0, device=indicator_repr.device, dtype=torch.bool)
            prototype_ids = torch.full((batch,), -1, device=indicator_repr.device, dtype=torch.long)
            min_dist = torch.full((batch,), float('inf'), device=indicator_repr.device)
            return empty_mask, empty_mask, prototype_ids, min_dist

        active_centers = self.centers[self.active_mask]  # [E, D]
        active_radii = self.radii[self.active_mask]      # [E]
        active_indices, active_predictions = self._active_patch_metadata()
        abs_diff = torch.abs(indicator_repr.unsqueeze(1) - active_centers.unsqueeze(0))  # [B, E_active, D]
        linf_dist = abs_diff.amax(dim=-1)  # [B, E_active]
        matches = linf_dist <= active_radii.view(1, -1)  # [B, E_active]
        has_match = matches.any(dim=1)  # [B]

        tau_matches = active_predictions.unsqueeze(0) == base_predictions.unsqueeze(1)  # [B, E_active]
        allocation_mask = torch.where(has_match.unsqueeze(1), matches, tau_matches)  # [B, E_active]
        has_allocation = allocation_mask.any(dim=1)
        first_selected_local = torch.argmax(allocation_mask.float(), dim=1)
        prototype_ids = torch.where(
            has_allocation,
            active_indices[first_selected_local],
            torch.full_like(first_selected_local, -1),
        )
        min_dist = linf_dist.min(dim=1).values
        return matches, allocation_mask, prototype_ids, min_dist

    def _apply_patch_library_vectorized(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Evaluate all active patch modules in parallel for one batch."""
        if not self.active_mask.any():
            return torch.zeros(x_flat.size(0), 0, self.feature_dim, device=x_flat.device, dtype=x_flat.dtype)

        if self.training or not self._packed_patch_weights:
            self._refresh_active_patch_cache()

        if self.patch_num_hidden_layers <= 1 or self.patch_hidden_dim <= 0:
            weights = self._packed_patch_weights[0]  # [E_active, D_out, D_in]
            biases = self._packed_patch_biases[0]  # [E_active, D_out]
            return torch.einsum('bi,eoi->beo', x_flat, weights) + biases.unsqueeze(0)  # [B, E_active, D_out]

        hidden = x_flat.unsqueeze(1).expand(-1, self._packed_patch_weights[0].size(0), -1)  # [B, E_active, D_in]
        num_linear_layers = len(self._packed_patch_weights)
        for layer_idx, (weights, biases) in enumerate(zip(self._packed_patch_weights, self._packed_patch_biases)):
            hidden = torch.einsum('bei,eoi->beo', hidden, weights) + biases.unsqueeze(0)  # [B, E_active, D_out]
            if layer_idx < num_linear_layers - 1:
                hidden = torch.relu(hidden)
        return hidden

    def compute_deployment_masks(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feat = self._get_features(x)
        base_logits = self._classify(feat)
        base_predictions = torch.argmax(base_logits, dim=1)
        return self.compute_deployment_masks_from_features(x, feat, base_logits, base_predictions)

    def compute_deployment_masks_from_features(
        self,
        x: torch.Tensor,
        feat: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        base_predictions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        indicator_repr = self._indicator_representation(x, feat)
        if base_logits is None:
            base_logits = self._classify(feat)
        if base_predictions is None:
            base_predictions = torch.argmax(base_logits, dim=1)
        matches, allocation_mask, prototype_ids, min_dist = self._compute_indicator_and_allocation(
            indicator_repr,
            base_predictions,
        )
        return {
            'feat': feat,
            'indicator_repr': indicator_repr,
            'base_logits': base_logits,
            'base_predictions': base_predictions,
            'matches': matches,
            'allocation_mask': allocation_mask,
            'prototype_ids': prototype_ids,
            'min_dist': min_dist,
        }

    def apply_allocation_mask(self, feat: torch.Tensor, allocation_mask: torch.Tensor) -> torch.Tensor:
        patch_outputs = self._apply_patch_library_vectorized(feat)  # [B, E_active, D]
        if patch_outputs.size(1) == 0:
            return torch.zeros_like(feat)
        weighted_patch = allocation_mask.to(dtype=patch_outputs.dtype).unsqueeze(-1) * patch_outputs  # [B, E_active, D]
        return weighted_patch.sum(dim=1)  # [B, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(x)['logits']

    def forward_with_intermediates(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        deployment = self.compute_deployment_masks(x)
        x_flat = deployment['feat']
        allocation_mask = deployment['allocation_mask']
        matches = deployment['matches']
        prototype_ids = deployment['prototype_ids']
        min_dist = deployment['min_dist']
        base_logits = deployment['base_logits']
        base_predictions = deployment['base_predictions']
        indicator_repr = deployment['indicator_repr']
        patch = self.apply_allocation_mask(x_flat, allocation_mask)

        route_weight = allocation_mask.any(dim=1).float().unsqueeze(1)
        gated_patch = patch
        logits = self._classify(x_flat + gated_patch)
        return {
            'shallow_feat': x_flat,
            'route_feat': indicator_repr,
            'patch': patch,
            'route_weight': route_weight,
            'prototype_ids': prototype_ids,
            'deep_feat': x_flat,
            'gated_patch': gated_patch,
            'logits': logits,
            'base_logits': base_logits,
            'base_predictions': base_predictions,
            'indicator_match': matches.any(dim=1) if matches.numel() > 0 else torch.zeros(x_flat.size(0), device=x_flat.device, dtype=torch.bool),
            'min_dist': min_dist,
        }
