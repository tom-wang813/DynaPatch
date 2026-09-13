"""Composable DynaPatch model."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import torch
import torch.nn as nn

from src.models.dynapatch.decomposition import BackboneDecomposition
from src.models.dynapatch.confusion_pair_bank import ConfusionPairBank
from src.models.dynapatch.fallback import FallbackPolicy
from src.models.dynapatch.hypernet import HyperNetworkPatchGenerator
from src.models.dynapatch.patch_operator import ResidualPatchOperator
from src.models.dynapatch.prototype_bank import PrototypeBank
from src.models.dynapatch.router import DistanceRouter


class DynaPatchModel(nn.Module):
    """Async or sync DynaPatch model assembled from reusable components."""

    def __init__(
        self,
        decomposition: BackboneDecomposition,
        router: DistanceRouter,
        hypernet: HyperNetworkPatchGenerator,
        patch_operator: ResidualPatchOperator,
        prototype_bank: PrototypeBank | ConfusionPairBank | None = None,
        fallback_policy: FallbackPolicy | None = None,
        async_mode: str = "parallel",
        route_mode: str = "distance",
    ) -> None:
        super().__init__()
        self.decomposition = decomposition
        self.router = router
        self.hypernet = hypernet
        self.patch_operator = patch_operator
        self.prototype_bank = prototype_bank
        self.fallback_policy = fallback_policy or FallbackPolicy("zero")
        self.async_mode = async_mode
        self.route_mode = route_mode
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stream_patch: torch.cuda.Stream | None = None
        self.stream_deep: torch.cuda.Stream | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the decomposed DynaPatch forward pass."""
        return self.forward_with_intermediates(x)["logits"]

    def forward_with_intermediates(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the forward pass and expose intermediate tensors for training losses."""
        shallow_feat = self.decomposition.extract_shallow(x)
        route_feat = self.decomposition.router_features(shallow_feat)  # [batch, shallow_dim]

        if getattr(self.decomposition, "insertion_point", "cross_layer") == "same_layer":
            patch, route_weight, prototype_ids = self._forward_same_layer(shallow_feat, route_feat)
            if getattr(self.patch_operator, "budget", None) == "contract":
                raise RuntimeError(
                    "repair.budget='contract' is undefined for insertion_point='same_layer': the "
                    "region (w_a - w_b).d > -(z_a - z_b) assumes the patch enters a single affine "
                    "map. Use patch_site='last_affine' instead of failing open."
                )
            gated_patch = self.patch_operator(patch, route_weight)
            patched_shallow = self.decomposition.apply_insertion_patch(shallow_feat, gated_patch)
            deep_feat = self.decomposition.extract_deep(patched_shallow)
            logits = self.decomposition.classify(deep_feat, None)
        else:
            if getattr(self.hypernet, "condition_source", "shallow") == "deep":
                patch, route_weight, deep_feat, prototype_ids = self._forward_deep_conditioned(
                    shallow_feat, route_feat
                )
            elif x.is_cuda:
                patch, route_weight, deep_feat, prototype_ids = self._forward_cuda(shallow_feat, route_feat)
            else:
                patch, route_weight, deep_feat, prototype_ids = self._forward_cpu(shallow_feat, route_feat)

            # `contract` needs the UNPATCHED logits and the frozen final Linear to build this
            # input's admissible region. Both are already on hand here: the base logits are one
            # extra head evaluation on the same deep_feat, and the weight comes from the same
            # split `classify()` uses, so the two can never disagree about which map is the tail.
            if getattr(self.patch_operator, "budget", None) == "contract":
                _, last = self.decomposition.classifier_split()
                if last is None:
                    raise RuntimeError(
                        "repair.budget='contract' requires a single affine tail; this head has no "
                        "final Linear. It is only valid with repair.patch_site='last_affine'."
                    )
                base_logits = self.decomposition.classify(deep_feat, None)
                gated_patch = self.patch_operator(patch, route_weight, base_logits, last.weight)
            else:
                gated_patch = self.patch_operator(patch, route_weight)
            logits = self.decomposition.classify(deep_feat, gated_patch)
        return {
            "shallow_feat": shallow_feat,
            "route_feat": route_feat,
            "patch": patch,
            "route_weight": route_weight,
            "prototype_ids": prototype_ids,
            "deep_feat": deep_feat,
            "gated_patch": gated_patch,
            "logits": logits,
        }

    def _forward_same_layer(
        self,
        shallow_feat: torch.Tensor,
        route_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a patch at the insertion boundary and defer deep execution until after patching."""
        context, prototype_ids = self._prototype_context(route_feat)
        patch = self.hypernet(shallow_feat, context)
        route_weight = self._route(route_feat, patch)
        return patch, route_weight, prototype_ids

    def seed_support(self, x: torch.Tensor) -> None:
        """Seed the router support set from an input batch."""
        with torch.no_grad():
            shallow_feat = self.decomposition.extract_shallow(x)
            route_feat = self.decomposition.router_features(shallow_feat)
            self.router.seed_support(route_feat)
            if isinstance(self.prototype_bank, PrototypeBank):
                self.prototype_bank.seed_from_features(route_feat)

    def trainable_parameters(self):
        """Return only repair-module parameters."""
        for module in (self.router, self.hypernet, self.patch_operator, self.fallback_policy):
            yield from module.parameters()

    def patch_from_shallow(
        self,
        shallow_feat: torch.Tensor,
        route_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a patch outside the normal forward pass.

        Deployment builds its support bank and its direct patches straight from cached shallow
        features, so the conditioning source has to be honoured here as well as in `forward`.
        For the default shallow source this is exactly the call it replaces.
        """
        context, _prototype_ids = self._prototype_context(route_feat)
        if getattr(self.hypernet, "condition_source", "shallow") == "deep":
            deep_feat = self.decomposition.extract_deep(shallow_feat)  # [batch, patch_dim]
            return self.hypernet(deep_feat, context)
        return self.hypernet(shallow_feat, context)

    def _forward_deep_conditioned(
        self,
        shallow_feat: torch.Tensor,
        route_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Condition the patch on the representation it corrects.

        The hypernet reads `deep_feat` rather than the shallow map, so the deep path has to
        finish before the patch can be generated.  That removes the stream overlap the shallow
        variant relies on -- the accuracy gain, if any, is paid for in deploy latency.
        """
        context, prototype_ids = self._prototype_context(route_feat)
        deep_feat = self.decomposition.extract_deep(shallow_feat)  # [batch, patch_dim]
        patch = self.hypernet(deep_feat, context)
        if self.training:
            route_weight = self._route(route_feat, patch)
        else:
            route_weight = torch.ones(patch.size(0), 1, device=patch.device, dtype=patch.dtype)
        return patch, route_weight, deep_feat, prototype_ids

    def _forward_cpu(
        self,
        shallow_feat: torch.Tensor,
        route_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            context, prototype_ids = self._prototype_context(route_feat)
            patch = self.hypernet(shallow_feat, context)
            route_weight = self._route(route_feat, patch)
            deep_feat = self.decomposition.extract_deep(shallow_feat)
            return patch, route_weight, deep_feat, prototype_ids

        if self.async_mode == "none":
            context, prototype_ids = self._prototype_context(route_feat)
            patch_future: Future[torch.Tensor] = self.executor.submit(self.hypernet, shallow_feat, context)
            deep_feat = self.decomposition.extract_deep(shallow_feat)
            patch = patch_future.result()
            route_weight = torch.ones(patch.size(0), 1, device=patch.device)
            return patch, route_weight, deep_feat, prototype_ids

        future = self.executor.submit(self._background_patch_and_route, shallow_feat, route_feat)
        deep_feat = self.decomposition.extract_deep(shallow_feat)
        patch, route_weight, prototype_ids = future.result()
        return patch, route_weight, deep_feat, prototype_ids

    def _forward_cuda(
        self,
        shallow_feat: torch.Tensor,
        route_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            context, prototype_ids = self._prototype_context(route_feat)
            patch = self.hypernet(shallow_feat, context)
            route_weight = self._route(route_feat, patch)
            deep_feat = self.decomposition.extract_deep(shallow_feat)
            return patch, route_weight, deep_feat, prototype_ids

        if self.stream_patch is None:
            self.stream_patch = torch.cuda.Stream()
            self.stream_deep = torch.cuda.Stream()

        current_stream = torch.cuda.current_stream()
        self.stream_patch.wait_stream(current_stream)
        self.stream_deep.wait_stream(current_stream)

        route_weight: torch.Tensor | None = None
        patch: torch.Tensor | None = None
        prototype_ids: torch.Tensor | None = None
        context, prototype_ids = self._prototype_context(route_feat)

        if self.async_mode in {"parallel", "sync"}:
            with torch.cuda.stream(self.stream_patch):
                patch = self.hypernet(shallow_feat, context)
                if self.async_mode == "parallel":
                    route_weight = self._route(route_feat, patch)
        else:
            with torch.cuda.stream(self.stream_patch):
                patch = self.hypernet(shallow_feat, context)
                route_weight = torch.ones(patch.size(0), 1, device=patch.device)

        with torch.cuda.stream(self.stream_deep):
            deep_feat = self.decomposition.extract_deep(shallow_feat)

        current_stream.wait_stream(self.stream_patch)
        current_stream.wait_stream(self.stream_deep)

        if patch is None:
            raise RuntimeError("Patch branch failed to produce a patch tensor.")
        if route_weight is None:
            route_weight = self._route(route_feat, patch)
        if prototype_ids is None:
            prototype_ids = torch.full((patch.size(0),), -1, device=patch.device, dtype=torch.long)

        return patch, route_weight, deep_feat, prototype_ids

    def _background_patch_and_route(
        self,
        shallow_feat: torch.Tensor,
        route_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            context, prototype_ids = self._prototype_context(route_feat)
            patch = self.hypernet(shallow_feat, context)
            route_weight = self._route(route_feat, patch)
            return patch, route_weight, prototype_ids

    def _route(self, route_feat: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """Return routing weights under the configured route mode."""
        if self.route_mode == "always_on" or self.async_mode == "none":
            return torch.ones(patch.size(0), 1, device=patch.device)
        if self.route_mode == "soft_distance":
            route_weight, _ = self.router.soft_forward(route_feat)
            return route_weight
        route_weight, _ = self.router(route_feat)
        return route_weight

    def _prototype_context(self, route_feat: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return conditioning context and prototype ids for the shared hypernet."""
        if self.prototype_bank is None:
            prototype_ids = torch.full((route_feat.size(0),), -1, device=route_feat.device, dtype=torch.long)
            return None, prototype_ids
        context, prototype_ids = self.prototype_bank.lookup(route_feat)
        return context, prototype_ids
