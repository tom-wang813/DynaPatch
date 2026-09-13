"""Backbone decomposition into shallow, deep, and classifier stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneDecomposition:
    """Holds decomposed backbone modules and routing helpers."""

    model_type: str
    insertion_point: str = "cross_layer"
    # WHERE THE PATCH ENTERS THE CLASSIFIER HEAD. Certifiability is a property of THIS, not of the
    # architecture: all four backbones end in a single Linear, but the closed-form bound
    # `Delta z = W d` needs the map from the injection point to the logits to be a SINGLE AFFINE
    # map. Confirmed from checkpoints 2026-07-31:
    #   resnet50   fc (C,2048)                                   -- nothing after injection
    #   densenet121 classifier (C,1024)                           -- nothing after injection
    #   vgg16      .0(4096,25088) ReLU .3(4096,4096) ReLU .6(C,4096)  -- 3 Linears + 2 ReLUs after
    #   convnext   .0=LayerNorm(768) .1=Flatten .2=Linear(C,768)  -- injected BEFORE the LayerNorm
    # "deep_feat" (default) reproduces every pre-2026-07-31 run byte for byte.
    # "last_affine" injects at the input of the FINAL Linear instead, which makes the bound exact
    # on all four families. For ResNet/DenseNet the final Linear is the whole head, so the two
    # settings are provably identical there -- that equality is the smoke test, not an assumption.
    patch_site: str = "deep_feat"
    shallow: nn.Module | None = None
    deep: nn.Module | None = None
    classifier: nn.Module | None = None
    avgpool: nn.Module | None = None
    base_model: nn.Module | None = None
    features: nn.Module | None = None
    feature_layers: nn.Module | None = None
    norm: nn.Module | None = None
    permute: nn.Module | None = None
    flatten: nn.Module | None = None

    @property
    def patch_dim(self) -> int | None:
        """Return the hypernet output width required at the configured insertion point."""
        if self.insertion_point == "same_layer":
            if self.model_type in {"ResNet", "DenseNet", "VGG", "Swin", "ConvNeXt", "MLP"}:
                return self.shallow_feature_dim()
            if self.model_type == "ViT":
                return self.shallow_feature_dim()
        if self.patch_site == "last_affine":
            _, last = self.classifier_split()
            if last is not None:
                return int(last.in_features)
        if self.classifier is not None and hasattr(self.classifier, "in_features"):
            return int(self.classifier.in_features)
        return None

    def classifier_split(self) -> tuple[nn.Module | None, nn.Linear | None]:
        """Split the head into (everything before the final Linear, that final Linear).

        Returns `(None, head)` when the head IS a bare Linear -- ResNet's `fc` and DenseNet's
        `classifier` -- so the caller's `last_affine` path collapses to the existing behaviour
        with no special case. Returns `(None, None)` when no Linear can be found, which the
        caller must treat as "this backbone has no affine boundary" rather than silently
        proceeding; a silent fallback here is how 22 patch tensors were once dropped without an
        error (see scripts/analyze_gate_sweep.py).
        """
        head = self.classifier
        if head is None:
            return None, None
        if isinstance(head, nn.Linear):
            return None, head
        if not isinstance(head, nn.Sequential):
            return None, None
        idx = [i for i, m in enumerate(head) if isinstance(m, nn.Linear)]
        if not idx:
            return None, None
        cut = idx[-1]
        pre = nn.Sequential(*list(head)[:cut]) if cut > 0 else None
        return pre, head[cut]

    def shallow_feature_dim(self, probe_resolution: int = 224) -> int:
        """Infer the channel width at the insertion boundary.

        Every convolutional backbone here uses a dry forward rather than module introspection.
        Introspection walks the DIRECT children of the shallow Sequential, and at every stage
        boundary those children are themselves Sequentials that expose no channel attribute --
        so the walk falls through to the stem and silently returns the stem width. That is not
        hypothetical: it returned 96 for ConvNeXt (fixed 2026-07-27) and 64 for ResNet-50 split
        at layer1/2/3 (found 2026-07-29, which is why `insertion_point=same_layer` could never
        have run at those splits -- the patch came out 64-wide against a 1024-wide feature map).
        A forward pass cannot be wrong about the shape it produces.
        """
        if self.model_type in {"ResNet", "VGG", "DenseNet", "ConvNeXt"}:
            source = self.shallow
            if source is None and self.model_type == "ConvNeXt" and self.features is not None:
                source = self.features[:2]
            if source is None:
                raise RuntimeError(
                    f"Unable to infer {self.model_type} shallow feature dimension without a "
                    f"shallow module."
                )
            with torch.no_grad():
                dev = next(source.parameters()).device
                probe = source(torch.zeros(1, 3, probe_resolution, probe_resolution, device=dev))
            return int(probe.shape[1])  # [1, C, H, W] -> C
        if self.model_type == "Swin" and self.features is not None:
            first_stage = self.features[0]
            if hasattr(first_stage, "out_dim"):
                return int(first_stage.out_dim)
        if self.model_type == "ViT" and self.base_model is not None:
            return int(self.base_model.hidden_dim)
        if self.model_type == "MLP" and self.classifier is not None and hasattr(self.classifier, "in_features"):
            return int(self.classifier.in_features)
        raise NotImplementedError(f"Unsupported shallow feature dimension inference for {self.model_type}")

    def extract_shallow(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shallow features `z = f_in(x)`."""
        if self.model_type in {"ResNet", "DenseNet", "VGG"} and self.shallow is not None:
            return self.shallow(x)
        if self.model_type == "MLP" and self.shallow is not None:
            if x.dim() > 2:
                x = torch.flatten(x, 1)
            return self.shallow(x)
        if self.model_type == "ViT" and self.base_model is not None:
            tokens = self.base_model._process_input(x)
            batch_size = tokens.shape[0]
            cls = self.base_model.class_token.expand(batch_size, -1, -1)
            return torch.cat([cls, tokens], dim=1)
        if self.model_type == "Swin" and self.features is not None:
            return self.features[0](x).permute(0, 3, 1, 2)  # [b, c, h, w]
        if self.model_type == "ConvNeXt":
            if self.shallow is not None:
                return self.shallow(x)
            if self.features is not None:
                return self.features[:2](x)
            raise NotImplementedError(f"Unsupported shallow extraction for {self.model_type}")
        if self.model_type == "MLP":
            if x.dim() > 2:
                x = torch.flatten(x, 1)
            return x
        raise NotImplementedError(f"Unsupported shallow extraction for {self.model_type}")

    def extract_deep(self, shallow_feat: torch.Tensor) -> torch.Tensor:
        """Extract deep features `h = f_mid(z)`."""
        if self.model_type == "ResNet" and self.deep is not None:
            deep_feat = self.deep(shallow_feat)
            return torch.flatten(deep_feat, 1)  # [batch, d]
        if self.model_type == "VGG" and self.deep is not None and self.avgpool is not None:
            deep_feat = self.deep(shallow_feat)
            pooled = self.avgpool(deep_feat)
            return torch.flatten(pooled, 1)  # [batch, d]
        if self.model_type == "DenseNet" and self.deep is not None:
            deep_feat = self.deep(shallow_feat)
            deep_feat = F.relu(deep_feat, inplace=False)
            pooled = F.adaptive_avg_pool2d(deep_feat, (1, 1))
            return torch.flatten(pooled, 1)  # [batch, d]
        if self.model_type == "ViT" and self.base_model is not None:
            encoded = self.base_model.encoder(shallow_feat)
            return encoded[:, 0]  # [batch, d]
        if self.model_type == "Swin" and self.features is not None and self.norm is not None:
            feat = shallow_feat.permute(0, 2, 3, 1)  # [batch, h, w, c]
            feat = self.features[1:](feat)
            feat = self.norm(feat)
            feat = self.permute(feat)
            feat = self.avgpool(feat)
            return self.flatten(feat)
        if self.model_type == "ConvNeXt" and self.avgpool is not None:
            if self.deep is not None:
                feat = self.deep(shallow_feat)
            elif self.features is not None:
                feat = self.features[2:](shallow_feat)
            else:
                raise NotImplementedError(f"Unsupported deep extraction for {self.model_type}")
            return self.avgpool(feat)
        if self.model_type == "MLP" and self.deep is not None:
            return self.deep(shallow_feat)
        raise NotImplementedError(f"Unsupported deep extraction for {self.model_type}")

    def router_features(self, shallow_feat: torch.Tensor) -> torch.Tensor:
        """Project shallow features into the router space."""
        if self.model_type == "MLP":
            return shallow_feat
        if self.model_type != "ViT":
            pooled = F.adaptive_avg_pool2d(shallow_feat, (1, 1))
            return torch.flatten(pooled, 1)  # [batch, shallow_dim]
        return shallow_feat[:, 0]  # [batch, shallow_dim]

    def classify(self, deep_feat: torch.Tensor, patch: torch.Tensor | None = None) -> torch.Tensor:
        """Apply an optional late residual patch and classify."""
        if self.classifier is None:
            raise RuntimeError("Classifier is not initialized in backbone decomposition.")

        if self.model_type == "MLP":
            return self.classifier(deep_feat if patch is None else (deep_feat + patch))

        if patch is None:
            if self.model_type == "ConvNeXt":
                return self.classifier(deep_feat)
            deep_flat = torch.flatten(deep_feat, 1) if self.model_type != "ViT" else deep_feat
            return self.classifier(deep_flat)

        if self.patch_site == "last_affine":
            pre, last = self.classifier_split()
            if last is None:
                raise RuntimeError(
                    f"patch_site='last_affine' requires a Linear in the {self.model_type} head; "
                    f"found {type(self.classifier).__name__}. Refusing to fall back silently."
                )
            # Run the frozen non-affine prefix FIRST, then inject. For ResNet/DenseNet `pre` is
            # None and this is bit-identical to the branch below.
            h = deep_feat if pre is None else pre(deep_feat)
            h = torch.flatten(h, 1) if self.model_type != "ViT" else h  # [batch, last.in_features]
            if h.size(1) != patch.size(1):
                raise RuntimeError(
                    f"patch width {patch.size(1)} != affine-boundary width {h.size(1)} for "
                    f"{self.model_type}; set model.out_dim to {h.size(1)}."
                )
            return last(h + patch)

        if self.model_type == "ConvNeXt":
            patch_4d = patch.view(-1, deep_feat.size(1), 1, 1)  # [batch, d, 1, 1]
            return self.classifier(deep_feat + patch_4d)

        deep_flat = torch.flatten(deep_feat, 1) if self.model_type != "ViT" else deep_feat  # [batch, d]
        return self.classifier(deep_flat + patch)

    def apply_insertion_patch(self, hidden_feat: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """Apply a patch at the configured insertion boundary."""
        if self.insertion_point != "same_layer":
            return hidden_feat
        if self.model_type == "MLP":
            return hidden_feat + patch
        if self.model_type != "ViT":
            patch_view = patch.view(patch.size(0), patch.size(1), 1, 1)
            return hidden_feat + patch_view
        patched = hidden_feat.clone()
        patched[:, 0, :] = patched[:, 0, :] + patch
        return patched


def build_decomposition(
    base_model: nn.Module,
    model_type: str,
    split_depth: int | None = None,
    split_layer: str | None = None,
    insertion_point: str = "cross_layer",
    patch_site: str = "deep_feat",
) -> BackboneDecomposition:
    """Build a backbone decomposition compatible with DynaPatch."""
    insertion_point = str(insertion_point)
    patch_site = str(patch_site)
    if patch_site not in {"deep_feat", "last_affine"}:
        raise ValueError(f"Unsupported patch_site: {patch_site}")
    if model_type == "ResNet":
        children = list(base_model.children())
        split_idx = _resolve_resnet_split_idx(split_depth=split_depth, split_layer=split_layer)
        return BackboneDecomposition(
            model_type=model_type,
            insertion_point=insertion_point,
            patch_site=patch_site,
            shallow=nn.Sequential(*children[:split_idx]),
            deep=nn.Sequential(*children[split_idx:-1]),
            classifier=base_model.fc,
            base_model=base_model,
        )
    if model_type == "VGG":
        features = list(base_model.features.children())
        split_idx = _resolve_vgg_split_idx(split_depth=split_depth, split_layer=split_layer)
        return BackboneDecomposition(
            model_type=model_type,
            insertion_point=insertion_point,
            patch_site=patch_site,
            shallow=nn.Sequential(*features[:split_idx]),
            deep=nn.Sequential(*features[split_idx:]),
            avgpool=base_model.avgpool,
            classifier=base_model.classifier,
            base_model=base_model,
        )
    if model_type == "DenseNet":
        features = list(base_model.features.children())
        split_idx = _resolve_densenet_split_idx(split_depth=split_depth, split_layer=split_layer)
        return BackboneDecomposition(
            model_type=model_type,
            insertion_point=insertion_point,
            patch_site=patch_site,
            shallow=nn.Sequential(*features[:split_idx]),
            deep=nn.Sequential(*features[split_idx:]),
            classifier=base_model.classifier,
            base_model=base_model,
        )
    if model_type == "ViT":
        return BackboneDecomposition(
            model_type=model_type,
            insertion_point=insertion_point,
            patch_site=patch_site,
            classifier=base_model.heads,
            base_model=base_model,
        )
    if model_type == "Swin":
        return BackboneDecomposition(
            model_type=model_type,
            insertion_point=insertion_point,
            patch_site=patch_site,
            features=base_model.features,
            norm=base_model.norm,
            permute=base_model.permute,
            avgpool=base_model.avgpool,
            flatten=base_model.flatten,
            classifier=base_model.head,
            base_model=base_model,
        )
    if model_type == "ConvNeXt":
        features = list(base_model.features.children())
        split_idx = _resolve_convnext_split_idx(split_depth=split_depth, split_layer=split_layer)
        return BackboneDecomposition(
            model_type=model_type,
            insertion_point=insertion_point,
            patch_site=patch_site,
            shallow=nn.Sequential(*features[:split_idx]),
            deep=nn.Sequential(*features[split_idx:]),
            features=base_model.features,
            avgpool=base_model.avgpool,
            classifier=base_model.classifier,
            base_model=base_model,
        )
    if model_type == "MLP":
        layers = list(base_model.feature_layers.children())
        split_idx = _resolve_mlp_split_idx(split_depth=split_depth, split_layer=split_layer, total_layers=len(layers))
        return BackboneDecomposition(
            model_type=model_type,
            insertion_point=insertion_point,
            patch_site=patch_site,
            shallow=nn.Sequential(*layers[:split_idx]),
            deep=nn.Sequential(*layers[split_idx:]),
            feature_layers=base_model.feature_layers,
            classifier=base_model.classifier,
            base_model=base_model,
        )
    raise NotImplementedError(f"Unsupported model type: {model_type}")


def _resolve_resnet_split_idx(split_depth: int | None, split_layer: str | None) -> int:
    """Resolve a ResNet split boundary while preserving old split_depth behavior."""
    if split_depth is not None:
        return int(split_depth)
    if split_layer is None:
        return 4
    mapping = {
        "stem": 4,
        "layer1": 5,
        "layer2": 6,
        "layer3": 7,
        "layer4": 8,
    }
    try:
        return mapping[str(split_layer).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported ResNet split_layer: {split_layer}") from exc


def _resolve_vgg_split_idx(split_depth: int | None, split_layer: str | None) -> int:
    """Resolve a VGG split boundary while preserving old split_depth behavior."""
    if split_depth is not None:
        return int(split_depth)
    if split_layer is None:
        return 5
    mapping = {
        "block1": 5,
        "block2": 10,
        "block3": 17,
        "block4": 24,
        "block5": 31,
    }
    try:
        return mapping[str(split_layer).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported VGG split_layer: {split_layer}") from exc


def _resolve_densenet_split_idx(split_depth: int | None, split_layer: str | None) -> int:
    """Resolve a DenseNet split boundary using feature-stage names."""
    if split_depth is not None:
        return int(split_depth)
    if split_layer is None:
        return 11
    mapping = {
        "stem": 4,
        "layer1": 5,
        "block1": 5,
        "layer2": 7,
        "block2": 7,
        "layer3": 9,
        "block3": 9,
        "layer4": 11,
        "block4": 11,
    }
    try:
        return mapping[str(split_layer).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported DenseNet split_layer: {split_layer}") from exc


def _resolve_convnext_split_idx(split_depth: int | None, split_layer: str | None) -> int:
    """Resolve a ConvNeXt stage boundary."""
    if split_depth is not None:
        return int(split_depth)
    if split_layer is None:
        return 2
    mapping = {
        "layer1": 2,  # 96-d
        "layer2": 4,  # 192-d
        "layer3": 6,  # 384-d
        "layer4": 8,  # 768-d
    }
    try:
        return mapping[str(split_layer).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported ConvNeXt split_layer: {split_layer}") from exc


def _resolve_mlp_split_idx(split_depth: int | None, split_layer: str | None, total_layers: int) -> int:
    """Resolve an MLP split boundary over feature extractor submodules."""
    if split_depth is not None:
        return int(split_depth)
    if split_layer is None:
        return max(total_layers // 2, 1)
    mapping = {
        "layer1": min(2, total_layers),
        "block1": min(2, total_layers),
        "layer2": min(4, total_layers),
        "block2": min(4, total_layers),
        "layer3": min(6, total_layers),
        "block3": min(6, total_layers),
    }
    try:
        return mapping[str(split_layer).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported MLP split_layer: {split_layer}") from exc
