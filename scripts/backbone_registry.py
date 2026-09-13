#!/usr/bin/env python3
"""Registry mapping (dataset, backbone) -> frozen backbone checkpoint directory.

Extracted from the original repo's `dump_base_logits.py` (a script belonging to an abandoned
certified-non-regression analysis direction that is not part of the current paper and was not
carried into this anonymized reproduction repo). `dump_prior_features.py` needs only this
dict, so it is kept here as a small, direction-agnostic constant rather than pulling in the
dead script's whole certificate-bound docstring/purpose.

Paths are relative to the repository root (`outputs/exp_<dataset>_<backbone>_backbone_public_*`).
"""

BACKBONE = {
    ("gtsrb", "resnet50"): "outputs/exp_gtsrb_resnet50_backbone_public_v2",
    ("gtsrb", "convnext_tiny"): "outputs/exp_gtsrb_convnext_tiny_backbone_public_v2",
    ("gtsrb", "densenet121"): "outputs/exp_gtsrb_densenet121_backbone_public_v2",
    ("gtsrb", "vgg16"): "outputs/exp_gtsrb_vgg16_backbone_public_v2",
    ("tt100k_signs", "resnet50"): "outputs/exp_tt100k_signs_resnet50_backbone_public_v7",
    ("tt100k_signs", "convnext_tiny"): "outputs/exp_tt100k_signs_convnext_tiny_backbone_public_v7",
    ("tt100k_signs", "densenet121"): "outputs/exp_tt100k_signs_densenet121_backbone_public_v7",
    ("tt100k_signs", "vgg16"): "outputs/exp_tt100k_signs_vgg16_backbone_public_v7",
    ("lisa_signs", "resnet50"): "outputs/exp_lisa_signs_resnet50_backbone_public_v7a",
    ("lisa_signs", "convnext_tiny"): "outputs/exp_lisa_signs_convnext_tiny_backbone_public_v7a",
    ("lisa_signs", "densenet121"): "outputs/exp_lisa_signs_densenet121_backbone_public_v1",
    ("lisa_signs", "vgg16"): "outputs/exp_lisa_signs_vgg16_backbone_public_v1",
}
