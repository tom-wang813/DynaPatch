"""Baseline models and utilities for safety comparisons."""

from .arachne import FeatureBank, SearchConfig, greedy_coordinate_search, select_topk_parameters
from .distrep_pso import DistrRepConfig, DistrRepPSO, PSOConfig
from .head_repair import (
    HeadOnlyFineTuneBaseline,
    HeadOnlySafetyBaseline,
    LastLayerDeltaBaseline,
    configure_baseline_model,
    get_classifier_module,
    set_classifier_module,
)

__all__ = [
    "DistrRepConfig",
    "DistrRepPSO",
    "PSOConfig",
    "HeadOnlyFineTuneBaseline",
    "HeadOnlySafetyBaseline",
    "LastLayerDeltaBaseline",
    "FeatureBank",
    "SearchConfig",
    "configure_baseline_model",
    "get_classifier_module",
    "greedy_coordinate_search",
    "select_topk_parameters",
    "set_classifier_module",
]
