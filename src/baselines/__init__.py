"""Baseline models and utilities for safety comparisons."""

from .distrep_pso import DistrRepConfig, DistrRepPSO, PSOConfig
from .head_repair import (
    HeadOnlyFineTuneBaseline,
    HeadOnlySafetyBaseline,
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
    "configure_baseline_model",
    "get_classifier_module",
    "set_classifier_module",
]
