"""Selective rejection baselines over direct-patch prediction rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DirectPredictionRow:
    split: str
    dataset_index: int
    label: int
    base_pred: int
    patched_pred: int
    base_conf: float
    patched_conf: float
    patch_norm: float
    base_correct: bool
    patched_correct: bool


def beneficial(row: DirectPredictionRow) -> bool:
    return (not row.base_correct) and row.patched_correct


def harmful(row: DirectPredictionRow) -> bool:
    return row.base_correct and (not row.patched_correct)


def neutral(row: DirectPredictionRow) -> bool:
    return not beneficial(row) and not harmful(row)


def confidence_score(row: DirectPredictionRow) -> float:
    return float(row.base_conf)


def patched_margin_proxy(row: DirectPredictionRow) -> float:
    return (2.0 * float(row.patched_conf)) - 1.0


def base_margin_proxy(row: DirectPredictionRow) -> float:
    return (2.0 * float(row.base_conf)) - 1.0


def margin_gain(row: DirectPredictionRow) -> float:
    return patched_margin_proxy(row) - base_margin_proxy(row)


def temperature_scaled_confidence(row: DirectPredictionRow, temperature: float) -> float:
    tau = max(float(temperature), 1.0e-6)
    return float(row.base_conf) ** (1.0 / tau)
