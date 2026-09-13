"""Utilities for managing deployment-time repair banks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_repair_bank(path: str | Path) -> dict[str, Any]:
    """Load a serialized repair bank payload from disk."""
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Repair bank payload must be a dictionary.")
    if "route_feat" not in payload or "patches" not in payload:
        raise ValueError("Repair bank payload must contain `route_feat` and `patches`.")
    return payload


def save_repair_bank(
    *,
    artifact_dir: str | Path,
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist a repair bank tensor payload plus a small JSON summary."""
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    bank_path = root / "managed_repair_bank.pt"
    summary_path = root / "managed_repair_bank_summary.json"
    torch.save(payload, bank_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return bank_path, summary_path
