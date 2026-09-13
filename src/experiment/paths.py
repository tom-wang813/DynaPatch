"""Artifact path helpers for experiment runs."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root based on this module location."""
    return Path(__file__).resolve().parents[2]


def resolve_artifact_root(root_value: str) -> Path:
    """Resolve an artifact root relative to the repository root."""
    root = Path(root_value)
    if root.is_absolute():
        return root
    return repo_root() / root
