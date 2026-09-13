"""Run record serialization for experiment lineage."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class RunRecord:
    """Machine-readable summary of one experiment run."""

    experiment_id: str
    stage: str
    artifact_root: str
    config_path: str
    env_path: str
    git_path: str
    notes_path: str
    metrics_path: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a plain dictionary."""
        return asdict(self)


def write_resolved_config(cfg: DictConfig, destination: Path) -> None:
    """Write the resolved Hydra config for traceability."""
    destination.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def write_json(payload: dict[str, Any], destination: Path) -> None:
    """Write a JSON payload with stable formatting."""
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def collect_env_payload() -> dict[str, Any]:
    """Collect runtime environment metadata."""
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count() if cuda_available else 0
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        torch_version = None
        cuda_available = False
        cuda_version = None
        device_count = 0
        torch_error = f"{type(exc).__name__}: {exc}"
    else:
        torch_error = None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "torch_import_error": torch_error,
    }


def collect_git_payload(cwd: Path) -> dict[str, Any]:
    """Collect git metadata if the repository is under version control."""
    def _run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _run_git(["status", "--short"]),
    }


def write_notes_stub(cfg: DictConfig, destination: Path) -> None:
    """Write a tri-layer note stub for the current run."""
    experiment_id = cfg.experiment.id
    stage = cfg.experiment.stage
    track = str(cfg.experiment.get("track", "diagnostic"))
    notes = "\n".join(
        [
            "# Run Notes",
            "",
            "## Context",
            "",
            f"Goal:",
            f"- Execute `{experiment_id}` for `{stage}`.",
            f"- Record class: `{track}`.",
            "",
            "Hypothesis:",
            f"- {cfg.experiment.notes}",
            "",
            "Boundary:",
            "- This note was auto-generated at run bootstrap and should be completed after execution.",
            "",
            "## Execution",
            "",
            "Environment:",
            "- See `env.json` and `git.json`.",
            "",
            "Trace:",
            "- Add links to checkpoints, bug-bank artifacts, and metric files here.",
            "",
            "## Insight",
            "",
            "Observation:",
            "- Pending execution.",
            "",
            "Interpretation:",
            "- Pending execution.",
            "",
            "Next Action:",
            "- Replace this stub with actual run outcomes after training or evaluation completes.",
            "",
        ]
    )
    destination.write_text(notes, encoding="utf-8")
