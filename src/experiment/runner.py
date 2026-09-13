"""Experiment bootstrap and artifact recording."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from omegaconf import DictConfig

from src.experiment.paths import repo_root, resolve_artifact_root
from src.experiment.record import (
    RunRecord,
    collect_env_payload,
    collect_git_payload,
    write_json,
    write_notes_stub,
    write_resolved_config,
)
from src.experiment.seed import seed_everything


class ExperimentRunner:
    """Prepare artifact directories and metadata for one experiment run."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.root = repo_root()
        self.artifact_root = resolve_artifact_root(cfg.artifacts.root)

    def bootstrap(self) -> RunRecord:
        """Create artifact directories and write initial metadata files."""
        seed_everything(int(self.cfg.seed))
        self._ensure_layout()
        self._snapshot_external_inputs()

        config_path = self.artifact_root / "config_resolved.yaml"
        env_path = self.artifact_root / "env.json"
        git_path = self.artifact_root / "git.json"
        notes_path = self.artifact_root / "notes.md"
        metrics_path = self.artifact_root / "metrics.json"

        write_resolved_config(self.cfg, config_path)
        write_json(collect_env_payload(), env_path)
        write_json(collect_git_payload(self.root), git_path)
        if not notes_path.exists():
            write_notes_stub(self.cfg, notes_path)
        if not metrics_path.exists():
            write_json({"status": "bootstrapped"}, metrics_path)

        return RunRecord(
            experiment_id=str(self.cfg.experiment.id),
            stage=str(self.cfg.experiment.stage),
            artifact_root=str(self.artifact_root),
            config_path=str(config_path),
            env_path=str(env_path),
            git_path=str(git_path),
            notes_path=str(notes_path),
            metrics_path=str(metrics_path),
            extra={"track": str(self.cfg.experiment.get("track", "diagnostic"))},
        )

    def save_metrics(self, metrics: dict[str, Any]) -> Path:
        """Persist metrics for the current run."""
        metrics_path = self.artifact_root / "metrics.json"
        write_json(metrics, metrics_path)
        return metrics_path

    def _ensure_layout(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "logs", "bug_bank", "predictions", "figures"):
            (self.artifact_root / name).mkdir(exist_ok=True)

    def _snapshot_external_inputs(self) -> None:
        """Copy mutable manifest inputs into the artifact root and rebind config paths."""
        if "data" not in self.cfg:
            return

        bug_bank_root = self.artifact_root / "bug_bank"
        bug_bank_root.mkdir(exist_ok=True)

        for key in (
            "bug_indices_path",
            "bug_train_indices_path",
            "bug_val_indices_path",
            "bug_eval_indices_path",
            "clean_eval_indices_path",
        ):
            path_value = self.cfg.data.get(key)
            if path_value is None:
                continue

            source = Path(str(path_value))
            if not source.exists():
                continue

            # Freeze external split manifests inside the run directory so later replays
            # do not depend on mutable files under `artifacts/`.
            destination = bug_bank_root / source.name
            shutil.copy2(source, destination)
            self.cfg.data[key] = str(destination)
