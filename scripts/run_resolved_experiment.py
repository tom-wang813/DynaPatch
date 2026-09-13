#!/usr/bin/env python3
"""Run a resolved train/deploy config with lightweight overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.deploy_eval import run_deploy_eval  # noqa: E402
from src.experiment.train_stage3 import run_stage3_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deployment-checkpoint-path", default=None)
    parser.add_argument("--experiment-suffix", default=None)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Dot-path overrides applied after loading config, e.g. method.name=fixed_patch runtime.device=cuda:1",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.artifacts.root = str(args.output_root)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.experiment_suffix:
        cfg.experiment.id = f"{cfg.experiment.id}_{args.experiment_suffix}"
        cfg.experiment.name = f"{cfg.experiment.name}_{args.experiment_suffix}"
    if args.deployment_checkpoint_path is not None:
        cfg.deployment.checkpoint_path = str(args.deployment_checkpoint_path)
    if args.overrides:
        for override in args.overrides:
            if "=" not in override:
                raise ValueError(f"Override must be key=value, got: {override!r}")
            key, value = override.split("=", 1)
            OmegaConf.update(cfg, key, value, merge=True)

    stage = str(cfg.experiment.stage)
    if stage == "stage3_repair":
        run_stage3_experiment(cfg)
        return
    if stage == "deploy_eval":
        run_deploy_eval(cfg)
        return
    raise ValueError(f"Unsupported experiment.stage: {stage}")


if __name__ == "__main__":
    main()
