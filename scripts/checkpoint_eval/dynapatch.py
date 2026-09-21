#!/usr/bin/env python3
"""DynaPatch itself, from a repair checkpoint -- the same unified CLI shape as the 6 baseline
scripts in this folder (headft/fullft/arachne/distrep/nnpatch/patchnas.py), so all 7 methods are
driven the same way. Python replacement for scripts/deploy_from_checkpoints.sh (no shell driver);
calls src.experiment.deploy_eval.run_deploy_eval(cfg) directly instead of shelling out to
scripts/run_resolved_experiment.py.

`--mode checkpoint` is the only mode: DynaPatch's repair checkpoints are always obtained by
training (scripts/run_resolved_experiment.py's stage3_repair stage, or TRAINING.md's from-scratch
path) -- there's no meaningful "train inline" for a single cell here, unlike the baselines.

For one (dataset, backbone, seed) cell, runs three deploy-eval passes with the SAME overrides
deploy_from_checkpoints.sh used:
  main           bug_train/bug_eval/clean_eval splits -> RR_seen/RR_held/Reg/CReg (ungated)
  deploy_direct  + deployment.save_route_features=true, dumped under outputs/effect_dump*/
  deploy_direct_calib   same, but against bug_val instead of bug_eval (the gate's calibration pass)
The gate itself is fit once over every cell's dump, not per-cell -- see this script's --fit-gate.

Usage:
  uv run python scripts/checkpoint_eval/dynapatch.py --dataset gtsrb --backbone resnet50 \
      --mode checkpoint --output-root outputs/from_checkpoints
  uv run python scripts/gate_protocol_b.py --min-pos 1   # gated numbers, once all cells are deployed
"""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.ROOT))
from src.experiment.deploy_eval import run_deploy_eval  # noqa: E402

METHOD = "DynaPatch"


def resolved_cfg(dataset: str, backbone: str, seed: int, output_root: Path,
                  checkpoint_path: Path, device: str, *, held_suffix: str,
                  save_route_features: bool) -> OmegaConf:
    """One deploy.yaml, pointed at this cell's checkpoint/splits/device -- the same dot-path
    overrides deploy_from_checkpoints.sh applied via scripts/run_resolved_experiment.py."""
    cfg = OmegaConf.load(common.ROOT / f"configs/shuffled_split_source/{dataset}/{backbone}/deploy.yaml")
    sdir = common.split_dir(dataset, backbone, seed)
    cfg.artifacts.root = str(output_root)
    cfg.deployment.checkpoint_path = str(checkpoint_path)
    cfg.model.checkpoint_path = str(common.backbone_checkpoint_path(dataset, backbone))
    OmegaConf.update(cfg, "data.bug_indices_path", str(sdir / f"{dataset}_bug_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.bug_train_indices_path", str(sdir / f"{dataset}_bug_train_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.bug_eval_indices_path", str(sdir / f"{dataset}_{held_suffix}.json"), merge=True)
    OmegaConf.update(cfg, "data.clean_eval_indices_path", str(sdir / f"{dataset}_clean_eval_indices.json"), merge=True)
    OmegaConf.update(cfg, "loss.lambda_robust", 0.0, merge=True)
    OmegaConf.update(cfg, "loss.lambda_field", 0.0, merge=True)
    OmegaConf.update(cfg, "loss.lambda_clean_replay", 1.0, merge=True)
    OmegaConf.update(cfg, "runtime.device", device, merge=True)
    if save_route_features:
        OmegaConf.update(cfg, "deployment.save_route_features", True, merge=True)
    return cfg


def repair_checkpoint_path(dataset: str, backbone: str, seed: int) -> Path:
    return common.ROOT / f"artifacts/checkpoints/dynapatch/{dataset}_{backbone}_s{seed}/repair_best.pt"


def main() -> None:
    ap = common.base_argparser(__doc__)
    ap.add_argument("--dump-tag", default="",
                     help="suffix for the gate-evidence dump tree, outputs/effect_dump<tag>_v8_s<seed>/ "
                          "(matches DUMP_TAG in the old deploy_from_checkpoints.sh).")
    a = ap.parse_args()
    if a.mode != "checkpoint":
        raise SystemExit(f"[{METHOD}] --mode train doesn't apply here -- DynaPatch's repair "
                          f"checkpoints only ever come from actual training (see TRAINING.md).")

    ckpt = repair_checkpoint_path(a.dataset, a.backbone, a.seed)
    if not ckpt.exists():
        raise SystemExit(f"[{METHOD}] no repair checkpoint at {ckpt} -- see artifacts/checkpoints/MANIFEST.md.")

    out_root = Path(a.output_root) / a.dataset / a.backbone / f"s{a.seed}" / "deploy"
    print(f"[{METHOD}] deploy-eval (ungated RR/Reg/CReg) -> {out_root}")
    run_deploy_eval(resolved_cfg(a.dataset, a.backbone, a.seed, out_root, ckpt, a.device,
                                  held_suffix="bug_eval_indices", save_route_features=False))

    gate_root = common.ROOT / f"outputs/effect_dump{a.dump_tag}_v8_s{a.seed}/{a.dataset}/{a.backbone}"
    for tag, held_suffix in (("deploy_direct", "bug_eval_indices"), ("deploy_direct_calib", "bug_val_indices")):
        out = gate_root / tag
        print(f"[{METHOD}] gate-evidence dump ({tag}) -> {out}")
        run_deploy_eval(resolved_cfg(a.dataset, a.backbone, a.seed, out, ckpt, a.device,
                                      held_suffix=held_suffix, save_route_features=True))

    summary = common.summarize(out_root / "predictions", a.dataset)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed} (ungated): {summary}")
    print(f"[{METHOD}] gated (DPGate) numbers need scripts/gate_protocol_b.py run once over every "
          f"cell's dump -- not per-cell; see its own --help.")


if __name__ == "__main__":
    main()
