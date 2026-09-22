#!/usr/bin/env python3
"""FixedPatch (FPa): score a trained FixedPatch checkpoint (scripts/repro/rq2_train_fixedpatch.py) --
same DynaPatchModel/deploy_eval path as dynapatch.py, just with `method.name=fixed_patch` so
build_stage3_bundle assembles a ConstantPatchGenerator instead of the hypernetwork. Ungated only
(FixedPatch has no gate of its own in the paper -- it's the RQ2 "no input-conditioning" ablation
compared directly against DynaPatch-NoGate).

Usage:
  uv run python scripts/checkpoint_eval/fixedpatch.py --dataset gtsrb --backbone resnet50 \
      --mode checkpoint --output-root outputs/repro/ckpt_eval
"""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.ROOT))
from src.experiment.deploy_eval import run_deploy_eval  # noqa: E402

METHOD = "FixedPatch"


def repair_checkpoint_path(dataset: str, backbone: str, seed: int) -> Path:
    return common.ROOT / f"artifacts/checkpoints/baselines/{METHOD}/{dataset}_{backbone}_s{seed}/repair_best.pt"


def resolved_cfg(dataset: str, backbone: str, seed: int, output_root: Path,
                  checkpoint_path: Path, device: str, *, save_route_features: bool = False) -> OmegaConf:
    cfg = OmegaConf.load(common.ROOT / f"configs/shuffled_split_source/{dataset}/{backbone}/deploy.yaml")
    sdir = common.split_dir(dataset, backbone, seed)
    cfg.artifacts.root = str(output_root)
    cfg.deployment.checkpoint_path = str(checkpoint_path)
    cfg.model.checkpoint_path = str(common.backbone_checkpoint_path(dataset, backbone))
    OmegaConf.update(cfg, "method.name", "fixed_patch", merge=True)
    OmegaConf.update(cfg, "data.bug_indices_path", str(sdir / f"{dataset}_bug_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.bug_train_indices_path", str(sdir / f"{dataset}_bug_train_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.bug_eval_indices_path", str(sdir / f"{dataset}_bug_eval_indices.json"), merge=True)
    OmegaConf.update(cfg, "data.clean_eval_indices_path", str(sdir / f"{dataset}_clean_eval_indices.json"), merge=True)
    OmegaConf.update(cfg, "runtime.device", device, merge=True)
    if save_route_features:
        OmegaConf.update(cfg, "deployment.save_route_features", True, merge=True)
    return cfg


def main() -> None:
    ap = common.base_argparser(__doc__)
    ap.add_argument("--dump-route-features", action="store_true",
                     help="also dump base/patched logits per sample (for RQ2's Delta-ell_M "
                          "analysis, scripts/repro/rq2_norm_and_direction.py)")
    a = ap.parse_args()
    if a.mode != "checkpoint":
        raise SystemExit(f"[{METHOD}] --mode train doesn't apply here -- use "
                          f"scripts/repro/rq2_train_fixedpatch.py to (re)train.")

    ckpt = repair_checkpoint_path(a.dataset, a.backbone, a.seed)
    if not ckpt.exists():
        raise SystemExit(f"[{METHOD}] no checkpoint at {ckpt} -- run scripts/repro/rq2_train_fixedpatch.py first.")

    out_root = Path(a.output_root) / a.dataset / a.backbone / f"s{a.seed}" / "deploy"
    run_deploy_eval(resolved_cfg(a.dataset, a.backbone, a.seed, out_root, ckpt, a.device,
                                  save_route_features=a.dump_route_features))

    summary = common.summarize(out_root / "predictions", a.dataset)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed}: {summary}")


if __name__ == "__main__":
    main()
