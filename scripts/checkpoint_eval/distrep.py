#!/usr/bin/env python3
"""DistrRep (Li Calsi et al., ICST'23 PyTorch re-implementation): a whole repaired-backbone
state_dict, from a checkpoint or from scratch via the 3-phase PSO in src.baselines.distrep_pso.

`--mode checkpoint` loads artifacts/checkpoints/baselines/DistrRep/<ds>_<bb>_s<seed>/
distrep_repaired.pt (a plain backbone state_dict, see scripts/run_distrep_pso.py). `--mode train`
is not implemented here yet -- the PSO driver's phase wiring (partition count, per-phase budgets)
lives entirely in scripts/run_distrep_pso.py and hasn't been ported; use --mode checkpoint, or
scripts/run_distrep_pso.py directly, until that's done.

Usage:
  uv run python scripts/checkpoint_eval/distrep.py --dataset gtsrb --backbone resnet50 \
      --mode checkpoint --output-root outputs/ckpt_eval
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

METHOD = "DistrRep"


def main() -> None:
    ap = common.base_argparser(__doc__)
    a = ap.parse_args()
    device = torch.device(a.device)

    cfg = common.load_cfg(a.dataset, a.backbone)
    base_model = common.build_frozen_backbone(cfg, device)

    if a.mode != "checkpoint":
        raise SystemExit(f"[{METHOD}] --mode train is not implemented yet -- see module docstring; "
                          f"use --mode checkpoint or scripts/run_distrep_pso.py directly.")

    ckpt_path = common.checkpoint_dir(METHOD, a.dataset, a.backbone, a.seed) / "distrep_repaired.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"[{METHOD}] no checkpoint at {ckpt_path} -- place it there per "
                          f"artifacts/checkpoints/MANIFEST.md.")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    patched_model = copy.deepcopy(base_model)
    patched_model.load_state_dict(state_dict)
    patched_model = patched_model.to(device).eval()
    print(f"[{METHOD}] loaded checkpoint {ckpt_path}")

    loaders = common.build_eval_loaders(cfg, a.dataset, a.backbone, a.seed)
    out_dir = Path(a.output_root) / METHOD / a.dataset / a.backbone / f"s{a.seed}" / "predictions"
    for split_name, (indices, loader) in loaders.items():
        rows = common.score_split(base_model, patched_model, indices, loader, device)
        common.write_predictions(out_dir, split_name, rows)
        print(f"[{METHOD}] wrote {len(rows)} rows for {split_name}")

    summary = common.summarize(out_dir, a.dataset)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed}: {summary}")


if __name__ == "__main__":
    main()
