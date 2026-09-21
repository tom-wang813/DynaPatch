#!/usr/bin/env python3
"""HeadFT and FullFT: last-layer / whole-model fine-tuning baselines, from a checkpoint or trained
from scratch, then scored the same way for both.

`--mode checkpoint` loads artifacts/checkpoints/baselines/{HeadFT,FullFT}/<ds>_<bb>_s<seed>/
baseline_last.pt (a `{"state_dict": ..., "mode": ...}` produced by
scripts/train_head_repair_baseline.py). `--mode train` instead calls that mode's real training
loop here via src.baselines.head_repair.configure_baseline_model (same freeze/unfreeze logic the
original driver script uses -- see its own docstring for why the mode name and architecture
string it's given must be exact).

Usage:
  uv run python scripts/checkpoint_eval/headft.py --baseline head_ft \
      --dataset gtsrb --backbone resnet50 --mode checkpoint --output-root outputs/ckpt_eval
  uv run python scripts/checkpoint_eval/headft.py --baseline full_ft \
      --dataset gtsrb --backbone resnet50 --mode train --output-root outputs/ckpt_eval
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.ROOT))
from src.baselines.head_repair import configure_baseline_model  # noqa: E402

# HeadFT/FullFT checkpoint dirs and the `train_head_repair_baseline.py --mode` value each maps to.
BASELINE_MODE = {"head_ft": "head_only", "full_ft": "full_finetune"}
METHOD_NAME = {"head_ft": "HeadFT", "full_ft": "FullFT"}
# head_repair.py's classifier lookup wants "convnext", not the "convnext_tiny" setting name.
ARCH_ALIAS = {"convnext_tiny": "convnext"}


def main() -> None:
    ap = common.base_argparser(__doc__)
    ap.add_argument("--baseline", required=True, choices=list(BASELINE_MODE))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    a = ap.parse_args()
    device = torch.device(a.device)
    method = METHOD_NAME[a.baseline]

    cfg = common.load_cfg(a.dataset, a.backbone)
    base_model = common.build_frozen_backbone(cfg, device)

    if a.mode == "checkpoint":
        ckpt_path = common.checkpoint_dir(method, a.dataset, a.backbone, a.seed) / "baseline_last.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"[{method}] no checkpoint at {ckpt_path} -- run with --mode train, "
                              f"or place the checkpoint there per artifacts/checkpoints/MANIFEST.md.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        patched_model = copy.deepcopy(base_model)
        patched_model.load_state_dict(ckpt["state_dict"])
        patched_model = patched_model.to(device).eval()
        print(f"[{method}] loaded checkpoint {ckpt_path} (trained mode={ckpt.get('mode')})")
    else:
        arch = ARCH_ALIAS.get(str(cfg.model.architecture), str(cfg.model.architecture))
        model = copy.deepcopy(base_model).to(device)
        bundle = configure_baseline_model(model, architecture=arch, mode=BASELINE_MODE[a.baseline])
        model = bundle.model.to(device)
        optimizer = torch.optim.AdamW(bundle.trainable_parameters, lr=a.lr, weight_decay=a.weight_decay)

        loaders = common.build_eval_loaders(cfg, a.dataset, a.backbone, a.seed)
        train_indices, _ = loaders["repair_support_seen"]
        from torch.utils.data import DataLoader, Subset
        from src.data.factory import build_dataset
        train_dataset = build_dataset(cfg, train=False)
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=16,
                                   shuffle=True, num_workers=4)

        model.train()
        for epoch in range(a.epochs):
            total_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * len(y)
                n += len(y)
            print(f"[{method}] epoch {epoch + 1}/{a.epochs} loss={total_loss / max(n, 1):.4f}")
        patched_model = model.eval()

    loaders = common.build_eval_loaders(cfg, a.dataset, a.backbone, a.seed)
    out_dir = Path(a.output_root) / method / a.dataset / a.backbone / f"s{a.seed}" / "predictions"
    for split_name, (indices, loader) in loaders.items():
        rows = common.score_split(base_model, patched_model, indices, loader, device)
        common.write_predictions(out_dir, split_name, rows)
        print(f"[{method}] wrote {len(rows)} rows for {split_name}")

    summary = common.summarize(out_dir, a.dataset)
    print(f"[{method}] {a.dataset}/{a.backbone} s{a.seed}: {summary}")


if __name__ == "__main__":
    main()
