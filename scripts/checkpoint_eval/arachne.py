#!/usr/bin/env python3
"""Arachne (Sohn, Kang & Yoo, TOSEM'22 PyTorch re-implementation): patch the last nn.Linear of the
frozen classifier's weight (bias is untouched -- Arachne only searches weight values), from a
checkpoint or from scratch via src.baselines.arachne_de.repair().

`--mode checkpoint` loads artifacts/checkpoints/baselines/Arachne/<ds>_<bb>_s<seed>/
arachne_patched_classifier.pt (a `{weight, bias, target_layer_shape}` dict, see
scripts/run_arachne_de.py). `--mode train` runs the real bidirectional-localisation + DE search
here instead.

Usage:
  uv run python scripts/checkpoint_eval/arachne.py --dataset gtsrb --backbone resnet50 \
      --mode checkpoint --output-root outputs/ckpt_eval
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.ROOT))
from src.baselines.arachne_de import ArachneConfig, repair  # noqa: E402

METHOD = "Arachne"


def last_linear(module: nn.Module) -> nn.Linear:
    found = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not found:
        raise RuntimeError("no nn.Linear found in the classifier")
    return found[-1]


def classifier_module(model: torch.nn.Module, architecture: str) -> nn.Module:
    """Same per-architecture classifier lookup as src.baselines.head_repair, without importing
    its architecture-string alias table (arachne always wants the raw torchvision attribute)."""
    if architecture == "resnet50":
        return model.fc
    if architecture in ("densenet121",):
        return model.classifier
    if architecture in ("vgg16", "convnext"):
        return model.classifier
    raise ValueError(f"Unsupported architecture for Arachne: {architecture}")


@torch.no_grad()
def cache_layer_inputs(model: nn.Module, target: nn.Linear, loader, device):
    feats, labels = [], []
    box = {}

    def hook(_m, inp, _out):
        box["x"] = inp[0].detach()

    h = target.register_forward_hook(hook)
    try:
        for x, y in loader:
            model(x.to(device))
            feats.append(box["x"].float().cpu())
            labels.append(y)
    finally:
        h.remove()
    return torch.cat(feats), torch.cat(labels)


def main() -> None:
    ap = common.base_argparser(__doc__)
    ap.add_argument("--num-places", type=int, default=64)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--patch-aggr", type=float, default=10.0)
    ap.add_argument("--bound-scale", type=float, default=128.0,
                     help="DE candidate range = init +/- bound_scale*|init|. 128 (not the "
                          "ArachneConfig dataclass's own 2.0 default) is what the shipped "
                          "artifacts/checkpoints/baselines/Arachne/ checkpoints were retrained "
                          "at -- bound_scale=2.0 was an unswept default (see "
                          "artifacts/checkpoints/baselines/Arachne_bound_scale2_backup/ for the "
                          "old checkpoints), documented in this repo's own git history as giving "
                          "RR_held 0.125 -> 0.375 on gtsrb/resnet50 at 128, verified across all "
                          "12 settings 2026-09-22 (mean RR_held 0.036 -> 0.146).")
    ap.add_argument("--save-checkpoint", default=None,
                     help="directory to save the trained patch as <dir>/<dataset>_<backbone>_s<seed>/"
                          "arachne_patched_classifier.pt (only for --mode train; --mode checkpoint "
                          "already has one on disk).")
    ap.add_argument("--de-seed", type=int, default=0,
                     help="ArachneConfig.seed for the DE search's own RNG (src/baselines/"
                          "arachne_de.py defaults this to 0 -- NOT passing it means every "
                          "invocation of --mode train is fully deterministic, not an independent "
                          "draw; vary this explicitly to actually test run-to-run variance.")
    a = ap.parse_args()
    device = torch.device(a.device)

    cfg = common.load_cfg(a.dataset, a.backbone)
    base_model = common.build_frozen_backbone(cfg, device)
    arch = "convnext" if a.backbone == "convnext_tiny" else a.backbone

    if a.mode == "checkpoint":
        ckpt_path = common.checkpoint_dir(METHOD, a.dataset, a.backbone, a.seed) / "arachne_patched_classifier.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"[{METHOD}] no checkpoint at {ckpt_path} -- run with --mode train, "
                              f"or place the checkpoint there per artifacts/checkpoints/MANIFEST.md.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        patched_model = copy.deepcopy(base_model)
        target = last_linear(classifier_module(patched_model, arch))
        target.weight.data.copy_(ckpt["weight"].to(device))
        target.bias.data.copy_(ckpt["bias"].to(device))
        patched_model = patched_model.to(device).eval()
        print(f"[{METHOD}] loaded checkpoint {ckpt_path}")
    else:
        loaders = common.build_eval_loaders(cfg, a.dataset, a.backbone, a.seed)
        fail_idx, fail_loader = loaders["repair_support_seen"]
        clean_idx, clean_loader = loaders["clean_eval"]
        patched_model = copy.deepcopy(base_model).to(device).eval()
        target = last_linear(classifier_module(patched_model, arch))
        feats_fail, labels_fail = cache_layer_inputs(patched_model, target, fail_loader, device)
        feats_ok, labels_ok = cache_layer_inputs(patched_model, target, clean_loader, device)
        preserve_cap = 2048
        acfg = ArachneConfig(num_places=a.num_places, pop_size=a.pop_size, max_iter=a.max_iter,
                              patch_aggr=a.patch_aggr, bound_scale=a.bound_scale, seed=a.de_seed)
        w_new, info = repair(target, feats_fail.to(device), labels_fail.to(device),
                              feats_ok[:preserve_cap].to(device), labels_ok[:preserve_cap].to(device), acfg)
        print(f"[{METHOD}] search info: {info}")
        target.weight.data.copy_(w_new)

        if a.save_checkpoint:
            save_dir = Path(a.save_checkpoint) / f"{a.dataset}_{a.backbone}_s{a.seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "arachne_patched_classifier.pt"
            torch.save({"weight": target.weight.detach().cpu(),
                        "bias": target.bias.detach().cpu(),
                        "target_layer_shape": list(target.weight.shape),
                        "bound_scale": a.bound_scale, "de_seed": a.de_seed, "search_info": info},
                       save_path)
            print(f"[{METHOD}] saved checkpoint -> {save_path}")

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
