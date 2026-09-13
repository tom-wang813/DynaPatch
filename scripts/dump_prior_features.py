#!/usr/bin/env python3
"""Cache frozen-backbone features for the two prior-art patch baselines.

NN-Patching (Kauschke & Fuernkranz) trains a patch model plus an error estimator on the deployed
network's INTERNAL REPRESENTATION; PatchNAS (Fang et al., AAAI'23) freezes the deployed network
and builds a lightweight patch network on the features of the FAULTY STAGE. Neither needs the
deployed weights to change, so both reduce to "train small heads on frozen features" once the
features are on disk -- which is what this script puts there. One forward pass per split per
setting; everything downstream is seconds.

Two taps per input:
  final  -- the input of the classification layer. This is the representation our own patch is
            injected into, so a baseline placed here is compared at the SAME injection point.
  stage  -- the output of the last convolutional stage before the final block, global-average
            pooled. This stands in for PatchNAS's "faulty stage": the paper searches an
            architecture for one stage of the frozen network, and a per-architecture stage choice
            is the closest tap available without re-implementing their supernet.

Populations: repair_support_seen (bug_train), repair_holdout_unseen (bug_eval), clean_calib,
clean_test, plus a sample of CLEAN TRAINING images, which the error estimator needs as its
negative class and which our own patch already sees through its clean-replay loss (parity).

Usage:
  python scripts/dump_prior_features.py --setting gtsrb/resnet50 --seed 101 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_registry import BACKBONE   # noqa: E402

N_CLEAN_TRAIN = 4096      # negatives for the error estimator; capped for disk and time


def taps(model, arch: str):
    """(head_module, stage_module) for each architecture.

    head: the final Linear -- hooking its INPUT gives the penultimate feature.
    stage: the last conv stage before the classifier head, pooled by the hook.
    """
    if arch == "resnet50":
        return model.fc, model.layer3
    if arch == "densenet121":
        return model.classifier, model.features.denseblock3
    if arch == "vgg16":
        return model.classifier[-1], model.features
    if arch == "convnext":
        return model.classifier[-1], model.features[5]
    raise ValueError(f"no tap defined for {arch}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing dump (needed to add fields, e.g. __logits, to "
                         "npz files written before this flag existed)")
    a = ap.parse_args()
    ds, bb = a.setting.split("/")

    out = ROOT / "artifacts/prior_feats" / f"{ds}_{bb}_s{a.seed}.npz"
    if out.exists() and not a.force:
        print(f"[skip done] {out.name}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    exp = BACKBONE[(ds, bb)]
    cfg_p, ckpt = ROOT / exp / "config_resolved.yaml", ROOT / exp / "checkpoints/backbone_last.pt"
    if not (cfg_p.exists() and ckpt.exists()):
        print(f"[skip] missing backbone under {exp}")
        return

    from omegaconf import OmegaConf
    from src.data.factory import build_dataset                # noqa: E402
    from src.models.backbones.factory import build_backbone   # noqa: E402

    cfg = OmegaConf.load(cfg_p)
    dev = torch.device(a.device)
    model = build_backbone(architecture=str(cfg.model.architecture),
                           num_classes=int(cfg.dataset.num_classes),
                           pretrained_weights=None).to(dev).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=False)

    head, stage = taps(model, str(cfg.model.architecture))
    buf: dict[str, list] = {"final": [], "stage": []}

    def on_head(_m, inp, _out):
        buf["final"].append(inp[0].detach().float().flatten(1).cpu())

    def on_stage(_m, _inp, out):
        o = out.detach().float()
        buf["stage"].append((o.mean(dim=(2, 3)) if o.ndim == 4 else o.flatten(1)).cpu())

    head.register_forward_hook(on_head)
    stage.register_forward_hook(on_stage)

    eval_ds = build_dataset(cfg, train=False)
    train_ds = build_dataset(cfg, train=True)
    sp = ROOT / f"artifacts/bug_sets/v8_splits_seed{a.seed}/{ds}_{bb}"
    idx_of = lambda n: json.loads((sp / f"{ds}_{n}_indices.json").read_text())["indices"]
    pops = {"bug_train": (eval_ds, idx_of("bug_train")),
            "bug_eval": (eval_ds, idx_of("bug_eval")),
            "clean_calib": (eval_ds, idx_of("clean_calib")),
            "clean_test": (eval_ds, idx_of("clean_test"))}
    rng = np.random.default_rng(a.seed)
    n_tr = len(train_ds)
    pops["clean_train"] = (train_ds,
                           sorted(rng.choice(n_tr, size=min(N_CLEAN_TRAIN, n_tr),
                                             replace=False).tolist()))

    store: dict[str, np.ndarray] = {}
    for name, (dset, idx) in pops.items():
        buf["final"].clear(); buf["stage"].clear()
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(dset, idx),
                                         batch_size=a.batch_size, shuffle=False,
                                         num_workers=a.workers)
        ys, preds, logit_list = [], [], []
        with torch.no_grad():
            for x, y in dl:
                logits = model(x.to(dev))
                preds.append(logits.argmax(1).cpu().numpy())
                logit_list.append(logits.detach().float().cpu().numpy())
                ys.append(np.asarray(y))
        store[f"{name}__final"] = torch.cat(buf["final"]).numpy().astype(np.float32)
        store[f"{name}__stage"] = torch.cat(buf["stage"]).numpy().astype(np.float32)
        store[f"{name}__y"] = np.concatenate(ys).astype(np.int64)
        store[f"{name}__pred"] = np.concatenate(preds).astype(np.int64)
        # the deployed (base) model's raw logits, kept rather than discarded so that
        # Delta z(x) = patched_logits - base_logits can be computed downstream for
        # NN-Patching/PatchNAS, the same way outputs/effect_dump_ep40ns already does for DynaPatch.
        store[f"{name}__logits"] = np.concatenate(logit_list).astype(np.float32)
        store[f"{name}__idx"] = np.asarray(idx, dtype=np.int64)
        print(f"  {name:12s} n={len(idx):6d} final={store[f'{name}__final'].shape[1]:5d} "
              f"stage={store[f'{name}__stage'].shape[1]:5d} "
              f"base_acc={float((store[f'{name}__pred'] == store[f'{name}__y']).mean()):.3f}",
              flush=True)
    np.savez_compressed(out, **store)
    print(f"[done] {out.name}")


if __name__ == "__main__":
    main()
