#!/usr/bin/env python3
"""Run the PyTorch re-implementation of Arachne on one setting and write standard prediction CSVs.

Arachne repairs the weights of a dense layer, so we target the LAST nn.Linear of the frozen
backbone's classifier and capture its input with a hook -- that keeps it correct for VGG-16,
whose classifier is a multi-layer MLP rather than a single Linear. The backbone is frozen, so
those inputs are cached once and every candidate evaluation during the search is one matmul.

Output layout matches the other baselines (predictions/*.csv), so analyze_mainline.py can read it
by adding one METHODS entry.

Usage:
  uv run python scripts/run_arachne_de.py --setting gtsrb/resnet50 --seed 101 --k full \
      --output-root outputs/fewshot_arachnede_v8_s101_kfull/gtsrb/resnet50
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines.arachne_de import ArachneConfig, repair  # noqa: E402
from src.experiment.stage3 import build_stage3_bundle  # noqa: E402
from src.models.backbones.factory import build_backbone, load_backbone_checkpoint  # noqa: E402

# Cached inputs to the frozen target layer. Independent of every search hyper-parameter, so one
# build per (setting, seed, k, checkpoint) serves an entire sweep. See the note at its use site.
FEAT_CACHE = Path(__file__).resolve().parents[1] / "outputs" / "_arachne_feat_cache"


def freeze_eval(dec) -> None:
    """Put every module the decomposition holds into eval mode.

    `BackboneDecomposition` is a dataclass, not an `nn.Module`, so the modules it holds are NOT
    submodules of the assembled model and `model.eval()` does not reach them. They stay in
    TRAINING mode, and every BatchNorm then normalises with batch statistics instead of the
    running estimates it was fitted with.

    Measured on tt100k_signs/resnet50, clean_test, before this call: the decomposition path agreed
    with an ordinary backbone forward on 0.375 of rows; after it, 1.000. The damage tracks
    normalisation type exactly -- ResNet and DenseNet (BatchNorm) collapsed to base accuracies of
    0.22-0.44 where the split defines them to be 1.000, while ConvNeXt (LayerNorm) and VGG-16 (no
    BatchNorm in the torchvision head) were unaffected. That is why the defect looked dataset-
    specific rather than architectural.
    """
    for m in (dec.shallow, dec.deep, dec.classifier, dec.avgpool, dec.base_model, dec.features,
              dec.feature_layers, dec.norm, dec.permute, dec.flatten):
        if isinstance(m, nn.Module):
            m.eval()


def assert_base_is_frozen_base(model, plain: nn.Module, loader, device, n: int = 512) -> None:
    """Refuse to run unless our `base` really is the frozen base model.

    The whole baseline is a comparison against the base model; if the path we read it from is not
    the path everyone else reads it from, every number this script writes is about a different
    model. This is cheap and it fails loudly, which is the only reason the defect above survived
    an entire experimental campaign.
    """
    d, seen, agree = model.decomposition, 0, 0
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch["input"]).to(device)
            ours = d.classify(d.extract_deep(d.extract_shallow(x)), None).argmax(1)
            agree += int((ours == plain(x).argmax(1)).sum()); seen += len(x)
            if seen >= n:
                break
    if agree != seen:
        raise SystemExit(f"[GUARD base] the decomposition path disagrees with an ordinary "
                         f"backbone forward on {seen - agree}/{seen} rows -- refusing to write "
                         f"predictions attributed to the frozen base.")
    print(f"[arachne] base path verified against the frozen backbone on {seen} rows", flush=True)


def last_linear(module: nn.Module) -> nn.Linear:
    found = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not found:
        raise RuntimeError("no nn.Linear found in the classifier")
    return found[-1]


@torch.no_grad()
def cache_layer_inputs(model, loader, target: nn.Linear, device):
    """Cache (input to `target`, label, base prediction, dataset index) over a loader."""
    feats, labels, preds, idxs = [], [], [], []
    box = {}
    seen = 0          # running offset for the fallback index; see the note at its use site

    def hook(_m, inp, _out):
        box["x"] = inp[0].detach()

    h = target.register_forward_hook(hook)
    try:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
                extra = batch[2] if len(batch) > 2 else None
            else:
                x, y, extra = batch["input"], batch["label"], batch.get("index")
            x = x.to(device)
            logits = model.decomposition.classify(
                model.decomposition.extract_deep(model.decomposition.extract_shallow(x)), None
            )
            feats.append(box["x"].float().cpu())
            labels.append(y.cpu())
            preds.append(logits.argmax(1).cpu())
            # When the loader does not carry a dataset index, fall back to a position -- but a
            # GLOBAL one. This used to be `torch.arange(len(y))`, which restarts at 0 every batch,
            # so the written `dataset_index` column collided across batches and could not be joined
            # against any split manifest. Rows stay in loader order either way.
            idxs.append(extra.cpu() if torch.is_tensor(extra) else torch.arange(seen, seen + len(y)))
            seen += len(y)
    finally:
        h.remove()
    return (torch.cat(feats), torch.cat(labels), torch.cat(preds), torch.cat(idxs))


def write_predictions(path: Path, split: str, idxs, labels, base_pred, base_logits, new_logits):
    path.parent.mkdir(parents=True, exist_ok=True)
    bp = base_logits.softmax(1)
    np_ = new_logits.softmax(1)
    new_pred = new_logits.argmax(1)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "dataset_index", "label", "base_pred", "patched_pred",
                    "base_confidence", "patched_confidence", "base_correct", "patched_correct",
                    "repaired", "regressed"])
        for i in range(len(labels)):
            y, b, n = int(labels[i]), int(base_pred[i]), int(new_pred[i])
            bc, nc = b == y, n == y
            w.writerow([split, int(idxs[i]), y, b, n,
                        float(bp[i, b]), float(np_[i, n]), bc, nc,
                        (not bc) and nc, bc and (not nc)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, help="ds/bb")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", default="full")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--num-places", type=int, default=64)
    ap.add_argument("--localise", choices=["pareto", "topn"], default="pareto",
                    help="pareto = TOSEM'22 as published (num-places is a CAP on the front, and "
                         "the front is 1-14 positions in practice, so raising it is a no-op). "
                         "topn = CAPACITY CONTROL, not Arachne: same bi-objective ranking with "
                         "the Pareto step removed, so num-places becomes the real capacity. Rows "
                         "produced with topn must never be labelled Arachne.")
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--patch-aggr", type=float, default=10.0)
    # bound_scale sets the DE candidate range to init +/- bound_scale*|init|. It lived only in
    # the dataclass, so it was never swept -- and the analogous knob in the greedy TopKSearch
    # baseline (step_scale) turned out to be the binding constraint there, taking RR_held from
    # 0.042 to 0.417 on gtsrb/resnet50. Exposed so Arachne gets the same treatment.
    ap.add_argument("--bound-scale", type=float, default=2.0)
    ap.add_argument("--preserve-split", choices=["clean_eval", "clean_train"],
                    default="clean_eval",
                    help="which clean rows feed Arachne's preservation objective. "
                         "clean_eval is the SHIPPED behaviour and is the REPORTING set "
                         "(clean_test): the search then optimises on 20-100%% of the rows Reg "
                         "is measured on. clean_train uses the dataset's train split, which "
                         "is disjoint from every reported population and matches what "
                         "DynaPatch's clean-replay already does.")
    ap.add_argument("--preserve-cap", type=int, default=2048,
                    help="rows handed to the preservation objective")
    ap.add_argument("--no-feature-cache", action="store_true",
                    help="recompute the target-layer features instead of reusing the cache")
    ap.add_argument("--device", default="cuda:0")
    # Retry knob for a memory-pressured box. The shipped configs ask for 0-12 dataloader workers
    # depending on the cell; each one is forked off a ~2.5 GB parent, and when the host is short
    # of RAM the OOM killer takes a worker and the parent dies with BrokenPipeError -- no CUDA
    # error, no traceback of its own. Overriding to 0 only changes data-loading throughput and
    # the permutation of the shuffled bug_train loader; every population this script evaluates is
    # a SET reduction over those rows, and the two eval loaders are shuffle=False, so no reported
    # number depends on it. Default None leaves the config untouched.
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    ds, bb = args.setting.split("/")
    root = Path(__file__).resolve().parent.parent
    splits = root / f"artifacts/bug_sets/shuffled_split_seed{args.seed}/{ds}_{bb}"
    sub = (splits / f"{ds}_bug_train_indices.json" if args.k == "full"
           else root / f"outputs/fewshot_shuffled_splits/s{args.seed}/{ds}_{bb}_k{args.k}/{ds}_bug_train_indices.json")

    cfg = OmegaConf.load(root / f"configs/shuffled_split_source/{ds}/{bb}/train.yaml")
    for key, val in [
        ("data.bug_indices_path", str(splits / f"{ds}_bug_indices.json")),
        ("data.bug_eval_indices_path", str(splits / f"{ds}_bug_eval_indices.json")),
        ("data.bug_train_indices_path", str(sub)),
        # 2026-07-31 FIX. This used to be `sub` as well, copied from run_fewshot_{safepatch,
        # distrep}.sh -- but those scripts write their held-out predictions from a LATER stage that
        # reads bug_eval directly, whereas this script evaluates straight off
        # build_stage3_bundle -> build_repair_dataloaders, and that builder PREFERS
        # bug_val_indices_path over bug_eval_indices_path (src/data/factory.py:344). So
        # `loaders["bug_eval"]` was the evidence set, and every RR_held this script ever wrote was a
        # second pass over the samples the repair was fitted on. Detected by the label multiset of
        # repair_holdout_unseen matching repair_support_seen exactly on gtsrb/resnet50.
        ("data.bug_val_indices_path", str(splits / f"{ds}_bug_eval_indices.json")),
        ("data.clean_eval_indices_path", str(splits / f"{ds}_clean_test_indices.json")),
        ("runtime.device", args.device),
    ] + ([("runtime.num_workers", args.num_workers)] if args.num_workers is not None else []):
        OmegaConf.update(cfg, key, val, merge=True)
    if ds == "tt100k_signs" and bb == "vgg16":
        OmegaConf.update(cfg, "model.checkpoint_path",
                         "outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt",
                         merge=True)

    device = torch.device(args.device)
    import time as _t
    _T0 = _t.time()
    def _lap(msg):  # phase timing: the 27-53 min/cell was never accounted for
        print(f"[timing] {msg}: +{_t.time()-_lap.prev:.1f}s (total {_t.time()-_T0:.1f}s)", flush=True)
        _lap.prev = _t.time()
    _lap.prev = _T0
    _backbone, model, loaders, _ = build_stage3_bundle(cfg, device)
    _lap("build_stage3_bundle")
    model.eval()
    freeze_eval(model.decomposition)      # model.eval() does not reach it -- see freeze_eval
    plain = build_backbone(architecture=str(cfg.model.architecture),
                           num_classes=int(cfg.dataset.num_classes), pretrained_weights=None)
    plain = load_backbone_checkpoint(plain, str(cfg.model.checkpoint_path)).to(device).eval()
    _lap("load plain backbone")
    assert_base_is_frozen_base(model, plain, loaders["clean_eval"], device)
    _lap("assert_base_is_frozen_base (full clean_eval forward)")
    target = last_linear(model.decomposition.classifier)
    print(f"[arachne] target layer {tuple(target.weight.shape)}", flush=True)

    # FEATURE CACHE. Measured on gtsrb/resnet50: cache_layer_inputs over the three loaders costs
    # 1710-2686 s, while the DE search it feeds costs 1.3 s. Sweeping a search hyper-parameter
    # therefore paid a ~45 min tax per value for ~1 s of actual search. The cached tensors are the
    # inputs to the frozen target layer under the frozen backbone, so they do not depend on any
    # search setting -- only on (setting, seed, k) and the checkpoint. Keyed accordingly, with the
    # checkpoint mtime+size in the key so a rebuilt backbone invalidates the cache rather than
    # silently serving features from a different model.
    ckpt = Path(str(cfg.model.checkpoint_path))
    stat = ckpt.stat() if ckpt.exists() else None
    key = (f"{ds}_{bb}_s{args.seed}_k{args.k}_{int(stat.st_mtime) if stat else 0}"
           f"_{stat.st_size if stat else 0}_pres-{args.preserve_split}")
    cache_f = FEAT_CACHE / f"{key}.npz"
    if cache_f.exists() and not args.no_feature_cache:
        z = np.load(cache_f)
        t = lambda n: torch.as_tensor(z[n])  # noqa: E731
        train_f, train_y, train_p = t("train_f"), t("train_y"), t("train_p")
        clean_f, clean_y, clean_p, clean_i = t("clean_f"), t("clean_y"), t("clean_p"), t("clean_i")
        pres_f, pres_y, pres_p = t("pres_f"), t("pres_y"), t("pres_p")
        held_f, held_y, held_p, held_i = t("held_f"), t("held_y"), t("held_p"), t("held_i")
        _lap(f"feature cache HIT ({cache_f.name})")
    else:
        train_f, train_y, train_p, _ = cache_layer_inputs(model, loaders["bug_train"], target, device)
        clean_f, clean_y, clean_p, clean_i = cache_layer_inputs(model, loaders["clean_eval"], target, device)
        held_f, held_y, held_p, held_i = cache_layer_inputs(model, loaders["bug_eval"], target, device)
        # The PRESERVATION set is a separate object from the REPORTING set. Under
        # --preserve-split clean_train it comes from the dataset's train split, so nothing the
        # search optimises appears in any reported population.
        if args.preserve_split == "clean_train":
            pres_f, pres_y, pres_p, _ = cache_layer_inputs(model, loaders["clean_train"], target, device)
            # Only `preserve_cap` base-correct rows are ever handed to the search, but the
            # train split is 4.5k-27k images and VGG's tap is 25088-d, so caching all of it
            # would put ~16 GB of npz on the NFS mount for no benefit. Keep a margin over the
            # cap. The backbone was TRAINED on this split so it is ~99.9% correct here;
            # a 2x margin yields the full cap of base-correct rows with room to spare.
            keep = min(len(pres_y), args.preserve_cap * 2)
            pres_f, pres_y, pres_p = pres_f[:keep], pres_y[:keep], pres_p[:keep]
        else:
            pres_f, pres_y, pres_p = clean_f, clean_y, clean_p
        _lap("cache_layer_inputs (MISS)")
        if not args.no_feature_cache:
            FEAT_CACHE.mkdir(parents=True, exist_ok=True)
            tmp = cache_f.with_suffix(".tmp.npz")   # atomic: never leave a half-written cache
            np.savez(tmp, train_f=train_f.cpu().numpy(), train_y=train_y.cpu().numpy(),
                     train_p=train_p.cpu().numpy(), clean_f=clean_f.cpu().numpy(),
                     clean_y=clean_y.cpu().numpy(), clean_p=clean_p.cpu().numpy(),
                     pres_f=pres_f.cpu().numpy(), pres_y=pres_y.cpu().numpy(),
                     pres_p=pres_p.cpu().numpy(),
                     clean_i=clean_i.cpu().numpy(), held_f=held_f.cpu().numpy(),
                     held_y=held_y.cpu().numpy(), held_p=held_p.cpu().numpy(),
                     held_i=held_i.cpu().numpy())
            tmp.replace(cache_f)
            _lap("wrote feature cache")

    fail = train_p != train_y
    ok = pres_p == pres_y            # preservation is defined on base-correct rows
    print(f"[arachne] evidence failures={int(fail.sum())} "
          f"preserve set={int(ok.sum())} from {args.preserve_split} "
          f"(cap {args.preserve_cap}) report clean={len(clean_y)} "
          f"heldout={len(held_y)}", flush=True)

    acfg = ArachneConfig(num_places=args.num_places, localise=args.localise,
                         pop_size=args.pop_size,
                         max_iter=args.max_iter, patch_aggr=args.patch_aggr,
                         bound_scale=args.bound_scale, seed=args.seed)
    W, info = repair(target.to(device), train_f[fail].to(device), train_y[fail].to(device),
                     pres_f[ok][:args.preserve_cap].to(device),
                     pres_y[ok][:args.preserve_cap].to(device), acfg)
    _lap("DE repair")
    print(f"[arachne] {info}", flush=True)

    b = target.bias.detach().cpu()
    W0, Wn = target.weight.detach().cpu(), W.cpu()

    ckpt_dir = Path(args.output_root) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": Wn, "bias": b, "target_layer_shape": list(Wn.shape)},
               ckpt_dir / "arachne_patched_classifier.pt")

    out = Path(args.output_root) / "predictions"
    for name, f_, y_, p_, i_ in [("repair_holdout_unseen", held_f, held_y, held_p, held_i),
                                 ("repair_support_seen", train_f, train_y, train_p,
                                  torch.arange(len(train_y))),
                                 ("clean_eval", clean_f, clean_y, clean_p, clean_i)]:
        write_predictions(out / f"{name}_predictions.csv", name, i_, y_, p_,
                          f_ @ W0.t() + b, f_ @ Wn.t() + b)
    # The search configuration is serialised whole, not as a hand-listed subset. A table built
    # from these trees has to be able to show whether this baseline was tuned or left at library
    # defaults, and until now it could not: no tree recorded bound_scale, patch_aggr, num_places,
    # pop_size or max_iter, which is how bound_scale=2.0 reached a reported table unswept
    # (RR_held 0.125 -> 0.375 at 128). asdict() cannot drift out of sync with the dataclass the
    # way an enumerated list of keys does. Read back by scripts/guards.py:recorded_baseline_params.
    (Path(args.output_root) / "metrics.json").write_text(json.dumps(
        {"setting": args.setting, "seed": args.seed, "k": args.k,
         "preserve_split": args.preserve_split, "preserve_cap": args.preserve_cap,
         "n_preserve": int(ok.sum()),
         "baseline_config": {"method": "Arachne", **asdict(acfg)},
         "target_layer_shape": list(target.weight.shape), **info}, indent=2))
    print(f"[arachne] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
