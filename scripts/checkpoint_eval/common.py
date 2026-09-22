"""Shared config/data/scoring plumbing for the per-method checkpoint-eval scripts in this folder.

Each sibling script (headft.py, arachne.py, ...) supports two ways to get a patched model for one
(dataset, backbone, seed) setting -- `--mode checkpoint` (load the artifact already sitting under
`artifacts/checkpoints/baselines/<Method>/...`) or `--mode train` (run that method's own real
training/search routine from src/baselines/, then score the result the same way) -- and both modes
end up here for everything that is NOT method-specific: which config to load, how the three
evaluation loaders are built, how predictions are written, and how RR_seen/RR_held/Reg/CReg are
computed from them.

Reuses the same building blocks the rest of the repo's data/model plumbing uses
(src/data/factory.py, src/models/backbones/factory.py) rather than re-deriving dataset/backbone
construction here -- see the module docstrings there for the config fields each one reads.

Loader note: build_repair_dataloaders() (src/data/factory.py) shuffles bug_train and drops the
original dataset index per row, which is fine for training but means predictions can't be joined
back to a split manifest. The loaders here wrap the same Subset()-over-test_dataset construction
with shuffle=False instead, so `indices[i]` is the dataset_index for row i of loader i directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.factory import build_dataset, _load_index_manifest  # noqa: E402
from src.models.backbones.factory import build_backbone, load_backbone_checkpoint  # noqa: E402

# (loader_key, index_file_suffix) -- index_file_suffix is the `<dataset>_<suffix>.json` under
# artifacts/bug_sets/shuffled_split_seed<seed>/<dataset>_<backbone>/
SPLITS = (
    ("repair_support_seen", "bug_train_indices"),
    ("repair_holdout_unseen", "bug_eval_indices"),
    ("clean_eval", "clean_test_indices"),
)


def split_dir(dataset: str, backbone: str, seed: int) -> Path:
    return ROOT / f"artifacts/bug_sets/shuffled_split_seed{seed}/{dataset}_{backbone}"


def checkpoint_dir(method: str, dataset: str, backbone: str, seed: int = 101) -> Path:
    return ROOT / f"artifacts/checkpoints/baselines/{method}/{dataset}_{backbone}_s{seed}"


def backbone_checkpoint_path(dataset: str, backbone: str) -> Path:
    return ROOT / f"artifacts/checkpoints/backbones/{dataset}_{backbone}/backbone_last.pt"


def load_cfg(dataset: str, backbone: str):
    """The setting's train.yaml, pointed at the manifest-listed frozen-backbone checkpoint."""
    cfg = OmegaConf.load(ROOT / f"configs/shuffled_split_source/{dataset}/{backbone}/train.yaml")
    OmegaConf.update(cfg, "model.checkpoint_path", str(backbone_checkpoint_path(dataset, backbone)), merge=True)
    return cfg


def build_frozen_backbone(cfg, device: torch.device) -> torch.nn.Module:
    model = build_backbone(
        architecture=str(cfg.model.architecture),
        num_classes=int(cfg.dataset.num_classes),
        pretrained_weights=None,
    )
    return load_backbone_checkpoint(model, str(cfg.model.checkpoint_path)).to(device).eval()


def build_eval_loaders(cfg, dataset: str, backbone: str, seed: int,
                        batch_size: int = 64, num_workers: int = 4) -> dict[str, tuple[list[int], DataLoader]]:
    """{'repair_support_seen'|'repair_holdout_unseen'|'clean_eval': (dataset_indices, loader)},
    loaders in shuffle=False manifest order so row i of the loader is dataset_indices[i]."""
    test_dataset = build_dataset(cfg, train=False)
    sdir = split_dir(dataset, backbone, seed)
    out = {}
    for split_name, suffix in SPLITS:
        path = sdir / f"{dataset}_{suffix}.json"
        indices = _load_index_manifest(str(path), field_name=split_name)
        loader = DataLoader(Subset(test_dataset, indices), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
        out[split_name] = (indices, loader)
    return out


N_CLEAN_TRAIN = 4096   # same cap scripts/dump_prior_features.py uses for the estimator's negatives


def taps(model, arch: str):
    """(head_module, stage_module) per architecture -- same taps as dump_prior_features.py, for
    NN-Patching/PatchNAS's "final"/"stage" features. Kept here rather than imported from that
    script only because that script's main() has other, heavier import-time dependencies
    (backbone_registry) that this checkpoint-eval folder deliberately doesn't take on."""
    if arch == "resnet50":
        return model.fc, model.layer3
    if arch == "densenet121":
        return model.classifier, model.features.denseblock3
    if arch == "vgg16":
        return model.classifier[-1], model.features
    if arch == "convnext":
        return model.classifier[-1], model.features[5]
    raise ValueError(f"no tap defined for {arch}")


@torch.no_grad()
def live_prior_features(cfg, dataset: str, backbone: str, seed: int,
                         device: torch.device) -> dict[str, "np.ndarray"]:
    """Live equivalent of artifacts/prior_feats/<ds>_<bb>_s<seed>.npz, extracted from the CURRENT
    canonical backbone checkpoint (artifacts/checkpoints/backbones/) rather than whatever
    checkpoint path scripts/backbone_registry.py named when that npz was cached -- the registry
    points at an outputs/exp_..._public_v2 tree that no longer exists on disk, so the cached
    features may not be from the same backbone weights the rest of this pipeline uses. Same key
    scheme as the npz: {pop}__final, {pop}__stage, {pop}__y, {pop}__pred."""
    import numpy as np

    arch = "convnext" if backbone == "convnext_tiny" else backbone
    model = build_frozen_backbone(cfg, device)
    head, stage = taps(model, arch)
    buf: dict[str, list] = {"final": [], "stage": []}

    def on_head(_m, inp, _out):
        buf["final"].append(inp[0].detach().float().flatten(1).cpu())

    def on_stage(_m, _inp, out):
        o = out.detach().float()
        buf["stage"].append((o.mean(dim=(2, 3)) if o.ndim == 4 else o.flatten(1)).cpu())

    head.register_forward_hook(on_head)
    stage.register_forward_hook(on_stage)

    sdir = split_dir(dataset, backbone, seed)
    eval_ds = build_dataset(cfg, train=False)
    train_ds = build_dataset(cfg, train=True)
    pops = {
        "bug_train": (eval_ds, _load_index_manifest(str(sdir / f"{dataset}_bug_train_indices.json"), field_name="bug_train")),
        "bug_eval": (eval_ds, _load_index_manifest(str(sdir / f"{dataset}_bug_eval_indices.json"), field_name="bug_eval")),
        "clean_calib": (eval_ds, _load_index_manifest(str(sdir / f"{dataset}_clean_calib_indices.json"), field_name="clean_calib")),
        "clean_test": (eval_ds, _load_index_manifest(str(sdir / f"{dataset}_clean_test_indices.json"), field_name="clean_test")),
    }
    rng = np.random.default_rng(seed)   # same seed dump_prior_features.py uses for this sample
    n_tr = len(train_ds)
    pops["clean_train"] = (train_ds, sorted(rng.choice(n_tr, size=min(N_CLEAN_TRAIN, n_tr),
                                                       replace=False).tolist()))

    store: dict[str, "np.ndarray"] = {}
    for name, (dset, idx) in pops.items():
        buf["final"].clear(); buf["stage"].clear()
        loader = DataLoader(Subset(dset, idx), batch_size=64, shuffle=False, num_workers=4)
        ys, preds, logit_list = [], [], []
        for x, y in loader:
            logits = model(x.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            logit_list.append(logits.detach().float().cpu().numpy())
            ys.append(np.asarray(y))
        store[f"{name}__final"] = torch.cat(buf["final"]).numpy().astype(np.float32)
        store[f"{name}__stage"] = torch.cat(buf["stage"]).numpy().astype(np.float32)
        store[f"{name}__y"] = np.concatenate(ys).astype(np.int64)
        store[f"{name}__pred"] = np.concatenate(preds).astype(np.int64)
        # base (deployed) model's own logits -- for computing Delta z(x) = patched - base later,
        # the same way outputs/effect_dump_*/deploy_direct already carries these for DynaPatch
        # (see scripts/analysis_patch_reassignment.py).
        store[f"{name}__logits"] = np.concatenate(logit_list).astype(np.float32)
        store[f"{name}__idx"] = np.asarray(idx, dtype=np.int64)
    return store


def crit_classes(dataset: str) -> set[int]:
    cfg = json.loads((ROOT / f"artifacts/risk/{dataset}_safety_risk_matrix.json").read_text())
    out: set[int] = set()
    for ids in cfg["critical_signs"].values():
        out.update(int(i) for i in ids)
    return out


@torch.no_grad()
def score_split(base_model: torch.nn.Module, patched_model: torch.nn.Module,
                 indices: list[int], loader: DataLoader, device: torch.device) -> list[dict]:
    """Row-per-sample base vs. patched predictions, in dataset_index order."""
    rows, seen = [], 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        base_logits = base_model(x)
        patched_logits = patched_model(x)
        base_pred = base_logits.argmax(1).cpu()
        patched_pred = patched_logits.argmax(1).cpu()
        base_conf = base_logits.softmax(1).cpu()
        patched_conf = patched_logits.softmax(1).cpu()
        for i in range(len(y)):
            idx = indices[seen + i]
            label, b, n = int(y[i]), int(base_pred[i]), int(patched_pred[i])
            bc, nc = b == label, n == label
            rows.append({
                "dataset_index": idx, "label": label, "base_pred": b, "patched_pred": n,
                "base_confidence": float(base_conf[i, b]), "patched_confidence": float(patched_conf[i, n]),
                "base_correct": bc, "patched_correct": nc,
                "repaired": (not bc) and nc, "regressed": bc and (not nc),
            })
        seen += len(y)
    return rows


def write_predictions(out_dir: Path, split_name: str, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split_name}_predictions.csv"
    fields = ["dataset_index", "label", "base_pred", "patched_pred", "base_confidence",
              "patched_confidence", "base_correct", "patched_correct", "repaired", "regressed"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _tf(x: object) -> bool:
    return str(x).lower() == "true"


def summarize(pred_dir: Path, dataset: str) -> dict[str, float] | None:
    """RR_seen/RR_held/Reg/CReg -- same definitions as scripts/table_from_checkpoints.py's cell()."""
    files = {name: pred_dir / f"{name}_predictions.csv" for name, _ in SPLITS}
    if not all(p.exists() for p in files.values()):
        return None
    sn = list(csv.DictReader(files["repair_support_seen"].open()))
    hd = list(csv.DictReader(files["repair_holdout_unseen"].open()))
    cl = list(csv.DictReader(files["clean_eval"].open()))
    cr = crit_classes(dataset)
    cc = [r for r in cl if int(r["label"]) in cr]
    return {
        "rr_seen": sum(_tf(r["patched_correct"]) for r in sn) / max(len(sn), 1),
        "rr_held": sum(_tf(r["patched_correct"]) for r in hd) / max(len(hd), 1),
        "reg": sum(1 for r in cl if _tf(r["base_correct"]) and not _tf(r["patched_correct"])) / max(len(cl), 1),
        "creg": sum(1 for r in cc if _tf(r["base_correct"]) and not _tf(r["patched_correct"])) / max(len(cc), 1),
        "n_seen": len(sn), "n_held": len(hd), "n_clean": len(cl), "n_crit": len(cc),
    }


def base_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--dataset", required=True, choices=["gtsrb", "tt100k_signs", "lisa_signs"])
    ap.add_argument("--backbone", required=True, choices=["resnet50", "convnext_tiny", "densenet121", "vgg16"])
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--mode", choices=["checkpoint", "train"], default="checkpoint",
                     help="checkpoint: load artifacts/checkpoints/baselines/<Method>/... and just "
                          "score it. train: run the method's own training/search from scratch, "
                          "then score exactly the same way.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    return ap
