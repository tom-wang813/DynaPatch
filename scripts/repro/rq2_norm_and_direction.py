#!/usr/bin/env python3
"""RQ2 (mechanism part): alignment/magnitude/margin-gain (Table rq2_norm) and direction/magnitude
reassignment (Table rq2_direction), computed directly from paper.tex's own equations
(logit_change, alignment, repair_margin_gain -- Methodology's "Patch Behavior Metrics" paragraph),
restricted to the FAILURE population (bug_train union bug_eval, i.e. every input the base model
got wrong -- by construction of the split manifests, base_pred != label everywhere in this
population already).

Delta-ell_M(x) = patched_logits - base_logits, UNROUTED/UNGATED (the raw effect of the patch
itself, before any accept/reject decision -- DPNoGate's ungated dump already is this; NNPatch/
PatchNAS's patch head output here deliberately skips their own routing estimator, since the
question is what the patch DOES, not whether that method would have applied it).

alpha_M(x)  = (dl[y] - dl[y_ori]) / (sqrt(2) * ||dl||_2)              -- Eq. alignment
dm_rep(x)   = dl[y] - dl[y_ori]                                       -- Eq. repair_margin_gain
where y = label, y_ori = base_pred (argmax base_logits).
Table rq2_norm reports MEAN alpha_M, MEDIAN ||dl||_2, MEDIAN dm_rep (matches paper.tex's own
text: "mean directional alignment", "median logit-change norm", "median margin gain").

Table rq2_direction ("direction/magnitude reassignment"): paper.tex's main text does not give the
exact reassignment procedure -- OPERATIONALIZED HERE (flagged, not silently assumed) as: shuffle
the failure population once per (method, setting); pair each failure x with the shuffled x'
(no self-pairs by construction of a derangement-style shift-by-1 on the shuffle); direction-
reassigned dl' = ||dl(x)|| * unit(dl(x')), magnitude-reassigned dl' = ||dl(x')|| * unit(dl(x));
patched_logits' = base_logits(x) + dl'; recompute predicted class; RR' = accuracy of predicted
class against label over the failure population; report RR' - RR (paper's sign convention: a
negative number is a decrease).

Usage:
  uv run python scripts/repro/rq2_norm_and_direction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/checkpoint_eval"))

import common  # noqa: E402
import prior_patch_common as bpp  # noqa: E402

SEED = 101
EFFECT_DUMP = ROOT / f"outputs/effect_dump_v8_s{SEED}"
FIXEDPATCH_DUMP = ROOT / "outputs/repro/ckpt_eval/FixedPatch_routefeat"
RNG_SEED = 0

ALL_SETTINGS = [
    (ds, bb)
    for ds in ("gtsrb", "tt100k_signs", "lisa_signs")
    for bb in ("resnet50", "convnext_tiny", "densenet121", "vgg16")
]


def _load_dump_split(dump_dir: Path, split: str) -> dict:
    pred_dir = dump_dir / "predictions"
    import csv
    rows = list(csv.DictReader((pred_dir / f"{split}_predictions.csv").open()))
    base_logits = np.load(pred_dir / f"base_logits_{split}.npy")
    patched_logits = np.load(pred_dir / f"patched_logits_{split}.npy")
    label = np.array([int(r["label"]) for r in rows])
    base_pred = np.array([int(r["base_pred"]) for r in rows])
    return {"base_logits": base_logits, "patched_logits": patched_logits, "label": label, "base_pred": base_pred}


def _cat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def dpnogate_data(dataset: str, backbone: str) -> dict | None:
    d = EFFECT_DUMP / dataset / backbone / "deploy_direct"
    if not d.exists():
        return None
    return _cat([_load_dump_split(d, s) for s in ("repair_support_seen", "repair_holdout_unseen")])


def fixedpatch_data(dataset: str, backbone: str) -> dict | None:
    d = FIXEDPATCH_DUMP / dataset / backbone / f"s{SEED}" / "deploy"
    if not d.exists():
        return None
    return _cat([_load_dump_split(d, s) for s in ("repair_support_seen", "repair_holdout_unseen")])


def prior_patch_data(method: str, dataset: str, backbone: str) -> dict | None:
    """Raw (unrouted) patch logits for NNPatch ("final" feature) / PatchNAS ("stage" feature)."""
    ckpt_dir = common.checkpoint_dir(method, dataset, backbone, SEED)
    if not (ckpt_dir / "patch_head.pt").exists():
        return None
    feature_key = "final" if method == "NNPatch" else "stage"
    cfg = common.load_cfg(dataset, backbone)
    d = common.live_prior_features(cfg, dataset, backbone, SEED, torch.device("cuda:0"))
    pops = ("bug_train", "bug_eval")
    f = {p: d[f"{p}__{feature_key}"] for p in pops}
    patch_ckpt = torch.load(ckpt_dir / "patch_head.pt", map_location="cpu")
    depth, width, act = patch_ckpt["arch"] if isinstance(patch_ckpt["arch"], (list, tuple)) else (0, 0, "relu")
    net = bpp.head(int(patch_ckpt["din"]), int(patch_ckpt["dout"]), int(depth), int(width), str(act))
    net.load_state_dict(patch_ckpt["state_dict"])
    net.to(bpp.DEV).eval()
    _, apps = bpp.standardise(f["bug_train"], [f[p] for p in pops])
    patch_logits = {p: bpp.predict_logits(net, x) for p, x in zip(pops, apps)}
    return {
        "base_logits": np.concatenate([d[f"{p}__logits"] for p in pops]),
        "patched_logits": np.concatenate([patch_logits[p] for p in pops]),
        "label": np.concatenate([d[f"{p}__y"] for p in pops]),
        "base_pred": np.concatenate([d[f"{p}__pred"] for p in pops]),
    }


def norm_metrics(data: dict) -> dict:
    base_logits, patched_logits = data["base_logits"], data["patched_logits"]
    label, base_pred = data["label"], data["base_pred"]
    n = base_logits.shape[0]
    rows = np.arange(n)
    dl = patched_logits - base_logits
    dl_y = dl[rows, label]
    dl_yori = dl[rows, base_pred]
    norm = np.linalg.norm(dl, axis=1)
    dm_rep = dl_y - dl_yori
    alpha = dm_rep / (np.sqrt(2.0) * (norm + 1e-12))
    return {
        "alignment_mean": float(alpha.mean()),
        "norm_median": float(np.median(norm)),
        "margin_gain_median": float(np.median(dm_rep)),
        "n": n,
        "_alpha": alpha.tolist(), "_norm": norm.tolist(), "_dm_rep": dm_rep.tolist(),
    }


def direction_magnitude_reassignment(data: dict, seed: int = RNG_SEED) -> dict:
    base_logits, patched_logits = data["base_logits"], data["patched_logits"]
    label, base_pred = data["label"], data["base_pred"]
    n = base_logits.shape[0]
    if n < 2:
        return {"rr": None, "rr_direction": None, "rr_magnitude": None,
                "direction_decrease": None, "magnitude_decrease": None, "n": n}
    dl = patched_logits - base_logits
    norm = np.linalg.norm(dl, axis=1, keepdims=True)
    unit = dl / (norm + 1e-12)

    rng = np.random.default_rng(seed)
    partner_idx = rng.permutation(n)
    fixed = np.flatnonzero(partner_idx == np.arange(n))
    for i in fixed:  # break any self-pairing left by the random permutation
        j = (i + 1) % n
        partner_idx[i], partner_idx[j] = partner_idx[j], partner_idx[i]

    def final_pred(dl_alt: np.ndarray) -> np.ndarray:
        return (base_logits + dl_alt).argmax(axis=1)

    rr = float((final_pred(dl) == label).mean())

    dl_dir = norm * unit[partner_idx]                    # own magnitude, partner's direction
    dl_mag = norm[partner_idx] * unit                    # own direction, partner's magnitude
    rr_direction = float((final_pred(dl_dir) == label).mean())
    rr_magnitude = float((final_pred(dl_mag) == label).mean())

    return {"rr": rr, "rr_direction": rr_direction, "rr_magnitude": rr_magnitude,
            "direction_decrease": rr_direction - rr, "magnitude_decrease": rr_magnitude - rr, "n": n}


def main() -> None:
    getters = {
        "FixedPatch": fixedpatch_data,
        "DynaPatch-NoGate": dpnogate_data,
        "NNPatch": lambda ds, bb: prior_patch_data("NNPatch", ds, bb),
        "PatchNAS": lambda ds, bb: prior_patch_data("PatchNAS", ds, bb),
    }
    results = {m: {} for m in getters}
    for dataset, backbone in ALL_SETTINGS:
        label = f"{dataset}/{backbone}"
        print(f"=== {label} ===")
        for method, getter in getters.items():
            data = getter(dataset, backbone)
            if data is None:
                print(f"  {method}: no data")
                results[method][label] = None
                continue
            nm = norm_metrics(data)
            rd = direction_magnitude_reassignment(data)
            results[method][label] = {**nm, **rd}
            print(f"  {method}: {results[method][label]}")

    out_path = ROOT / "outputs/repro/rq2_norm_direction_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
