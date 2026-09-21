#!/usr/bin/env python3
"""NN-Patching (Kauschke & Fuernkranz): a patch head + error estimator on the frozen backbone's
penultimate ("final") feature, from a checkpoint or trained from scratch.

Unlike the other 5 methods, NN-Patching/PatchNAS operate on features CACHED by
scripts/dump_prior_features.py (artifacts/prior_feats/<ds>_<bb>_s<seed>.npz), not on raw images --
that cache must exist first (`uv run python scripts/dump_prior_features.py --setting <ds>/<bb>
--seed <seed>`). Reuses the pure scoring helpers (head/predict/prob1/route/calib_tau/metrics) from
scripts/baseline_prior_patches.py rather than re-deriving them -- those functions, not the training
loop shape, are what a fresh rewrite would risk drifting from.

`--mode checkpoint` loads artifacts/checkpoints/baselines/NNPatch/<ds>_<bb>_s<seed>/
{patch_head,estimator_head}.pt. `--mode train` fits both heads from scratch on the cached features
(same recipe as baseline_prior_patches.py's nn_patching(), without --early-stop).

Usage:
  uv run python scripts/checkpoint_eval/nnpatch.py --dataset gtsrb --backbone resnet50 \
      --mode checkpoint --output-root outputs/ckpt_eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.ROOT))
import scripts.baseline_prior_patches as bpp  # noqa: E402

METHOD = "NNPatch"
FEATURE_KEY = "final"   # NN-Patching taps the penultimate feature; PatchNAS taps "stage" instead.
POPS = ("bug_train", "bug_eval", "clean_calib", "clean_test")


def load_npz(dataset: str, backbone: str, seed: int) -> dict:
    p = common.ROOT / f"artifacts/prior_feats/{dataset}_{backbone}_s{seed}.npz"
    if not p.exists():
        raise SystemExit(f"[{METHOD}] no cached features at {p} -- run "
                          f"scripts/dump_prior_features.py --setting {dataset}/{backbone} "
                          f"--seed {seed} first.")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def build_head_from_ckpt(ckpt: dict) -> nn.Module:
    depth, width, act = ckpt["arch"] if isinstance(ckpt["arch"], (list, tuple)) else (0, 0, "relu")
    net = bpp.head(int(ckpt["din"]), int(ckpt["dout"]), int(depth), int(width), str(act))
    net.load_state_dict(ckpt["state_dict"])
    return net.to(bpp.DEV).eval()


def build_estimator_from_ckpt(ckpt: dict, din: int) -> nn.Module:
    arch = ckpt.get("arch", "linear")
    net = bpp.head(din, 2, 0, 0, "relu") if arch == "linear" else bpp.head(din, 2, 2, 256, "relu")
    net.load_state_dict(ckpt["state_dict"])
    return net.to(bpp.DEV).eval()


def main() -> None:
    ap = common.base_argparser(__doc__)
    ap.add_argument("--tau", type=float, default=0.5,
                     help="routing threshold (score > tau applies the patch). 0.5 matches the "
                          "paper's ungated NNPatch/PatchNAS rows (Table rq2_ungated_summary).")
    ap.add_argument("--features", choices=["live", "cached"], default="live",
                     help="live (default): extract features from the current canonical backbone "
                          "checkpoint (artifacts/checkpoints/backbones/), matching every other "
                          "method in this folder. cached: use artifacts/prior_feats/*.npz -- "
                          "faster, but that cache's own backbone_registry.py path "
                          "(outputs/exp_..._public_v2) no longer exists on disk, so it may not "
                          "be the same backbone weights.")
    ap.add_argument("--per-sample", default=None,
                     help="directory to also write per-sample prediction CSVs + base/patched "
                          "logits into (both operating points, tagged tau/matched), in the same "
                          "layout scripts/baseline_prior_patches.py:dump_per_sample uses -- what "
                          "scripts/analysis_patch_reassignment.py reads.")
    a = ap.parse_args()
    bpp.CRIT[0] = bpp.critical(a.dataset)
    if a.features == "live":
        cfg = common.load_cfg(a.dataset, a.backbone)
        d = common.live_prior_features(cfg, a.dataset, a.backbone, a.seed, torch.device(a.device))
    else:
        d = load_npz(a.dataset, a.backbone, a.seed)
    n_classes = int(max(d["clean_test__y"].max(), d["bug_train__y"].max(),
                         d["bug_eval__y"].max(), d["clean_train__y"].max()) + 1)

    ckpt_dir = common.checkpoint_dir(METHOD, a.dataset, a.backbone, a.seed)
    f = {p: d[f"{p}__{FEATURE_KEY}"] for p in POPS}
    ftr = d[f"clean_train__{FEATURE_KEY}"]
    apps = [f[p] for p in POPS]

    if a.mode == "checkpoint":
        patch_ckpt = torch.load(ckpt_dir / "patch_head.pt", map_location="cpu")
        est_ckpt = torch.load(ckpt_dir / "estimator_head.pt", map_location="cpu")
        net = build_head_from_ckpt(patch_ckpt)
        est_net = build_estimator_from_ckpt(est_ckpt, f["bug_train"].shape[1])
        # standardisation is a fixed function of the DATA (mean/std), not a learned parameter, so
        # re-deriving it here from the same population baseline_prior_patches.py's fit() used
        # reproduces the training-time normalisation exactly -- patch head: bug_train alone;
        # estimator: bug_train + clean_train (see estimator()'s own X = concat(f_bug, f_clean)).
        _, patch_apps = bpp.standardise(f["bug_train"], apps)
        _, est_apps = bpp.standardise(np.concatenate([f["bug_train"], ftr]), apps)
        patch = {p: bpp.predict(net, x) for p, x in zip(POPS, patch_apps)}
        patch_logits = {p: bpp.predict_logits(net, x) for p, x in zip(POPS, patch_apps)}
        score = dict(zip(POPS, [bpp.prob1(est_net, x) for x in est_apps]))
        print(f"[{METHOD}] loaded checkpoint {ckpt_dir}")
    else:
        bpp.CKPT_ROOT[0] = None
        bpp.NCLS[0] = n_classes
        r = bpp.nn_patching(d)
        patch, score, patch_logits = r["patch"], r["score"], r["patch_logits"]

    m = bpp.metrics(d, patch, score, a.tau)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed} tau={a.tau} (ungated, Table rq2 op point): "
          f"RR_seen={m['RR_seen']:.4f} RR_held={m['RR_held']:.4f} Reg={m['Reg']:.4f} "
          f"CReg={m['CReg']:.4f} route_rate={m['route_rate']:.4f}")

    # Table rq1_rr's NNPatch/PatchNAS columns are NOT at tau=0.5 -- baseline_prior_patches.py's
    # own main() reports them at a threshold calibrated on clean_calib to match DynaPatch's own
    # Reg for this cell (`calib_tau`, its "matched" operating point). Reproduced here the same way,
    # with the same 0.016 fallback when outputs/paper_tables.json (DynaPatch's per-cell Reg) isn't
    # present -- comparing tau=0.5's numbers against Table rq1_rr would be the wrong operating point.
    bpp.load_ours_reg()
    target_reg = bpp.OURS_REG.get(f"{a.dataset}/{a.backbone}", 0.016)
    tau_matched = bpp.calib_tau(d, patch, score, target_reg)
    m_matched = bpp.metrics(d, patch, score, tau_matched)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed} tau={tau_matched:.4f} "
          f"(matched to Reg<={target_reg:.4f}, Table rq1_rr op point): "
          f"RR_seen={m_matched['RR_seen']:.4f} RR_held={m_matched['RR_held']:.4f} "
          f"Reg={m_matched['Reg']:.4f} CReg={m_matched['CReg']:.4f} "
          f"route_rate={m_matched['route_rate']:.4f}")

    if a.per_sample:
        r = {"patch": patch, "score": score, "patch_logits": patch_logits}
        for tval, tag in ((a.tau, "tau"), (tau_matched, "matched")):
            bpp.dump_per_sample(Path(a.per_sample), METHOD, a.dataset, a.backbone, a.seed, d, r, tval, tag)
        print(f"[{METHOD}] wrote per-sample dumps to {a.per_sample}")

    out_dir = Path(a.output_root) / METHOD / a.dataset / a.backbone / f"s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "summary.json").write_text(json.dumps({"ungated": m, "matched": m_matched}, indent=2))
    print(f"[{METHOD}] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
