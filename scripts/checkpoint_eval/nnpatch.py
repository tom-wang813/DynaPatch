#!/usr/bin/env python3
"""NN-Patching (Kauschke & Fuernkranz): a patch head + error estimator on the frozen backbone's
penultimate ("final") feature, from a checkpoint or trained from scratch. Features are extracted
live from the canonical backbone checkpoint (`common.live_prior_features`), matching every other
method in this folder. Reuses the pure scoring helpers (head/predict/prob1/route/calib_tau/metrics)
from scripts/checkpoint_eval/prior_patch_common.py rather than re-deriving them -- those functions,
not the training loop shape, are what a fresh rewrite would risk drifting from.

`--mode checkpoint` loads artifacts/checkpoints/baselines/NNPatch/<ds>_<bb>_s<seed>/
{patch_head,estimator_head}.pt. `--mode train` fits both heads from scratch on the cached features
(same recipe as prior_patch_common.py's nn_patching(), without --early-stop).

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

import prior_patch_common as bpp  # noqa: E402

METHOD = "NNPatch"
FEATURE_KEY = "final"   # NN-Patching taps the penultimate feature; PatchNAS taps "stage" instead.
POPS = ("bug_train", "bug_eval", "clean_calib", "clean_test")


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
    ap.add_argument("--tau", type=float, default=float("-inf"),
                     help="routing threshold (score > tau applies the patch) for the 'ungated' "
                          "row. Default -inf (always route, i.e. patch applied unconditionally) "
                          "-- this is what matches paper.tex's ungated NNPatch/PatchNAS Reg "
                          "(~0.82-0.86, Table rq2_ungated_summary): verified against a tau=0.5 "
                          "default, which under-routes (route_rate 5-20%) and gives Reg ~40x too "
                          "low. 'ungated' here means the same thing it means for DynaPatch-NoGate "
                          "-- no threshold/estimator gate at all, patch applied to every sample.")
    a = ap.parse_args()
    bpp.CRIT[0] = bpp.critical(a.dataset)
    cfg = common.load_cfg(a.dataset, a.backbone)
    d = common.live_prior_features(cfg, a.dataset, a.backbone, a.seed, torch.device(a.device))
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
        # re-deriving it here from the same population prior_patch_common.py's fit() used
        # reproduces the training-time normalisation exactly -- patch head: bug_train alone;
        # estimator: bug_train + clean_train (see estimator()'s own X = concat(f_bug, f_clean)).
        _, patch_apps = bpp.standardise(f["bug_train"], apps)
        _, est_apps = bpp.standardise(np.concatenate([f["bug_train"], ftr]), apps)
        patch = {p: bpp.predict(net, x) for p, x in zip(POPS, patch_apps)}
        score = dict(zip(POPS, [bpp.prob1(est_net, x) for x in est_apps]))
        print(f"[{METHOD}] loaded checkpoint {ckpt_dir}")
    else:
        bpp.CKPT_ROOT[0] = None
        bpp.NCLS[0] = n_classes
        r = bpp.nn_patching(d)
        patch, score = r["patch"], r["score"]

    m = bpp.metrics(d, patch, score, a.tau)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed} tau={a.tau} (unconditional/ungated, "
          f"Table rq2_ungated op point): RR_seen={m['RR_seen']:.4f} RR_held={m['RR_held']:.4f} "
          f"Reg={m['Reg']:.4f} CReg={m['CReg']:.4f} route_rate={m['route_rate']:.4f}")

    # Table rq1_rr's NNPatch/PatchNAS columns use the method's OWN natural operating point:
    # tau=0.5, the estimator's own decision boundary -- paper.tex (Methodology, baseline list)
    # describes NNPatch/PatchNAS as "uses an error estimator to decide whether to apply the
    # patch", not as a threshold recalibrated against a different method's regression budget.
    # FIX (2026-09-22): an earlier version of this table used calib_tau (below, kept as an extra
    # diagnostic only, NOT a paper table's operating point) matched to DynaPatch's own per-cell
    # Reg -- that gave NNPatch/PatchNAS RQ1 means of 0.133/0.173, far below paper's 0.373/0.317.
    # tau=0.5's means (0.330/0.259) are much closer to paper's -- verified across all 12 settings.
    m_natural = bpp.metrics(d, patch, score, 0.5)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed} tau=0.5 (natural estimator threshold, "
          f"Table rq1_rr op point): RR_seen={m_natural['RR_seen']:.4f} "
          f"RR_held={m_natural['RR_held']:.4f} Reg={m_natural['Reg']:.4f} "
          f"CReg={m_natural['CReg']:.4f} route_rate={m_natural['route_rate']:.4f}")

    # Reg-matched calib_tau: NOT used by any of the 9 paper tables (see above) -- kept only as an
    # extra diagnostic (e.g. "what if we budgeted NNPatch/PatchNAS the same Reg as DynaPatch").
    bpp.load_ours_reg()
    target_reg = bpp.OURS_REG.get(f"{a.dataset}/{a.backbone}", 0.016)
    tau_matched = bpp.calib_tau(d, patch, score, target_reg)
    m_matched = bpp.metrics(d, patch, score, tau_matched)
    print(f"[{METHOD}] {a.dataset}/{a.backbone} s{a.seed} tau={tau_matched:.4f} "
          f"(Reg-matched diagnostic, not a paper table op point): "
          f"RR_seen={m_matched['RR_seen']:.4f} RR_held={m_matched['RR_held']:.4f} "
          f"Reg={m_matched['Reg']:.4f} CReg={m_matched['CReg']:.4f} "
          f"route_rate={m_matched['route_rate']:.4f}")

    out_dir = Path(a.output_root) / METHOD / a.dataset / a.backbone / f"s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "summary.json").write_text(json.dumps(
        {"ungated": m, "natural": m_natural, "matched": m_matched}, indent=2))
    print(f"[{METHOD}] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
