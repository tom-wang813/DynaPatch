#!/usr/bin/env python3
"""Leave-one-backbone-out test of intervention-response signals as a repair gate.

Question
--------
Can we predict whether a proposed repair will help or harm an unseen input using
*intervention-level* signals (how the patch moved the decision) rather than
*input-level* signals (where the input sits in latent space), and does that
prediction survive a change of backbone?

Why this is not one classifier
------------------------------
Measured over all 36 cells of outputs/effect_dump_v8_s*/: regression events occur
only on clean inputs (8,526 on clean_eval) and repair events only on failure inputs
(2,887 on repair_holdout_unseen). The two label classes are perfectly separated by
which split a row came from, so a single head fit on the pooled rows learns
"clean vs failure" -- a population classifier with a near-1 and meaningless AUROC.
This script therefore never pools: it runs two strictly within-population arms,

    HARM  on clean_eval           : among clean inputs, which does the patch break?
    HELP  on repair_holdout_unseen: among known failures, which does the patch fix?

Why AUROC is not the verdict
----------------------------
The shipped latent gate was killed by scripts/analyze_gate_sweep.py:140 -- 15 of 24
active cells merely "slid the curve" (bought Reg by giving back RR_held). AUROC cannot
see that failure. Two guards are therefore built in:

  * MAGNITUDE-MATCHED STRATA. rho = dm/(|m_base|+eps) is monotone in patch strength, so
    thresholding it can reproduce curve-sliding exactly. We recompute AUROC *within*
    quintiles of ||delta|| -- if it collapses to 0.5 the signal is only magnitude.
  * DEPLOYMENT SIMULATION. Suppressing the patch on a row is exact and needs no re-run
    (a suppressed clean row cannot regress; a suppressed failure row cannot be repaired),
    so we can trace the true Reg/RR_held curve of the gate and compare it against the
    magnitude-scaling null, then apply the pre-registered trichotomy.

Everything reads existing dumps; nothing is retrained.

Usage:
  .venv/bin/python scripts/analyze_response_gate_lobo.py
  .venv/bin/python scripts/analyze_response_gate_lobo.py --arms harm --feature-sets response mag
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DSS = ["gtsrb", "tt100k_signs", "lisa_signs"]
BBS = ["resnet50", "convnext_tiny", "densenet121", "vgg16"]
SEEDS = [101, 202, 303]
# deploy_direct is the HELD pass: repair_holdout_unseen has ~146 rows/cell there versus
# ~29 in deploy_direct_calib, so the HELP arm is only adequately powered on this pass.
PASS_DIR = "deploy_direct"
# DUMP_TAG selects WHICH trained patch the response features come from. "" is the shipped
# 12-epoch patch (outputs/effect_dump_v8_s*); "_ep40ns" is the 40-epoch no-early-stop patch
# dumped 2026-08-30. The feature cache is tagged with it so the two never mix.
import os as _os
DUMP_TAG = _os.environ.get("DUMP_TAG", "")
EPS = 1e-6

# split -> (label column, positive meaning)
ARMS = {
    "harm": ("clean_eval", "regressed"),
    # The dataset's TRAIN split, dumped 2026-09-01 into outputs/effect_dump_ct_ep40ns_v8_s*.
    # Disjoint from BOTH reported populations, so it can enlarge the gate's negative class
    # without touching anything the paper reports. Read it with DUMP_TAG=_ct_ep40ns.
    "harm_train": ("clean_train", "regressed"),
    "help": ("repair_holdout_unseen", "repaired"),
    # The honest HELP question. Measured over all 36 cells:
    #   regressed == flip AND base_correct   -- 8,526 rows, zero exceptions
    #   P(repaired | flip) == 2,887/3,842 == 0.751
    # So on clean inputs the outcome is a *definition*, not a prediction: any model that sees
    # `flip` scores AUROC 1.000 for free. On failure inputs the flip is only necessary, and the
    # open question is whether it lands on the right class. `help_flipped` restricts to rows the
    # patch already flips, which deletes the tautology and leaves the real problem.
    "help_flipped": ("repair_holdout_unseen", "repaired"),
    # the reported failures, needed to state what authorization does to RR_repair. Same shape as
    # `help`: every row is base-wrong, so the final prediction is correct exactly when the
    # candidate is admitted AND the candidate is right.
    "seen": ("repair_support_seen", "patched_correct"),
}
# arm -> predicate over the derived features, applied after caching
FILTERS = {"help_flipped": lambda f: f["flip"] > 0.5}

# The deployment-bridge arm. Every arm above conditions on something a runtime gate cannot
# observe: `help_flipped` is drawn from repair_holdout_unseen, i.e. inputs we already know the
# base model got wrong. That is legitimate for offline repair evaluation but it is NOT
# deployment-time authorization -- at runtime there is no y_t and no oracle telling us the base
# was failing. `commit` removes that oracle: it unions the two splits and conditions ONLY on the
# runtime-observable event flip=1 ("the patch wants to change the answer"), leaving a natural
# mixture of base-correct and base-wrong inputs. The label is `patched_correct` -- should this
# proposed decision change be committed? The mixture contains good flips (repairs), bad flips
# (regressions) and useless flips (wrong -> still wrong), and carries no split identity to steal.
#
# CAVEAT: this is a RECONSTRUCTED pool, not a natural one. clean_eval holds every base-correct
# test input while only the held-out third of the failures (bug_eval) is available, so the base
# error prevalence is understated. AUROC is prevalence-free, so the incremental comparison is
# unaffected; any threshold or calibration read off this population would not be.
MULTI_SPLIT = {"commit": (["clean_eval", "repair_holdout_unseen"], "patched_correct")}


# --------------------------------------------------------------------------- io


def cell_dir(seed: int, ds: str, bb: str) -> Path:
    return ROOT / f"outputs/effect_dump{DUMP_TAG}_v8_s{seed}/{ds}/{bb}/{PASS_DIR}/predictions"


def route_dir(seed: int, ds: str, bb: str) -> Path:
    return ROOT / f"outputs/effect_dump_routefeat_v8_s{seed}/{ds}/{bb}/{PASS_DIR}/predictions"


def read_csv_col(path: Path, col: str) -> np.ndarray:
    with path.open(newline="") as fh:
        return np.array([r[col] == "True" for r in csv.DictReader(fh)], dtype=bool)


def read_csv_int(path: Path, col: str) -> np.ndarray:
    with path.open(newline="") as fh:
        return np.array([int(r[col]) for r in csv.DictReader(fh)], dtype=np.int64)


def chunked_norm(path: Path, chunk: int = 2048) -> np.ndarray:
    """||delta||_2 per row without materialising patch_vec.

    patch_vec is (N, 2048) and there are 36 cells; loading them all as float64 is ~7 GB over
    an NFS mount. Only the norm is ever used, so stream it.
    """
    arr = np.load(path, mmap_mode="r")
    out = np.empty(len(arr), dtype=np.float64)
    for i in range(0, len(arr), chunk):
        blk = np.asarray(arr[i:i + chunk], dtype=np.float32)
        out[i:i + chunk] = np.linalg.norm(blk, axis=1)
    return out


def load_split(seed: int, ds: str, bb: str, split: str, label_col: str) -> dict | None:
    """Load one (cell, split) as raw tensors plus the binary outcome label."""
    d = cell_dir(seed, ds, bb)
    # `clean_train` dumps store the patch NORM directly instead of the full delta tensor
    # (deploy_eval: the train split would be 2.7 GB per VGG cell). Either form is accepted;
    # only the norm is ever consumed.
    vec, norm = d / f"patch_vec_{split}.npy", d / f"patch_norm_{split}.npy"
    need = [f"base_logits_{split}.npy", f"patched_logits_{split}.npy",
            f"dataset_indices_{split}.npy", f"{split}_predictions.csv"]
    if not all((d / f).exists() for f in need) or not (vec.exists() or norm.exists()):
        return None
    base = np.load(d / f"base_logits_{split}.npy").astype(np.float64)
    patched = np.load(d / f"patched_logits_{split}.npy").astype(np.float64)
    dnorm = np.load(norm).astype(np.float64) if norm.exists() else chunked_norm(vec)
    idx = np.load(d / f"dataset_indices_{split}.npy")
    y = read_csv_col(d / f"{split}_predictions.csv", label_col)
    truth = read_csv_int(d / f"{split}_predictions.csv", "label")
    if not (len(base) == len(patched) == len(dnorm) == len(idx) == len(y)):
        raise SystemExit(f"row-count mismatch in {d} [{split}] -- refusing to guess alignment")

    return {"base": base, "patched": patched, "dnorm": dnorm, "idx": idx, "y": y,
            "truth": truth}


def load_latent(seed: int, ds: str, bb: str, split: str, idx: np.ndarray) -> np.ndarray | None:
    """G_latent control, aligned to `idx` by dataset index rather than by row order.

    The route-feature dump is a separate run from the logit dump, so row order is assumed
    to match only after it has been checked; when it does not, we permute or give up.

    Returned as a read-only memmap, not a materialised array: the clean_eval latents are
    12.5k x up-to-2048 per cell and eagerly loading all 36 as float64 costs ~4 GB for a control
    that leave-one-backbone-out cannot even use. matrix() casts only the slice it needs.
    """
    rd = route_dir(seed, ds, bb)
    rf, ri = rd / f"route_features_{split}.npy", rd / f"dataset_indices_{split}.npy"
    if not (rf.exists() and ri.exists()):
        return None
    ridx = np.load(ri)
    if np.array_equal(ridx, idx):
        return np.load(rf, mmap_mode="r")
    order = {int(v): k for k, v in enumerate(ridx)}
    if set(order) != set(int(v) for v in idx):
        return None
    return np.load(rf, mmap_mode="r")[[order[int(v)] for v in idx]]


# ---------------------------------------------------------------------- features


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_features(rec: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return {name: column} and the ||delta|| vector used for magnitude stratification.

    Every feature is dimensionless by construction (probabilities, ratios, indicators, or
    entropies divided by log C). That is the whole cross-backbone argument: a raw latent
    coordinate means nothing across architectures, but "the patch closed 40% of this
    input's margin" means the same thing everywhere. The two exceptions are marked
    `*_r` -- they are rank-normalised within the cell, which a deployed system can do with
    its own unlabelled calibration data, but which does use cell-level statistics.
    """
    base, patched, dnorm = rec["base"], rec["patched"], rec["dnorm"]
    n, C = base.shape
    logC = np.log(C)
    pB, pP = softmax(base), softmax(patched)

    order = np.argsort(-base, axis=1)
    c = order[:, 0]                                   # base top-1
    j = order[:, 1]                                   # base runner-up
    rows = np.arange(n)

    m_base = base[rows, c] - base[rows, j]            # top-1 vs top-2 margin (logits)
    dlog = patched - base                             # (n, C) per-class logit displacement
    dm = dlog[rows, c] - dlog[rows, j]                # margin displacement on the base pair
    rho = dm / (np.abs(m_base) + EPS)                 # dimensionless: fraction of margin moved

    # worst-case displacement: the margin against whichever class the patch favours most
    dlog_masked = dlog.copy()
    dlog_masked[rows, c] = -np.inf
    dm_worst = dlog[rows, c] - dlog_masked.max(axis=1)
    rho_worst = dm_worst / (np.abs(m_base) + EPS)

    H = lambda p: -(p * np.log(p + 1e-12)).sum(axis=1) / logC
    kl = (pP * (np.log(pP + 1e-12) - np.log(pB + 1e-12))).sum(axis=1) / logC

    rank = lambda v: (np.argsort(np.argsort(v)) / max(n - 1, 1))

    feats = {
        # --- base-only (G_output): what the model thought before any patch existed
        "pB_max": pB[rows, c],
        "pB_margin": pB[rows, c] - pB[rows, j],
        "H_base": H(pB),
        "m_base_r": rank(m_base),
        # --- intervention response (G_response): what the patch did to this input
        "rho": np.clip(rho, -10, 10),
        "rho_worst": np.clip(rho_worst, -10, 10),
        "dp_c": pP[rows, c] - pB[rows, c],
        # post-repair confidence. THE headline scalar as of 2026-08-19 (section 8): standalone it
        # beats dp_max decisively (0.900/0.876 vs 0.757/0.728 inside the base_wrong stratum), and
        # dp_max's apparent edge on the commit arm was population identity, not information.
        "pP_max": pP.max(axis=1),
        # kept because `pre_mag + dp_max` and `pre_mag + pP_max` span the SAME linear space
        # (pre_mag already carries pB_max, and pP_max = pB_max + dp_max; verified equal on 40/40
        # cells, delta +-0.000). The two are interchangeable ONLY when nested like that -- used
        # ALONE they are different features and must never be renamed into each other.
        "dp_max": pP.max(axis=1) - pB.max(axis=1),
        "kl": np.clip(kl, 0, 50),
        "dH": H(pP) - H(pB),
        "flip": (np.argmax(patched, axis=1) != c).astype(np.float64),
        "dm_r": rank(dm),
        # --- magnitude only (G_mag): the curve-sliding null
        "dnorm_r": rank(dnorm),
        # raw (unranked) scalars behind the three `*_r` columns above, kept only so a fixed
        # reference distribution can be fit once (e.g. on repair data) and applied unchanged to
        # a single incoming row -- see analysis_gate_norank_ablation.py. Additive: no existing
        # feature set reads these, so this changes no shipped number by itself.
        "m_base_raw": m_base,
        "dm_raw": dm,
        "dnorm_raw": dnorm,
        # --- label-PRIVILEGED pre-intervention features (Ishimoto et al. TOSEM'25 use these).
        # LPS is the predicted probability of the ground-truth label and loss needs y, so neither
        # is available to a runtime gate facing an unlabelled x_t. They are included only to give
        # the rival family its strongest possible form -- an upper bound, not a fair baseline.
        "lps": pB[rows, rec["truth"]],
        "nll": -np.log(pB[rows, rec["truth"]] + 1e-12) / logC,
    }
    return feats, dnorm


FEATURE_SETS = {
    "output":   ["pB_max", "pB_margin", "H_base", "m_base_r"],
    "response": ["rho", "rho_worst", "dp_c", "dp_max", "kl", "dH", "flip", "dm_r"],
    # --- the five pre-registered LOBO baselines for the incremental-information question
    "pre":            ["pB_max", "pB_margin", "H_base", "m_base_r"],
    "pre_mag":        ["pB_max", "pB_margin", "H_base", "m_base_r", "dnorm_r"],
    "pre_mag_response": ["pB_max", "pB_margin", "H_base", "m_base_r", "dnorm_r",
                         "rho", "rho_worst", "dp_c", "dp_max", "kl", "dH", "dm_r"],
    # label-privileged variants: the rival family at its strongest
    "preO_mag":       ["pB_max", "pB_margin", "H_base", "m_base_r", "lps", "nll", "dnorm_r"],
    "preO_mag_response": ["pB_max", "pB_margin", "H_base", "m_base_r", "lps", "nll", "dnorm_r",
                          "rho", "rho_worst", "dp_c", "dp_max", "kl", "dH", "dm_r"],
    "mag_response":   ["dnorm_r", "rho", "rho_worst", "dp_c", "dp_max", "kl", "dH", "dm_r"],
    # ablation: how much of `response` is just the tautological flip indicator?
    "response_noflip": ["rho", "rho_worst", "dp_c", "dp_max", "kl", "dH", "dm_r"],
    "flip":     ["flip"],
    # the minimal representation: one dimensionless scalar, the patch-induced change in max
    # softmax probability. 4/4 unseen backbones significant, 65-119% of the 7-d increment.
    # --- canonical as of 2026-08-19: the post-repair state, not the change
    "pre_mag_pPmax": ["pB_max", "pB_margin", "H_base", "m_base_r", "dnorm_r", "pP_max"],
    "pPmax":    ["pP_max"],
    # the pre-proposal control the whole axis rests on
    "pBmax":    ["pB_max"],
    # DEPRECATED SPELLINGS. `pre_mag_dpmax` == `pre_mag_pPmax` (same span, same numbers).
    # `dpmax` is NOT `pPmax` -- different feature, different results. Do not conflate.
    "dpmax":    ["dp_max"],
    "rho":      ["rho"],
    "mag":      ["dnorm_r"],
    "latent":   None,      # filled from route_features when the dump lands
}


# ------------------------------------------------------------------- classifier


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300) -> np.ndarray:
    """Standardised L2 logistic regression, Newton steps with a damped Hessian.

    Deliberately linear and low-capacity: this is a signal-existence test, not a gate
    design. If a linear probe on eight dimensionless numbers transfers across backbones,
    that is the finding; adding capacity would only blur what is being measured.
    """
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    yf = y.astype(np.float64)
    # class-balanced weights: harm is ~3% of clean_eval, unweighted fitting would just
    # predict "never harmful" and score a meaningless accuracy
    pos = max(yf.sum(), 1.0)
    neg = max(len(yf) - yf.sum(), 1.0)
    sw = np.where(y, len(yf) / (2 * pos), len(yf) / (2 * neg))
    reg = l2 * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (sw * (p - yf)) + reg @ w
        Hm = (Z * (sw * p * (1 - p))[:, None]).T @ Z + reg + 1e-6 * np.eye(Z.shape[1])
        try:
            step = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return np.concatenate([w[:-1] / sd, [w[-1] - (mu / sd) @ w[:-1]]])


def score(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ w[:-1] + w[-1]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank AUROC with tie correction; nan when a cell has only one class."""
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(np.float64) + 1.0
    # average ranks over ties so a constant score scores exactly 0.5, not 1.0
    order = np.argsort(s)
    ss = s[order]
    i = 0
    while i < len(ss):
        k = i
        while k + 1 < len(ss) and ss[k + 1] == ss[i]:
            k += 1
        if k > i:
            r[order[i:k + 1]] = np.mean(r[order[i:k + 1]])
        i = k + 1
    return float((r[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


# ------------------------------------------------------------------- experiment


CACHE = ROOT / "outputs" / f"_response_gate_cache{DUMP_TAG}"
_MEM: dict[str, dict[tuple, dict]] = {}


def collect(arm: str, refresh: bool = False) -> dict[tuple, dict]:
    """Derived features per cell, memoised in-process and on disk.

    The derived columns are a few hundred KB per cell versus ~200 MB of raw tensors on an
    NFS mount, so caching turns a multi-minute rerun into a second. Latents are re-checked
    on every call (their dump may still be in flight) and are never cached.
    """
    if not refresh and arm in _MEM:
        return _MEM[arm]
    CACHE.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple, dict] = {}
    for seed, ds, bb in itertools.product(SEEDS, DSS, BBS):
        cf = CACHE / f"{arm}_{ds}_{bb}_s{seed}.npz"
        if cf.exists() and not refresh:
            z = np.load(cf, allow_pickle=False)
            names = [k for k in z.files if k not in ("__y", "__dnorm", "__idx")]
            entry = {"feats": {k: z[k] for k in names}, "dnorm": z["__dnorm"],
                     "y": z["__y"].astype(bool), "idx": z["__idx"], "latent": None}
        else:
            if arm in MULTI_SPLIT:
                splits, label_col = MULTI_SPLIT[arm]
                parts = []
                for sp in splits:
                    rec = load_split(seed, ds, bb, sp, label_col)
                    if rec is None:
                        continue
                    fe, dn = build_features(rec)
                    m = fe["flip"] > 0.5          # the only runtime-observable condition
                    parts.append(({k: v[m] for k, v in fe.items()}, dn[m],
                                  rec["y"][m], rec["idx"][m]))
                if not parts:
                    continue
                feats = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
                dnorm = np.concatenate([p[1] for p in parts])
                yv = np.concatenate([p[2] for p in parts])
                idxv = np.concatenate([p[3] for p in parts])
            else:
                split, label_col = ARMS[arm]
                rec = load_split(seed, ds, bb, split, label_col)
                if rec is None:
                    continue
                feats, dnorm = build_features(rec)
                yv, idxv = rec["y"], rec["idx"]
            np.savez_compressed(cf, __y=yv, __dnorm=dnorm, __idx=idxv, **feats)
            entry = {"feats": feats, "dnorm": dnorm, "y": yv, "idx": idxv, "latent": None}
        if arm not in MULTI_SPLIT:
            entry["latent"] = load_latent(seed, ds, bb, ARMS[arm][0], entry["idx"])
        # caches written before 2026-08-19 predate the pP_max column; derive it rather than
        # invalidating ~650 cached cells (pP_max = pB_max + dp_max, exactly).
        if "pP_max" not in entry["feats"]:
            entry["feats"]["pP_max"] = entry["feats"]["pB_max"] + entry["feats"]["dp_max"]
        if arm in FILTERS:
            m = FILTERS[arm](entry["feats"])
            entry = {"feats": {k: v[m] for k, v in entry["feats"].items()},
                     "dnorm": entry["dnorm"][m], "y": entry["y"][m], "idx": entry["idx"][m],
                     "latent": None if entry["latent"] is None else entry["latent"][m]}
        cells[(seed, ds, bb)] = entry
    _MEM[arm] = cells
    return cells


def folds(keys: list[tuple], protocol: str):
    """Yield (fold name, train keys, test keys).

    `lobo` is the headline protocol: hold out an entire backbone. `lodo` holds out a dataset
    within a single backbone -- the only protocol under which G_latent is even *defined*,
    because router latents are 2048/768/1024/512-dimensional on resnet50/convnext/densenet/vgg16
    and therefore not the same space across backbones. Comparing response against latent under
    lodo is the apples-to-apples control; under lobo, latent has no apples.
    """
    if protocol == "lobo":
        for bb in BBS:
            tr = [k for k in keys if k[2] != bb]
            te = [k for k in keys if k[2] == bb]
            if tr and te:
                yield f"hold {bb}", tr, te
    else:
        for bb in BBS:
            for ds in DSS:
                tr = [k for k in keys if k[2] == bb and k[1] != ds]
                te = [k for k in keys if k[2] == bb and k[1] == ds]
                if tr and te:
                    yield f"{bb} / hold {ds}", tr, te


def fit_pca(Xtr: np.ndarray, max_k: int = 64) -> tuple[np.ndarray, np.ndarray] | None:
    """PCA basis for the G_latent control, fit on TRAINING rows only.

    Router latents are 512-2048 dimensional while a leave-one-dataset-out training fold holds
    only ~640 flipped rows, so fitting the raw latent is degenerate -- it would overfit and then
    report a transfer failure that is really a sample-size failure. Projecting onto the top
    min(64, n/10) training components gives the latent gate its best honest shot, and also keeps
    the Newton solve from inverting a 2049x2049 Hessian 300 times per fold.
    """
    k = int(min(max_k, max(2, len(Xtr) // 10), Xtr.shape[1]))
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return mu, Vt[:k].T


def apply_pca(X: np.ndarray, proj: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mu, V = proj
    return (X - mu) @ V


def matrix(cell: dict, names: list[str] | None) -> np.ndarray:
    if names is None:
        return np.asarray(cell["latent"], dtype=np.float64)   # materialise the memmap here only
    return np.column_stack([cell["feats"][n] for n in names])


def stratified_auroc(y: np.ndarray, s: np.ndarray, dnorm: np.ndarray, q: int = 5) -> float:
    """AUROC recomputed inside quintiles of ||delta||, pooled by positive-pair count.

    This is the load-bearing control. If the gate's whole signal is "the patch moved this
    input a lot", it dies here, and response-gating is latent-gating in new coordinates.
    """
    edges = np.quantile(dnorm, np.linspace(0, 1, q + 1))
    edges[-1] += 1e-9
    num = den = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dnorm >= lo) & (dnorm < hi)
        if m.sum() < 20:
            continue
        a = auroc(y[m], s[m])
        if np.isnan(a):
            continue
        wgt = float(y[m].sum() * (~y[m]).sum())
        num += a * wgt
        den += wgt
    return num / den if den > 0 else float("nan")


def matched_pair_auroc(y: np.ndarray, s: np.ndarray, dnorm: np.ndarray,
                       frac: float = 0.2) -> tuple[float, int]:
    """AUROC restricted to (positive, negative) pairs of near-identical patch magnitude.

    The quintile control needs ~20 rows per stratum, which gtsrb and lisa cells do not have
    (24 of 36 cells returned nan). This is the same question asked pairwise: of all pos/neg
    pairs, keep the `frac` closest in |delta| and ask how often the gate still ranks the
    positive higher. It conditions on magnitude without ever binning, so every row contributes
    and small cells stay measurable. 0.5 means the signal was magnitude all along.
    """
    p, n = np.flatnonzero(y), np.flatnonzero(~y)
    if len(p) == 0 or len(n) == 0:
        return float("nan"), 0
    diff = s[p][:, None] - s[n][None, :]
    gap = np.abs(dnorm[p][:, None] - dnorm[n][None, :])
    keep = gap <= np.quantile(gap, frac)
    k = int(keep.sum())
    if k == 0:
        return float("nan"), 0
    return float(((diff > 0)[keep].sum() + 0.5 * (diff == 0)[keep].sum()) / k), k


def bootstrap_ci(y: np.ndarray, s: np.ndarray, dnorm: np.ndarray, reps: int = 500,
                 frac: float = 0.2, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for matched_pair_auroc, resampling positives and negatives separately."""
    p, n = np.flatnonzero(y), np.flatnonzero(~y)
    if len(p) < 3 or len(n) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pi, ni = rng.choice(p, len(p), True), rng.choice(n, len(n), True)
        idx = np.concatenate([pi, ni])
        yb = np.concatenate([np.ones(len(pi), bool), np.zeros(len(ni), bool)])
        v, _ = matched_pair_auroc(yb, s[idx], dnorm[idx], frac)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < reps // 4:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def magnitude_control(arm: str, feature_sets: list[str], protocol: str = "lobo",
                      frac: float = 0.2) -> None:
    """Per-setting magnitude-matched control, seeds pooled, with bootstrap CIs.

    Reported per setting (dataset x backbone) rather than per cell: pooling the three seeds is
    what makes gtsrb and lisa measurable at all, and per-setting is the project's reporting unit.
    A CI whose lower bound clears 0.5 is the claim "this is not magnitude" actually earning its
    keep on that setting.
    """
    cells = collect(arm)
    print(f"\n{'=' * 96}\nMAGNITUDE-MATCHED CONTROL  ({arm}, {protocol}, closest {frac:.0%} of "
          f"pos/neg pairs by |delta|)")
    print("0.5 == the gate was ranking by patch magnitude all along. CI is 500-rep bootstrap.")

    for fs in feature_sets:
        names = FEATURE_SETS[fs]
        if names is None:
            continue
        print(f"\n-- feature set: {fs}   null: {null}")
        hdr = (f"{'setting':<26}{'n':>7}{'pos':>6}{'pairs':>9}{'matched AUROC':>15}"
               f"{'95% CI':>18}   ")
        print(hdr + "\n" + "-" * (len(hdr) + 8))
        won = tot = 0
        for ds, bb in itertools.product(DSS, BBS):
            keys = [k for k in cells if k[1] == ds and k[2] == bb]
            if not keys:
                continue
            # one probe per held-out backbone, exactly as in the headline protocol
            tr = [k for k in cells if k[2] != bb] if protocol == "lobo" else \
                 [k for k in cells if k[2] == bb and k[1] != ds]
            if not tr:
                continue
            Xtr = np.vstack([matrix(cells[k], names) for k in tr])
            ytr = np.concatenate([cells[k]["y"] for k in tr])
            if ytr.sum() == 0 or (~ytr).sum() == 0:
                continue
            w = fit_logreg(Xtr, ytr)
            ss = np.concatenate([score(w, matrix(cells[k], names)) for k in keys])
            yy = np.concatenate([cells[k]["y"] for k in keys])
            dd = np.concatenate([cells[k]["dnorm"] for k in keys])
            a, k_pairs = matched_pair_auroc(yy, ss, dd, frac)
            lo, hi = bootstrap_ci(yy, ss, dd, frac=frac)
            flag = ""
            tot += 1
            if not np.isnan(lo) and lo > 0.5:
                won += 1
                flag = "  clears 0.5"
            elif not np.isnan(lo):
                flag = "  !! includes 0.5"
            print(f"{ds + '/' + bb:<26}{len(yy):>7}{int(yy.sum()):>6}{k_pairs:>9}"
                  f"{a:>15.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}{flag}")
        print("-" * (len(hdr) + 8))
        print(f"  {won}/{tot} settings with a 95% CI strictly above 0.5")


def incremental(arm: str, pairs: list[tuple[str, str]], protocol: str = "lobo",
                reps: int = 1000) -> None:
    """Nested-model comparison: what does the augmented feature set add over the base set?

    This is the experiment the RQ actually rests on. The rival family (Ishimoto et al.,
    CAIN'23 / TOSEM'25) predicts per-sample repaired/broken from PRE-intervention quantities --
    entropy, PCS/margin, confidence, and (label-privileged) LPS and loss. The open question is
    not "can outcome be predicted" (it can, and they showed it) but whether the response the
    proposed repair induces carries information those pre-intervention quantities do not.

    So the headline number is a difference of AUROCs between nested probes, evaluated on a
    backbone neither probe was fit on, with a paired bootstrap CI. A CI strictly above zero on
    an unseen architecture is what would move this from an empirical finding to a method.
    """
    cells = collect(arm)

    def fold_scores(fs: str) -> dict[tuple, np.ndarray]:
        names = FEATURE_SETS[fs]
        outp: dict[tuple, np.ndarray] = {}
        for fold_name, tr, te in folds(list(cells), protocol):
            X = np.vstack([matrix(cells[k], names) for k in tr])
            y = np.concatenate([cells[k]["y"] for k in tr])
            if y.sum() == 0 or (~y).sum() == 0:
                continue
            w = fit_logreg(X, y)
            for k in te:
                outp[k] = score(w, matrix(cells[k], names))
        return outp

    need = sorted({f for pr in pairs for f in pr})
    S = {fs: fold_scores(fs) for fs in need}

    print(f"\n{'=' * 96}\nINCREMENTAL INFORMATION  ({arm}, {protocol}, per setting, seeds pooled)")
    print("Does the response to a proposed repair say anything that pre-intervention uncertainty")
    print("and repair magnitude do not? Delta > 0 on a held-out backbone is the claim.")
    for base, aug in pairs:
        print(f"\n-- {aug}  vs  {base}")
        hdr = (f"{'setting':<26}{'n':>6}{'AUC base':>10}{'AUC aug':>10}{'delta':>9}"
               f"{'95% CI':>21}")
        print(hdr + "\n" + "-" * (len(hdr) + 3))
        rng = np.random.default_rng(0)
        won = tot = 0
        for ds, bb in itertools.product(DSS, BBS):
            ks = [k for k in cells if k[1] == ds and k[2] == bb and k in S[base] and k in S[aug]]
            if not ks:
                continue
            y = np.concatenate([cells[k]["y"] for k in ks])
            a = np.concatenate([S[base][k] for k in ks])
            b = np.concatenate([S[aug][k] for k in ks])
            if y.sum() == 0 or (~y).sum() == 0:
                continue
            d0 = auroc(y, b) - auroc(y, a)
            pi_, ni_ = np.flatnonzero(y), np.flatnonzero(~y)
            bs = []
            for _ in range(reps):
                pp, nn = rng.choice(pi_, len(pi_), True), rng.choice(ni_, len(ni_), True)
                ix = np.concatenate([pp, nn])
                yb = np.concatenate([np.ones(len(pp), bool), np.zeros(len(nn), bool)])
                bs.append(auroc(yb, b[ix]) - auroc(yb, a[ix]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            tot += 1
            star = "*" if lo > 0 else " "
            won += lo > 0
            print(f"{ds + '/' + bb:<26}{len(y):>6}{auroc(y, a):>10.3f}{auroc(y, b):>10.3f}"
                  f"{d0:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>19}{star:>2}")
        print("-" * (len(hdr) + 3))
        print(f"  {won}/{tot} settings with a 95% CI on the delta strictly above 0")


def run_arm(arm: str, feature_sets: list[str], out: dict, protocol: str = "lobo") -> None:
    cells = collect(arm)
    if not cells:
        print(f"  [{arm}] no cells found -- skipped")
        return
    split, label_col = (MULTI_SPLIT[arm][0][0] + "+", MULTI_SPLIT[arm][1]) \
        if arm in MULTI_SPLIT else ARMS[arm]

    n_tot = sum(len(c["y"]) for c in cells.values())
    n_pos = sum(int(c["y"].sum()) for c in cells.values())
    print(f"\n{'=' * 96}\nARM: {arm.upper()}   split={split}   label={label_col}")
    print(f"cells={len(cells)}  n={n_tot:,}  positives={n_pos:,} ({n_pos / n_tot:.2%})")

    # population guard: the HELP arm must be scored on failures the patch never trained on
    if arm == "help":
        bad = []
        for (seed, ds, bb) in cells:
            sup = load_split(seed, ds, bb, "repair_support_seen", "repaired")
            if sup is not None:
                ov = np.intersect1d(cells[(seed, ds, bb)]["idx"], sup["idx"])
                if len(ov):
                    bad.append((seed, ds, bb, len(ov)))
        if bad:
            print(f"  !! EVAL-POPULATION LEAK: {len(bad)} cell(s) share indices with "
                  f"repair_support_seen -- {bad[:3]}")
        else:
            print("  eval-population guard: held n support = 0 on every cell  [OK]")

    for fs in feature_sets:
        names = FEATURE_SETS[fs]
        usable = {k: v for k, v in cells.items()
                  if names is not None or v["latent"] is not None}
        if not usable:
            print(f"\n-- {fs}: route_features not dumped yet -- skipped")
            continue
        if len(usable) < len(cells):
            print(f"\n-- {fs}: only {len(usable)}/{len(cells)} cells have latents")

        if names is None and protocol == "lobo":
            dims = sorted({(k[2], usable[k]["latent"].shape[1]) for k in usable})
            print(f"\n-- {fs}: UNDEFINED under leave-one-backbone-out. Router latents live in "
                  f"different spaces per backbone -- {', '.join(f'{b}={d}' for b, d in dims)} -- "
                  f"so a probe fit on three backbones cannot even be applied to the fourth. "
                  f"Re-run with --protocol lodo for the fair within-backbone comparison.")
            continue

        print(f"\n-- feature set: {fs}  ({'route_features' if names is None else len(names)} dims)")
        col = "held-out backbone" if protocol == "lobo" else "backbone / held dataset"
        hdr = f"{col:<26}{'cells':>7}{'AUROC med':>12}{'[min':>9}{'max]':>9}{'strat-AUROC':>14}"
        print(hdr + "\n" + "-" * len(hdr))

        for fold_name, tr, te in folds(list(usable), protocol):
            Xtr = np.vstack([matrix(usable[k], names) for k in tr])
            proj = fit_pca(Xtr) if names is None else None
            if proj is not None:
                Xtr = apply_pca(Xtr, proj)
            ytr = np.concatenate([usable[k]["y"] for k in tr])
            if ytr.sum() == 0 or (~ytr).sum() == 0:
                continue
            w = fit_logreg(Xtr, ytr)

            aucs, strats = [], []
            for k in te:
                cell = usable[k]
                Xte = matrix(cell, names)
                s = score(w, apply_pca(Xte, proj) if proj is not None else Xte)
                a = auroc(cell["y"], s)
                if not np.isnan(a):
                    aucs.append(a)
                    strats.append(stratified_auroc(cell["y"], s, cell["dnorm"]))
                out.setdefault(arm, {}).setdefault(fs, []).append(
                    {"fold": fold_name, "protocol": protocol, "cell": list(k), "auroc": a,
                     "strat_auroc": strats[-1] if aucs else None,
                     "n": int(len(cell["y"])), "pos": int(cell["y"].sum())})
            if not aucs:
                continue
            st = [x for x in strats if not np.isnan(x)]
            print(f"{fold_name:<26}{len(aucs):>7}{np.median(aucs):>12.3f}"
                  f"{min(aucs):>9.3f}{max(aucs):>9.3f}"
                  f"{(np.median(st) if st else float('nan')):>14.3f}")


# ------------------------------------------------------------ deployment simulation


def deployment_sim(feature_sets: list[str], out: dict, null: str = "mag") -> None:
    """Trace the true Reg / RR_held curve of a gate, against a null that suppresses as many flips.

    `null` selects WHICH null, and the choice decides the verdict:
      mag  suppress the q fraction of flips with the LARGEST ||delta||. The original null.
           Measured 2026-08-19: this is a WEAK null -- it is not aimed at regression at all.
      pre  suppress the q fraction of flips with the HIGHEST p^B_max. The pre-proposal control,
           i.e. "do not touch a prediction the base model was already confident about". This is
           the null the paper's pre-/post-proposal axis actually has to beat, and it is much
           harder: `dpmax` scores 35/108 against `mag` and 0/108 against `pre`
           (scripts/probe_gate_on_other_repairs.py, note/RESULTS_ALL_RQ.md section 10).
    Reporting only `mag` overstates the gate; both columns must appear in the paper.

    Suppressing the patch on a row is exactly simulable offline: a clean row whose patch is
    suppressed reverts to base and cannot be a regression; a failure row whose patch is
    suppressed reverts to base and cannot be a repair. So for a gate that suppresses the
    top-q fraction by predicted harm we get the exact operating point, and the
    pre-registered verdict of scripts/analyze_gate_sweep.py:140 applies:
        gate helps  iff  dReg < -0.002 AND dRR_held > -0.02
    The null suppresses the same q fraction ranked by ||delta|| instead -- if the gate
    cannot beat that, it is sliding the curve.
    """
    harm_all = collect("harm")
    help_all = collect("help")
    train_pop = collect("help_flipped")
    keys = sorted(set(harm_all) & set(help_all))
    if not keys:
        print("\nno cells with both arms -- deployment simulation skipped")
        return

    print(f"\n{'=' * 96}\nDEPLOYMENT SIMULATION  (exact; suppression needs no re-run)")
    print("Only FLIPPED rows are actionable: a patch that leaves the prediction unchanged can")
    print("neither regress a clean input nor repair a failing one. So the gate's real job is the")
    print("single-population question 'this patch is about to change the answer -- will the new")
    print("answer be right?', which is also what dissolves the disjoint-population problem.")
    print("The scorer is trained on help_flipped from the OTHER three backbones.")
    print("verdict rule: analyze_gate_sweep.py:140 -- helps iff dReg < -0.002 and dRR_held > -0.02")

    for fs in feature_sets:
        names = FEATURE_SETS[fs]
        if names is None and any(harm[k]["latent"] is None for k in keys):
            print(f"\n-- {fs}: route_features not dumped yet -- skipped")
            continue
        print(f"\n-- feature set: {fs}   null: {null}")
        hdr = (f"{'setting (held bb)':<30}{'q':>6}{'dReg':>10}{'dRR_held':>11}"
               f"{'dReg_null':>11}{'dRR_null':>10}   verdict")
        print(hdr + "\n" + "-" * len(hdr))
        tally: dict[str, int] = {}

        for held_bb in BBS:
            tr = [k for k in train_pop if k[2] != held_bb and len(train_pop[k]["y"])]
            te = [k for k in keys if k[2] == held_bb]
            if not tr:
                continue
            Xtr = np.vstack([matrix(train_pop[k], names) for k in tr])
            proj = fit_pca(Xtr) if names is None else None
            if proj is not None:
                Xtr = apply_pca(Xtr, proj)
            ytr = np.concatenate([train_pop[k]["y"] for k in tr])
            if ytr.sum() == 0 or (~ytr).sum() == 0:
                continue
            w = fit_logreg(Xtr, ytr)     # high score == flip predicted to land on the right class

            for k in te:
                hc, pc = harm_all[k], help_all[k]
                # a non-flipped row is a no-op in both populations; gating it changes nothing
                fh, fp = hc["feats"]["flip"] > 0.5, pc["feats"]["flip"] > 0.5
                nh, npp = len(hc["y"]), len(pc["y"])
                reg0, rr0 = hc["y"].mean(), pc["y"].mean()
                Xh, Xp = matrix(hc, names), matrix(pc, names)
                if proj is not None:
                    Xh, Xp = apply_pca(Xh, proj), apply_pca(Xp, proj)
                sh, sp = score(w, Xh), score(w, Xp)
                # null statistic: higher == suppressed first
                if null == "pre":
                    nh_stat, np_stat = hc["feats"]["pB_max"], pc["feats"]["pB_max"]
                else:
                    nh_stat, np_stat = hc["dnorm"], pc["dnorm"]
                dn_pool = np.concatenate([nh_stat[fh], np_stat[fp]])
                if not len(dn_pool):
                    continue
                for q in (0.05, 0.10, 0.20):
                    # gate: veto the q fraction of FLIPS least likely to land correctly,
                    # one shared threshold -- deployment cannot tell the populations apart
                    thr = np.quantile(np.concatenate([sh[fh], sp[fp]]), q)
                    keep_h, keep_p = ~fh | (sh > thr), ~fp | (sp > thr)
                    reg, rr = hc["y"][keep_h].sum() / nh, pc["y"][keep_p].sum() / npp
                    # null: veto the same fraction of flips, ranked by patch magnitude
                    tn = np.quantile(dn_pool, 1 - q)
                    kh, kp = ~fh | (nh_stat <= tn), ~fp | (np_stat <= tn)
                    regn, rrn = hc["y"][kh].sum() / nh, pc["y"][kp].sum() / npp
                    dreg, drr = reg - reg0, rr - rr0
                    v = ("gate helps" if dreg < -0.002 and drr > -0.02
                         else "slides curve" if dreg < -0.002 else "no Reg gain")
                    # only credit the gate if it also beats the null on Reg
                    if v == "gate helps" and dreg >= regn - reg0:
                        v = f"= {null} null"
                    tally[v] = tally.get(v, 0) + 1
                    print(f"{k[1] + '/' + k[2] + ' s' + str(k[0]):<30}{q:>6.2f}"
                          f"{dreg:>+10.4f}{drr:>+11.3f}{regn - reg0:>+11.4f}{rrn - rr0:>+10.3f}"
                          f"   {v}")
        print("-" * len(hdr))
        print(f"  tally: " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
        out.setdefault("deployment", {}).setdefault(null, {})[fs] = tally


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["harm", "help"],
                    choices=list(ARMS) + list(MULTI_SPLIT))
    ap.add_argument("--feature-sets", nargs="*", default=["output", "response", "rho", "mag", "latent"],
                    choices=list(FEATURE_SETS))
    ap.add_argument("--no-deploy-sim", action="store_true")
    ap.add_argument("--nulls", nargs="+", default=["mag"], choices=["mag", "pre"],
                    help="which suppression null(s) to score the gate against; see deployment_sim")
    ap.add_argument("--magnitude-control", action="store_true",
                    help="per-setting magnitude-matched pairwise AUROC with bootstrap CIs; "
                         "measurable on the gtsrb/lisa cells the quintile control cannot reach")
    ap.add_argument("--match-frac", type=float, default=0.2)
    ap.add_argument("--incremental", nargs="*", default=None,
                    help="nested comparisons as base:augmented, e.g. pre_mag:pre_mag_response")
    ap.add_argument("--protocol", default="lobo", choices=["lobo", "lodo"],
                    help="lobo = leave-one-backbone-out (headline); "
                         "lodo = leave-one-dataset-out within a backbone (the only protocol "
                         "under which G_latent is defined)")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the derived-feature cache from the raw dumps")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    print("LEAVE-ONE-BACKBONE-OUT: do intervention-response signals predict repair outcome?")
    print(f"pass={PASS_DIR}  seeds={SEEDS}  backbones={BBS}")
    print("\nreminder: AUROC is necessary, not sufficient. Read the strat-AUROC column (signal")
    print("surviving inside a fixed ||delta|| stratum) and the deployment simulation before")
    print("concluding anything -- the latent gate died on the curve, not on AUROC.")

    out: dict = {}
    if a.refresh:
        for arm in ARMS:
            collect(arm, refresh=True)
    for arm in a.arms:
        run_arm(arm, a.feature_sets, out, a.protocol)
    if a.incremental is not None:
        prs = [tuple(x.split(":")) for x in a.incremental] or [
            ("pre_mag", "pre_mag_response"), ("preO_mag", "preO_mag_response")]
        for arm in a.arms:
            incremental(arm, prs, a.protocol)
    if a.magnitude_control:
        for arm in a.arms:
            magnitude_control(arm, a.feature_sets, a.protocol, a.match_frac)
    if not a.no_deploy_sim:
        for _null in a.nulls:
            deployment_sim(a.feature_sets, out, null=_null)

    if a.json_out:
        a.json_out.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
