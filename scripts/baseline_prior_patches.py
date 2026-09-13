#!/usr/bin/env python3
"""NN-Patching and PatchNAS on the shipped settings, as the two nearest patch-based rivals.

Both are patch-based repair with a FROZEN deployed model, which is exactly our design point, so
they are the comparison the paper actually owes -- more so than full fine-tuning, which changes
the weights we are not allowed to touch.

  NN-Patching (Kauschke & Fuernkranz).  A patch model trained on the deployed network's internal
  representation, plus an ERROR ESTIMATOR that decides when to hand the input to the patch.
  Reproduced here as: patch = a classifier on the penultimate feature fitted on the reported
  failures; estimator = a binary classifier on the same feature, failures against clean training
  images. At inference the estimator routes: p(error) > tau -> patch output, else deployed output.

  PatchNAS (Fang et al., AAAI'23).  The deployed model is frozen and a lightweight patch network
  is SEARCHED for the faulty stage, with the patch activated per input.
  Reproduced here as: a small search over patch heads on the faulty-stage feature (depth, width,
  activation), selected on a validation slice of the reported failures, with the same style of
  learned activation probe on the stage feature.

DEVIATIONS, stated rather than hidden:
  * neither original operates on cached features; both are re-implemented on the frozen taps
    dumped by scripts/dump_prior_features.py, which fixes the injection point to ours and makes
    the comparison about the METHOD rather than about where it is allowed to attach;
  * PatchNAS's supernet search is replaced by an exhaustive sweep of a 12-point head space. This
    is smaller than their space and can only understate them -- said in the paper;
  * both are given the same failures, the same clean data and the same evaluation populations as
    our patch, and their routing threshold is swept so they appear on the frontier rather than at
    one hand-picked operating point.

Usage:
  OMP_NUM_THREADS=1 .venv/bin/python scripts/baseline_prior_patches.py [--settings ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DSS = ["gtsrb", "tt100k_signs", "lisa_signs"]
BBS = ["resnet50", "convnext_tiny", "densenet121", "vgg16"]
SEEDS = [101, 202, 303]
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PatchNAS search space: depth x width x activation. 12 points, swept exhaustively.
NAS_SPACE = [(d, w, act) for d in (1, 2) for w in (128, 256, 512) for act in ("relu", "gelu")]


def load(ds: str, bb: str, seed: int) -> dict | None:
    p = ROOT / f"artifacts/prior_feats/{ds}_{bb}_s{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def head(din: int, dout: int, depth: int, width: int, act: str) -> nn.Module:
    if depth == 0:
        return nn.Linear(din, dout)
    a = nn.ReLU if act == "relu" else nn.GELU
    layers: list[nn.Module] = [nn.Linear(din, width), a()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), a()]
    layers += [nn.Linear(width, dout)]
    return nn.Sequential(*layers)


def fit(net: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 1e-3,
        wd: float = 1e-4, val: tuple | None = None) -> nn.Module:
    """Train the patch head. With `val`, keep the parameters that do best on a held-out slice.

    Without it the head runs a fixed 300 steps on a few dozen rows and memorises them -- RR_repair
    comes out at 1.000 and the head that generalises best is not the one that gets deployed. Early
    stopping is part of both original methods and its absence would be our handicap, not theirs.
    """
    net = net.to(DEV)
    xb = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    yb = torch.as_tensor(y, dtype=torch.long, device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    if val is not None:
        vx = torch.as_tensor(val[0], dtype=torch.float32, device=DEV)
        vy = torch.as_tensor(val[1], dtype=torch.long, device=DEV)
    best, best_state = -1.0, None
    for e in range(epochs):
        net.train()
        opt.zero_grad()
        lossf(net(xb), yb).backward()
        opt.step()
        if val is not None and (e + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                acc = float((net(vx).argmax(1) == vy).float().mean())
            if acc > best:
                best, best_state = acc, {k: v.detach().clone()
                                         for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net.eval()


@torch.no_grad()
def predict(net: nn.Module, X: np.ndarray) -> np.ndarray:
    return net(torch.as_tensor(X, dtype=torch.float32, device=DEV)).argmax(1).cpu().numpy()


@torch.no_grad()
def predict_logits(net: nn.Module, X: np.ndarray) -> np.ndarray:
    """The patch head's raw, pre-argmax output -- what `predict` discards.

    Needed for the mechanism analysis (Delta z(x) = patched_logits - base_logits, and the
    true-vs-wrong margin improvement Delta m(x)); `predict` only ever kept the argmax.
    """
    return net(torch.as_tensor(X, dtype=torch.float32, device=DEV)).cpu().numpy().astype(np.float32)


@torch.no_grad()
def prob1(net: nn.Module, X: np.ndarray) -> np.ndarray:
    z = net(torch.as_tensor(X, dtype=torch.float32, device=DEV))
    return torch.softmax(z, 1)[:, 1].cpu().numpy()


def standardise(train: np.ndarray, others: list[np.ndarray]):
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-6
    return (train - mu) / sd, [(o - mu) / sd for o in others]


EST_ARCH = ["linear"]


def estimator(f_bug: np.ndarray, f_clean: np.ndarray, applies: list[np.ndarray]):
    """Binary error estimator: reported failures against clean training images.

    Capacity is a flag, not a constant: a linear probe is the cheapest reading of "error
    estimator", and if these baselines lose only because their estimator is under-powered then
    the comparison says nothing. `--estimator mlp` re-runs everything with a 2x256 head so the
    claim can be checked rather than asserted.
    """
    X = np.concatenate([f_bug, f_clean])
    y = np.concatenate([np.ones(len(f_bug), int), np.zeros(len(f_clean), int)])
    Xs, apps = standardise(X, applies)
    # class weighting: failures are a few dozen rows against thousands of clean ones
    w = torch.as_tensor([1.0, max(1.0, len(f_clean) / max(len(f_bug), 1))], dtype=torch.float32,
                        device=DEV)
    net = (head(X.shape[1], 2, 0, 0, "relu") if EST_ARCH[0] == "linear"
           else head(X.shape[1], 2, 2, 256, "relu")).to(DEV)
    xb = torch.as_tensor(Xs, dtype=torch.float32, device=DEV)
    yb = torch.as_tensor(y, dtype=torch.long, device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    for _ in range(400):
        opt.zero_grad()
        lossf(net(xb), yb).backward()
        opt.step()
    net.eval()
    return [prob1(net, a) for a in apps]


def critical(ds: str) -> set[int]:
    j = json.loads((ROOT / f"artifacts/risk/{ds}_safety_risk_matrix.json").read_text())
    return {int(i) for v in j["critical_signs"].values() for i in v}


def route(base_pred, patch_pred, score, tau):
    on = score > tau
    return np.where(on, patch_pred, base_pred)


def calib_tau(d: dict, patch: dict, score: dict, target_reg: float) -> float:
    """Cheapest routing threshold whose regression on clean_calib stays within the budget.

    clean_calib is the only clean population that is neither reported on nor used to fit the
    patch, so it is where a deployed system would place this threshold. Their paper tunes the
    estimator on validation data; this is the same act, on the split this project reserves for it.
    """
    y, base, pp = d["clean_calib__y"], d["clean_calib__pred"], patch["clean_calib"]
    sc = score["clean_calib"]
    for t in np.unique(np.concatenate([[-1.0], np.quantile(sc, np.linspace(0, 1, 201)), [2.0]])):
        if float((route(base, pp, sc, t) != y).mean()) <= target_reg:
            return float(t)
    return 2.0


def metrics(d: dict, patch: dict, score: dict, tau: float) -> dict:
    out = {}
    for pop, key in (("bug_train", "RR_seen"), ("bug_eval", "RR_held")):
        pred = route(d[f"{pop}__pred"], patch[pop], score[pop], tau)
        out[key] = float((pred == d[f"{pop}__y"]).mean())
    y = d["clean_test__y"]
    pred = route(d["clean_test__pred"], patch["clean_test"], score["clean_test"], tau)
    out["Reg"] = float((pred != y).mean())
    # regression restricted to clean inputs whose true class is safety-critical: the same
    # CReg the rest of the paper reports, so this row is comparable to every other row.
    cm = np.isin(y, list(CRIT[0]))
    out["CReg"] = float((pred[cm] != y[cm]).mean()) if cm.any() else float("nan")
    out["route_rate"] = float((score["clean_test"] > tau).mean())
    return out


def nn_patching(d: dict) -> dict:
    """Patch and estimator both on the penultimate feature."""
    pops = ["bug_train", "bug_eval", "clean_calib", "clean_test"]
    f = {p: d[f"{p}__final"] for p in pops}
    ftr = d["clean_train__final"]
    Xs, apps = standardise(f["bug_train"], [f[p] for p in pops])
    y = d["bug_train__y"]
    tr, va = holdout(len(y))
    net = fit(head(Xs.shape[1], NCLS[0], 0, 0, "relu"), Xs[tr], y[tr],
              val=(Xs[va], y[va]) if len(va) >= 3 else None) if EARLY[0] \
        else fit(head(Xs.shape[1], NCLS[0], 0, 0, "relu"), Xs, y)
    patch = {p: predict(net, a) for p, a in zip(pops, apps)}
    patch_logits = {p: predict_logits(net, a) for p, a in zip(pops, apps)}
    sc = estimator(f["bug_train"], ftr, [f[p] for p in pops])
    score = dict(zip(pops, sc))
    return {"patch": patch, "patch_logits": patch_logits, "score": score}


def patchnas(d: dict) -> dict:
    """Search a lightweight head on the faulty-stage feature; activate it with a stage probe."""
    pops = ["bug_train", "bug_eval", "clean_calib", "clean_test"]
    f = {p: d[f"{p}__stage"] for p in pops}
    ftr = d["clean_train__stage"]
    y = d["bug_train__y"]
    Xs, apps = standardise(f["bug_train"], [f[p] for p in pops])
    n = len(y)
    # selection split: 70/30 of the reported failures when there are enough of them, otherwise
    # the fit itself is the only signal available and that is said in the log.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    ntr = max(1, int(0.7 * n))
    tr, va = perm[:ntr], perm[ntr:]
    small = len(va) < 3
    best, best_net = None, None
    for depth, width, act in NAS_SPACE:
        net = fit(head(Xs.shape[1], NCLS[0], depth, width, act), Xs[tr], y[tr])
        sel = float((predict(net, Xs[tr if small else va]) == y[tr if small else va]).mean())
        if best is None or sel > best[0]:
            best, best_net = (sel, (depth, width, act)), net
    # refit the selected architecture. With --early-stop the refit keeps a validation slice too,
    # otherwise the searched head is chosen on validation and then trained to memorisation anyway.
    if EARLY[0] and len(va) >= 3:
        net = fit(head(Xs.shape[1], NCLS[0], *best[1]), Xs[tr], y[tr], val=(Xs[va], y[va]))
    else:
        net = fit(head(Xs.shape[1], NCLS[0], *best[1]), Xs, y)
    patch = {p: predict(net, a) for p, a in zip(pops, apps)}
    patch_logits = {p: predict_logits(net, a) for p, a in zip(pops, apps)}
    sc = estimator(f["bug_train"], ftr, [f[p] for p in pops])
    return {"patch": patch, "patch_logits": patch_logits, "score": dict(zip(pops, sc)),
            "arch": best[1], "select_on_train": small}


NCLS = [0]
EARLY = [False]


def holdout(n: int, frac: float = 0.3):
    """A fixed 70/30 split of the reported failures, for early stopping."""
    perm = np.random.default_rng(0).permutation(n)
    k = max(1, int((1 - frac) * n))
    return perm[:k], perm[k:]
CRIT: list[set] = [set()]
# our patch's ungated regression per setting (median over its 3 seeds), read from the shipped
# trees by scripts/paper_tables.py; used only to set the baselines' safety budget.
OURS_REG: dict[str, float] = {}


def load_ours_reg() -> None:
    import statistics as st
    p = ROOT / "outputs/paper_tables.json"
    if not p.exists():
        return
    j = json.loads(p.read_text()).get("t4", {}).get("DynaPatch (ours)", {})
    for k, v in j.items():
        OURS_REG[k] = float(st.median(v["Reg"]))


def dump_per_sample(out_root: Path, name: str, ds: str, bb: str, seed: int,
                   d: dict, r: dict, tau: float, tag: str) -> None:
    """Per-sample predictions, in the schema every other baseline in this project writes.

    These two methods were only ever summarised (`metrics()` averaged `route(...)` and threw the
    vector away), which is why they were the only methods with no per-sample data -- not a
    limitation of the artefacts. Nothing is recomputed here: `route` is the same call `metrics`
    makes, at the same tau.

    One extra column beyond the shared schema: `routed`. These are ROUTER methods, so "the patch
    was never applied" and "the patch was applied and did not help" are different events and a
    failure-type analysis that merges them is measuring the estimator, not the patch.

    Also writes, per population, `base_logits_{pop}.npy` and `patched_logits_{pop}.npy` (row order
    matches `dataset_index` in the CSV) so that Delta z(x) = patched_logits - base_logits and the
    true-vs-wrong margin improvement can be computed downstream, the same way
    outputs/effect_dump_ep40ns already carries these two arrays for DynaPatch. `patched_logits` is
    the patch head's own raw output regardless of whether the router applied it on that row --
    routing is recorded separately in `routed`, not baked into the logits. Skipped (with a
    warning, not silently) when `d` predates the `__logits` field added to
    scripts/dump_prior_features.py -- rerun that script with --force first.
    """
    dirp = out_root / f"{ds}_{bb}_s{seed}" / name.replace("/", "_") / tag
    dirp.mkdir(parents=True, exist_ok=True)
    for pop in ("bug_train", "bug_eval", "clean_calib", "clean_test"):
        y, base, pp = d[f"{pop}__y"], d[f"{pop}__pred"], r["patch"][pop]
        sc = r["score"][pop]
        on = sc > tau
        pred = np.where(on, pp, base)
        pd.DataFrame({
            "split": pop, "dataset_index": d[f"{pop}__idx"], "label": y,
            "base_pred": base, "patched_pred": pred,
            "patch_pred_raw": pp,          # what the patch head said, before routing
            "routed": on,
            "estimator_score": sc,
            "base_correct": base == y, "patched_correct": pred == y,
            "repaired": (base != y) & (pred == y),
            "regressed": (base == y) & (pred != y),
        }).to_csv(dirp / f"{pop}_predictions.csv", index=False)
        base_logits = d.get(f"{pop}__logits")
        patch_logits = r.get("patch_logits", {}).get(pop)
        if base_logits is None or patch_logits is None:
            print(f"  [no logits] {ds}/{bb} s{seed} {name} {tag} {pop} -- "
                  f"rerun dump_prior_features.py --force to add __logits")
            continue
        np.save(dirp / f"base_logits_{pop}.npy", base_logits)
        np.save(dirp / f"patched_logits_{pop}.npy", patch_logits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--json", default="outputs/baseline_prior_patches.json")
    ap.add_argument("--estimator", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--torch-seed", type=int, default=None,
                    help="Seed torch before each cell. WITHOUT this both methods are "
                         "NONDETERMINISTIC: every head here (NN-Patching's patch head, the 12 "
                         "architectures PatchNAS searches plus its refit, and the MLP error "
                         "estimator) is randomly initialised, and only numpy was ever seeded. "
                         "Measured over two identical invocations, 200 of 360 reported values "
                         "moved: RR_held by up to .056, Reg by up to .023 (100% of its own "
                         "value on the small cells). Per-setting Reg/CReg are therefore not "
                         "reportable from a single draw -- use --repeats.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Independent draws per cell (torch-seed + i). Every draw is kept in the "
                         "JSON under `draws`, so the spread can be reported rather than hidden.")
    ap.add_argument("--per-sample", default=None,
                    help="directory to write per-sample prediction CSVs into (both operating "
                         "points: `tau` = the fixed threshold, `matched` = the clean_calib "
                         "threshold at our own Reg). Costs nothing -- the vectors already exist.")
    ap.add_argument("--early-stop", action="store_true",
                    help="hold out 30%% of the reported failures and keep the best patch head")
    a = ap.parse_args()
    EST_ARCH[0] = a.estimator
    EARLY[0] = a.early_stop
    load_ours_reg()
    res: dict = {}
    for ds, bb in itertools.product(DSS, BBS):
        for seed in SEEDS:
            d = load(ds, bb, seed)
            if d is None:
                continue
            CRIT[0] = critical(ds)
            NCLS[0] = int(max(d["clean_test__y"].max(), d["bug_train__y"].max(),
                              d["bug_eval__y"].max(), d["clean_train__y"].max()) + 1)
            for name, fn in (("NN-Patching", nn_patching), ("PatchNAS", patchnas)):
              draws = []
              for rep in range(a.repeats):
                if a.torch_seed is not None:
                    torch.manual_seed(a.torch_seed + rep)
                    torch.cuda.manual_seed_all(a.torch_seed + rep)
                r = fn(d)
                m = metrics(d, r["patch"], r["score"], a.tau)
                m["n_classes"] = NCLS[0]
                if "arch" in r:
                    m["arch"] = list(r["arch"]); m["select_on_train"] = bool(r["select_on_train"])
                # the swept routing curve, so both can be placed on the RQ3 frontier
                grid = np.unique(np.quantile(r["score"]["clean_test"],
                                             np.linspace(0.0, 1.0, 41)))
                # second operating point: threshold placed on clean_calib against the
                # regression our own patch produces on this cell, so the comparison is at
                # matched safety instead of at their default 0.5.
                tgt = OURS_REG.get(f"{ds}/{bb}", 0.016)
                tc = calib_tau(d, r["patch"], r["score"], tgt)
                mc = metrics(d, r["patch"], r["score"], tc)
                m["matched"] = {"target_reg": tgt, "tau": tc, **mc}
                m["curve"] = [[float(t)] + [metrics(d, r["patch"], r["score"], float(t))[k]
                                            for k in ("RR_held", "Reg")] for t in grid]
                if a.per_sample:
                    for tval, tg in ((a.tau, "tau"), (tc, "matched")):
                        dump_per_sample(Path(a.per_sample), name, ds, bb, seed, d, r, tval, tg)
                print(f"{name:12s} {ds}/{bb:14s} s{seed}  RR_seen={m['RR_seen']:.3f} "
                      f"RR_held={m['RR_held']:.3f} Reg={m['Reg']:.3f} "
                      f"route={m['route_rate']:.3f}  |matched Reg<={m['matched']['target_reg']:.3f}: "
                      f"RR_held={m['matched']['RR_held']:.3f} Reg={m['matched']['Reg']:.3f}"
                      + (f"  arch={m.get('arch')}" if "arch" in m else ""), flush=True)
                draws.append(m)
              # A single draw is not reportable: with --torch-seed unset these methods moved
              # 200 of 360 values between two identical invocations (Reg by up to 100% of its
              # own value). Keep every draw and report the median with its range.
              base = dict(draws[0])
              if a.repeats > 1:
                  base["draws"] = draws
                  for k in ("RR_seen", "RR_held", "Reg", "CReg", "route_rate"):
                      v = [dd[k] for dd in draws if k in dd]
                      if v:
                          base[k] = float(np.median(v))
                          base[f"{k}__min"], base[f"{k}__max"] = float(min(v)), float(max(v))
                  mv = [dd["matched"] for dd in draws if "matched" in dd]
                  if mv:
                      base["matched"] = dict(mv[0])
                      for k in ("RR_held", "Reg", "CReg"):
                          v = [x[k] for x in mv if k in x]
                          if v:
                              base["matched"][k] = float(np.median(v))
                              base["matched"][f"{k}__min"] = float(min(v))
                              base["matched"][f"{k}__max"] = float(max(v))
              res.setdefault(name, {}).setdefault(f"{ds}/{bb}", {})[str(seed)] = base
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(f"[written] {a.json}")


if __name__ == "__main__":
    main()
