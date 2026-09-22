"""Minimal scoring helpers for NN-Patching / PatchNAS checkpoint-eval (nnpatch.py, patchnas.py).

Fresh, minimal reimplementation of just the pieces those two scripts' `--mode checkpoint` path
calls (`head`, `predict`, `predict_logits`, `standardise`, `prob1`, `metrics`, `calib_tau`,
`critical`, `CRIT`, `DEV`, `load_ours_reg`/`OURS_REG`) -- the original
`scripts/baseline_prior_patches.py` this replaces was removed. Architecture (`head`) is reverse
engineered from the actual shipped checkpoint state_dicts:
  - patch_head.pt, arch=(0, 0, 'relu'): bare `nn.Linear(din, dout)` (keys "weight"/"bias", no
    Sequential wrapper -- depth=0 means no hidden layers).
  - estimator_head.pt, arch="mlp": `nn.Sequential(Linear(din,256), ReLU, Linear(256,256), ReLU,
    Linear(256,2))` (keys "0.*", "2.*", "4.*"), i.e. head(din, 2, depth=2, width=256, act='relu').
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CRIT: list[set[int] | None] = [None]     # mutable single-slot, set via CRIT[0] = critical(dataset)
CKPT_ROOT: list[Path | None] = [None]    # unused in --mode checkpoint; kept for signature parity
NCLS: list[int | None] = [None]

OURS_REG: dict[str, float] = {}
_OURS_REG_LOOKUP = ROOT / "outputs/repro/dynapatch_reg_lookup.json"
_OURS_REG_FALLBACK = 0.016  # same fallback the original script used when the lookup isn't present


def critical(dataset: str) -> set[int]:
    cfg = json.loads((ROOT / f"artifacts/risk/{dataset}_safety_risk_matrix.json").read_text())
    out: set[int] = set()
    for ids in cfg["critical_signs"].values():
        out.update(int(i) for i in ids)
    return out


def head(din: int, dout: int, depth: int, width: int, act: str = "relu") -> nn.Module:
    """depth=0 -> bare nn.Linear(din, dout). depth>0 -> Sequential(Linear, Act, ..., Linear)."""
    if depth == 0:
        return nn.Linear(din, dout)
    activation = {"relu": nn.ReLU}.get(act, nn.ReLU)
    layers: list[nn.Module] = []
    d = din
    for _ in range(depth):
        layers += [nn.Linear(d, width), activation()]
        d = width
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


def standardise(fit_data: np.ndarray, apps: list[np.ndarray]) -> tuple[tuple[np.ndarray, np.ndarray], list[np.ndarray]]:
    """Fit mean/std on `fit_data`, apply to every array in `apps`."""
    mean = fit_data.mean(axis=0, keepdims=True)
    std = fit_data.std(axis=0, keepdims=True) + 1e-8
    return (mean, std), [(a - mean) / std for a in apps]


@torch.no_grad()
def predict(net: nn.Module, x: np.ndarray) -> np.ndarray:
    logits = net(torch.as_tensor(x, dtype=torch.float32, device=DEV))
    return logits.argmax(dim=1).cpu().numpy()


@torch.no_grad()
def predict_logits(net: nn.Module, x: np.ndarray) -> np.ndarray:
    return net(torch.as_tensor(x, dtype=torch.float32, device=DEV)).cpu().numpy()


@torch.no_grad()
def prob1(net: nn.Module, x: np.ndarray) -> np.ndarray:
    logits = net(torch.as_tensor(x, dtype=torch.float32, device=DEV))
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def _route(score: np.ndarray, tau: float) -> np.ndarray:
    return score > tau


def metrics(d: dict, patch: dict[str, np.ndarray], score: dict[str, np.ndarray], tau: float) -> dict[str, float]:
    """RR_seen/RR_held/Reg/CReg/route_rate at routing threshold `tau` (score[pop] > tau applies
    the patch prediction, else the base prediction stands) -- same definitions as
    scripts/checkpoint_eval/common.py's summarize()."""
    def merged(pop: str) -> tuple[np.ndarray, np.ndarray]:
        route = _route(score[pop], tau)
        base = d[f"{pop}__pred"]
        pat = patch[pop]
        return np.where(route, pat, base), route

    m_bt, r_bt = merged("bug_train")
    m_be, r_be = merged("bug_eval")
    m_ct, r_ct = merged("clean_test")
    y_bt, y_be, y_ct = d["bug_train__y"], d["bug_eval__y"], d["clean_test__y"]
    base_ct = d["clean_test__pred"]

    rr_seen = float(np.mean(m_bt == y_bt)) if len(y_bt) else 0.0
    rr_held = float(np.mean(m_be == y_be)) if len(y_be) else 0.0
    reg_mask = (base_ct == y_ct) & (m_ct != y_ct)
    reg = float(reg_mask.mean()) if len(y_ct) else 0.0
    crit = CRIT[0]
    if crit:
        crit_idx = np.isin(y_ct, list(crit))
        creg = float(reg_mask[crit_idx].mean()) if crit_idx.sum() else 0.0
    else:
        creg = 0.0
    route_rate = float(np.mean(np.concatenate([r_bt, r_be, r_ct]))) if (len(r_bt) + len(r_be) + len(r_ct)) else 0.0
    return {"RR_seen": rr_seen, "RR_held": rr_held, "Reg": reg, "CReg": creg, "route_rate": route_rate}


def calib_tau(d: dict, patch: dict[str, np.ndarray], score: dict[str, np.ndarray],
               target_reg: float, grid: np.ndarray | None = None) -> float:
    """Smallest tau (most aggressive routing) on `clean_calib` whose resulting Reg <= target_reg.
    Falls back to 1.0 (never patch) if no grid point satisfies the budget."""
    grid = grid if grid is not None else np.linspace(0.0, 1.0, 201)
    base = d["clean_calib__pred"]
    pat = patch["clean_calib"]
    y = d["clean_calib__y"]
    sc = score["clean_calib"]
    for tau in grid:
        route = sc > tau
        merged = np.where(route, pat, base)
        reg_mask = (base == y) & (merged != y)
        reg = float(reg_mask.mean()) if len(y) else 0.0
        if reg <= target_reg:
            return float(tau)
    return 1.0


def load_ours_reg() -> None:
    """Populate OURS_REG from outputs/repro/dynapatch_reg_lookup.json (written by
    scripts/repro/rq1_aggregate.py's DynaPatch pass) if present; empty otherwise, in which case
    callers fall back to _OURS_REG_FALLBACK per-cell."""
    OURS_REG.clear()
    if _OURS_REG_LOOKUP.exists():
        OURS_REG.update(json.loads(_OURS_REG_LOOKUP.read_text()))
