#!/usr/bin/env python3
"""M3 -- does merging DistRep's pre-integration experts shift correction direction, or shrink
its magnitude, on the held-out failures the experts could repair?

Scope, stated up front because it is easy to overclaim from a single-setting case study: this
answers ONLY "what does integration lose relative to the experts themselves" on
gtsrb/resnet50 seed 101 (the only setting/seed with expert-level logits -- a full-budget DistRep
PSO run costs ~46min PER CELL, see note/RESEARCH_STATE.md "DistRep expert logits"). It does NOT
explain why DistRep underperforms DynaPatch overall, and must not be cited as if it did -- that
comparison needs the other 11 settings, which do not exist for this baseline.

Method, in the same logit space as the M1 series: for each held-out failure x,

    Delta z_i(x) = expert_i's patched_logits(x) - base_logits(x)      for each of the 5 experts
    Delta z_m(x) = merged model's patched_logits(x) - base_logits(x)

`base_logits(x)` is the SAME frozen deployed model for every expert and the merged model, so all
six Delta z's live in the same space and are directly comparable -- no scale-mismatch caveat like
the NN-Patching/PatchNAS one (those used a freshly trained head; DistRep's experts and merged
model are all perturbations of the SAME frozen backbone's own classifier).

For each x where AT LEAST ONE expert repairs it (argmax flips to y), reference direction
`Delta z_best(x)` is the repairing expert with the largest true-vs-wrong margin gain
(Delta m_i(x), same definition as scripts/analysis_m1_correction_direction.py) -- the strongest
available fix for that input. Then, split by whether the MERGED model retains the repair or not:

    cos_to_best(x)   = cosine similarity of Delta z_m(x) to Delta z_best(x) (unit-normalised)
    magnitude_ratio(x) = ||Delta z_m(x)|| / ||Delta z_best(x)||

A retained-repair group with high cos_to_best and magnitude_ratio near 1, versus a lost-repair
group with either low cos_to_best (direction shifted) or a magnitude_ratio far from 1 (shrunk or
overshot), is the raw signal this script reports -- no test is run given n ~= 30, this is
descriptive only.

    .venv/bin/python scripts/analysis_m3_distrep_expert_merge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402

BASE = ROOT / "outputs/distrep_expert_oracle_gtsrb_resnet50_s101"
SPLIT = "repair_holdout_unseen"
N_EXPERTS = 5


def load_logits(dirp: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    base = np.load(dirp / f"base_logits_{SPLIT}.npy").astype(np.float64)
    patched = np.load(dirp / f"patched_logits_{SPLIT}.npy").astype(np.float64)
    t = pd.read_csv(dirp / f"{SPLIT}_predictions.csv")
    return base, patched, t


def dm(base: np.ndarray, patched: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    ar = np.arange(len(y))
    return (patched[ar, y] - patched[ar, yhat]) - (base[ar, y] - base[ar, yhat])


def main() -> None:
    merged_base, merged_patched, t = load_logits(BASE / "predictions")
    y = t.label.to_numpy()
    yhat = t.base_pred.to_numpy()
    n = len(t)
    fail = (t.base_pred != t.label).to_numpy()
    if not fail.all():
        raise SystemExit(f"expected all {SPLIT} rows to be base-wrong failures, {fail.sum()}/{n} were")

    merged_dz = merged_patched - merged_base
    merged_dm = dm(merged_base, merged_patched, y, yhat)
    merged_repaired = merged_patched.argmax(1) == y

    expert_dz, expert_dm, expert_repaired = [], [], []
    for i in range(N_EXPERTS):
        edirp = BASE / f"expert_predictions/expert_{i:02d}"
        ebase, epatched, et = load_logits(edirp)
        if not (et.dataset_index.to_numpy() == t.dataset_index.to_numpy()).all():
            raise SystemExit(f"expert_{i:02d}: dataset_index order mismatch vs merged predictions")
        if not np.allclose(ebase, merged_base):
            raise SystemExit(f"expert_{i:02d}: base_logits differ from merged's -- should be the "
                             f"same frozen deployed model")
        edz = epatched - ebase
        edm = dm(ebase, epatched, y, yhat)
        erep = epatched.argmax(1) == y
        expert_dz.append(edz)
        expert_dm.append(edm)
        expert_repaired.append(erep)
    expert_repaired = np.stack(expert_repaired)          # [5, n]
    expert_dm = np.stack(expert_dm)                       # [5, n]
    expert_dz = np.stack(expert_dz)                       # [5, n, n_classes]

    any_expert_repairs = expert_repaired.any(axis=0)
    n_repairing = expert_repaired.sum(axis=0)

    rows = []
    for j in range(n):
        if not any_expert_repairs[j]:
            continue
        repairing_experts = np.where(expert_repaired[:, j])[0]
        best_i = repairing_experts[np.argmax(expert_dm[repairing_experts, j])]
        dz_best = expert_dz[best_i, j]
        dz_m = merged_dz[j]
        nb, nm = np.linalg.norm(dz_best), np.linalg.norm(dz_m)
        cos_to_best = float(np.dot(dz_best, dz_m) / (nb * nm + 1e-12))
        magnitude_ratio = float(nm / (nb + 1e-12))
        rows.append({
            "dataset_index": int(t.dataset_index.iloc[j]), "true_label": int(y[j]),
            "base_pred": int(yhat[j]), "n_experts_repair": int(n_repairing[j]),
            "best_expert": int(best_i), "best_expert_dm": float(expert_dm[best_i, j]),
            "merged_dm": float(merged_dm[j]), "merged_repairs": bool(merged_repaired[j]),
            "cos_to_best_expert": cos_to_best, "magnitude_ratio_merged_over_best": magnitude_ratio,
        })
    cell = pd.DataFrame(rows)

    summary_rows = []
    for group_name, group in (("retained (merged also repairs)", cell[cell.merged_repairs]),
                              ("lost (merged fails)", cell[~cell.merged_repairs])):
        if len(group) == 0:
            summary_rows.append({"group": group_name, "n": 0, "mean_cos_to_best_expert": float("nan"),
                                "median_cos_to_best_expert": float("nan"),
                                "mean_magnitude_ratio": float("nan"),
                                "median_magnitude_ratio": float("nan")})
            continue
        summary_rows.append({
            "group": group_name, "n": len(group),
            "mean_cos_to_best_expert": float(group.cos_to_best_expert.mean()),
            "median_cos_to_best_expert": float(group.cos_to_best_expert.median()),
            "mean_magnitude_ratio": float(group.magnitude_ratio_merged_over_best.mean()),
            "median_magnitude_ratio": float(group.magnitude_ratio_merged_over_best.median()),
        })
    summary = pd.DataFrame(summary_rows)

    RM.write_section(
        "M3", "M3 — DistRep expert-to-merged retention: direction shift vs magnitude shrinkage "
        "(raw, single setting/seed case study)",
        f"""
**Scope: gtsrb/resnet50, seed 101 only** (the only setting/seed with expert-level logits -- a
full-budget DistRep PSO run costs ~46 minutes PER CELL). This does NOT explain why DistRep
underperforms DynaPatch overall (that needs all 12 settings, which do not exist for this
baseline's expert-level data) -- it explains only what integration loses relative to the experts
themselves, on this one setting.

For each held-out failure at least one of the 5 pre-integration experts repairs
({int(any_expert_repairs.sum())} of {n} failures), the reference direction `Delta z_best(x)` is
the repairing expert with the largest true-vs-wrong margin gain for that input (the strongest
available fix). `cos_to_best_expert` is the cosine similarity of the MERGED model's own
`Delta z_m(x)` to that reference direction; `magnitude_ratio_merged_over_best` is
`||Delta z_m(x)|| / ||Delta z_best(x)||`. Both use the SAME frozen base model's logits for every
expert and the merged model (verified: base_logits identical across all 6), so there is no
scale-mismatch caveat here (unlike the NN-Patching/PatchNAS comparisons in the M1 series).

Split by whether the merged model retains the repair or loses it. No statistical test is run
(n~{int(any_expert_repairs.sum())}, single setting/seed) -- this is descriptive only.
""",
        [("per_failure", cell.sort_values("dataset_index")),
         ("retained_vs_lost", summary)],
    )

    out = ROOT / "outputs" / "m3_distrep_expert_merge"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_failure.csv", index=False)
    summary.to_csv(out / "retained_vs_lost_summary.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/per_failure.csv  ({len(cell)} rows)")
    print(f"[written] {out.relative_to(ROOT)}/retained_vs_lost_summary.csv")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
