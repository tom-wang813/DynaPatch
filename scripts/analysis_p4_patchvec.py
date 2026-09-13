#!/usr/bin/env python3
"""P4 -- does the generator actually specialise, in its own output space?

Every other analysis infers specialisation from outcomes. This looks at the thing the generator
produces: the per-input delta `patch_vec` (2048-d, already on disk in the ep40ns dump trees).

The question: are two deltas generated for inputs that fail the SAME way closer to each other
than two deltas generated for inputs that fail differently?

    d_within    d(p_i, p_j)  for  failure_type(i) == failure_type(j)
    d_between   d(p_i, p_j)  for  failure_type(i) != failure_type(j)

Both cosine and normalised L2 are reported: cosine sees direction only, and a generator that
scales one shared direction by input-dependent magnitudes would look specialised under L2 and
flat under cosine. That contrast is itself informative, so neither metric is dropped.

Controls that decide whether any gap means anything
---------------------------------------------------
1. **A label-shuffle null.** The same pair distances with the failure-type labels permuted
   within the cell. A gap that survives against this is not an artefact of unequal group sizes
   -- the between set is always the larger one, and larger sets have larger mean pair distance
   under almost any distribution.
2. **Class as well as ordered pair.** `failure_type` is (true -> predicted); `true_class` is the
   coarser grouping. If the structure is only class-level, the generator conditions on what the
   input IS, not on how it fails.
3. **Both populations.** `seen` deltas come from inputs the generator was trained on; `held`
   deltas do not. Structure present only on `seen` is memorisation.

A null result here is reported as a null result. The generator can be doing something useful
without its output space showing group structure under a distance this crude.

    .venv/bin/python scripts/analysis_p4_patchvec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402

SEEDS = (101, 202, 303)
SETTINGS = [(d, b) for d in ("gtsrb", "tt100k_signs", "lisa_signs")
            for b in ("resnet50", "convnext_tiny", "densenet121", "vgg16")]
SPLITS = {"repair_support_seen": "seen", "repair_holdout_unseen": "held"}
TAG = "_ep40ns"
NPERM = 200
MAX_N = 600           # cap vectors per cell; ~180k pairs, ample for a mean


def pair_stats(P: np.ndarray, lab: np.ndarray, rng: np.random.Generator) -> dict | None:
    """Mean within/between pair distance, plus the same gap under label permutation."""
    n = len(P)
    if n < 6 or len(np.unique(lab)) < 2:
        return None
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    cos = 1.0 - Pn @ Pn.T                                  # [n, n] cosine distance
    # Gram trick ALWAYS. The direct form allocates [n, n, 2048]: at n=400 that is 2.6 GB of
    # float64, and this analysis reached 14.7 GB beside a DistRep PSO on a shared 62 GB box --
    # the exact pairing that OOM-killed four cells on 2026-09-01. Never reintroduce it.
    sq = (P ** 2).sum(1)
    l2 = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0))
    l2 = l2 / (np.median(l2[np.triu_indices(n, 1)]) + 1e-12)   # scale-free

    iu = np.triu_indices(n, 1)
    i0, i1 = iu
    out: dict = {"n_vec": n, "n_groups": int(len(np.unique(lab)))}

    def gap(mask_same, v):
        a, b = v[mask_same], v[~mask_same]
        if len(a) < 3 or len(b) < 3:
            return None, None, None
        return float(a.mean()), float(b.mean()), float(b.mean() - a.mean())

    same = lab[i0] == lab[i1]
    for nm, D in (("cos", cos), ("l2", l2)):
        v = D[iu]                       # flattened ONCE: recomputing it per permutation is
        w, b, g = gap(same, v)          # what made this analysis quadratic in NPERM
        if w is None:
            return None
        out |= {f"{nm}_within": w, f"{nm}_between": b, f"{nm}_gap": g}
        perm = []
        for _ in range(NPERM):
            s = rng.permutation(lab)
            sm = s[i0] == s[i1]
            aa, bb = v[sm], v[~sm]
            if len(aa) >= 3 and len(bb) >= 3:
                perm.append(bb.mean() - aa.mean())
        if perm:
            perm = np.asarray(perm)
            out[f"{nm}_gap_null_mean"] = float(perm.mean())
            out[f"{nm}_gap_null_sd"] = float(perm.std())
            out[f"{nm}_z"] = float((g - perm.mean()) / (perm.std() + 1e-12))
            out[f"{nm}_p_perm"] = float((np.abs(perm) >= abs(g)).mean())
    out["n_pairs_within"] = int(same.sum())
    out["n_pairs_between"] = int((~same).sum())
    return out


def main() -> None:
    rows = []
    for ds, bb in SETTINGS:
        # 2026-09-05: vgg16/convnext_tiny adopted repair.patch_site=last_affine (see
        # note/RQ1_DATA.md's header and note/RESEARCH_STATE.md); resnet50/densenet121 are
        # untouched (last_affine is bit-identical to deep_feat there).
        tag = "_lastaffine" if bb in ("vgg16", "convnext_tiny") else TAG
        for s in SEEDS:
            d = ROOT / f"outputs/effect_dump{tag}_v8_s{s}/{ds}/{bb}/deploy_direct/predictions"
            if not d.is_dir():
                continue
            for split, tag in SPLITS.items():
                pv = d / f"patch_vec_{split}.npy"
                pc = d / f"{split}_predictions.csv"
                if not (pv.is_file() and pc.is_file()):
                    continue
                P = np.load(pv).astype(np.float64)
                t = pd.read_csv(pc)
                if len(t) != len(P):
                    rows.append({"setting": f"{ds}/{bb}", "seed": s, "split": tag,
                                 "grouping": "-", "note": f"row mismatch {len(t)} vs {len(P)}"})
                    continue
                fail = (t.base_pred != t.label).to_numpy()
                P, t = P[fail], t[fail]
                if len(P) > MAX_N:
                    sel = np.random.default_rng(0).choice(len(P), MAX_N, replace=False)
                    P, t = P[sel], t.iloc[sel]
                for gname, lab in (("failure_type",
                                    (t.label.astype(str) + "->" + t.base_pred.astype(str))
                                    .to_numpy()),
                                   ("true_class", t.label.to_numpy())):
                    st = pair_stats(P, lab, np.random.default_rng(0))
                    if st is None:
                        rows.append({"setting": f"{ds}/{bb}", "seed": s, "split": tag,
                                     "grouping": gname, "n_vec": len(P),
                                     "note": "too few vectors or one group only"})
                        continue
                    rows.append({"setting": f"{ds}/{bb}", "seed": s, "split": tag,
                                 "grouping": gname, **st, "note": ""})
    if not rows:
        raise SystemExit("no patch_vec dumps found")
    cell = pd.DataFrame(rows)

    RM.write_section("P4", "P4 — patch-vector geometry: does the generator specialise? (raw)",
                     f"""
The generator's own output: `patch_vec`, the 2048-d delta it produces per input, taken from the
`effect_dump{TAG}` trees. Rows are restricted to inputs the deployed model got WRONG (a delta for
a correctly-classified input has no failure type).

For each (setting, seed, split, grouping): the mean distance between deltas of the SAME group
versus DIFFERENT groups, and whether that gap survives a label permutation.

| column | meaning |
|---|---|
| `grouping` | `failure_type` = (true -> predicted); `true_class` = true label only |
| `n_vec` / `n_groups` | vectors compared, and distinct groups among them (capped at {MAX_N}) |
| `cos_within` / `cos_between` / `cos_gap` | mean cosine distance within, between, and between-minus-within |
| `l2_within` / `l2_between` / `l2_gap` | the same for L2, divided by the cell's median pair distance so cells are comparable |
| `*_gap_null_mean` / `*_gap_null_sd` | the same gap over {NPERM} within-cell label permutations |
| `*_z` / `*_p_perm` | the observed gap against that null |
| `n_pairs_within` / `n_pairs_between` | pair counts; the between set is always larger, which is exactly why the permutation null is needed |

Both metrics are kept on purpose: a generator that rescales ONE shared direction per input looks
specialised in L2 and flat in cosine. `seen` deltas come from inputs the generator trained on and
`held` deltas do not — structure present only on `seen` is memorisation, not specialisation.
""", [("", cell.sort_values(["grouping", "split", "setting", "seed"]))])

    out = ROOT / "outputs" / "p4_patchvec"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_cell.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/per_cell.csv  ({len(cell)} rows)")


if __name__ == "__main__":
    main()
