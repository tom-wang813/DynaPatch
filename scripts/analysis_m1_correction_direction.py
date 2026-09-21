#!/usr/bin/env python3
"""M1 -- do NN-Patching, PatchNAS, and DynaPatch-NoGate correct different failures differently?

The hypothesis under test (2026-09-04): DynaPatch's advantage is not that its corrections are
BIGGER, it is that different failures get DIFFERENT correction directions -- input-conditioned
patch freedom lets the same failure type be corrected differently input by input, and that is
what the other two patch-based baselines cannot do as well.

Everything here is computed in a single shared space: the deployed model's own logit space
(dimension = n_classes for that setting), which every method's corrected prediction lives in
regardless of where the method injects its patch. For each failure input x:

    Delta z(x) = z_patched(x) - z_base(x)                        (n_classes-d correction vector)
    Delta m(x) = [z'_y - z'_yhat] - [z_y - z_yhat]                (true-vs-wrong margin improvement)

where y is the true label and yhat is the deployed (base) model's wrong prediction. Delta m(x) is
exactly the projection of Delta z(x) onto the fixed two-sparse direction (e_y - e_yhat) -- margin
improvement and "did the correction point the right way" are the same computation read two ways,
so no separate metric needs to be defined for "pointing toward the true class".

Primary question -- direction, not magnitude
----------------------------------------------
Reuses the exact within/between-group cosine-distance method scripts/analysis_p4_patchvec.py
already validated on the generator's feature-space `patch_vec` (label-shuffle null, both cosine
and L2, both `failure_type` and `true_class` groupings, both `seen` and `held` populations) --
applied here to `Delta z(x)` instead, and computed for THREE methods side by side so the
within/between gap itself is comparable across methods, not just present-or-absent for one.

    A method that uses ONE shared direction per failure type: cos_within tiny, cos_between
    large, gap large and it survives the permutation null -- but the SAME direction serves every
    input of that type, which does not distinguish it from a per-type lookup table.
    A method whose correction is genuinely per-input, even within one failure type: cos_within
    is NOT near zero (same-type deltas still are not identical), which is a different, and more
    interesting, form of structure than a clean within/between split. Both are reported; neither
    is assumed in advance.

Secondary reads -- magnitude and margin, kept separate so "more input-adapted direction" cannot
be confounded with "just moves more"
-----------------------------------------------------------------------------------------------
Per (method, setting, seed, split): mean/median ||Delta z(x)||, mean/median Delta m(x), and the
fraction of failures with Delta m(x) > 0 (does the correction even point toward the true class
over the wrong one, regardless of whether it repairs the prediction outright).

Data sources (verified present, not assumed; see note/RESEARCH_STATE.md "Data pipeline changes"
for how these were produced)
-----------------------------------------------------------------------------------------------
  DynaPatch-NoGate   outputs/effect_dump_ep40ns_v8_s{seed}/<ds>/<bb>/deploy_direct/predictions/
                     {base,patched}_logits_{repair_support_seen,repair_holdout_unseen}.npy
  NN-Patching        outputs/prior_patch_persample/<ds>_<bb>_s<seed>/NN-Patching/tau/
                     {base,patched}_logits_{bug_train,bug_eval}.npy
  PatchNAS           outputs/prior_patch_persample/<ds>_<bb>_s<seed>/PatchNAS/tau/
                     {base,patched}_logits_{bug_train,bug_eval}.npy

`patched_logits` for NN-Patching/PatchNAS is the patch head's raw, un-routed output (see
scripts/baseline_prior_patches.py:predict_logits) -- the same "what would the patch mechanism
produce for this failure" question DynaPatch-NoGate answers by construction (no gate). Using the
`tau` operating-point directory is arbitrary: `patched_logits` is identical between `tau` and
`matched` (same trained head, only the routing threshold differs), verified 2026-09-04.

    .venv/bin/python scripts/analysis_m1_correction_direction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM                     # noqa: E402
from analysis_p4_patchvec import pair_stats   # noqa: E402  (reuse the validated within/between test)

SEEDS = (101, 202, 303)
SETTINGS = [(d, b) for d in ("gtsrb", "tt100k_signs", "lisa_signs")
            for b in ("resnet50", "convnext_tiny", "densenet121", "vgg16")]
SPLIT_TAG = {"seen": "seen", "held": "held"}
NPERM = 200
MAX_N = 600           # matches P4; pair_stats is O(n^2), this population never exceeds it here
MIN_TYPE_N = 5         # same "estimable failure type" floor AUDIT.md already uses project-wide
FIGURE_CELL = ("gtsrb/convnext_tiny", 101, "held")   # fixed choice, see plot_direction_pca()


def purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Fraction of points whose cluster's majority true label matches their own."""
    total = 0
    for c in np.unique(cluster_labels):
        vals, counts = np.unique(true_labels[cluster_labels == c], return_counts=True)
        total += int(counts.max())
    return total / len(true_labels)


def cluster_alignment(dz: np.ndarray, fail_type: np.ndarray) -> dict | None:
    """Spherical k-means (k-means on unit-normalised Delta z) with k = number of observed
    failure types, scored against the TRUE failure-type labels by purity, ARI, and NMI.

    This is a stronger and more legible claim than the pairwise cosine gap in `pair_stats`: it
    asks whether clustering on direction ALONE, with no knowledge of the labels beyond how many
    groups to form, recovers which failure type each input belongs to. Types with fewer than
    MIN_TYPE_N members are dropped first (same floor as note/AUDIT.md's "estimable" failure
    types) so the cluster count is not dominated by singletons.
    """
    vals, counts = np.unique(fail_type, return_counts=True)
    keep_types = vals[counts >= MIN_TYPE_N]
    mask = np.isin(fail_type, keep_types)
    if mask.sum() < 2 * len(keep_types) or len(keep_types) < 2:
        return None
    dz_k, lab_k = dz[mask], fail_type[mask]
    unit = dz_k / (np.linalg.norm(dz_k, axis=1, keepdims=True) + 1e-12)
    k = len(keep_types)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(unit)
    return {
        "n": int(mask.sum()), "n_types": int(k),
        "purity": float(purity(lab_k, km.labels_)),
        "ari": float(adjusted_rand_score(lab_k, km.labels_)),
        "nmi": float(normalized_mutual_info_score(lab_k, km.labels_)),
    }


def plot_direction_pca(cells: dict, figure_dir: Path) -> None:
    """One fixed (setting, seed, split), three methods side by side: a 2-D PCA projection of the
    unit-normalised Delta z(x) vectors, coloured by failure type (types with fewer than
    MIN_TYPE_N members shown in light gray, unlabelled). This is the direct visual counterpart of
    `cluster_alignment` -- the same population, plotted rather than scored, so cluster structure
    (or its absence) can be read off directly instead of taken on a metric's word.

    FIGURE_CELL is fixed rather than picked by best-looking result, to avoid cherry-picking: it
    is a mid-size setting (217 base-wrong on `held`, 27 failure types with n>=5, per
    note/AUDIT.md) chosen for legibility before any number in this script was looked at.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import tab20
    from sklearn.decomposition import PCA

    setting, seed, split = FIGURE_CELL
    ds, bb = setting.split("/")
    methods = list(cells.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.6), sharex=False,
                             sharey=False)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        cell = cells[method]
        if cell is None:
            ax.set_title(f"{method}\n(no data)")
            ax.axis("off")
            continue
        dz, fail_type = cell["dz"], cell["fail_type"]
        unit = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
        xy = PCA(n_components=2, random_state=0).fit_transform(unit)
        vals, counts = np.unique(fail_type, return_counts=True)
        keep = set(vals[counts >= MIN_TYPE_N])
        big = np.array([t in keep for t in fail_type])
        ax.scatter(xy[~big, 0], xy[~big, 1], s=14, c="#C9CDD3", linewidths=0, label=None)
        big_types = sorted(keep)
        cmap = {t: tab20(i % 20) for i, t in enumerate(big_types)}
        for t in big_types:
            m = fail_type == t
            ax.scatter(xy[m, 0], xy[m, 1], s=20, color=cmap[t], linewidths=0, label=t)
        ax.set_title(f"{method}\n(n={len(fail_type)}, {len(big_types)} types n>={MIN_TYPE_N} coloured)",
                     fontsize=9)
        ax.set_xlabel("PC1 of unit Delta z(x)", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Correction direction, PCA of unit Delta z(x) — {setting} s{seed} {split}",
                fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Each point is one failure input's correction direction. Same colour = same failure "
             "type (true label -> base prediction). Gray = failure type with fewer than "
             f"{MIN_TYPE_N} members.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    slug = setting.replace("/", "_")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"direction_pca_{slug}_s{seed}_{split}.{suffix}", dpi=220,
                   bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dynapatch_dir(ds: str, bb: str, seed: int) -> tuple[Path, dict]:
    # 2026-09-05: vgg16/convnext_tiny adopted repair.patch_site=last_affine (see
    # note/RQ1_DATA.md's header and note/RESEARCH_STATE.md); resnet50/densenet121 are
    # untouched (last_affine is bit-identical to deep_feat there, so effect_dump_ep40ns_v8
    # is correct for them either way -- there is no separate lastaffine tree for them).
    tag = "_lastaffine" if bb in ("vgg16", "convnext_tiny") else "_ep40ns"
    d = ROOT / f"outputs/effect_dump{tag}_v8_s{seed}/{ds}/{bb}/deploy_direct/predictions"
    return d, {"seen": "repair_support_seen", "held": "repair_holdout_unseen"}


def prior_patch_dir(name: str) -> "callable":
    def _f(ds: str, bb: str, seed: int) -> tuple[Path, dict]:
        d = ROOT / f"outputs/prior_patch_persample/{ds}_{bb}_s{seed}/{name}/tau"
        return d, {"seen": "bug_train", "held": "bug_eval"}
    return _f


METHODS = {
    "DynaPatch-NoGate": dynapatch_dir,
    "NN-Patching": prior_patch_dir("NN-Patching"),
    "PatchNAS": prior_patch_dir("PatchNAS"),
}


def load_cell(dir_fn, ds: str, bb: str, seed: int, split: str):
    """Delta z(x), Delta m(x), and the failure-type labels for one (method, setting, seed, split).

    Returns None if any required artefact is missing -- reported as a note row, never silently
    dropped from the output.
    """
    d, pop_of = dir_fn(ds, bb, seed)
    pop = pop_of[split]
    bl, pl, pc = (d / f"base_logits_{pop}.npy", d / f"patched_logits_{pop}.npy",
                 d / f"{pop}_predictions.csv")
    if not (bl.is_file() and pl.is_file() and pc.is_file()):
        return None, "missing artefact"
    base = np.load(bl).astype(np.float64)
    patched = np.load(pl).astype(np.float64)
    t = pd.read_csv(pc)
    if not (len(t) == len(base) == len(patched)):
        return None, f"row mismatch csv={len(t)} base={len(base)} patched={len(patched)}"
    fail = (t.base_pred != t.label).to_numpy()
    if fail.sum() < 6:
        return None, f"too few failures ({int(fail.sum())})"
    base, patched, t = base[fail], patched[fail], t[fail].reset_index(drop=True)
    y = t.label.to_numpy()
    yhat = t.base_pred.to_numpy()
    n = len(t)
    ar = np.arange(n)
    dz = patched - base
    dm = (patched[ar, y] - patched[ar, yhat]) - (base[ar, y] - base[ar, yhat])
    repaired = patched.argmax(1) == y
    return {"dz": dz, "dm": dm, "y": y, "yhat": yhat, "n": n, "repaired": repaired}, None


def main() -> None:
    direction_rows: list[dict] = []
    magnitude_rows: list[dict] = []
    cluster_rows: list[dict] = []
    figure_cells: dict = {}
    for method, dir_fn in METHODS.items():
        for ds, bb in SETTINGS:
            for seed in SEEDS:
                for split in ("seen", "held"):
                    cell, err = load_cell(dir_fn, ds, bb, seed, split)
                    if err is not None:
                        direction_rows.append({"method": method, "setting": f"{ds}/{bb}",
                                               "seed": seed, "split": split, "grouping": "-",
                                               "note": err})
                        continue
                    dz, dm, y, yhat, n = (cell["dz"], cell["dm"], cell["y"], cell["yhat"],
                                          cell["n"])
                    fail_type = (pd.Series(y).astype(str) + "->" + pd.Series(yhat).astype(str)).to_numpy()

                    if (f"{ds}/{bb}", seed, split) == FIGURE_CELL:
                        figure_cells[method] = {"dz": dz, "fail_type": fail_type}

                    ca = cluster_alignment(dz, fail_type)
                    if ca is None:
                        cluster_rows.append({"method": method, "setting": f"{ds}/{bb}",
                                             "seed": seed, "split": split,
                                             "note": f"fewer than 2 failure types with "
                                                     f"n>={MIN_TYPE_N} members"})
                    else:
                        cluster_rows.append({"method": method, "setting": f"{ds}/{bb}",
                                             "seed": seed, "split": split, **ca, "note": ""})

                    # secondary reads: magnitude and margin, on the FULL failing population,
                    # never capped -- these are just means, not O(n^2).
                    dz_norm = np.linalg.norm(dz, axis=1)
                    magnitude_rows.append({
                        "method": method, "setting": f"{ds}/{bb}", "seed": seed, "split": split,
                        "n": n,
                        "mean_dz_norm": float(dz_norm.mean()), "median_dz_norm": float(np.median(dz_norm)),
                        "mean_dm": float(dm.mean()), "median_dm": float(np.median(dm)),
                        "frac_dm_positive": float((dm > 0).mean()),
                    })

                    # primary read: direction structure, capped for the O(n^2) pair test.
                    dz_c, y_c, yhat_c = dz, y, yhat
                    if n > MAX_N:
                        sel = np.random.default_rng(0).choice(n, MAX_N, replace=False)
                        dz_c, y_c, yhat_c = dz[sel], y[sel], yhat[sel]
                    for gname, lab in (
                        ("failure_type",
                         (pd.Series(y_c).astype(str) + "->" + pd.Series(yhat_c).astype(str)).to_numpy()),
                        ("true_class", y_c),
                    ):
                        st = pair_stats(dz_c, lab, np.random.default_rng(0))
                        if st is None:
                            direction_rows.append({"method": method, "setting": f"{ds}/{bb}",
                                                   "seed": seed, "split": split,
                                                   "grouping": gname, "n_vec": len(dz_c),
                                                   "note": "too few vectors or one group only"})
                            continue
                        direction_rows.append({"method": method, "setting": f"{ds}/{bb}",
                                               "seed": seed, "split": split,
                                               "grouping": gname, **st, "note": ""})

    if not direction_rows:
        raise SystemExit("no logit dumps found for any method")
    direction = pd.DataFrame(direction_rows)
    magnitude = pd.DataFrame(magnitude_rows)
    cluster = pd.DataFrame(cluster_rows)

    RM.write_section(
        "M1", "M1 — correction direction and margin improvement: NN-Patching vs PatchNAS vs "
        "DynaPatch-NoGate (raw)",
        f"""
Same test as P4 (`scripts/analysis_p4_patchvec.py`: within/between-failure-type cosine and L2
pair distance, plus a {NPERM}-permutation label-shuffle null), applied to `Delta z(x) =
patched_logits(x) - base_logits(x)` instead of the generator's own `patch_vec`, and computed for
three patch-based methods side by side. Rows are restricted to inputs the deployed model got
WRONG (a correction for a correctly-classified input has no failure type). `patched_logits` is
each method's raw, un-routed patch output -- what the patch mechanism itself produces for that
failure, independent of any deployment-time gate.

**Direction table** (primary): one row per (method, setting, seed, split, grouping).

| column | meaning |
|---|---|
| `grouping` | `failure_type` = (true -> base prediction); `true_class` = true label only |
| `n_vec` / `n_groups` | correction vectors compared, and distinct groups among them (capped at {MAX_N}) |
| `cos_within` / `cos_between` / `cos_gap` | mean cosine distance of `Delta z(x)` within vs between groups, and between-minus-within |
| `l2_within` / `l2_between` / `l2_gap` | the same for L2, divided by the cell's median pair distance |
| `*_gap_null_mean` / `*_gap_null_sd` / `*_z` / `*_p_perm` | the same gap over {NPERM} within-cell label permutations, and the observed gap against that null |
| `n_pairs_within` / `n_pairs_between` | pair counts; the between set is always larger |

A method whose corrections collapse to one shared direction per failure type has `cos_within`
near zero. `cos_within` clearly above zero, even where `cos_gap` is large and survives the null,
means the correction still varies input by input inside one failure type -- a stronger claim than
"this method separates failure types" and the one the 2026-09-04 hypothesis is actually about.

**Magnitude/margin table** (secondary): one row per (method, setting, seed, split), NOT capped.

| column | meaning |
|---|---|
| `mean_dz_norm` / `median_dz_norm` | correction magnitude \\|\\|Delta z(x)\\|\\| |
| `mean_dm` / `median_dm` | true-vs-wrong margin improvement Delta m(x) = Delta z(x) . (e_y - e_yhat) |
| `frac_dm_positive` | fraction of failures where the correction points toward the true class over the base model's wrong prediction, whether or not it flips the argmax |

Reported separately from direction on purpose: a method can move more (`mean_dz_norm`) without
that motion being better aimed (`cos_within`/`mean_dm`), and the reverse.

**Caveat that must travel with the magnitude/margin table, not with the direction table.**
DynaPatch's `patched_logits` come from the SAME frozen classifier head as `base_logits` (the
patch is injected upstream in feature space, then read out through the deployed model's own,
unmodified final layer). NN-Patching's and PatchNAS's `patched_logits` come from a FRESHLY
TRAINED head (`scripts/baseline_prior_patches.py:head()`, fit with AdamW from a random init) that
has no reason to share the frozen head's logit scale. So `mean_dz_norm` / `mean_dm` /
`median_dm` are **not comparable in absolute units across methods** -- only within one method
(e.g. `seen` vs `held`, or against that method's own repaired/regressed outcomes). The direction
table is far less exposed to this: cosine similarity normalises each `Delta z(x)` to a unit
vector before comparing, so it reads relative angle, not absolute scale, and the class axes it
compares are still the same n_classes label indices for every method. `frac_dm_positive` (does
the correction point toward the true class over the wrong one AT ALL, sign only) is the one
magnitude-table column that stays meaningful across methods despite the scale mismatch.

**Cluster-alignment table** (a more legible restatement of the direction claim): one row per
(method, setting, seed, split). Spherical k-means -- k-means on unit-normalised `Delta z(x)`,
which is exactly cosine-based clustering since `||a-b||^2 = 2 - 2cos(a,b)` for unit vectors -- run
with `k` = the number of observed failure types (types with fewer than {MIN_TYPE_N} members
dropped first, same floor as `note/AUDIT.md`'s "estimable" types), scored against the TRUE
failure-type labels the clustering was never shown.

| column | meaning |
|---|---|
| `n` / `n_types` | correction vectors clustered, and the number of clusters (= number of failure types kept) |
| `purity` | fraction of vectors whose cluster's majority failure type matches their own true failure type; 1/`n_types` is chance level for balanced groups |
| `ari` | adjusted Rand index between the cluster assignment and the true failure-type labels; 0 = chance, 1 = perfect |
| `nmi` | normalised mutual information between the same two labellings; 0 = chance, 1 = perfect |

This asks a stronger question than the pairwise cosine gap above: does clustering on direction
ALONE, with no label information beyond how many groups to form, recover which failure type each
input belongs to. See `note/figures/m1_correction_direction/` for the same population plotted
directly (2-D PCA of the unit vectors, coloured by failure type) rather than scored.
""",
        [("direction", direction.sort_values(["grouping", "split", "method", "setting", "seed"])),
         ("magnitude_margin", magnitude.sort_values(["split", "method", "setting", "seed"])),
         ("cluster_alignment", cluster.sort_values(["split", "method", "setting", "seed"]))],
    )

    out = ROOT / "outputs" / "rq2" / "m1_correction_direction"
    out.mkdir(parents=True, exist_ok=True)
    direction.to_csv(out / "direction_per_cell.csv", index=False)
    magnitude.to_csv(out / "magnitude_margin_per_cell.csv", index=False)
    cluster.to_csv(out / "cluster_alignment_per_cell.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/direction_per_cell.csv  ({len(direction)} rows)")
    print(f"[written] {out.relative_to(ROOT)}/magnitude_margin_per_cell.csv  ({len(magnitude)} rows)")
    print(f"[written] {out.relative_to(ROOT)}/cluster_alignment_per_cell.csv  ({len(cluster)} rows)")

    figure_dir = ROOT / "note" / "figures" / "m1_correction_direction"
    if len(figure_cells) == len(METHODS):
        plot_direction_pca(figure_cells, figure_dir)
        print(f"[written] {figure_dir.relative_to(ROOT)}/direction_pca_*.{{png,pdf}}")
    else:
        missing = set(METHODS) - set(figure_cells)
        print(f"[skip figure] FIGURE_CELL {FIGURE_CELL} missing for: {sorted(missing)}")


if __name__ == "__main__":
    main()
