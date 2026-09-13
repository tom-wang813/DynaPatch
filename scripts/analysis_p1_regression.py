#!/usr/bin/env python3
"""P1 -- where repair lands and where regression comes from, per class.

The advisor's hypothetical was: "other methods fail because they repair only one or a few types
of classes, introducing large regression on other classes; our method could avoid it". This
tests that sentence directly, on the unit it is actually about -- the class -- rather than on
the aggregate rates the tables already report.

Four quantities, all per (method, setting, seed), then reported per setting:

  RR_c        repair rate among held-out failures whose TRUE class is c
  Reg_c       regression rate among clean_test inputs the deployed model got right, true class c
  Coverage(t) fraction of classes with RR_c >= t. Reaching many classes at a modest rate is a
              different capability from acing a few, and the aggregate RR cannot tell them apart.
  Conc@20     share of all repairs (or regressions) contributed by the top 20% of classes.
              1.0 means every repair came from one fifth of the label space.

Why the class and not the ordered pair (y -> y_hat): on the repair side the ordered pair leaves
1-6 estimable types per LISA cell (note/AUDIT.md). The class is estimable in 12/12.

Rows come from outputs/sample_frame.csv.gz, so the reported configuration of every method is
whatever the paper table reports -- there is no second selection here.

    .venv/bin/python scripts/analysis_p1_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import names as N  # noqa: E402

TAUS = (0.3, 0.5, 0.7)
OUT = ROOT / "outputs" / "p1_regression"


def conc(counts: np.ndarray, frac: float = 0.2) -> float:
    """Share of the total contributed by the top `frac` of classes (by count).

    KEPT FOR REFERENCE ONLY -- do not report it. It is collinear with the event COUNT: a method
    with 5 regressions over 43 classes scores 1.00 by arithmetic, not by concentrating. Measured
    on this data, every method with few regressions scores ~1.0 and the ranking it produces is
    the ranking of 1/n. Use `spread_vs_null` instead.
    """
    if counts.sum() == 0:
        return float("nan")
    k = max(1, int(round(frac * len(counts))))
    return float(np.sort(counts)[::-1][:k].sum() / counts.sum())


def spread_vs_null(event_classes: np.ndarray, pool_classes: np.ndarray,
                   B: int = 400, rng: int = 0) -> tuple[float, float]:
    """How concentrated are these events, at their own count?

    Observed: how many distinct classes the events landed in.
    Null: draw the SAME NUMBER of events uniformly from the population that was at risk
    (`pool_classes`, which carries the real class imbalance) and count distinct classes.

    Returns (ratio, z). ratio = observed / expected. Below 1 means the events pile into fewer
    classes than chance would give at that count -- which is the claim "this method only fixes a
    few kinds of error" actually makes. A raw top-k share cannot say this because it moves with
    the count; this does not.
    """
    n = len(event_classes)
    if n == 0 or len(pool_classes) == 0:
        return float("nan"), float("nan")
    obs = len(np.unique(event_classes))
    g = np.random.default_rng(rng)
    draws = np.array([len(np.unique(g.choice(pool_classes, size=n, replace=False)))
                      for _ in range(B)]) if n <= len(pool_classes) else np.array([obs])
    mu, sd = draws.mean(), draws.std()
    return float(obs / mu) if mu else float("nan"), float((obs - mu) / sd) if sd else 0.0


def per_class_rows(d: pd.DataFrame, method: str, setting: str, seed: int) -> list[dict]:
    """Per-class event counts backing `repair_spread`/`reg_spread` -- for the concentration figure.

    One row per (method, setting, seed, kind, true_class). `kind` is 'repair' (pool = held-out
    failures, event = repaired) or 'regression' (pool = clean_test correct, event = regressed).
    Kept separate from `per_cell`'s aggregated stats so per_cell.csv does not grow one row per
    class; this is the only place the raw per-class counts are persisted.
    """
    held = d[d.split == "held"]
    clean = d[d.split == "clean"]
    rows: list[dict] = []

    h = held[~held.base_correct]
    if len(h):
        g = h.groupby("true_class").repaired.agg(["sum", "size"])
        for cls, r in g.iterrows():
            rows.append({"method": method, "setting": setting, "seed": seed, "kind": "repair",
                         "true_class": int(cls), "n_events": int(r["sum"]), "n_pool": int(r["size"])})

    c = clean[clean.base_correct]
    if len(c):
        g = c.groupby("true_class").regressed.agg(["sum", "size"])
        for cls, r in g.iterrows():
            rows.append({"method": method, "setting": setting, "seed": seed, "kind": "regression",
                         "true_class": int(cls), "n_events": int(r["sum"]), "n_pool": int(r["size"])})
    return rows


def per_cell(d: pd.DataFrame) -> dict:
    held = d[d.split == "held"]
    clean = d[d.split == "clean"]
    out: dict = {}

    # --- repair, per true class among the deployed model's failures
    h = held[~held.base_correct]
    if len(h):
        g = h.groupby("true_class").repaired.agg(["sum", "size"])
        rr = g["sum"] / g["size"]
        g5 = g[g["size"] >= 5]
        out |= {"n_classes_failed": len(g),
                "RR_macro": float(rr.mean()),
                "RR_median_class": float(rr.median()),
                "RR_p10_class": float(rr.quantile(0.10)),
                "RR_iqr_class": float(rr.quantile(.75) - rr.quantile(.25)),
                "repair_conc20": conc(g["sum"].to_numpy())}
        rp = h[h.repaired]
        out["repair_spread"], out["repair_spread_z"] = spread_vs_null(
            rp.true_class.to_numpy(), h.true_class.to_numpy())
        out["n_repaired"] = int(len(rp))
        for t in TAUS:
            # coverage on classes with enough failures to estimate a rate at all
            r5 = (g5["sum"] / g5["size"]) if len(g5) else pd.Series(dtype=float)
            out[f"cov{t}"] = float((r5 >= t).mean()) if len(r5) else float("nan")
        out["n_classes_est"] = int(len(g5))

    # --- regression, per true class among clean inputs the deployed model got right
    c = clean[clean.base_correct]
    if len(c):
        g = c.groupby("true_class").regressed.agg(["sum", "size"])
        rg = g["sum"] / g["size"]
        out |= {"Reg_overall": float(c.regressed.mean()),
                "Reg_macro": float(rg.mean()),
                "Reg_max_class": float(rg.max()),
                "n_classes_hit": int((g["sum"] > 0).sum()),
                "frac_classes_hit": float((g["sum"] > 0).mean()),
                "reg_conc20": conc(g["sum"].to_numpy())}
        rg_ = c[c.regressed]
        out["reg_spread"], out["reg_spread_z"] = spread_vs_null(
            rg_.true_class.to_numpy(), c.true_class.to_numpy())
        out["n_regressed"] = int(len(rg_))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "outputs/sample_frame.csv.gz")
    recs = []
    class_rows = []
    for (m, stg, sd), d in df.groupby(["method", "setting", "seed"]):
        r = per_cell(d)
        if r:
            recs.append({"method": m, "setting": stg, "seed": sd, **r})
        class_rows.extend(per_class_rows(d, m, stg, sd))
    cell = pd.DataFrame(recs)
    cell.to_csv(OUT / "per_cell.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "per_class.csv", index=False)

    # per SETTING (median over seeds), never a grand mean -- means hid a pathological cell and
    # three sign reversals in this project before.
    st = cell.groupby(["method", "setting"]).median(numeric_only=True).reset_index()
    st.to_csv(OUT / "per_setting.csv", index=False)

    import raw_md as RM
    keep = ["method", "setting", "seed", "n_classes_failed", "n_classes_est", "n_repaired",
            "RR_macro", "RR_median_class", "RR_p10_class", "RR_iqr_class",
            "cov0.3", "cov0.5", "cov0.7", "repair_spread", "repair_spread_z",
            "n_regressed", "Reg_overall", "Reg_macro", "Reg_max_class",
            "n_classes_hit", "frac_classes_hit", "reg_spread", "reg_spread_z"]
    raw = cell[[c for c in keep if c in cell]].sort_values(["method", "setting", "seed"])
    RM.write_section("P1", "P1 — per-class repair and regression (raw, per cell)", """
Unit of every rate is the **true class**. One row per (method, setting, seed).

| column | meaning |
|---|---|
| `n_classes_failed` | distinct true classes among the deployed model's held-out failures |
| `n_classes_est` | of those, how many have >=5 failures (a rate is estimable) |
| `n_repaired` | held-out failures this method fixed |
| `RR_macro` / `RR_median_class` / `RR_p10_class` / `RR_iqr_class` | per-class repair rate: mean, median, 10th pct, IQR |
| `cov0.3/0.5/0.7` | fraction of estimable classes with per-class RR >= tau |
| `repair_spread` | distinct classes the repairs landed in, divided by the expected number if the SAME NUMBER of repairs were drawn at random from the failure pool. <1 = concentrated. Count-robust; a raw top-k share is not. |
| `repair_spread_z` | the same contrast as a z-score against the null draws |
| `n_regressed` | clean_test inputs the deployed model got right and this method broke |
| `Reg_overall` / `Reg_macro` / `Reg_max_class` | regression rate overall, averaged over classes, and its worst class |
| `n_classes_hit` / `frac_classes_hit` | classes receiving at least one regression |
| `reg_spread` / `reg_spread_z` | the same count-matched null, for regressions against the clean pool |

Populations: repair columns are `bug_eval` (held), regression columns are `clean_test` (clean).
They are disjoint and are never pooled. DistRep covers 9/12 settings (see note/AUDIT.md).
""", [("", raw)])

    ours = N.display("DP") if hasattr(N, "display") else "DynaPatch (gated)"
    ref = N.gate_label_natural() if hasattr(N, "gate_label_natural") else "DynaPatch (gated)"
    cols = ["RR_median_class", "cov0.3", "cov0.5", "cov0.7", "repair_spread", "n_repaired",
            "Reg_overall", "Reg_max_class", "reg_spread", "n_regressed"]

    lines = ["# P1 — per-class repair and regression", "",
             "Unit: the TRUE CLASS. Each cell is the median over the three seeds; each table is "
             "read per setting, never as a grand mean.", "",
             "## Table 1 — every method, median over the 12 settings", "",
             "| method | RR med/class | cov@.3 | cov@.5 | cov@.7 | repair spread | n rep | "
             "Reg | max class Reg | reg spread | n reg |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    tab = st.groupby("method")[cols].median()
    order = tab.sort_values("RR_median_class", ascending=False).index
    for m in order:
        r = tab.loc[m]
        lines.append(f"| {m} | {r['RR_median_class']:.3f} | {r['cov0.3']:.2f} | "
                     f"{r['cov0.5']:.2f} | {r['cov0.7']:.2f} | {r['repair_spread']:.2f} | "
                     f"{r['n_repaired']:.0f} | {r['Reg_overall']:.4f} | "
                     f"{r['Reg_max_class']:.3f} | {r['reg_spread']:.2f} | "
                     f"{r['n_regressed']:.0f} |")

    # per-setting win counts against ours, the project's reporting rule
    lines += ["", f"## Table 2 — {ref} vs each baseline, counted over settings", "",
              "`W/T/L` counts SETTINGS, not samples. A tie is a difference below 0.005.", "",
              "| baseline | coverage@.5 W/T/L | Reg W/T/L (lower better) | "
              "repair spread W/T/L (higher = repairs reach more classes at matched count) |",
              "|---|---|---|---|"]
    o = st[st.method == ref].set_index("setting")
    for m in sorted(set(st.method) - {ref}):
        b = st[st.method == m].set_index("setting")
        common = o.index.intersection(b.index)
        if len(common) < 6:
            lines.append(f"| {m} | only {len(common)}/12 settings — not counted | | |")
            continue
        cells = []
        for col, better_low in (("cov0.5", False), ("Reg_overall", True),
                                ("repair_spread", False)):
            x, y = o.loc[common, col], b.loc[common, col]
            d = (y - x) if better_low else (x - y)
            w = int((d > 0.005).sum()); l = int((d < -0.005).sum())
            cells.append(f"{w}/{len(common)-w-l}/{l}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Table 3 — per setting, coverage@.5 (fraction of failing classes "
              "repaired at 50%+)", "",
              "| setting | " + " | ".join(order) + " |",
              "|---" * (len(order) + 1) + "|"]
    for stg in sorted(st.setting.unique()):
        row = st[st.setting == stg].set_index("method")
        vals = [f"{row.loc[m,'cov0.5']:.2f}" if m in row.index else "—" for m in order]
        lines.append(f"| {stg} | " + " | ".join(vals) + " |")

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\n[written] {OUT.relative_to(ROOT)}/REPORT.md, per_cell.csv, per_setting.csv, "
          "per_class.csv")


if __name__ == "__main__":
    main()
