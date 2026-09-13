#!/usr/bin/env python3
"""The paper's "Gate Classification Performance" table, for the gate the paper ACTUALLY ships.

Why this script replaces the RQ2 numbers
----------------------------------------
Two different classifiers were both being called "the gate":

  gate_performance.py   pre_mag_pPmax   6 features   BINARY (class-balanced logistic)
  gate_zoo.py           L4 pre+post    12 features   3-CLASS over gain in {-1,0,+1}

Only the second one produced a single RR / Reg / CReg number in the paper. Reporting the
accuracy / precision / recall / F1 of the first one next to the RR / Reg of the second is a
category error: they are different objects with different inputs and different heads. This
script reports the classification behaviour of the SHIPPED object, so section "Evaluation
Metrics" and section "Results" describe one gate.

What is being classified
------------------------
Population = the rows where the patch proposes a change (`flip = 1`) on the reporting set,
i.e. S_clean^test + S_held. A row the patch leaves alone is not a decision the gate makes.

Label, matching the paper's u:
    u = 1   beneficial   the base was wrong and the patch lands on the right class  (gain +1)
    u = 0   otherwise    harmful (a clean input broken, gain -1) or useless (still wrong, 0)

Decision: apply the patch iff score(x) > theta, with score = P(u beneficial) - P(u harmful)
from the same 3-class model gate_zoo fits.

Threshold convention -- the SAME object as the RR / Reg tables
--------------------------------------------------------------
The deployer names r, the fraction of the ungated regression that must be removed. theta is
the veto quantile that attains r on the curve. There is no second, incompatible operating
point living only in this table, and no calibration step: the shipped gate is used as a
RANKER, so the probability scale never enters any reported number.

Protocol: leave-one-backbone-out, identical to gate_zoo. Nothing scores a row that took part
in its own fit.

Zero GPU. Usage:
  .venv/bin/python scripts/gate_report.py
  .venv/bin/python scripts/gate_report.py --features "L2 pre-strong"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_zoo as Z  # noqa: E402
import names as N  # noqa: E402
import probe_gonogo_pre_vs_prepost as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "outputs" / "gate_report.csv"
RS = (0.60, 0.80, 0.90)


def confusion(pred: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """Binary report with u = 1 meaning 'this patch application is beneficial'."""
    tp = int((pred & (u == 1)).sum())
    fp = int((pred & (u == 0)).sum())
    fn = int((~pred & (u == 1)).sum())
    tn = int((~pred & (u == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": tp + fp + fn + tn,
        "prevalence": (tp + fn) / max(tp + fp + fn + tn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": prec,
        "recall": rec,
        "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def theta_for_r(c: dict, h: dict, sc: np.ndarray, sh: np.ndarray, r: float) -> float | None:
    """Smallest veto that removes at least fraction r of the ungated regression.

    Identical construction to gate_zoo.operating_points: the veto pool is the flipped rows of
    both populations, and vetoing a row reverts it to the base prediction.
    """
    reg0 = c["y"].sum() / c["n"]
    if reg0 <= 0:
        return None
    pool = np.concatenate([sc[c["flip"]], sh[h["flip"]]])
    if not len(pool):
        return None
    for q in Z.QGRID:
        t = -np.inf if q <= 0 else float(np.quantile(pool, q))
        keep = ~c["flip"] | (sc > t)
        if (reg0 - c["y"][keep].sum() / c["n"]) / reg0 >= r - 1e-9:
            return t
    return None


def rows_for(cells: dict, feats: list[str]) -> list[dict]:
    """Fit leave-one-backbone-out, then report the confusion at each target r."""
    out: list[dict] = []
    for held_bb in Z.BBS:
        tr = [k for k in cells if k[2] != held_bb]
        X, g = [], []
        for k in tr:
            for side in ("clean", "held"):
                d = cells[k][side]
                X.append(G.mat(d["f"], feats, d["flip"]))
                g.append(d["gain"][d["flip"]])
        mdl = G.fit_gain(np.vstack(X), np.concatenate(g))
        if mdl is None:
            continue
        for k in (k for k in cells if k[2] == held_bb):
            seed, ds, bb = k
            c, h = cells[k]["clean"], cells[k]["held"]
            sc = G.score(mdl, G.mat(c["f"], feats))
            sh = G.score(mdl, G.mat(h["f"], feats))
            # the decision population: only rows the patch wants to change
            u = np.concatenate([c["gain"][c["flip"]] == 1, h["gain"][h["flip"]] == 1]).astype(int)
            s = np.concatenate([sc[c["flip"]], sh[h["flip"]]])
            for r in RS:
                t = theta_for_r(c, h, sc, sh, r)
                if t is None:
                    continue
                out.append({"setting": f"{ds}/{bb}", "seed": seed, "r": r, "theta": t,
                            **confusion(s > t, u)})
    return out


def pooled(rows: list[dict], setting: str, r: float) -> dict[str, float] | None:
    """Pool the three bug-split seeds by summing the confusion, not by averaging rates."""
    sel = [x for x in rows if x["setting"] == setting and abs(x["r"] - r) < 1e-9]
    if not sel:
        return None
    tp, fp, fn, tn = (sum(x[k] for x in sel) for k in ("TP", "FP", "FN", "TN"))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n": tp + fp + fn + tn, "seeds": len(sel),
            "prevalence": (tp + fn) / max(tp + fp + fn + tn, 1),
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "precision": prec, "recall": rec,
            "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=N.SHIPPED_GATE,
                    help="gate_zoo LEARNED key; default is the shipped gate")
    ap.add_argument("--out", default=None,
                    help="csv path; defaults to outputs/gate_report.csv for the shipped gate")
    a = ap.parse_args()
    key = N.assert_gate_admissible(a.features)
    feats = Z.LEARNED[key]

    cells = Z.attach_criticality(Z.attach_idx(Z.add_derived(G.build_cells())))
    rows = rows_for(cells, feats)

    print(f"gate: {N.gate_display(key)}   {len(feats)} features, 3-class over gain in "
          f"{{-1,0,+1}}, leave-one-backbone-out")
    print(f"population: rows the patch proposes to change (flip=1) on "
          f"S_clean^test + S_held; u=1 means the application is beneficial")
    print(f"cells: {len(cells)}   reported rows: {len(rows)}\n")

    settings = [f"{d}/{b}" for d, b, _ in N.SETTING_ORDER]
    for r in RS:
        print(f"=== r = {r:.2f} of the ungated regression removed ===")
        hdr = (f"{'setting':<10}{'n':>7}{'prev':>7}{'acc':>8}{'prec':>8}"
               f"{'rec':>8}{'F1':>8}{'TP':>7}{'FP':>6}{'FN':>6}{'TN':>6}")
        print(hdr)
        print("-" * len(hdr))
        agg = []
        for stg in settings:
            p = pooled(rows, stg, r)
            lab = N.SETTING_LABEL[stg]
            if p is None:
                print(f"{lab:<10}{'-- not estimable (ungated Reg is 0)':>50}")
                continue
            agg.append(p)
            print(f"{lab:<10}{p['n']:>7}{p['prevalence']:>7.2f}{p['accuracy']:>8.3f}"
                  f"{p['precision']:>8.3f}{p['recall']:>8.3f}{p['f1']:>8.3f}"
                  f"{p['TP']:>7}{p['FP']:>6}{p['FN']:>6}{p['TN']:>6}")
        if agg:
            print("-" * len(hdr))
            med = lambda k: float(np.median([x[k] for x in agg]))  # noqa: E731
            print(f"{'MEDIAN':<10}{'':>7}{med('prevalence'):>7.2f}{med('accuracy'):>8.3f}"
                  f"{med('precision'):>8.3f}{med('recall'):>8.3f}{med('f1'):>8.3f}")
        print()

    out_csv = Path(a.out) if a.out else OUT_CSV
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} per-seed rows)")


if __name__ == "__main__":
    main()
