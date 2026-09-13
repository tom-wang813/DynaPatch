#!/usr/bin/env python3
"""RQ4 (controllability): does sweeping the gate's veto threshold trace a well-behaved
repair-regression frontier, not just a single (NoGate, Gated) point pair?

RQ4_DATA.md currently only reports RQ4.1-4.5's single theta=0 natural-threshold point versus
NoGate -- it never shows that the threshold is actually a CONTINUOUS, well-behaved knob. This
script reads the full q-sweep (`outputs/gate_curves_protocolC_ep40ns.csv`, 59 veto-quantile
values per setting, already computed -- no new experiment) and checks the two things that make
"controllable" a real claim rather than an assertion about two points:

  1. Monotonicity: as q (veto strictness) increases, does RR_held only ever stay flat or
     decrease, never increase? A frontier with an increasing RR_held step would mean vetoing
     MORE proposals sometimes repairs MORE failures -- not controllable, a modeling artefact.
  2. The figure: one repair-regression curve per setting (Reg on x, RR_held on y, traced across
     q), with the two points RQ4_DATA.md already reports (Ungated at q=0, the deployed theta=0
     point) marked on their own curve -- shows the trade-off is smooth and the deployed point
     sits ON the frontier, not an isolated pair invented separately from the sweep.

**Scope note, explicitly NOT done here (2026-09-06, user confirmed)**: no cross-method
comparison (NN-Patching/PatchNAS's transplanted-gate curve, via
`scripts/analysis_gate_transplant_curve.py`) -- DynaPatch's own frontier only, this pass.

**Framing caveat that must travel with any read of this curve**: this is a DESCRIPTIVE
characterization of the already-computed frontier, not a threshold-selection result. The
project dropped r-targeted threshold selection project-wide (2026-09-04) because choosing a
theta by searching the same population RR/Reg are then reported on is an eval-set selection bug
(see note/RESEARCH_STATE.md). Reading monotonicity or curve shape off the full 59-point sweep
does not select anything -- no point on this curve is being proposed as a new deployment
threshold -- but any per-setting numeric summary derived from it (e.g. "RR retained at X% Reg
removed") would reopen exactly that risk if quoted as a result rather than a description, so this
script deliberately reports monotonicity only, not any single interpolated operating value.

    .venv/bin/python scripts/analysis_rq4_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import names as N                    # noqa: E402
import raw_md as RM                  # noqa: E402

CURVE = ROOT / "outputs/gate_curves_protocolC_ep40ns.csv"
NATURAL = ROOT / N.GATE_NATURAL_POINT
EPS = 1e-9


def main() -> None:
    curves = pd.read_csv(CURVE)
    curves["setting"] = curves.dataset + "/" + curves.backbone
    nat = pd.read_csv(NATURAL)

    rows = []
    for ds, bb, lab in N.SETTING_ORDER:
        stg = f"{ds}/{bb}"
        c = curves[curves.setting == stg].sort_values("q").reset_index(drop=True)
        if c.empty:
            rows.append({"setting": stg, "label": lab, "n_q": 0, "note": "missing"})
            continue
        rr = c.RR_held.to_numpy()
        reg = c.Reg.to_numpy()
        d_rr = np.diff(rr)
        d_reg = np.diff(reg)
        # A "violation" is RR_held INCREASING as q increases (should only ever fall or hold).
        rr_violations = int((d_rr > EPS).sum())
        # Reg should also never increase as q increases (more veto -> less regression, by
        # construction of q as a veto threshold) -- sanity check, not the main claim.
        reg_violations = int((d_reg > EPS).sum())
        rows.append({
            "setting": stg, "label": lab, "n_q": len(c),
            "rr_at_q0": float(rr[0]), "rr_at_q1": float(rr[-1]),
            "reg_at_q0": float(reg[0]), "reg_at_q1": float(reg[-1]),
            "n_rr_violations": rr_violations, "max_rr_violation": float(d_rr.max() if len(d_rr) else 0.0),
            "n_reg_violations": reg_violations,
            "note": "",
        })
    summary = pd.DataFrame(rows)
    out = ROOT / "outputs" / "rq4_frontier"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "monotonicity_per_setting.csv", index=False)

    RM.write_section(
        "RQ4Frontier",
        "RQ4 -- repair-regression frontier monotonicity, all 59 veto-quantile steps per "
        "setting (raw)",
        f"""
As the gate's veto threshold `q` sweeps its full range (0 to 1, 59 steps,
`outputs/gate_curves_protocolC_ep40ns.csv`), `RR_held` and `Reg` should each be non-increasing
in `q` (more veto strictness should never repair more or regress more). `n_rr_violations`/
`n_reg_violations` count adjacent-step increases (should be 0 for a clean frontier);
`max_rr_violation` is the largest single-step RR_held increase observed, if any.
""",
        [("monotonicity", summary)],
    )

    n_clean = int((summary.n_rr_violations == 0).sum())
    print(summary.to_string(index=False))
    print(f"\n{n_clean}/{len(summary)} settings: RR_held is perfectly non-increasing in q "
         f"across all 59 steps.")
    print(f"[written] {out.relative_to(ROOT)}/monotonicity_per_setting.csv")

    plot_frontier(curves, nat, out.parent.parent / "note" / "figures" / "rq4_frontier")


def plot_frontier(curves: pd.DataFrame, nat: pd.DataFrame, figure_dir: Path) -> None:
    """12-panel small multiples: one repair-regression frontier per setting, Ungated (q=0) and
    the deployed theta=0 point marked on the same curve the monotonicity check above verifies."""
    import matplotlib.pyplot as plt

    nat_mean = nat.groupby("setting", as_index=False)[["RR_held", "Reg"]].mean()
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=False, sharey=False)
    for ax, (ds, bb, lab) in zip(axes.flat, N.SETTING_ORDER):
        stg = f"{ds}/{bb}"
        c = curves[curves.setting == stg].sort_values("q")
        ax.plot(c.Reg, c.RR_held, color="#2F6F73", lw=1.8, zorder=2)
        ax.scatter([c.Reg.iloc[0]], [c.RR_held.iloc[0]], marker="o", s=60,
                  facecolor="white", edgecolor="#888888", linewidth=1.5, zorder=3,
                  label="Ungated (q=0)")
        p = nat_mean[nat_mean.setting == stg]
        if len(p):
            ax.scatter([float(p.Reg.iloc[0])], [float(p.RR_held.iloc[0])], marker="*", s=140,
                      color="#C66A45", zorder=4, label="theta=0 (deployed)")
        ax.set_title(lab, fontsize=10, fontweight="bold")
        ax.set_xlabel("Reg", fontsize=8)
        ax.set_ylabel("RR_held", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Repair-regression frontier, all 59 veto-threshold steps per setting",
                fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"frontier_grid.{suffix}", dpi=200, bbox_inches="tight",
                   facecolor="white")
    plt.close(fig)
    print(f"[written] {figure_dir.relative_to(ROOT)}/frontier_grid.{{png,pdf}}")


if __name__ == "__main__":
    main()
