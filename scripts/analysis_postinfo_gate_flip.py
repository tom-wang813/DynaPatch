#!/usr/bin/env python3
"""Why does adding post-repair evidence make DynaPatch's OWN gate's natural-threshold
(theta=0) decision slightly WORSE (RQ3.6: RR_held delta -0.0208), when the SAME evidence
addition makes the transplanted gate on NN-Patching/PatchNAS's patches strongly BETTER
(+0.13-0.14), and makes the classifier itself more discriminating either way (RQ3.8's
cross-validated macro-F1 improves for all three)?

Per-sample decisions are already on disk from two separate `gate_protocol_b.py --per-sample`
runs (2026-09-05): `outputs/gate_persample_pre/` (features = "L2 pre-strong", pre-only) and
`outputs/gate_persample/` (features = shipped "L4 pre+post"). Both carry `apply_natural`
(theta=0: commit iff score > 0) per (setting, seed, side, dataset_index). Joining them shows
exactly which rows FLIP decision when post-info is added, and whether that flip was a good
idea (using `gain`, already computed from the SAME ground truth both runs share):

    held rows   (repair candidates): gain=+1 means committing was correct (repairs)
    clean rows  (regression candidates): gain=-1 means committing was WRONG (causes damage)

A flip from veto (pre) to commit (pre+post) on a `gain=+1` held row is a genuine new repair.
A flip from commit (pre) to veto (pre+post) on a `gain=+1` held row is a repair post-info
newly TALKED THE GATE OUT OF -- exactly the mechanism that would explain a negative RR delta.

    .venv/bin/python scripts/analysis_postinfo_gate_flip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402


def load(dirp: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(dirp.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    pre = load(ROOT / "outputs/gate_persample_pre")
    both = load(ROOT / "outputs/gate_persample")
    key = ["setting", "seed", "side", "dataset_index"]
    m = pre[key + ["gate_score", "flip", "gain", "apply_natural"]].merge(
        both[key + ["gate_score", "flip", "gain", "apply_natural"]],
        on=key, suffixes=("_pre", "_both"))
    assert m.gain_pre.equals(m.gain_both), "ground truth must agree between the two runs"
    assert m.flip_pre.equals(m.flip_both), "which rows the patch proposes to change must agree"
    m = m[m.flip_pre].copy()  # only rows the patch actually proposes changing -- the gate's
    # decision problem is a no-op everywhere else, regardless of which feature set is used
    m["transition"] = pd.Categorical(
        m.apply_natural_pre.map({True: "commit", False: "veto"}) + " -> " +
        m.apply_natural_both.map({True: "commit", False: "veto"}))
    m["gain_label"] = m.gain_pre.map({1: "beneficial (+1)", 0: "no-effect (0)", -1: "harmful (-1)"})

    cell = m.groupby(["setting", "side", "transition", "gain_label"], as_index=False).size()
    setting_summary = cell.groupby(["side", "transition", "gain_label"], as_index=False)["size"].sum()
    overall = setting_summary.groupby(["side", "transition", "gain_label"], as_index=False)["size"].sum()

    # The headline number: net repairs GAINED vs LOST by adding post-info, on the held
    # population specifically (this is what RQ3.6's RR_held delta is made of).
    held = overall[overall.side == "held"].set_index(["transition", "gain_label"])["size"]

    def g(transition: str, gain_label: str) -> int:
        return int(held.get((transition, gain_label), 0))

    newly_committed_beneficial = g("veto -> commit", "beneficial (+1)")
    newly_vetoed_beneficial = g("commit -> veto", "beneficial (+1)")
    newly_committed_noneffect = g("veto -> commit", "no-effect (0)")
    newly_vetoed_noneffect = g("commit -> veto", "no-effect (0)")
    n_flip_pre = int(m[m.side == "held"].shape[0])

    clean = overall[overall.side == "clean"].set_index(["transition", "gain_label"])["size"]

    def gc(transition: str) -> int:
        return int(clean.get((transition, "harmful (-1)"), 0))

    newly_vetoed_harmful = gc("commit -> veto")     # newly PREVENTED regressions
    newly_committed_harmful = gc("veto -> commit")  # newly INTRODUCED regressions

    RM.write_section(
        "PostInfoGateFlip",
        "Why DynaPatch's own gate's post-info RR delta is negative -- per-sample decision "
        "flips, pre-only vs pre+post, at the natural threshold (raw)",
        f"""
Every row here is a `held` (repair candidate) or `clean` (regression candidate) input the
patch actually proposes to change (`flip=True` in both runs -- identical since both share the
same generator output, only the GATE's feature set differs). `transition` is the gate's
theta=0 decision under pre-only -> under pre+post; `gain_label` is the ground truth (shared by
both runs, asserted equal).

**On the held population** (n={n_flip_pre} proposed changes across 12 settings x 3 seeds):
adding post-info newly COMMITS {newly_committed_beneficial} genuinely beneficial repairs, but
newly VETOES {newly_vetoed_beneficial} genuinely beneficial repairs it would have committed
under pre-only -- a net of {newly_committed_beneficial - newly_vetoed_beneficial} on the
repairs that matter. On no-effect rows (gain=0, flipping changes nothing for RR_held either
way but is diagnostic of gate calibration): {newly_committed_noneffect} newly committed,
{newly_vetoed_noneffect} newly vetoed.

If `{newly_vetoed_beneficial}` (repairs post-info talks the gate OUT of) exceeds
`{newly_committed_beneficial}` (repairs post-info newly persuades the gate INTO), that is
the entire mechanism behind RQ3.6's negative RR_held delta for DynaPatch's own gate -- not a
paradox with RQ3.8's macro-F1 improvement, since F1 is computed over ALL three gain classes
(including harmful/no-effect, which post-info may be getting more right) while RR_held only
credits the beneficial-committed cell.

**On the clean population** (regression candidates): post-info newly VETOES (prevents)
{newly_vetoed_harmful} genuinely harmful commits pre-only would have made, and newly COMMITS
(introduces) {newly_committed_harmful} new harmful ones pre-only correctly avoided -- net
{newly_vetoed_harmful - newly_committed_harmful} regressions newly prevented. This is the
other half of the mechanism: **post-info shifts DynaPatch's own gate toward more caution
overall.** That extra caution is a clear net win on the safety axis (Reg: {newly_vetoed_harmful}
prevented vs {newly_committed_harmful} introduced) but a net loss on the repair axis (RR_held:
{newly_committed_beneficial} gained vs {newly_vetoed_beneficial} lost) -- a real
precision/recall-shaped trade-off, not a contradiction, and it explains why RQ3.6 shows
DynaPatch's own gate's Reg delta ALSO improving (-0.0011) alongside its RR_held delta going
negative: both are the same added-caution mechanism, read from opposite sides.
""",
        [("by_setting", cell.sort_values(["side", "setting", "transition", "gain_label"])),
         ("by_side_overall", overall.sort_values(["side", "transition", "gain_label"]))],
    )

    out = ROOT / "outputs" / "postinfo_gate_flip"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_setting.csv", index=False)
    overall.to_csv(out / "overall.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/*.csv")
    print("\nHeld population (repair candidates), transition x gain_label:")
    print(overall[overall.side == "held"].to_string(index=False))
    print(f"\nHeld: newly committed beneficial {newly_committed_beneficial}  "
         f"newly vetoed beneficial {newly_vetoed_beneficial}  "
         f"net {newly_committed_beneficial - newly_vetoed_beneficial}")
    print(f"Clean: newly vetoed (prevented) harmful {newly_vetoed_harmful}  "
         f"newly committed (introduced) harmful {newly_committed_harmful}  "
         f"net {newly_vetoed_harmful - newly_committed_harmful}")


if __name__ == "__main__":
    main()
