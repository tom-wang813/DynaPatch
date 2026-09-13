#!/usr/bin/env python3
"""Executable versions of the five checks that this project kept writing down and then skipping.

Why this file exists
--------------------
Each check below corresponds to a mistake that was made, diagnosed, written into a note, and
then made again. Writing "remember to check X" somewhere only works when someone happens to
reread it at the right moment; an assertion fires every time. The same reasoning produced
`names.py` after the two naming failures, and this file is its sibling.

The five, with the damage each one did:

1. `check_checkpoint`     `deploy_eval` drops a shape-mismatched patch tensor silently and still
                          writes a full metrics.json. Numbers from such a run describe the base
                          model, not the patch, and were read as if they described the patch.
2. `check_split_disjoint` A cross-method table evaluated a method on rows that were in that
                          method's own support set, i.e. re-tested it on its own evidence.
3. `check_seed_parity`    Means were compared across arms with unequal seed counts; four verdicts
                          in one day were wrong because of it.
4. `Ratio`                `RR_reported` was computed against three different denominators in three
                          places. Once unified to bug_train the anomaly it "showed" disappeared.
5. `check_baseline_defaults`
                          Rival hyperparameters were left at library defaults while ours were
                          tuned. Sweeping TopKSearch's `step_scale` moved RR_held from .035 to
                          .423 -- the reported gap was mostly our own configuration.

Use as a library:

    from guards import check_checkpoint, check_split_disjoint, Ratio, GuardError

Or audit a tree from the shell:

    python scripts/guards.py audit outputs/fewshot_safepatch_ep40_bank_v8_s101_kfull
    python scripts/guards.py audit outputs/ --glob '*_v8_*' --warn-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class GuardError(AssertionError):
    """A guard failed. The number this run produced must not be reported."""


class GuardWarning(UserWarning):
    """A guard is suspicious but not conclusive. Report the number with the caveat attached."""


def _fail(msg: str, warn_only: bool = False) -> str:
    if warn_only:
        print(f"WARN  {msg}", file=sys.stderr)
        return msg
    raise GuardError(msg)


# --------------------------------------------------------------------------------------
# 1. Checkpoint actually loaded
# --------------------------------------------------------------------------------------

def check_checkpoint(metrics_path: Path | str, warn_only: bool = False) -> dict:
    """Fail unless every key of the checkpoint was consumed.

    `deploy_eval` records the outcome of the load under `checkpoint_load_info` and then carries
    on regardless. A non-empty `skipped_keys` means a tensor in the checkpoint did not match the
    model and was dropped, so the "patched" arm of that run is partly or wholly the base model.
    `missing_keys` means the reverse: the model expected weights the checkpoint did not carry.

    Returns the load info on success so callers can log `checkpoint_epoch` alongside a number.
    """
    metrics_path = Path(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    info = metrics.get("checkpoint_load_info")
    if info is None:
        _fail(f"{metrics_path}: no checkpoint_load_info recorded; cannot verify the patch loaded",
              warn_only)
        return {}
    for field in ("skipped_keys", "missing_keys", "unexpected_keys"):
        keys = info.get(field) or []
        if keys:
            _fail(
                f"{metrics_path}: checkpoint_load_info.{field} is non-empty ({len(keys)} keys, "
                f"e.g. {keys[:3]}). The patch did not load cleanly; this run's numbers describe "
                f"an unknown mixture of base and patched model.",
                warn_only,
            )
    return info


def check_epochs(metrics_path: Path | str, expected: int | None = None,
                 warn_only: bool = False) -> int | None:
    """Report the epoch count that actually ran, never the one `config_resolved.yaml` claims.

    The baseline trees record 12 in their resolved config while having run 40. The only honest
    sources are the length of `history` and the `checkpoint_epoch` stamped at save time.
    """
    metrics_path = Path(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    history = metrics.get("history")
    real = len(history) if isinstance(history, list) else metrics.get("checkpoint_epoch")
    if real is None:
        _fail(f"{metrics_path}: neither history nor checkpoint_epoch present; epoch count unknown",
              warn_only)
        return None
    if expected is not None and real != expected:
        _fail(f"{metrics_path}: ran {real} epochs, caller expected {expected}. Do not compare a "
              f"converged arm against an under-trained one without saying so.", warn_only)
    return real


# --------------------------------------------------------------------------------------
# 2. Evaluation population disjoint from the method's own evidence
# --------------------------------------------------------------------------------------

SUPPORT_SPLITS = ("repair_support_seen", "bug_train", "bug_val")
HELD_SPLITS = ("repair_holdout_unseen",)


def check_split_disjoint(predictions_dir: Path | str, warn_only: bool = False) -> dict[str, int]:
    """Fail if the held-out rows overlap the rows the repair was fitted on.

    Identity is `dataset_index`, not row order: the same sample appears in several CSVs and the
    files are written independently. Returns `{"n_seen": ..., "n_held": ...}`, which callers are
    expected to print next to any held-out number -- a held-out rate over 40 samples and one over
    4000 are not the same claim, and pooling them across settings hid that for weeks.
    """
    predictions_dir = Path(predictions_dir)

    def ids(splits: Sequence[str]) -> set[int]:
        out: set[int] = set()
        for csv_path in sorted(predictions_dir.glob("*_predictions.csv")):
            with csv_path.open() as fh:
                for row in csv.DictReader(fh):
                    if row.get("split") in splits:
                        out.add(int(row["dataset_index"]))
        return out

    seen, held = ids(SUPPORT_SPLITS), ids(HELD_SPLITS)
    overlap = seen & held
    if overlap:
        _fail(
            f"{predictions_dir}: {len(overlap)} of {len(held)} held-out samples are also in the "
            f"method's own support set (e.g. {sorted(overlap)[:5]}). This is a re-test on its own "
            f"evidence, not a held-out result.",
            warn_only,
        )
    if not held:
        _fail(f"{predictions_dir}: no held-out rows found; a held-out number cannot be computed "
              f"from this tree", warn_only)
    return {"n_seen": len(seen), "n_held": len(held)}


# --------------------------------------------------------------------------------------
# 3. Seed parity
# --------------------------------------------------------------------------------------

def check_seed_parity(arms: dict[str, Iterable], warn_only: bool = False) -> list:
    """Fail unless every arm was run on the same seeds.

    Not merely the same *count*: two arms on three seeds each, but different three, are still
    paired incorrectly for a per-setting comparison. Returns the shared seed list, which is the
    only set a paired comparison may use.
    """
    got = {name: sorted(set(seeds)) for name, seeds in arms.items()}
    sizes = {name: len(s) for name, s in got.items()}
    if len(set(map(tuple, got.values()))) > 1:
        detail = "; ".join(f"{n}: {s} (n={sizes[n]})" for n, s in got.items())
        _fail(f"seed mismatch across arms -- {detail}. Comparing means over unequal seed sets "
              f"produced four wrong verdicts in a single day. Restrict to the shared seeds or "
              f"run the missing ones.", warn_only)
    return next(iter(got.values())) if got else []


# --------------------------------------------------------------------------------------
# 4. Ratios carry their denominator and split
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Ratio:
    """A rate that cannot be constructed without naming what it is over.

    `RR_reported` was computed against bug_train in one script, bug_train+bug_val in another, and
    a post-gate subset in a third; the resulting table appeared to show a gate beating its own
    ungated arm on a metric that can only fall. The certificate coverage headline was 80.5% of a
    denominator that was 80.4% empty-scope rows.

    Construct with all four fields and print `label()` next to the value:

        rr = Ratio(name="RR_reported", numerator=n_rep, denominator=n_bugs,
                   population="bug_train", note="gated arm, r=0.90")
    """

    name: str
    numerator: int
    denominator: int
    population: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.population or not self.population.strip():
            raise GuardError(
                f"{self.name}: a rate must name the population its denominator is drawn from "
                f"(e.g. 'bug_train', 'clean_eval', 'held-out failures, certified scope only')."
            )
        if self.denominator <= 0:
            raise GuardError(f"{self.name}: denominator is {self.denominator}; the rate is undefined")
        if self.numerator > self.denominator:
            raise GuardError(
                f"{self.name}: numerator {self.numerator} exceeds denominator {self.denominator}. "
                f"The two are almost certainly drawn from different populations."
            )

    @property
    def value(self) -> float:
        return self.numerator / self.denominator

    def label(self) -> str:
        tail = f", {self.note}" if self.note else ""
        return (f"{self.name}={self.value:.3f} "
                f"({self.numerator}/{self.denominator} over {self.population}{tail})")


def check_nonvacuous(rows: Sequence[dict], scope_field: str = "scope",
                     warn_only: bool = False) -> int:
    """Drop rows whose scope is empty before they inflate a coverage denominator.

    A certified-decision count of 590,770 was 80.4 per cent rows with an empty scope, for which
    the guarantee is vacuously true. The honest denominator was 151,465.
    """
    real = [r for r in rows if r.get(scope_field)]
    if len(real) < len(rows):
        share = 1 - len(real) / max(len(rows), 1)
        msg = (f"{len(rows) - len(real)} of {len(rows)} rows ({share:.1%}) have an empty "
               f"'{scope_field}' and are vacuously satisfied; report {len(real)} as the denominator")
        if share > 0.05:
            _fail(msg, warn_only)
        else:
            print(f"WARN  {msg}", file=sys.stderr)
    return len(real)


# --------------------------------------------------------------------------------------
# 5. Rival hyperparameters are not at library defaults
# --------------------------------------------------------------------------------------

# Values read off the dataclasses in src/baselines/ on 2026-09-03. A rival shipped at these
# values was not tuned, and a table comparing it against our swept method is not a comparison.
# `swept_in_paper` marks parameters the original paper explicitly sweeps -- leaving those at a
# default is the most damaging case, because the published method is defined by the sweep.
LIBRARY_DEFAULTS: dict[str, dict[str, tuple[object, bool]]] = {
    # method            param              (default, swept_in_paper)
    "TopKSearch":     {"top_k": (64, True), "rounds": (3, False), "step_scale": (0.5, True),
                       "clean_tradeoff": (0.25, False)},
    "Arachne":        {"num_places": (64, True), "patch_aggr": (10.0, True),
                       "bound_scale": (2.0, True), "pop_size": (100, False),
                       "max_iter": (100, False)},
    "DistRep":        {"n_particles": (40, False), "n_iterations": (40, False),
                       "max_delta": (0.05, True), "lambda_clean": (0.5, True),
                       "top_k_weights": (1024, True)},
}


def recorded_baseline_params(tree: Path | str) -> dict:
    """Pull whatever hyperparameters a baseline run actually wrote down.

    As of 2026-09-03 the answer is: very little. `config_resolved.yaml` carries the repair
    experiment's own settings but not the search algorithm's, and `metrics.json` carries only
    `selected_top_k` / `selected_parameter_count`. `step_scale`, `rounds`, `clean_tradeoff`,
    `bound_scale`, `patch_aggr` and the PSO settings are not persisted anywhere, so a table built
    from these trees cannot show that a rival was tuned -- or that it was not.

    That gap is the reason the `step_scale` default went unnoticed until a sweep was run by hand.
    Until the runners in `scripts/run_*.py` serialise their config dataclass into metrics.json,
    `check_baseline_defaults` can only see the one or two values below.
    """
    tree = Path(tree)
    metrics_path = tree / "metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = json.loads(metrics_path.read_text())
    out: dict = {}
    search = metrics.get("search_result") or {}
    if "selected_top_k" in search:
        out["top_k"] = search["selected_top_k"]
    for key in ("arachne_config", "search_config", "baseline_config", "repair_config"):
        if isinstance(metrics.get(key), dict):
            out.update(metrics[key])
    return out


def check_params_recorded(tree: Path | str, method: str, warn_only: bool = True) -> bool:
    """Warn when a baseline tree does not record the hyperparameters its result depends on.

    A number whose configuration was never written down cannot be defended, reproduced, or
    audited for the default-value problem. This is a weaker claim than "the rival was untuned",
    and it is the one the artifacts currently support.
    """
    known = set(LIBRARY_DEFAULTS.get(method, {}))
    got = set(recorded_baseline_params(tree))
    missing = known - got
    if missing:
        _fail(f"{tree}: {method} run records none of {sorted(missing)}; whether this baseline was "
              f"tuned or left at library defaults cannot be determined from the artifact",
              warn_only)
        return False
    return True


def check_baseline_defaults(method: str, params: dict, warn_only: bool = True) -> list[str]:
    """Warn for every rival hyperparameter still sitting at its library default.

    Sweeping `step_scale` away from 0.5 moved TopKSearch's RR_held from .035 to .423, and
    `bound_scale` moved Arachne's from .033 to .156. Both had been reported as the methods'
    performance. Defaults on a parameter the original paper sweeps are escalated to a failure,
    because in that case the default is not the method.
    """
    known = LIBRARY_DEFAULTS.get(method)
    if known is None:
        return []
    hits: list[str] = []
    for param, (default, swept) in known.items():
        if param in params and params[param] == default:
            msg = (f"{method}.{param} is at the library default {default!r}"
                   + (" -- and the original paper sweeps this parameter, so the default is not "
                      "the published method" if swept else ""))
            hits.append(msg)
            _fail(msg, warn_only=warn_only or not swept)
    return hits


def asymmetry_check(notes: Sequence[tuple[str, str]]) -> None:
    """Record, for every shortcut in a comparison, which side it favours.

    Pass `[(what_was_approximated, "us" | "rival" | "neither"), ...]`. If every entry favours us,
    the comparison is invalid regardless of the size of the gap. Six independent reproduction
    shortcuts on this project all happened to weaken the rival; that is not what random
    corner-cutting looks like.
    """
    sides = [side for _, side in notes]
    if notes and all(side == "us" for side in sides):
        raise GuardError(
            "every approximation in this comparison favours our method:\n  "
            + "\n  ".join(f"- {what}" for what, _ in notes)
            + "\nFix the comparison before using the table; disclosing it afterwards is not enough."
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def audit(roots: Sequence[Path], pattern: str, warn_only: bool) -> int:
    """Audit the deployment-stage trees under `roots`.

    Both guards are properties of an evaluation, so training-stage trees -- which carry neither a
    `checkpoint_load_info` nor held-out predictions -- are skipped rather than reported as
    problems. They are counted so a run that produced no evaluations at all is still visible.
    """
    failures = skipped = audited = 0
    trees = sorted({p.parent for root in roots for p in root.rglob("metrics.json")
                    if pattern == "*" or pattern in str(p)})
    if not trees:
        print(f"no metrics.json under {[str(r) for r in roots]} matching {pattern!r}")
        return 0
    for tree in trees:
        metrics = json.loads((tree / "metrics.json").read_text())
        if "checkpoint_load_info" not in metrics:
            skipped += 1
            continue
        audited += 1
        try:
            check_checkpoint(tree / "metrics.json", warn_only)
            pred = tree / "predictions"
            if pred.is_dir():
                counts = check_split_disjoint(pred, warn_only)
                print(f"ok    {tree}  n_seen={counts['n_seen']} n_held={counts['n_held']}")
            else:
                print(f"ok    {tree}  (checkpoint clean; no predictions dir to check splits)")
        except GuardError as exc:
            failures += 1
            print(f"FAIL  {exc}")
    print(f"\n{audited} evaluation trees audited, {failures} failures "
          f"({skipped} training-stage trees skipped)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="run the checkpoint and split guards over output trees")
    a.add_argument("roots", nargs="+", type=Path)
    a.add_argument("--glob", default="*", help="substring an output path must contain")
    a.add_argument("--warn-only", action="store_true",
                   help="print violations instead of raising (for surveying an existing tree)")
    args = ap.parse_args()
    if args.cmd == "audit":
        return 1 if audit(args.roots, args.glob, args.warn_only) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
