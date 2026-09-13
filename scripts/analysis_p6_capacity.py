#!/usr/bin/env python3
"""P6 -- how many parameters does each method actually change?

Every capacity statement in this project so far has been about ONE method: Arachne moves 1-14
weights, and the relaxed arm moves up to 1024. That number means nothing without the same count
for everyone else. This produces it, measured rather than assumed.

Two counts, because "capacity" is two different things here:

  n_changed     parameters of the DEPLOYED network that differ from the base checkpoint.
                Measured by diffing state_dicts where a checkpoint exists; taken from the run's
                own record where the method reports it (`num_places`, `top_k_weights`,
                `selected_parameter_count`). This is the number a weight-editing paper means.

  n_added       parameters that exist only because of the repair and are not in the base network
                at all -- a patch head, a routing estimator, a generator. Weight editing adds
                nothing; patch methods add most of their capacity here.

They are NOT interchangeable and must never be summed into one "capacity" column: 1024 edited
weights inside a trained classifier and a 1024-parameter added head are different interventions.

Also reported: `n_per_input`, the size of the correction that varies from input to input. For
every method except ours this is 0 -- the same edited weights act on every input. For DynaPatch
it is the delta's dimension. This is the column that makes "input-specific" a measurable
property rather than a description.

    .venv/bin/python scripts/analysis_p6_capacity.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import raw_md as RM  # noqa: E402

SETTINGS = [(d, b) for d in ("gtsrb", "tt100k_signs", "lisa_signs")
            for b in ("resnet50", "convnext_tiny", "densenet121", "vgg16")]
SEED = 101
ATOL = 0.0          # exact inequality: an edited weight differs bit-for-bit


def base_ckpt(ds: str, bb: str) -> Path | None:
    """The frozen backbone every method starts from.

    Read from the setting's own deploy.yaml rather than guessed: the shipped configs point at
    `outputs/exp_<ds>_<bb>_backbone_public_v*/checkpoints/backbone_last.pt`, and one setting has
    historically pointed at a path that did not exist (the tt100k/vgg16 _v1 bug). Guessing a
    location silently yields "no diff available" for every fine-tuning method, which is exactly
    the number this analysis exists to produce.
    """
    cfg = ROOT / f"configs/v8_source/{ds}/{bb}/deploy.yaml"
    if cfg.is_file():
        for ln in cfg.read_text().splitlines()[:20]:
            m = re.search(r"checkpoint_path:\s*(\S+backbone\S*\.pt)", ln)
            if m:
                q = ROOT / m.group(1)
                if q.is_file():
                    return q
    c = sorted(ROOT.glob(f"outputs/exp_{ds}_{bb}_backbone*/checkpoints/backbone_last.pt"))
    return c[-1] if c else None


def n_classes(state: dict) -> int | None:
    """Output width of the final 2-D weight -- the classification layer."""
    two_d = [v for v in state.values() if torch.is_tensor(v) and v.ndim == 2]
    return int(two_d[-1].shape[0]) if two_d else None


def feat_dim(state: dict) -> int | None:
    """Input width of the final 2-D weight -- what a patch head on the penultimate feature sees.

    NOT the same as `patch_dim`: our delta is injected earlier on VGG16 (the 25088-d flatten),
    while the classification layer there takes 4096.
    """
    two_d = [v for v in state.values() if torch.is_tensor(v) and v.ndim == 2]
    return int(two_d[-1].shape[1]) if two_d else None


def patch_dim(ds: str, bb: str) -> int | None:
    """Width of the delta the generator emits, read off a dumped patch_vec."""
    import numpy as np
    for tag in ("_ep40ns", ""):
        f = (ROOT / f"outputs/effect_dump{tag}_v8_s{SEED}/{ds}/{bb}/deploy_direct/predictions"
             / "patch_vec_clean_eval.npy")
        if f.is_file():
            return int(np.load(f, mmap_mode="r").shape[1])
    return None


def sd(p: Path) -> dict | None:
    try:
        o = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(o, dict) and k in o and isinstance(o[k], dict):
            o = o[k]
            break
    return o if isinstance(o, dict) else None


BUFFER = ("running_mean", "running_var", "num_batches_tracked")


def is_buffer(name: str) -> bool:
    return any(name.endswith(b) for b in BUFFER)


def diff_count(a: dict, b: dict) -> tuple[int, int, int, int]:
    """(changed trainable scalars, changed BN-buffer scalars, total scalars, tensors touched).

    The two must be separated. Head-only fine-tuning on gtsrb/resnet50 differs from the base
    checkpoint in 141,227 scalars, but only 88,107 of those are the head (`fc.weight` 88,064 +
    `fc.bias` 43): the other 53,120 are BatchNorm running statistics, which move during the
    forward passes even though the backbone's weights are frozen. Counting them as "parameters
    the method changes" overstates every BN-bearing backbone by ~60% and understates nothing --
    LastDelta on resnet50 shows 53,119 changed scalars of which ZERO are trainable.

    That BN drift is itself worth recording: a method described as freezing the backbone is not
    leaving the backbone bit-identical.
    """
    ch = buf = tot = nt = 0
    for k, va in a.items():
        vb = b.get(k)
        if not (torch.is_tensor(va) and torch.is_tensor(vb) and va.shape == vb.shape):
            continue
        if not va.is_floating_point():
            continue
        n = int(((va.float() - vb.float()).abs() > ATOL).sum())
        if is_buffer(k):
            buf += n
        else:
            ch += n
            tot += va.numel()
            nt += n > 0
    return ch, buf, tot, nt


def main() -> None:
    rows = []
    for ds, bb in SETTINGS:
        bp = base_ckpt(ds, bb)
        base = sd(bp) if bp else None
        for meth, tpl, note in [
            ("HeadFT", "fewshot_headonly_ep40_v8_s{s}_kfull/{ds}/{bb}", "fine-tunes the classifier head"),
            ("LastDelta", "fewshot_lastdelta_ep40_v8_s{s}_kfull/{ds}/{bb}", "additive delta on the last layer"),
            ("FullFT", "fewshot_distrep_ep40_v8_s{s}_kfull/{ds}/{bb}", "fine-tunes everything"),
            ("WeightedRetrain", "baselines_wr_ep40_v8_s{s}/{ds}/{bb}/weighted_retraining", "retrains with reweighted loss"),
            ("GreedyTopK", "topksearch_ss8.0_v8_s{s}/{ds}/{bb}", "internal; edits a fixed top-k"),
        ]:
            d = ROOT / "outputs" / tpl.format(s=SEED, ds=ds, bb=bb)
            m = {}
            f = d / "metrics.json"
            if f.is_file():
                try:
                    m = json.loads(f.read_text())
                except json.JSONDecodeError:
                    pass
            ch = buf = tot = nt = None
            ck = m.get("checkpoint_path")
            if ck and base is not None:
                new = sd(Path(ck))
                if new is not None:
                    ch, buf, tot, nt = diff_count(new, base)
            rows.append({"method": meth, "setting": f"{ds}/{bb}",
                         "n_changed": m.get("selected_parameter_count", ch),
                         "n_bn_buffer_drift": buf,
                         "n_total_model": tot, "n_tensors_touched": nt,
                         "n_added": 0, "n_per_input": 0, "source": "state_dict diff",
                         "note": note})

        # methods that record their own edited-weight count
        for meth, tpl, key, note in [
            ("Arachne", "arachne_ct_bs16.0_v8_s{s}_kfull/{ds}/{bb}", "num_places",
             "Pareto front of the bidirectional localisation"),
            ("DistRep", "distrepPSO_ct_full_v8_s{s}_kfull/{ds}/{bb}", "top_k_weights",
             "PSO over a fixed top-k of the last layer"),
        ]:
            f = ROOT / "outputs" / tpl.format(s=SEED, ds=ds, bb=bb) / "metrics.json"
            if not f.is_file():
                continue
            try:
                m = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            rows.append({"method": meth, "setting": f"{ds}/{bb}", "n_changed": m.get(key),
                         "n_total_model": None, "n_tensors_touched": 1, "n_added": 0,
                         "n_per_input": 0, "source": f"run record ({key})", "note": note})

        for n in (16, 64, 256, 1024):
            f = (ROOT / "outputs" / f"arachneTopN{n}_bs16.0_v8_s{SEED}_kfull"
                 / ds / bb / "metrics.json")
            if f.is_file():
                try:
                    m = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                rows.append({"method": f"Arachne-relaxed N={n}", "setting": f"{ds}/{bb}",
                             "n_changed": m.get("num_places"), "n_total_model": None,
                             "n_tensors_touched": 1, "n_added": 0, "n_per_input": 0,
                             "source": "run record (num_places)",
                             "note": "NOT Arachne: Pareto step removed"})

        # patch-type methods: nothing in the base network changes.
        # Both numbers are MEASURED, not tabulated. The first version hard-coded 45 classes for
        # TT100K, which actually has 181, and guessed the patch dimension per backbone; a wrong
        # constant here silently rescales every capacity number for the patch family.
        dim = patch_dim(ds, bb)
        ncls = n_classes(base) if base is not None else None
        feat = feat_dim(base) if base is not None else None
        for meth, added, per_in, note in [
            ("FixedPatch", dim, 0, "one delta shared by every input"),
            ("DynaPatch", None, dim, "delta GENERATED per input; n_added is the generator"),
            ("NN-Patching", (feat * ncls + ncls) if feat and ncls else None, 0,
             "linear patch head on the penultimate feature"),
            ("PatchNAS", None, 0, "searched head; width varies per cell, see arch in the JSON"),
        ]:
            rows.append({"method": meth, "setting": f"{ds}/{bb}", "n_changed": 0,
                         "n_bn_buffer_drift": 0, "n_total_model": None,
                         "n_tensors_touched": 0, "n_added": added, "n_per_input": per_in,
                         "n_classes": ncls, "patch_dim": dim, "feature_dim": feat,
                         "source": "architecture", "note": note})

    d = pd.DataFrame(rows)
    RM.write_section("P6", "P6 — how many parameters each method actually changes (raw)", f"""
Seed {SEED}. Produced because every capacity claim so far covered only Arachne (1-14 weights on
the front, up to 1024 relaxed) with no comparable count for anyone else.

| column | meaning |
|---|---|
| `n_changed` | parameters of the DEPLOYED network that differ from the base checkpoint. Measured by an exact state_dict diff where a checkpoint exists, or taken from the run's own record (`num_places`, `top_k_weights`, `selected_parameter_count`) |
| `n_bn_buffer_drift` | BatchNorm `running_mean`/`running_var` scalars that also moved. **Not parameters.** They drift during the method's forward passes even when the backbone is frozen, so a method advertised as freezing the backbone still does not leave it bit-identical |
| `n_total_model` | trainable scalars compared, so `n_changed` has a denominator |
| `n_tensors_touched` | how many tensors differ at all — separates "edits a few weights in one layer" from "moves the whole network" |
| `n_added` | parameters that exist ONLY because of the repair and are absent from the base network (patch head, estimator, generator) |
| `n_per_input` | the part of the correction that VARIES per input. 0 for every method except ours; for DynaPatch it is the delta's dimension |
| `source` | how the number was obtained — a measured diff and a self-reported count are not the same evidence |

**`n_changed` and `n_added` must not be summed.** 1024 edited weights inside a trained classifier
and a 1024-parameter added head are different interventions; a single "capacity" column would
hide which one a method is doing.

`n_per_input` is the column that makes "input-specific" measurable rather than a description.

Blank `n_changed` means no checkpoint was found to diff against, not zero.
""", [("", d.sort_values(["setting", "method"]))])

    out = ROOT / "outputs" / "p6_capacity"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "per_cell.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/per_cell.csv  ({len(d)} rows)")
    print(d.groupby("method")[["n_changed", "n_added", "n_per_input"]].median().to_string())


if __name__ == "__main__":
    main()
