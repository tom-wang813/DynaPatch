#!/usr/bin/env python3
"""DistrRep (Li Calsi et al., ICST 2023) -- the real three-phase PSO method, not a fine-tune.

WHY THIS EXISTS. The column the paper called `DistRep` was produced by
`run_fewshot_distrep.sh --mode full_finetune_distr`, which
`src/baselines/head_repair.py:107` resolves to "every parameter requires_grad" -- plain full
fine-tuning. That is a valid capability ceiling but it is NOT DistrRep, whose three phases are
distributed fault localisation over a SPARSE suspicious-weight set, per-partition PSO, and a PSO
over expert blending coefficients. The faithful re-implementation has been sitting in
`src/baselines/distrep_pso.py` unused since it was written (note/PITFALLS.md 2026-08-01).

This runner drives it and writes the same prediction CSVs every other baseline writes, so
`per_setting.py` / `compare_all_methods.py` / `make_main_table.py` pick it up with no changes.

Two deliberate differences from `run_arachne_de.py`:
  - DistrRep localises over ALL layers, so it cannot use the cached-last-layer-input trick and
    needs raw images and full forward passes. It is correspondingly slower.
  - Both PSO worker devices are pinned to the same GPU: this host has one live card (the T4 at
    47:00.0 is dead), and the default config asks for cuda:1 / cuda:2.

Usage:
    uv run python scripts/run_distrep_pso.py --setting gtsrb/resnet50 --seed 101 \
        --output-root outputs/fewshot_distrepPSO_v8_s101_kfull/gtsrb/resnet50
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines.distrep_pso import (  # noqa: E402
    DistrRepConfig,
    DistrRepPSO,
    DistrRepRunArtifacts,
    PSOConfig,
    materialise_expert,
)
from src.experiment.stage3 import build_stage3_bundle  # noqa: E402
from src.experiment.seed import seed_everything  # noqa: E402


def materialise(loader, device, cap: int | None = None):
    """Collect a loader into (inputs, labels, dataset_indices) on CPU, optionally capped."""
    xs, ys, idxs = [], [], []
    seen = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        xs.append(x)
        ys.append(y)
        idxs.append(torch.arange(seen, seen + x.size(0)))
        seen += x.size(0)
        if cap is not None and seen >= cap:
            break
    X = torch.cat(xs)[:cap] if cap else torch.cat(xs)
    Y = torch.cat(ys)[:cap] if cap else torch.cat(ys)
    I = torch.cat(idxs)[:cap] if cap else torch.cat(idxs)
    return X, Y, I


@torch.no_grad()
def predict(model, X, device, batch: int = 128):
    """(argmax, max softmax probability, raw logits).

    The confidence half was added 2026-08-19: the original dump wrote predictions only, which
    left DistRep the one repair the authorization layer could not be evaluated on at all
    (scripts/cross_repair_frontier.py). PSO does not save the repaired weights, so recovering
    it required re-running the repair rather than re-reading a checkpoint.

    The logits half was added 2026-09-04 for the same reason: `logits` was computed every call
    and discarded, which left DistRep's pre-integration experts the only patch-shaped baseline
    with no per-input Delta z(x) = patched_logits - base_logits (needed for
    scripts/analysis_m1_correction_direction.py-style mechanism analysis). Single-setting only
    (gtsrb/resnet50 s101) -- see note/RESEARCH_STATE.md M3 for why DistRep cannot cheaply cover
    all 12 settings.
    """
    out, conf, logit_list = [], [], []
    model.eval()
    for i in range(0, X.size(0), batch):
        logits = model(X[i:i + batch].to(device))
        out.append(logits.argmax(1).cpu())
        conf.append(logits.softmax(1).max(1).values.cpu())
        logit_list.append(logits.detach().float().cpu())
    return torch.cat(out), torch.cat(conf), torch.cat(logit_list)


def write_predictions(path: Path, split: str, idxs, labels, base_pred, new_pred,
                      base_conf, new_conf, base_logits=None, new_logits=None):
    """Column names match the other baseline trees (base_confidence / patched_confidence).

    When `base_logits`/`new_logits` are given (numpy arrays, row order matching `idxs`), also
    writes `base_logits_{split}.npy` / `patched_logits_{split}.npy` next to the CSV -- same
    layout as outputs/effect_dump_ep40ns_v8_s*/.../predictions/ and
    outputs/prior_patch_persample/.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "dataset_index", "label", "base_pred", "patched_pred",
                    "base_confidence", "patched_confidence",
                    "base_correct", "patched_correct", "repaired", "regressed"])
        for i in range(len(labels)):
            y, b, n = int(labels[i]), int(base_pred[i]), int(new_pred[i])
            w.writerow([split, int(idxs[i]), y, b, n,
                        float(base_conf[i]), float(new_conf[i]), b == y, n == y,
                        (b != y) and (n == y), (b == y) and (n != y)])
    if base_logits is not None and new_logits is not None:
        np.save(path.parent / f"base_logits_{split}.npy", np.asarray(base_logits, dtype=np.float32))
        np.save(path.parent / f"patched_logits_{split}.npy", np.asarray(new_logits, dtype=np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, help="ds/bb")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", default="full")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clean-cap", type=int, default=2048,
                    help="preservation-set size handed to PSO; matches run_arachne_de.py")
    ap.add_argument("--preserve-split", choices=["clean_eval", "clean_train"],
                    default="clean_eval",
                    help="which clean rows feed the PSO clean-preservation penalty. "
                         "clean_eval is the SHIPPED behaviour and is the REPORTING set "
                         "(clean_test). clean_train uses the dataset's train split, which is "
                         "disjoint from every reported population.")
    ap.add_argument("--n-partitions", type=int, default=5)
    ap.add_argument("--top-k-weights", type=int, default=1024)
    ap.add_argument("--particles", type=int, default=40)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--integration-particles", type=int, default=None)
    ap.add_argument("--integration-iterations", type=int, default=None)
    ap.add_argument("--rng-seed", type=int, default=None,
                    help="explicit RNG seed for partitioning and PSO; omitted preserves legacy behavior")
    ap.add_argument("--dump-experts", action="store_true",
                    help="write each pre-integration expert on held and clean test only")
    ap.add_argument("--experiment-config", default=None,
                    help="audit config copied into the output directory")
    # configs/v8_source/*/train.yaml ship runtime.num_workers=0 with batch_size=16. Measured on
    # the sibling Arachne runner, that made the feature/forward passes 1652 s instead of 22 s
    # (75x). Aggregate metrics are unaffected; only the row ORDER of the shuffled bug_train
    # loader changes, and this tree's dataset_index is a positional counter anyway.
    ap.add_argument("--num-workers", type=int, default=None)
    a = ap.parse_args()

    for name in ("n_partitions", "top_k_weights", "particles", "iterations"):
        if getattr(a, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("integration_particles", "integration_iterations"):
        value = getattr(a, name)
        if value is not None and value <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if a.rng_seed is not None:
        seed_everything(a.rng_seed)

    ds, bb = a.setting.split("/")
    splits = ROOT / f"artifacts/bug_sets/v8_splits_seed{a.seed}/{ds}_{bb}"
    sub = (splits / f"{ds}_bug_train_indices.json" if a.k == "full"
           else ROOT / f"outputs/fewshot_v8_splits/s{a.seed}/{ds}_{bb}_k{a.k}/{ds}_bug_train_indices.json")

    cfg = OmegaConf.load(ROOT / f"configs/v8_source/{ds}/{bb}/train.yaml")
    for key, val in [
        ("data.bug_indices_path", str(splits / f"{ds}_bug_indices.json")),
        ("data.bug_eval_indices_path", str(splits / f"{ds}_bug_eval_indices.json")),
        ("data.bug_train_indices_path", str(sub)),
        # factory.py:344 PREFERS bug_val_indices_path; pointing it at bug_eval is what makes
        # loaders["bug_eval"] the held-out set. Getting this wrong is exactly the bug that
        # invalidated the first Arachne sweep (PITFALLS 2026-07-31).
        ("data.bug_val_indices_path", str(splits / f"{ds}_bug_eval_indices.json")),
        ("data.clean_eval_indices_path", str(splits / f"{ds}_clean_test_indices.json")),
        ("runtime.device", a.device),
    ] + ([("runtime.num_workers", a.num_workers)] if a.num_workers is not None else []):
        OmegaConf.update(cfg, key, val, merge=True)
    if ds == "tt100k_signs" and bb == "vgg16":
        OmegaConf.update(cfg, "model.checkpoint_path",
                         "outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt",
                         merge=True)

    device = torch.device(a.device)
    backbone, _model, loaders, _ = build_stage3_bundle(cfg, device)
    backbone = backbone.to(device).eval()

    t0 = time.time()
    bug_X, bug_Y, _ = materialise(loaders["bug_train"], device)
    held_X, held_Y, held_I = materialise(loaders["bug_eval"], device)
    # PSO only ever sees `clean_cap` samples (memory: two worker threads each hold a copy), but Reg
    # must be reported on the SAME population as every other method or the column is not comparable
    # -- and a 512-sample estimate has granularity 1/512, which is coarser than the differences we
    # are reporting. So evaluate on the full clean split and hand PSO the capped prefix.
    cleanF_X, cleanF_Y, cleanF_I = materialise(loaders["clean_eval"], device)
    # The PSO penalty set and the Reg reporting set are two different objects. Under
    # --preserve-split clean_train the penalty is computed on the dataset's train split, so
    # nothing PSO optimises appears in any reported population.
    if a.preserve_split == "clean_train":
        presF_X, presF_Y, _ = materialise(loaders["clean_train"], device, cap=a.clean_cap)
        clean_X, clean_Y = presF_X[:a.clean_cap], presF_Y[:a.clean_cap]
    else:
        clean_X, clean_Y = cleanF_X[:a.clean_cap], cleanF_Y[:a.clean_cap]

    # RELEASE THE DATALOADER WORKERS. `_loader_kwargs` sets persistent_workers=True, so with
    # num_workers=4 the four worker processes stay alive for the WHOLE PSO run -- measured
    # 2026-09-01 at 4.9 GB resident each, 19.6 GB held for the ~1 h after their last use. They
    # are only needed for the materialise calls above. On a 62 GB shared box that difference is
    # what the OOM killer acted on (4 GTSRB cells lost at 19:16-19:26).
    for _ldr in list(loaders.values()):
        it = getattr(_ldr, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
        _ldr._iterator = None
    loaders.clear()
    del loaders
    import gc as _gc
    _gc.collect()

    bug_bp, bug_bc, bug_bl = predict(backbone, bug_X, device)
    held_bp, held_bc, held_bl = predict(backbone, held_X, device)
    cleanF_bp, cleanF_bc, cleanF_bl = predict(backbone, cleanF_X, device)
    clean_bp = (predict(backbone, clean_X, device)[0] if a.preserve_split == "clean_train"
                else cleanF_bp[:a.clean_cap])
    print(f"[distrepPSO] evidence={len(bug_Y)} heldout={len(held_Y)} "
          f"clean(PSO)={len(clean_Y)} from {a.preserve_split} clean(eval)={len(cleanF_Y)} "
          f"load={time.time()-t0:.1f}s", flush=True)

    pcfg = PSOConfig(n_particles=a.particles, n_iterations=a.iterations)
    integration_particles = (
        a.integration_particles
        if a.integration_particles is not None
        else max(a.particles * 3 // 4, 10)
    )
    integration_iterations = (
        a.integration_iterations
        if a.integration_iterations is not None
        else max(a.iterations * 3 // 4, 10)
    )
    dcfg = DistrRepConfig(
        n_partitions=a.n_partitions, top_k_weights=a.top_k_weights,
        pso_fix=pcfg,
        pso_integrate=PSOConfig(n_particles=integration_particles,
                                n_iterations=integration_iterations),
        # One live card on this host; the dataclass defaults ask for cuda:1 / cuda:2.
        device0=a.device, device1=a.device,
    )
    t1 = time.time()
    run_result = DistrRepPSO(dcfg).run(
        base_model=backbone,
        bug_inputs=bug_X, bug_labels=bug_Y,
        clean_inputs=clean_X, clean_labels=clean_Y,
        base_bug_preds=bug_bp, base_clean_preds=clean_bp,
        return_experts=a.dump_experts,
    )
    if a.dump_experts:
        if not isinstance(run_result, DistrRepRunArtifacts):
            raise RuntimeError("--dump-experts requested but DistrRep returned no expert artifacts")
        repaired_cpu = run_result.merged_model
        expert_deltas = run_result.expert_deltas
        if len(expert_deltas) != a.n_partitions:
            raise RuntimeError(
                f"expected {a.n_partitions} experts, received {len(expert_deltas)}"
            )
    else:
        if isinstance(run_result, DistrRepRunArtifacts):
            raise RuntimeError("unexpected expert artifacts without --dump-experts")
        repaired_cpu = run_result
        expert_deltas = []
    elapsed = time.time() - t1
    print(f"[distrepPSO] repair done in {elapsed:.1f}s", flush=True)

    output_root = Path(a.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if a.experiment_config is not None:
        experiment_config = Path(a.experiment_config)
        if not experiment_config.is_file():
            raise FileNotFoundError(experiment_config)
        (output_root / "experiment_config.yaml").write_text(experiment_config.read_text())
    if a.dump_experts:
        expert_manifest = []
        for expert_index, expert_delta in enumerate(expert_deltas):
            expert = materialise_expert(backbone, expert_delta, device)
            expert_out = output_root / "expert_predictions" / f"expert_{expert_index:02d}"
            for name, X, Y, I, BP, BC, BL in [
                ("repair_holdout_unseen", held_X, held_Y, held_I, held_bp, held_bc, held_bl),
                ("clean_eval", cleanF_X, cleanF_Y, cleanF_I, cleanF_bp, cleanF_bc, cleanF_bl),
            ]:
                np_, nc_, nl_ = predict(expert, X, device)
                write_predictions(expert_out / f"{name}_predictions.csv",
                                  name, I, Y, BP, np_, BC, nc_,
                                  base_logits=BL.numpy(), new_logits=nl_.numpy())
            suspicious, delta = expert_delta
            expert_manifest.append({
                "expert_index": expert_index,
                "suspicious_weights": int(delta.numel()),
                "parameter_tensors": len(suspicious.param_names),
                "reported_splits": ["repair_holdout_unseen", "clean_eval"],
                "training_predictions_saved": False,
            })
            del expert
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        (output_root / "expert_manifest.json").write_text(json.dumps({
            "oracle_status": "test-label oracle upper bound; not deployable",
            "selection_population": "repair_holdout_unseen and clean_eval only",
            "training_population": "bug_train and clean_train; predictions intentionally not exported",
            "experts": expert_manifest,
        }, indent=2))
        print(f"[distrepPSO] wrote {len(expert_manifest)} pre-integration expert test dumps",
              flush=True)

    repaired = repaired_cpu.to(device).eval()
    out = output_root / "predictions"
    for name, X, Y, I, BP, BC, BL in [
        ("repair_holdout_unseen", held_X, held_Y, held_I, held_bp, held_bc, held_bl),
        ("repair_support_seen", bug_X, bug_Y, torch.arange(len(bug_Y)), bug_bp, bug_bc, bug_bl),
        ("clean_eval", cleanF_X, cleanF_Y, cleanF_I, cleanF_bp, cleanF_bc, cleanF_bl),
    ]:
        np_, nc_, nl_ = predict(repaired, X, device)
        write_predictions(out / f"{name}_predictions.csv", name, I, Y, BP, np_, BC, nc_,
                          base_logits=BL.numpy(), new_logits=nl_.numpy())

    # The four hand-listed knobs below were the only ones any tree recorded; w, c1, c2, max_delta,
    # lambda_clean and clean_eval_interval were not, so a reader could not tell whether this
    # baseline had been tuned. `baseline_config` serialises both dataclasses whole -- the PSO
    # fields are flattened alongside so a checker sees them without knowing the nesting -- and
    # cannot fall out of sync with the dataclass the way an enumerated list does.
    # Read back by scripts/guards.py:recorded_baseline_params.
    (Path(a.output_root) / "metrics.json").write_text(json.dumps({
        "setting": a.setting, "seed": a.seed, "k": a.k, "method": "DistrRep-PSO (ICST'23 re-impl)",
        "n_partitions": a.n_partitions, "top_k_weights": a.top_k_weights,
        "particles": a.particles, "iterations": a.iterations,
        "clean_cap": a.clean_cap, "preserve_split": a.preserve_split,
        "rng_seed": a.rng_seed, "dump_experts": a.dump_experts,
        "n_exported_experts": len(expert_deltas),
        "experiment_config": a.experiment_config,
        "command": [sys.executable, *sys.argv],
        "reported_splits": ["repair_holdout_unseen", "clean_eval"],
        "split_paths": {
            "bug_train": str(sub),
            "bug_eval": str(splits / f"{ds}_bug_eval_indices.json"),
            "clean_eval": str(splits / f"{ds}_clean_test_indices.json"),
        },
        "baseline_config": {"method": "DistRep", **asdict(dcfg), **asdict(pcfg)},
        "repair_seconds": round(elapsed, 1),
    }, indent=2))
    print(f"[distrepPSO] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
