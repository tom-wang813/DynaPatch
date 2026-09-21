# Training from scratch

This covers retraining everything yourself — backbones, DynaPatch (DPGen+DPGate), and all 6 of
the paper's baselines — instead of using shipped checkpoints/results. See
**[README.md](README.md)** first if you only want to reproduce the paper's reported tables; that
path needs no GPU and no training.

## Setup

```bash
uv sync
```

GPU setup (driver/CUDA pinning notes): see README.md "Requirements".

## Data

`data/{gtsrb,tt100k_signs_clf,lisa_signs_clf}` is not tracked by git (~680 MB, third-party
licensed — see README.md "License"). GTSRB can be re-fetched automatically
(`torchvision.datasets.GTSRB(..., download=True)`, wired into the training scripts). TT100K-Signs
and LISA-Signs need to be prepared into the `{train,test}/<class>/*.jpg` `ImageFolder` layout
`src/data/factory.py` expects:

```bash
uv run python scripts/prepare_lisa_classification.py --raw-root data/lisa_raw
uv run python scripts/prepare_tt100k_classification.py --raw-root data/tt100k_raw
```

These convert an already-downloaded raw annotation format into the classification crops; you
must separately obtain the LISA Traffic Sign Dataset and TT100K raw data (their standard public
releases) first. If this checkout already has `data/` populated (e.g. received alongside this
repo), nothing further is needed here.

## One entrypoint per setting

```bash
bash scripts/reproduce_all.sh <dataset> <backbone> <seed> [outdir]
# e.g. bash scripts/reproduce_all.sh gtsrb resnet50 101 outputs/repro_run
```

Datasets: `gtsrb | tt100k_signs | lisa_signs`. Backbones: `resnet50 | convnext_tiny |
densenet121 | vgg16`. Seeds used in the paper: `101 202 303`.

This trains the frozen backbone if its checkpoint is missing (`scripts/train_backbone.py`,
shared across the 3 repair seeds), then runs DynaPatch (DPGen+DPGate) train+deploy -- **including
DPGate itself**: a second "calib" deploy-eval pass (identical to the report pass except
`bug_eval_indices_path` points at the held-out `bug_val` slice instead of `bug_eval`) plus an
automatic gate fit (`scripts/gate_protocol_b.py`) over every (dataset, backbone, seed) cell dumped
so far, under `outputs/effect_dump<tag>_v8_s<seed>/<dataset>/<backbone>/`. Fitting needs no GPU
and takes seconds; it just won't have much to fit until you've run more than one or two cells.
Then all 6 of the paper's baselines:

| Baseline | Driver | Status |
|---|---|---|
| DynaPatch | `scripts/run_resolved_experiment.py` | self-contained |
| Arachne (real, DE re-implementation) | `scripts/run_arachne_de.py` / `src/baselines/arachne_de.py` | self-contained |
| DistrRep (real, 3-phase PSO re-implementation) | `scripts/run_distrep_pso.py` / `src/baselines/distrep_pso.py` | self-contained |
| HeadFT | `scripts/train_head_repair_baseline.py --mode head_only` | self-contained |
| FullFT | `scripts/train_head_repair_baseline.py --mode full_finetune` | self-contained |
| NNPatch (NN-Patching) | `scripts/dump_prior_features.py` + `scripts/baseline_prior_patches.py` | self-contained (cross-setting; run separately, see below) |
| PatchNAS | same as NNPatch (`baseline_prior_patches.py` produces both) | self-contained |

`scripts/run_mainline_all.sh`, `scripts/queue_mainline_baselines.sh`, `scripts/queue_arachne_de.sh`,
`scripts/queue_distrep_pso.sh`, `scripts/queue_real_baselines.sh` orchestrate these across the 12
settings x 3 seeds.

## NNPatch / PatchNAS (cross-setting)

Unlike the other baselines, these operate on cached features across ALL settings at once rather
than per-setting:

```bash
uv run python scripts/dump_prior_features.py --setting <dataset>/<backbone> --seed <seed> --device cuda:0
uv run python scripts/baseline_prior_patches.py --estimator mlp
```

## Training drivers (reference)

`scripts/train_backbone.py` (Hydra-composed, `configs/{dataset,model,train,runtime}/`),
`scripts/train_head_repair_baseline.py` (HeadFT/FullFT, flat `--config` +
`--mode`), `scripts/run_arachne_de.py` / `scripts/run_distrep_pso.py` (Arachne(DE)/DistRep(PSO)),
`scripts/run_resolved_experiment.py` (DynaPatch train/deploy).

## Batch size

`train_head_repair_baseline.py`'s `train_loop.batch_size` for HeadFT/FullFT
uses `configs/shuffled_split_source/<ds>/<bb>/train.yaml`'s `train_loop.batch_size`. Set `BATCH=<value>` to
override.

## Roughly how long this takes

**Not benchmarked end-to-end on any fixed reference machine** — the numbers below are order-of-magnitude,
derived from each stage's own epoch/iteration budget, not a timed run. Get an exact number for your own
GPU with a single `bash scripts/reproduce_all.sh gtsrb resnet50 101` (resnet50/gtsrb is the cheapest
setting: small backbone, 43-class, ~39k images) before committing to the full 12 settings x 3 seeds grid.

Per (dataset, backbone, seed), `reproduce_all.sh` runs, in order:

| Stage | Budget | Notes |
|---|---|---|
| Backbone (`train_backbone.py`) | 5 epochs | only once per (dataset, backbone) — shared across all 3 seeds, skipped if `backbone_last.pt` already exists |
| DynaPatch train (DPGen+DPGate) | up to 50 epochs, early-stop-capable (`early_stop_min_epochs: 1`) | usually the fastest of the epoch-based stages since it only touches the frozen feature/patch layers, not the whole backbone |
| HeadFT | 40 epochs | `--mode head_only` |
| FullFT | 40 epochs | `--mode full_finetune`, all backbone params updated — slower per epoch than HeadFT |
| Arachne(DE) | population-based, no fixed epoch cap | `run_arachne_de.py --max_iter` |
| DistrRep(PSO) | population-based, no fixed epoch cap | called out in `reproduce_all.sh`'s own comments as **"the long pole"** — `SKIP_DISTREP=1` skips it so you can get the other 5 methods' numbers first and run this one separately/overnight |

NNPatch/PatchNAS are not included in that per-setting loop — they run once across *all* settings via
`dump_prior_features.py` + `baseline_prior_patches.py` (see above), so their cost doesn't multiply by 12.

Rule of thumb: on a single modern GPU (the paper's runs used an RTX 3090, sometimes shared with other
jobs — see "Batch size"), the epoch-based stages (backbone, DynaPatch, HeadFT, FullFT) together are
minutes-to-tens-of-minutes per setting/seed on GTSRB/LISA-Signs (~6k-39k images) and longer on
TT100K-Signs (181 classes); Arachne(DE)/DistrRep(PSO) are the dominant cost for the full 12x3 grid, with
DistrRep specifically flagged as the slowest. Budget accordingly, or use `DYNAPATCH_CHECKPOINT=<path>`
to skip DynaPatch's own training stage when you already have its checkpoint (see README.md
"Pretrained checkpoints").
