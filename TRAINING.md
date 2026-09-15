# Training from scratch

This covers retraining everything yourself — backbones, DynaPatch (DPGen+DPGate), and all 6 of
the paper's baselines — instead of using shipped checkpoints/results. See
**[README.md](README.md)** first if you only want to reproduce the paper's reported tables; that
path needs no GPU and no training.

## Setup

```bash
uv sync
```

GPU setup (driver/CUDA pinning notes): see README.md "GPU setup".

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
shared across the 3 repair seeds), then runs DynaPatch (DPGen+DPGate) train+deploy, then all 6
of the paper's baselines:

| Baseline | Driver | Status |
|---|---|---|
| DynaPatch | `scripts/run_resolved_experiment.py` | self-contained |
| Arachne (real, DE re-implementation) | `scripts/run_arachne_de.py` / `src/baselines/arachne_de.py` | self-contained |
| DistrRep (real, 3-phase PSO re-implementation) | `scripts/run_distrep_pso.py` / `src/baselines/distrep_pso.py` | self-contained |
| HeadFT | `scripts/train_head_repair_baseline.py --mode head_only` | self-contained |
| FullFT | `scripts/train_head_repair_baseline.py --mode full_finetune` | self-contained |
| NNPatch (NN-Patching) | `scripts/dump_prior_features.py` + `scripts/baseline_prior_patches.py` | self-contained (cross-setting; run separately, see below) |
| PatchNAS | same as NNPatch (`baseline_prior_patches.py` produces both) | self-contained |

`scripts/train_head_repair_baseline.py` also drives two secondary, non-headline variants used
elsewhere in the paper's ablations: `--mode full_finetune_distr` ("DistRep-original", a plain
full-fine-tune upper bound distinct from the real PSO method above) and
`scripts/train_arachne_baseline.py` ("TopKSearch", a greedy top-k weight search that predates
the real Arachne(DE) re-implementation and must never be reported as "Arachne" — see that
script's header). `reproduce_all.sh` runs both of these too.

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
`scripts/train_head_repair_baseline.py` (HeadFT/FullFT/DistRep-original, flat `--config` +
`--mode`), `scripts/train_arachne_baseline.py` (TopKSearch), `scripts/run_arachne_de.py` /
`scripts/run_distrep_pso.py` (Arachne(DE)/DistRep(PSO)), `scripts/run_resolved_experiment.py`
(DynaPatch train/deploy).

## Known, disclosed gap

`train_head_repair_baseline.py`'s `train_loop.batch_size` for HeadFT/FullFT/DistRep-original was
originally read from a resolved config produced by an earlier training run; that specific
resolved-config file's provenance could not be traced with certainty, so this repo falls back to
`configs/v8_source/<ds>/<bb>/train.yaml`'s own `train_loop.batch_size` (a real, in-repo number)
instead of guessing. This is disclosed rather than silently assumed to be bit-identical to the
paper's shipped numbers for this one hyperparameter — set `BATCH=<value>` to override if you know
the original value. This affects gradient step count for those 3 non-headline-method training
runs only; it does not affect DynaPatch, Arachne(DE), DistRep(PSO), NNPatch, or PatchNAS.
