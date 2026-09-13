# DynaPatch

**Deployment-Time Gating for Critical-Class Safety in Neural Repair**

Anonymized reproduction repository for DynaPatch (double-blind review artifact). DynaPatch is
a patch-based DNN repair method with two components on top of a **frozen** base classifier:

1. **DPGen** — an input-conditioned hypernetwork that generates a feature-space correction
   ("patch") without modifying the frozen backbone's weights.
2. **DPGate** — a runtime gate (multinomial logistic regression over pre- and post-patch
   prediction evidence) that decides whether to admit a candidate patch, to avoid trading
   repaired failures for new regressions.

It is evaluated on 3 datasets (GTSRB, TT100K-Signs, LISA-Signs) x 4 backbones (ResNet50,
ConvNeXt-tiny, VGG16, DenseNet121) = 12 settings, x 3 seeds, against 6 baselines: HeadFT,
FullFT, Arachne, DistrRep, NNPatch (NN-Patching), PatchNAS.

## What this repository actually supports

This repo ships two genuinely different reproduction paths. Read this section before assuming
either "GPU-free" or "full retrain" means everything.

### 1. GPU-free: recompute the paper's LaTeX tables from shipped results

`outputs/` ships the small, already-aggregated per-cell/per-setting CSV and JSON files (a few
MB total) that `scripts/paper_latex_tables.py` reads directly — no feature tensors, no
checkpoints, no GPU. This regenerates every RQ1-RQ4 LaTeX table in the paper's own house style:

```bash
uv sync
uv run python scripts/paper_latex_tables.py --outdir note/tables
```

**What this does NOT do**: it does not re-derive those CSVs from raw per-sample predictions.
The raw per-sample prediction dumps behind them (per-input logits/labels across 12 settings x
3 seeds x up to ~17 methods) are tens to ~150 GB in the original working repository and are
**not shipped here** — see "Known limitations" below. If you need to audit a specific number
end-to-end from raw predictions rather than trusting the shipped intermediate CSV, that requires
the full retrain path (below) plus the analysis scripts under `scripts/analysis_*.py` /
`scripts/gate_*.py`, which are included for reference but expect those large intermediate dumps.

### 2. Full retrain: backbones + DynaPatch + all 6 baselines, from data

Runs entirely LOCALLY — no external checkout needed. One entrypoint per (dataset, backbone,
seed) setting:

```bash
uv sync
bash scripts/reproduce_all.sh <dataset> <backbone> <seed> [outdir]
# e.g. bash scripts/reproduce_all.sh gtsrb resnet50 101 outputs/repro_run
```

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
settings x 3 seeds. `scripts/prepare_lisa_classification.py` / `prepare_tt100k_classification.py`
convert each dataset's raw detection/annotation format into the `ImageFolder` classification
crops `data/{lisa_signs_clf,tt100k_signs_clf}` expects, if starting from raw source data instead
of the shipped crops.

**Known, disclosed gap** (not silently smoothed over): `train_head_repair_baseline.py`'s
`train_loop.batch_size` for HeadFT/FullFT/DistRep-original was originally read from a resolved
config produced by an earlier training run; that specific resolved-config file's provenance
could not be traced with certainty, so this repo falls back to `configs/v8_source/<ds>/<bb>/
train.yaml`'s own `train_loop.batch_size` (a real, in-repo number) instead of guessing. This is
disclosed rather than silently assumed to be bit-identical to the paper's shipped numbers for
this one hyperparameter — set `BATCH=<value>` to override if you know the original value. See
"Known limitations" below.

## Method code

`src/models/dynapatch/` (hypernetwork, patch operator, router, deployment gate, prototype/repair
banks), `src/experiment/{train_stage3.py,deploy_eval.py,stage3.py,runner.py}` (training/eval
loops), `src/baselines/{arachne_de.py,distrep_pso.py,head_repair.py,arachne.py}` (baseline
implementations), `src/data/`, `src/training/`, `src/evaluation/`.

Training drivers: `scripts/train_backbone.py` (Hydra-composed, `configs/{dataset,model,train,
runtime}/`), `scripts/train_head_repair_baseline.py` (HeadFT/FullFT/DistRep-original, flat
`--config` + `--mode`), `scripts/train_arachne_baseline.py` (TopKSearch), `scripts/run_arachne_de.py`
/ `scripts/run_distrep_pso.py` (Arachne(DE)/DistRep(PSO)), `scripts/run_resolved_experiment.py`
(DynaPatch train/deploy).

## Data

`data/{gtsrb,tt100k_signs_clf,lisa_signs_clf}` — GTSRB (torchvision standard layout) and two
ImageFolder-format traffic-sign classification crops. `artifacts/bug_sets/v8_splits_seed{101,202,303}/`
carries the per-setting, per-seed JSON index manifests defining which images are
bug_train/bug_eval/clean_calib/clean_eval (required to reproduce exactly which images are
"bugs", independent of the raw image data). `artifacts/risk/*_safety_risk_matrix.json` defines
per-dataset critical-class sets used by the analysis scripts.

**`data/` is not tracked by git in this repository** (`.gitignore`) — ~680 MB across ~75k image
files, which is impractical to commit and is redistribution of third-party-licensed data (see
"License" below) rather than this project's own artifact. GTSRB can be re-fetched automatically
(`torchvision.datasets.GTSRB(..., download=True)`); TT100K-Signs and LISA-Signs need to be
prepared into the `{train,test}/<class>/*.jpg` `ImageFolder` layout `src/data/factory.py`
expects. If this checkout already has `data/` populated (e.g. you received it alongside this
repo), nothing further is needed for the full-retrain path.

## Setup

```bash
uv sync
```

### GPU setup

`torch`/`torchvision` are pinned to CUDA 12.4 builds (works with any NVIDIA driver reporting
CUDA Version >= 12.4 in `nvidia-smi`) rather than the newest available wheel, specifically
because an unpinned install silently falls back to CPU (`torch.cuda.is_available() == False`,
no error) on machines with an older driver — this was caught on this artifact's own test
machine. To pin a specific physical GPU: `export CUDA_VISIBLE_DEVICES=0` (the retrain scripts
also default `DEVICE=cuda:0`, override with `DEVICE=cuda:<n>` or `--device` as shown above).

## License

Code is released under the MIT License (`LICENSE`). The bundled datasets (GTSRB, TT100K-Signs,
LISA-Signs) remain under their **original licenses and terms**; redistribution here is for
reproduction convenience only and does not relicense them.

## Known limitations of this artifact (read before filing an issue)

1. **Large intermediate/raw prediction dumps are not shipped.** The GPU-free path reproduces the
   paper's tables from final per-cell CSVs, not from raw per-sample predictions. `scripts/analysis_*.py`,
   `scripts/gate_*.py` and similar are included for methodological transparency (they show exactly
   how each shipped CSV was computed) but will not run standalone without those larger dumps.
2. **Frozen backbone checkpoints (~2.2 GB) are not shipped as binaries.** `artifacts/checkpoints/manifest.json`
   / `MANIFEST.md` document what each checkpoint is; `scripts/validate_assets.py` will correctly
   report them as missing until they are supplied or retrained (`scripts/reproduce_all.sh` trains
   a missing backbone automatically).
3. **HeadFT/FullFT/DistRep-original's exact `train_loop.batch_size` is a documented fallback, not
   a verified-original value.** See "Full retrain" above. This affects gradient step count for
   those 3 non-headline-method training runs only; it does not affect DynaPatch, Arachne(DE),
   DistRep(PSO), NNPatch, or PatchNAS.
4. **TT100K-Signs / LISA-Signs raw source data and their exact download locations are not
   documented here.** `scripts/prepare_{lisa,tt100k}_classification.py` convert an already-
   downloaded raw annotation format into the classification crops this repo ships pre-built
   under `data/`; if you need to regenerate those crops from scratch rather than use the shipped
   ones, you must separately obtain the LISA Traffic Sign Dataset and TT100K raw data
   (their standard public releases), matching the raw layout those two prepare scripts expect
   (`--raw-root`, defaulting to `data/lisa_raw` / `data/tt100k_raw`).
