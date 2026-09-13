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

### 2. Full retrain: DynaPatch's own training + 2 of 6 baselines, from data + frozen backbones

Self-contained in this repo, given the shipped `data/` and a GPU:

```bash
# DynaPatch (DPGen + DPGate), one setting/seed at a time
uv run python scripts/run_resolved_experiment.py --config configs/v8_source/<dataset>/<backbone>/train.yaml
uv run python scripts/run_resolved_experiment.py --config configs/v8_source/<dataset>/<backbone>/deploy.yaml

# Real Arachne (our PyTorch differential-evolution re-implementation) and real DistRep (PSO)
uv run python scripts/run_arachne_de.py ...
uv run python scripts/run_distrep_pso.py ...
```

`scripts/run_mainline_all.sh`, `scripts/queue_mainline_baselines.sh`, `scripts/queue_arachne_de.sh`,
`scripts/queue_distrep_pso.sh` orchestrate these across the 12 settings x 3 seeds.

Frozen backbone checkpoints (12 = 3 datasets x 4 backbones) are needed as the starting point for
this path and are **not shipped in this repository** (~2.2 GB total; see "Known limitations").
Retraining the 4 backbones themselves, and running the other 4 baselines (HeadFT, FullFT,
DistrRep's original full-fine-tune variant, and the greedy "TopKSearch" baseline that predates
the real Arachne(DE) re-implementation and must never be called "Arachne" — see
`scripts/run_fewshot_arachne.sh`'s header), depend on driver scripts (`train_backbone.py`,
`prepare_*_classification.py`, `train_head_repair_baseline.py`, `train_arachne_baseline.py`)
that live in a separate, non-anonymized sibling repository and are **not included here** (see
"Known limitations"). `scripts/run_fewshot_{arachne,distrep,headrepair}.sh` and
`scripts/queue_real_baselines.sh` require `DYNAPATCH_BASELINE_REPO` to point at a checkout
providing those scripts; they fail fast with an explanatory message if it is unset.

## Method code

`src/models/dynapatch/` (hypernetwork, patch operator, router, deployment gate, prototype/repair
banks), `src/experiment/{train_stage3.py,deploy_eval.py,stage3.py,runner.py}` (training/eval
loops), `src/baselines/{arachne_de.py,distrep_pso.py,head_repair.py}` (baseline
implementations), `src/data/`, `src/training/`, `src/evaluation/`.

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

## License

Code is released under the MIT License (`LICENSE`). The bundled datasets (GTSRB, TT100K-Signs,
LISA-Signs) remain under their **original licenses and terms**; redistribution here is for
reproduction convenience only and does not relicense them.

## Known limitations of this artifact (read before filing an issue)

1. **Large intermediate/raw prediction dumps are not shipped.** The GPU-free path reproduces the
   paper's tables from final per-cell CSVs, not from raw per-sample predictions. `scripts/analysis_*.py`,
   `scripts/gate_*.py` and similar are included for methodological transparency (they show exactly
   how each shipped CSV was computed) but will not run standalone without those larger dumps.
2. **Frozen backbone checkpoints (~2.2 GB) are not shipped.** `artifacts/checkpoints/manifest.json`
   / `MANIFEST.md` document what each checkpoint is; `scripts/validate_assets.py` will correctly
   report them as missing until they are supplied.
3. **4 of 6 baselines' training drivers are not included.** HeadFT, FullFT, the DistrRep
   full-fine-tune variant, and the (mislabeled, non-headline) "TopKSearch" baseline are driven by
   scripts that live only in a separate, non-anonymized sibling repository. Only Arachne(DE) and
   DistRep(PSO) — the two real, paper-headline baseline re-implementations — are fully
   self-contained here.
Item 3 above is the most significant open gap in this artifact and should be resolved (either by
porting the missing drivers, or by scoping the paper's reproducibility claim explicitly to the
two paths above) before this repository is finalized as the camera-ready artifact link.
