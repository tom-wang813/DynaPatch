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

**This README covers reproducing the paper's reported tables from the shipped
checkpoints/results — no GPU required.** If you want to retrain everything from scratch
(backbones, DynaPatch, and all 6 baselines), see **[TRAINING.md](TRAINING.md)** instead.

## Setup

```bash
uv sync
```

### GPU setup (only needed if you also use checkpoints for inference, e.g. deploy-eval)

`torch`/`torchvision` are pinned to CUDA 12.4 builds (works with any NVIDIA driver reporting
CUDA Version >= 12.4 in `nvidia-smi`) rather than the newest available wheel, specifically
because an unpinned install silently falls back to CPU (`torch.cuda.is_available() == False`,
no error) on machines with an older driver — this was caught on this artifact's own test
machine. To pin a specific physical GPU: `export CUDA_VISIBLE_DEVICES=0`.

## Reproduce the paper's tables (GPU-free)

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
**not shipped here**. `scripts/analysis_*.py` / `scripts/gate_*.py` are included for
methodological transparency (they show exactly how each shipped CSV was computed) but will not
run standalone without those larger dumps.

## Using checkpoints directly

Frozen backbone checkpoints (~2.2 GB) are **not shipped as binaries** in this repository —
`artifacts/checkpoints/manifest.json` / `MANIFEST.md` document what each checkpoint is, and
`scripts/validate_assets.py` reports what is present vs missing:

```bash
uv run python scripts/validate_assets.py
```

If you have obtained the checkpoints separately (e.g. alongside this repository) and placed
them at the paths `artifacts/checkpoints/manifest.json` names, you can re-run deploy-time
evaluation directly against a trained DynaPatch checkpoint without retraining anything:

```bash
uv run python scripts/run_resolved_experiment.py \
  --config configs/v8_source/<dataset>/<backbone>/deploy.yaml \
  --output-root <outdir> \
  --deployment-checkpoint-path <path-to-repair_best.pt> \
  --overrides runtime.device=cuda:0
```

This writes fresh `repair_holdout_unseen_predictions.csv` / `clean_eval_predictions.csv` files
you can recompute RR/Reg/CReg from directly. If you don't have checkpoints and want to produce
them yourself, see **[TRAINING.md](TRAINING.md)**.

## Method code

`src/models/dynapatch/` (hypernetwork, patch operator, router, deployment gate, prototype/repair
banks), `src/experiment/{train_stage3.py,deploy_eval.py,stage3.py,runner.py}` (training/eval
loops), `src/baselines/{arachne_de.py,distrep_pso.py,head_repair.py,arachne.py}` (baseline
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
"License" below) rather than this project's own artifact. See TRAINING.md for how to obtain it
if it is not already present alongside this checkout.

## License

Code is released under the MIT License (`LICENSE`). The bundled datasets (GTSRB, TT100K-Signs,
LISA-Signs) remain under their **original licenses and terms**; redistribution here is for
reproduction convenience only and does not relicense them.
