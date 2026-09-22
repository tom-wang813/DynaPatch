# DynaPatch

**Input-Specific Patching with Deployment-Time Gating for DNN Repair**

This repository provides the anonymized implementation and reproduction package for **DynaPatch**.

DynaPatch repairs a frozen DNN using two components:

* **DPGen** synthesizes an input-specific patch for each input.
* **DPGate** determines whether the synthesized patch should be applied using information from both the original and patched predictions.

DynaPatch is evaluated on three traffic-sign datasets and four DNN architectures, covering 12 dataset-architecture settings, against 6 baselines: HeadFT, FullFT, Arachne, DistrRep, NNPatch, PatchNAS.

> **Reproduction artifacts:** `artifacts/` (checkpoints) and `data/` (datasets) are not in this repo — they're hosted externally (Figshare), see [Quick Start](#quick-start).

---

## Contents

* [Quick Start](#quick-start)
* [Reproduce from Released Checkpoints](#reproduce-from-released-checkpoints)
* [Train DynaPatch from Scratch](#train-dynapatch-from-scratch)
* [Repository Structure](#repository-structure)

---

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

Requires Python >= 3.10 and a CUDA-capable GPU. To select a GPU: `export CUDA_VISIBLE_DEVICES=0`.

### 2. Download the datasets and checkpoints

Download `data.tar.gz` and `artifacts.tar.gz` from:

**[https://figshare.com/s/2cca5fdded3c08779f52](https://figshare.com/s/2cca5fdded3c08779f52)**

Extract both into the repository root:

```bash
tar -xzf data.tar.gz -C .
tar -xzf artifacts.tar.gz -C .
```

This produces `data/{tt100k_signs_clf,lisa_signs_clf}/` and `artifacts/{checkpoints,bug_sets,risk}/`.
GTSRB isn't part of the download -- it's auto-fetched by `torchvision` the first time a GTSRB
script runs.

The repository is now ready for either checkpoint-based reproduction or training from scratch.

---

## Reproduce from Released Checkpoints

### Evaluate a single setting

```bash
uv run python scripts/checkpoint_eval/dynapatch.py \
    --dataset gtsrb --backbone resnet50 \
    --mode checkpoint --output-root outputs/ckpt_eval --gate
```

Same shape for the baselines: `headft.py --baseline {full_ft,head_ft}`, `arachne.py`,
`distrep.py`, `nnpatch.py`, `patchnas.py`, `fixedpatch.py` (needs training first, see below).

### Reproduce the paper's tables

```bash
uv run python scripts/repro/rq1_aggregate.py            # RQ1: all 12 settings x 7 methods
uv run python scripts/repro/rq2_train_fixedpatch.py --all
uv run python scripts/repro/rq2_norm_and_direction.py    # RQ2
uv run python scripts/repro/rq3_fit_eval_gate.py         # RQ3
uv run python scripts/repro/rq4_sweep_gate_lambda.py     # RQ4
```

Results are written to `outputs/repro/`. `repro_tables.tex` has all 9 tables filled from a real
run of the above, next to the paper's own published numbers for side-by-side diffing.

---

## Train DynaPatch from Scratch

For reproducing DynaPatch end to end (using the shipped frozen backbones), not just scoring
shipped checkpoints — trains DPGen, fits DPGate, evaluates ungated vs. gated:

```bash
uv run python scripts/repro/train_and_eval_dynapatch.py --dataset gtsrb --backbone resnet50
```

Same command works for the other 11 dataset/backbone settings by swapping `--dataset`/`--backbone`.

---

## Repository Structure

```text
├── src/
│   ├── models/dynapatch/       # DPGen and DPGate
│   ├── experiment/             # training and evaluation
│   ├── baselines/              # baseline repair methods
│   └── data/, training/, utils/
│
├── scripts/
│   ├── checkpoint_eval/        # evaluate released checkpoints, one script per method
│   ├── repro/                  # reproduce RQ1-RQ4; train-from-scratch entrypoint
│   ├── run_resolved_experiment.py
│   └── train_backbone.py
│
├── configs/                    # experiment configurations
├── artifacts/                  # downloaded: checkpoints, split manifests, risk matrices
├── outputs/repro/              # reproduced results
└── data/                       # gtsrb/ auto-downloaded by torchvision; tt100k_signs_clf/, lisa_signs_clf/ downloaded
```

---

## License

Code is released under the MIT License. The bundled datasets (GTSRB, TT100K-Signs, LISA-Signs)
remain under their original licenses and terms.
