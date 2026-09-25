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

`data/` and `artifacts/` must end up directly under the repository root (i.e. `DynaPatch/data/`,
`DynaPatch/artifacts/`, not nested any deeper) -- extract both archives there:

```bash
cd /path/to/DynaPatch   # repository root
tar -xzf data.tar.gz
tar -xzf artifacts.tar.gz
```

This produces `data/{gtsrb,tt100k_signs_clf,lisa_signs_clf}/` and
`artifacts/{checkpoints,bug_sets,risk}/`. All three datasets are included in `data.tar.gz`; nothing
is downloaded at run time.

The repository is now ready for either checkpoint-based reproduction or training from scratch.

---

## Reproduce from Released Checkpoints

All table reproduction uses seed `101` and the 12 dataset/backbone settings
(`gtsrb`, `tt100k_signs`, `lisa_signs` x `resnet50`, `convnext_tiny`, `densenet121`, `vgg16`).

### Evaluate a single setting

```bash
uv run python scripts/checkpoint_eval/dynapatch.py \
    --dataset gtsrb --backbone resnet50 \
    --mode checkpoint --output-root outputs/repro/ckpt_eval/DynaPatch --gate
```

`--gate` additionally applies the released DPGate (`artifacts/checkpoints/gates/`) and writes the
gated results next to the ungated ones; without it only the ungated patch is scored.

Same shape for the baselines: `headft.py --baseline {full_ft,head_ft}`, `arachne.py`,
`distrep.py`, `nnpatch.py`, `patchnas.py`. `--mode checkpoint` (default) scores the released
baseline checkpoint; `--mode train` retrains the baseline instead. `fixedpatch.py` has no released
checkpoint -- train it first (step 1 below).

### Reproduce the paper's tables

Run the steps **in this order** -- later steps read the outputs of earlier ones, and a missing
input is skipped silently rather than reported as an error.

```bash
# 1. Train the FixedPatch ablation (no released checkpoint). Must precede step 3,
#    otherwise RQ1 skips FixedPatch. Use --dataset X --backbone Y instead of --all for one setting.
uv run python scripts/repro/rq2_train_fixedpatch.py --all

# 2. Dump FixedPatch's patch logits for RQ2 (one run per setting).
for ds in gtsrb tt100k_signs lisa_signs; do
  for bb in resnet50 convnext_tiny densenet121 vgg16; do
    uv run python scripts/checkpoint_eval/fixedpatch.py --dataset $ds --backbone $bb \
        --mode checkpoint --output-root outputs/repro/ckpt_eval/FixedPatch_routefeat \
        --dump-route-features
  done
done

# 3. RQ1: all 12 settings x all methods. Also writes outputs/effect_dump_v8_s101/,
#    which steps 4-6 read. Use --settings gtsrb/resnet50 for one setting.
uv run python scripts/repro/rq1_aggregate.py

# 4-6. RQ2-RQ4
uv run python scripts/repro/rq2_norm_and_direction.py
uv run python scripts/repro/rq3_fit_eval_gate.py
uv run python scripts/repro/rq4_sweep_gate_lambda.py
```

Outputs, all under `outputs/repro/` (raw per-setting numbers; the paper tables aggregate them):

| Step | Paper | Output file |
|---|---|---|
| `rq1_aggregate.py` | RQ1 (repair/regression per method), RQ2 ungated comparison | `rq1_raw_cells.csv` |
| `rq2_norm_and_direction.py` | RQ2 (patch norm/alignment, direction/magnitude reassignment) | `rq2_norm_direction_raw.json` |
| `rq3_fit_eval_gate.py` | RQ3 (DPGate vs. input-only gate) | `rq3_raw_s101.json` |
| `rq4_sweep_gate_lambda.py` | RQ4 (gate lambda sweep) | `rq4_raw.json` |

Per-method intermediate results are kept under `outputs/repro/ckpt_eval/<method>/`.

---

## Train DynaPatch from Scratch

For reproducing DynaPatch end to end (using the shipped frozen backbones), not just scoring
shipped checkpoints — trains DPGen, fits DPGate, evaluates ungated vs. gated:

```bash
uv run python scripts/repro/train_and_eval_dynapatch.py --dataset gtsrb --backbone resnet50 --seed 101
```

- `--dataset`: `gtsrb`, `tt100k_signs`, `lisa_signs`
- `--backbone`: `resnet50`, `convnext_tiny`, `densenet121`, `vgg16`
- `--seed`: `101`, `202`, or `303` (the only seeds with full split manifests)

Same command works for the other 11 dataset/backbone settings by swapping `--dataset`/`--backbone`.

**Output:** the trained checkpoint is saved to
`artifacts/checkpoints/dynapatch/<dataset>_<backbone>_s<seed>/repair_best.pt` (the same canonical
path `checkpoint_eval/dynapatch.py` reads from), and a summary is written to
`outputs/repro/dynapatch_fromscratch_<dataset>_<backbone>_s<seed>.json` with two parts: `ungated`
(RR/Reg/CReg with the patch always applied) and `gate` (the fitted DPGate's own classification
metrics plus gated RR/Reg/CReg), also printed to stdout.

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
└── data/                       # downloaded: gtsrb/, tt100k_signs_clf/, lisa_signs_clf/
```

---

## License

Code is released under the MIT License. The bundled datasets (GTSRB, TT100K-Signs, LISA-Signs)
remain under their original licenses and terms.
