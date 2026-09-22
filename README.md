# DynaPatch

**Deployment-Time Gating for Critical-Class Safety in Neural Repair**

Anonymized reproduction repository for DynaPatch (double-blind review artifact).

A deployed classifier's known bugs can be patched, but a patch that fixes one input can silently
break others that used to work — a *regression*. If a regressed input belongs to a
safety-critical class (a stop sign, say), that's worse than the bug it fixed. DynaPatch repairs a
**frozen** classifier with two parts: **DPGen**, a hypernetwork that proposes an input-conditioned
patch, and **DPGate**, a runtime gate that applies it only when doing so doesn't look like it will
create a new failure. Evaluated on 3 datasets x 4 backbones (12 settings), against 6 baselines:
HeadFT, FullFT, Arachne, DistrRep, NNPatch, PatchNAS.

## Two ways to reproduce

| Path | Needs | Gives you |
|---|---|---|
| **[Checkpoint-only](#checkpoint-only-reproduction)** | Shipped checkpoints (`artifacts/checkpoints/`) + GPU | Every RQ1–RQ4 table, scored directly from the checkpoints this repo's results depend on — no training |
| **[Train from scratch](#train-from-scratch)** | Data + a frozen backbone checkpoint (or train that too) + GPU | DynaPatch's hypernetwork trained from nothing, its gate fit fresh, ungated vs. gated RR/Reg/CReg plus the gate's own classification metrics |

These are independent, not a ladder — pick the one that matches what you have. The first is what
you want to check "does the paper's number reproduce from what's checked in"; the second is what
you want to check "can a reviewer, given only data and a frozen backbone, reproduce DynaPatch
without any of our checkpoints".

## Requirements

```bash
uv sync
```

- Python `>=3.10`, dependency list in `pyproject.toml` (torch==2.6.0 / torchvision==0.21.0, pinned
  CUDA 12.4 builds — pin a physical GPU with `export CUDA_VISIBLE_DEVICES=0`).
- GPU needed for both paths below (checkpoint-only still runs a full forward pass per setting to
  score predictions; nothing here is precomputed-CSV-only).

## Getting the data and checkpoints

`artifacts/` (checkpoints + split manifests + risk matrices, ~9 GB) and `data/` (the three image
datasets, ~940 MB) are not tracked by git — they're hosted as a Zenodo record instead. One
command downloads, verifies (sha256), and extracts both:

```bash
python3 scripts/fetch_release_data.py
```

Needs only the Python standard library, so it works before `uv sync`. Use `--only artifacts` or
`--only data` to fetch just one (e.g. skip `data/` if you already have GTSRB/TT100K/LISA
elsewhere), and `--force` to re-fetch. The record it pulls from is `ZENODO_RECORD.json` (repo
root) — see that file for the DOI.

## Data

`data/{gtsrb,tt100k_signs_clf,lisa_signs_clf}` — GTSRB (torchvision standard layout) and two
`ImageFolder`-format traffic-sign classification crops, ~680 MB across ~75k image files.
`artifacts/bug_sets/shuffled_split_seed{101,202,303}/` carries the per-setting,
per-seed JSON index manifests defining which images are bug_train/bug_eval/bug_val/clean_calib/
clean_eval/clean_test — **only these 3 seeds have complete manifests** (bug_val and clean_calib
in particular, which gate fitting needs); other seeds cannot be used with either path below.
`artifacts/risk/*_safety_risk_matrix.json` defines each dataset's critical-class set.

## Checkpoints

`artifacts/checkpoints/manifest.json` / `MANIFEST.md` inventory every checkpoint this repo's
results depend on: 12 frozen backbones (one per dataset x backbone), DynaPatch's own repair
checkpoints + gates (seed 101, plus 202/303 for a subset used in reproducibility checks), and all
6 baselines' checkpoints at seed 101 under `artifacts/checkpoints/baselines/<Method>/
<dataset>_<backbone>_s101/`.

One baseline's checkpoint was retrained after an issue was found in the shipped one: **Arachne**
originally shipped at `bound_scale=2.0` (an unswept DE-search hyperparameter — this repo's own
prior work had already flagged that `bound_scale=128` gives a materially higher repair rate,
e.g. 0.125 -> 0.375 on gtsrb/resnet50, but the fix was never carried into the checkpoint). The
checkpoints under `artifacts/checkpoints/baselines/Arachne/` are now the `bound_scale=128`
retrain; the original `bound_scale=2.0` checkpoints are kept at
`artifacts/checkpoints/baselines/Arachne_bound_scale2_backup/` for comparison, not deleted.

## Checkpoint-only reproduction

`scripts/checkpoint_eval/` scores one (method, dataset, backbone, seed) cell at a time, from a
checkpoint, using a shared CLI shape (`common.py:base_argparser` — `--dataset --backbone --seed
--mode {checkpoint,train} --output-root --device`):

| Method | Script | Notes |
|---|---|---|
| FullFT / HeadFT | `headft.py --baseline {full_ft,head_ft}` | |
| Arachne | `arachne.py` | `--mode train --bound-scale <v>` re-runs the DE search; default is now 128 (see above) |
| DistrRep | `distrep.py` | |
| NNPatch | `nnpatch.py` | see "Two operating points" below |
| PatchNAS | `patchnas.py` | same |
| FixedPatch (DynaPatch ablation) | `fixedpatch.py` | no shipped checkpoint — train first with `scripts/repro/rq2_train_fixedpatch.py` |
| DynaPatch | `dynapatch.py` | `--gate` also applies the shipped `artifacts/checkpoints/gates/<ds>_<bb>_s101.json` |

Example, one cell:

```bash
uv run python scripts/checkpoint_eval/dynapatch.py --dataset gtsrb --backbone resnet50 \
    --mode checkpoint --output-root outputs/ckpt_eval --gate
```

### Two operating points for NNPatch/PatchNAS

These two baselines have their own error estimator that decides, per input, whether to apply the
patch. That estimator is scored at **two different thresholds** depending on which table you're
reproducing, and the two give very different numbers for the same checkpoint — this is expected,
not a bug:

- **RQ1** (`Table rq1_rr`/`rq1_summary`): `tau=0.5`, the estimator's own natural decision boundary
  — paper.tex describes both methods as "uses an error estimator to decide whether to apply the
  patch", with no cross-method recalibration. This is `nnpatch.py`/`patchnas.py`'s `--tau` default
  is actually `-inf` (see next point) so RQ1's `tau=0.5` value comes from the script's own
  internal `"natural"` pass, not the CLI flag.
- **RQ2** (`Table rq2_ungated_*`): unconditional/always-apply (`tau=-inf`, i.e. `route_rate=1.0`)
  — the same "no gate at all" concept as `DynaPatch-NoGate`. This IS the `--tau` CLI default.

A third value (`"matched"` in each script's `summary.json`, a Reg-budget-matched `calib_tau`) is
computed and saved but not used by any of the 9 paper tables — kept only as a diagnostic.

### Assembling the 9 tables

```bash
uv run python scripts/repro/rq1_aggregate.py                          # all 12 settings, 7 methods -> outputs/repro/rq1_raw_cells.csv
uv run python scripts/repro/rq2_train_fixedpatch.py --all             # train FixedPatch (no shipped checkpoint), needed before RQ2
uv run python scripts/repro/rq2_norm_and_direction.py                 # RQ2 direction/norm tables
uv run python scripts/repro/rq3_fit_eval_gate.py                      # RQ3 gate classification + effect tables (all 12 settings)
uv run python scripts/repro/rq4_sweep_gate_lambda.py                  # RQ4 lambda sweep
```

`rq1_aggregate.py` drives all 7 methods (subprocess per method, to keep global state like
`prior_patch_common.CRIT` from leaking across methods) and writes
`outputs/repro/{rq1_raw_cells.csv,dynapatch_reg_lookup.json}`. `rq3_fit_eval_gate.py` fits the
"pre-only" (DPInput) gate fresh on `bug_train + bug_val + clean_calib` — deliberately not
`clean_eval`, which is the population Reg/CReg get reported on; fitting and reporting on the same
population is a selection-on-the-evaluation-set leak. The "pre+post" (DPGate) row loads the
shipped `artifacts/checkpoints/gates/<ds>_<bb>_s101.json` directly (no refitting) — the script also
independently refits a pre+post gate the same way as pre-only, as an extra reproducibility check
that the fitting protocol itself works, not just that a frozen JSON loads.

`repro_tables.tex` (repo root) has all 9 tables filled in from a real run of the above, each
followed by the paper's own published numbers as a comment block for side-by-side diffing, plus a
per-table note on anything that doesn't match exactly and why (FixedPatch's 2 untrainable LISA
cells, RQ2's direction/magnitude reassignment being under-specified by the paper's main text and
so operationalized here explicitly, etc.). It is not included by paper.tex.

## Train from scratch

For a reviewer who has only the data and a frozen backbone checkpoint (no DynaPatch checkpoint, no
gate), and wants to reproduce DynaPatch — not score it — end to end:

```bash
# 0. only if the backbone itself also needs training:
uv run python scripts/train_backbone.py dataset=gtsrb model=resnet50 train=backbone_finetune runtime=local_gpu

# 1. train DynaPatch's hypernetwork, dump gate evidence, fit a gate fresh, evaluate ungated vs gated:
uv run python scripts/repro/train_and_eval_dynapatch.py --dataset gtsrb --backbone resnet50 --seed 101
```

`train_and_eval_dynapatch.py` is the single entrypoint for this path. It:

1. Trains DynaPatch's hypernetwork via `src/experiment/train_stage3.py::run_stage3_experiment`
   (the same code `scripts/run_resolved_experiment.py`'s `stage3_repair` stage calls), saves the
   checkpoint to the canonical `artifacts/checkpoints/dynapatch/<ds>_<bb>_s<seed>/repair_best.pt`.
2. Dumps gate evidence (base/patched logits over the held-out and calibration splits).
3. Fits a fresh 9-feature DPGate on that evidence (reusing `scripts/repro/rq3_fit_eval_gate.py`'s
   fitting code — not a second implementation) and reports ungated RR/Reg/CReg, gated RR/Reg/CReg,
   and the gate's own classification metrics (accuracy/precision/recall/F1).

**Only seeds 101/202/303 are supported** — the only ones with complete split manifests (see
"Data" above). `--seed 101` overwrites the shipped checkpoint at the canonical path; use 202 or
303 to train an independent copy without touching what's shipped.

Training epoch budget matters more than the stock config's default: `configs/shuffled_split_
source/<ds>/<bb>/train.yaml`'s own `train_loop` block (epochs=12, early-stop metric
`safety_balanced`) under-trains on several settings — the metric can pick epoch 1 as "best" before
the model has moved. Passing a longer budget and a repair-aware early-stop metric closes most of
the gap to the shipped checkpoint (verified at seed 202 across all 12 settings):

```bash
uv run python scripts/repro/train_and_eval_dynapatch.py --dataset gtsrb --backbone resnet50 --seed 202 \
    --override train_loop.epochs=50 \
    --override train_loop.early_stop_patience=10 \
    --override train_loop.early_stop_min_epochs=30 \
    --override train_loop.early_stop_metric=heldout_repaired \
    --force-retrain
```

`--override dot.path=value` is repeatable and applies on top of the stock config; `--force-retrain`
retrains even if a checkpoint already exists at the canonical path.

## Repository structure

```
├── src/                      method implementation
│   ├── models/dynapatch/     DPGen hypernetwork, patch operator, DPGate (gate.py), deployment policy
│   ├── experiment/           training loop (train_stage3.py), deploy-time eval (deploy_eval.py), stage3.py (shared config/bundle builder, incl. FixedPatch)
│   ├── baselines/            arachne_de.py, distrep_pso.py, head_repair.py
│   └── data/, training/, utils/
│
├── scripts/
│   ├── checkpoint_eval/      score one (method, dataset, backbone, seed) cell from a checkpoint — one script per method
│   ├── repro/                assemble RQ1-RQ4 from those scores; train-from-scratch entrypoint
│   ├── run_resolved_experiment.py   generic driver: resolve a config, apply --overrides, run stage3_repair or deploy_eval
│   ├── train_backbone.py     Hydra-composed frozen-backbone training
│   └── fetch_release_data.py   downloads+extracts artifacts/ and data/ from the Zenodo record
│
├── configs/
│   ├── config.yaml, dataset/, model/, train/, runtime/          Hydra groups, train_backbone.py only
│   └── shuffled_split_source/<dataset>/<backbone>/{train,deploy}.yaml   per-setting configs (DynaPatch + baselines)
│
├── artifacts/
│   ├── checkpoints/          manifest.json + MANIFEST.md, backbones/, baselines/<Method>/, dynapatch/, gates/
│   ├── bug_sets/shuffled_split_seed{101,202,303}/   per-seed split manifests
│   └── risk/                 *_safety_risk_matrix.json
│
├── outputs/repro/            everything scripts/repro/*.py writes — raw per-cell CSVs/JSONs, not just aggregates
├── repro_tables.tex          all 9 paper tables filled from a real run, with paper.tex's own numbers alongside for diffing
├── ZENODO_RECORD.json        DOI + per-file URL/sha256 that fetch_release_data.py reads
└── data/                     GTSRB / TT100K-Signs / LISA-Signs images, not tracked by git
```

## License

Code is released under the MIT License (`LICENSE`). The bundled datasets (GTSRB, TT100K-Signs,
LISA-Signs) remain under their **original licenses and terms**; redistribution here is for
reproduction convenience only and does not relicense them.
