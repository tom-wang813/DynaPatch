# Environment

Runtime environment for the anonymized DynaPatch reproduction repository.

## Python runtime

- Python: `>=3.10`
- Environment tool: `uv` (metadata in `pyproject.toml`)
- Dependencies: torch, torchvision, numpy, pandas, scipy, scikit-learn, omegaconf, pyyaml,
  pillow, matplotlib, seaborn, psutil — see `pyproject.toml`. No Hydra (the one script that used
  it, `main.py`, was dead code depending on a config file that did not exist, and is not
  included here).

## Setup

```bash
uv sync
```

## GPU-free path (recompute paper tables)

```bash
uv run python scripts/paper_latex_tables.py --outdir note/tables
```

Reads only the small CSV/JSON files already present under `outputs/` (a few MB total). CPU-only,
seconds to run.

## GPU path (retrain)

- DynaPatch (DPGen+DPGate) training/deploy-eval: `scripts/run_resolved_experiment.py` with
  `configs/v8_source/<dataset>/<backbone>/{train,deploy}.yaml`. Device selected via
  `runtime.device` in the config or `CUDA_VISIBLE_DEVICES`.
- Arachne(DE) / DistRep(PSO): `scripts/run_arachne_de.py` / `scripts/run_distrep_pso.py`.
- HeadFT / FullFT / DistrRep-fullFT / TopKSearch: require `DYNAPATCH_BASELINE_REPO` (external,
  not shipped here — see README.md "Known limitations").

Training/deploy-eval needs a starting frozen backbone checkpoint per (dataset, backbone); these
binaries are not shipped in this repository (~2.2 GB total across 12 settings) — see
`artifacts/checkpoints/MANIFEST.md`.

## Data expectations

- `data/gtsrb` — standard torchvision `GTSRB` layout (`torchvision.datasets.GTSRB`).
- `data/tt100k_signs_clf`, `data/lisa_signs_clf` — `ImageFolder` layout:
  `{train,test}/<class_name>/*.jpg`.
- `artifacts/bug_sets/v8_splits_seed{101,202,303}/<dataset>_<backbone>/` — JSON index manifests
  (`*_bug_train_indices.json`, `*_bug_eval_indices.json`, `*_clean_calib_indices.json`,
  `*_clean_eval_indices.json`, ...) recording exactly which images are in which split, per seed.
  Required for exact reproducibility; not re-derivable from the raw images alone.

## Reproduction basis

Training/deploy protocol: bug set shuffled before a 50/50 `S_repair`/`S_held` split, clean set
split into calibration/eval, `clean_eval` wired in every config, aggregated as mean/median over
seeds 101/202/303. The gate's reported operating point is `theta=0` (the model's own natural
class decision — score `P(gain=+1) - P(gain=-1) > 0`), not a target-`r` threshold search; see
`scripts/names.py:GATE_NATURAL_POINT` and `scripts/gate_protocol_b.py`.
