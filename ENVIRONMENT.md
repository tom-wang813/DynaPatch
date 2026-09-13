# Environment

Runtime environment for the anonymized DynaPatch reproduction repository.

## Python runtime

- Python: `>=3.10`
- Environment tool: `uv` (metadata in `pyproject.toml`)
- Dependencies: torch, torchvision, numpy, pandas, scipy, scikit-learn, omegaconf, pyyaml,
  pillow, matplotlib, seaborn, psutil, hydra-core — see `pyproject.toml`. Hydra is used only by
  `scripts/train_backbone.py` (`configs/{config.yaml,dataset,model,train,runtime}/`); every
  other training/deploy script uses a flat `--config <path.yaml>` + `--overrides` convention
  instead (`scripts/run_resolved_experiment.py`, `train_head_repair_baseline.py`,
  `train_arachne_baseline.py`).

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

Everything below runs locally; no external checkout is required.

```bash
bash scripts/reproduce_all.sh <dataset> <backbone> <seed> [outdir]
```

trains a missing frozen backbone (`scripts/train_backbone.py`, shared across the 3 repair
seeds), then DynaPatch train+deploy (`scripts/run_resolved_experiment.py`), then all 6 of the
paper's baselines: Arachne(DE)/DistRep(PSO) (`run_arachne_de.py`/`run_distrep_pso.py`, real
re-implementations), HeadFT/FullFT/DistRep-original (`train_head_repair_baseline.py --mode ...`),
and TopKSearch (`train_arachne_baseline.py`, NOT real Arachne). NNPatch/PatchNAS run separately
(`dump_prior_features.py` + `baseline_prior_patches.py`) since they operate across settings, not
per-setting. Device selected via `runtime.device`/`--device`/`DEVICE` env var depending on script.

Backbone checkpoints (~2.2 GB across 12 settings) are not shipped as binaries — see
`artifacts/checkpoints/MANIFEST.md`; `reproduce_all.sh` trains a missing one automatically.

Known, disclosed caveat: HeadFT/FullFT/DistRep-original's `train_loop.batch_size` falls back to
`configs/v8_source/<ds>/<bb>/train.yaml`'s own value (a real, in-repo number) rather than a
value whose original provenance could not be traced — see README.md "Known limitations". Set
`BATCH=<value>` to override.

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
