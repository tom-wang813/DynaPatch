# Environment

Runtime environment for the anonymized DynaPatch reproduction repository. For setup and
reproduction commands, see [README.md](README.md) (GPU-free tables, checkpoints) and
[TRAINING.md](TRAINING.md) (full retrain).

## Python runtime

- Python: `>=3.10`
- Environment tool: `uv` (metadata in `pyproject.toml`)
- Dependencies: torch, torchvision, numpy, pandas, scipy, scikit-learn, omegaconf, pyyaml,
  pillow, matplotlib, seaborn, psutil, hydra-core — see `pyproject.toml`. Hydra is used only by
  `scripts/train_backbone.py` (`configs/{config.yaml,dataset,model,train,runtime}/`); every
  other training/deploy script uses a flat `--config <path.yaml>` + `--overrides` convention
  instead (`scripts/run_resolved_experiment.py`, `train_head_repair_baseline.py`).

## Data expectations

- `data/gtsrb` — standard torchvision `GTSRB` layout (`torchvision.datasets.GTSRB`).
- `data/tt100k_signs_clf`, `data/lisa_signs_clf` — `ImageFolder` layout:
  `{train,test}/<class_name>/*.jpg`.
- `artifacts/bug_sets/shuffled_split_seed{101,202,303}/<dataset>_<backbone>/` — JSON index manifests
  (`*_bug_train_indices.json`, `*_bug_eval_indices.json`, `*_clean_calib_indices.json`,
  `*_clean_eval_indices.json`, ...) recording exactly which images are in which split, per seed.
  Required for exact reproducibility; not re-derivable from the raw images alone.

## Sanity checks

```bash
uv run python scripts/check_environment.py
uv run python scripts/validate_assets.py --fail-if-missing
uv run python scripts/check_seed_split_leak.py \
  artifacts/bug_sets/shuffled_split_seed101 \
  artifacts/bug_sets/shuffled_split_seed202 \
  artifacts/bug_sets/shuffled_split_seed303
```

## Reproduction basis

Training/deploy protocol: bug set shuffled before a 50/50 `S_repair`/`S_held` split, clean set
split into calibration/eval, `clean_eval` wired in every config, aggregated as mean/median over
seeds 101/202/303. The gate's reported operating point is `theta=0` (the model's own natural
class decision — score `P(gain=+1) - P(gain=-1) > 0`), not a target-`r` threshold search; see
`scripts/names.py:GATE_NATURAL_POINT` and `scripts/gate_protocol_b.py`.
