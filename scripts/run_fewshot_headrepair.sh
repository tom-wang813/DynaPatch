#!/usr/bin/env bash
# run_fewshot_headrepair.sh -- the SAME-SCENARIO rivals we never ran.
#
# DistRep (run_fewshot_distrep.sh) edits every weight, so it plays a different game: it is a
# development-time upper bound, not a rival under frozen deployment. `head_only` and
# `last_layer_delta` are the cheap repairs that ARE available when the backbone is frozen, and
# they are the honest test of whether a conditioned hypernet patch earns its complexity.
#
# Same cross-repo trick as run_fewshot_distrep.sh: executed in the ORIGINAL repo, every data
# path pointed at the review repo's splits/subsamples, output written back to the review repo.
# Hyperparameters mirror the DistRep column (12 epochs, ce clean replay @1.0) so the three
# baselines differ only in WHICH weights they are allowed to touch.
#
# Usage: bash scripts/run_fewshot_headrepair.sh <ds>/<bb> "<seeds>" "<ks>" <head_only|last_layer_delta>
#
# Env overrides (added 2026-07-29 for the coupling-vs-capacity control; same convention as
# run_fewshot_safepatch.sh). Defaults reproduce the shipped baseline exactly, so an unset
# environment leaves every existing result bit-identical:
#   CR_WEIGHT  clean-replay weight        (default 1.0; 0.0 removes the preservation term)
#   EPOCHS     training budget            (default 12)
#   OUT_TAG    suffix on the output tree  (default empty -- ALWAYS set it for an ablation,
#              otherwise the run overwrites the mainline baseline)
set -euo pipefail
RV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIG="${DYNAPATCH_BASELINE_REPO:?Set DYNAPATCH_BASELINE_REPO to a checkout that provides train_head_repair_baseline.py and the resolved baseline configs under outputs/baselines_rq4/ -- this driver script is not self-contained in this anonymized repro repo, see README.md}"
SETTING="$1"; SEEDS="$2"; KS="$3"; MODE="${4:-head_only}"
CR_WEIGHT="${CR_WEIGHT:-1.0}"; EPOCHS="${EPOCHS:-12}"; OUT_TAG="${OUT_TAG:-}"
# BATCH (added 2026-07-29): overrides train_loop.batch_size. Empty = whatever the DistRep resolved
# config says, which is what every recorded baseline used. This exists because matching EPOCHS is
# not the same as matching training: ours ships batch 8-32 while the baselines inherit 32-64, so
# at equal epochs we take 2-4x more gradient steps in all 12 settings. Aligning the budget means
# aligning STEPS, not epochs.
BATCH="${BATCH:-}"
case "$MODE" in
  head_only)        TAG="headonly" ;;
  last_layer_delta) TAG="lastdelta" ;;
  *) echo "unsupported mode: $MODE (use head_only | last_layer_delta)" >&2; exit 2 ;;
esac
ds="${SETTING%/*}"; bb="${SETTING#*/}"
cd "$ORIG"
PY=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_ALLOC_CONF=expandable_segments:True
src="outputs/baselines_rq4/${ds}/${bb}/distrrep/config_resolved.yaml"
train_bs="$($PY -c "from omegaconf import OmegaConf;print(int(OmegaConf.load('$src').train_loop.batch_size))")"

for seed in $SEEDS; do
  d="$RV/artifacts/bug_sets/v8_splits_seed${seed}/${ds}_${bb}"
  for k in $KS; do
    if [ "$k" = "full" ]; then sub="${d}/${ds}_bug_train_indices.json"
    else sub="$RV/outputs/fewshot_v8_splits/s${seed}/${ds}_${bb}_k${k}/${ds}_bug_train_indices.json"; fi
    # A missing subsample file means the ours-side run never happened for this cell; skipping
    # silently is what hid the empty k=full column once already, so make it loud.
    [ -f "$sub" ] || { echo "[skip missing sub] s${seed} k${k} ${ds}/${bb}"; continue; }
    out="$RV/outputs/fewshot_${TAG}${OUT_TAG}_v8_s${seed}_k${k}/${ds}/${bb}"
    if [ -f "$out/predictions/repair_holdout_unseen_predictions.csv" ]; then
      echo "[skip done] s${seed} k${k}"; continue; fi
    extra=()
    [ "$ds" = tt100k_signs ] && [ "$bb" = vgg16 ] && \
      extra+=("model.checkpoint_path=outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt")
    echo "=== [${TAG}] s${seed} k${k} ${ds}/${bb} @ $(date -Is) ==="
    $PY scripts/train_head_repair_baseline.py \
      --config "$src" --output-root "$out" \
      --mode "$MODE" --lr 1e-4 --clean-replay-weight "$CR_WEIGHT" --clean-replay-objective ce \
      --epochs "$EPOCHS" --num-workers 8 --clean-replay-subset-size 2048 --clean-replay-batch-limit "$train_bs" \
      --overrides \
        "data.bug_indices_path=${d}/${ds}_bug_indices.json" \
        "data.bug_eval_indices_path=${d}/${ds}_bug_eval_indices.json" \
        "data.bug_train_indices_path=${sub}" \
        "data.bug_val_indices_path=${sub}" \
        "data.clean_eval_indices_path=${d}/${ds}_clean_test_indices.json" \
        ${BATCH:+"train_loop.batch_size=${BATCH}"} \
        "runtime.device=cuda:0" "${extra[@]}" >/dev/null 2>&1 \
      && echo "  [done] s${seed} k${k}" || echo "  [FAIL] s${seed} k${k}"
  done
done
echo "${TAG} FEWSHOT COMPLETE ${SETTING} @ $(date -Is)"
