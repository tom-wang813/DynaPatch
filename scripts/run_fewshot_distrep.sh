#!/usr/bin/env bash
# run_fewshot_distrep.sh -- "DistRep-original" (weight-retraining upper bound, --mode
# full_finetune_distr = plain full fine-tune, NOT the real 3-phase PSO method -- that is
# scripts/run_distrep_pso.py / src/baselines/distrep_pso.py, a separate, faithful
# re-implementation). Runs entirely LOCALLY: every data path already lives in this repo.
# DistRep-original = gradient retrain (full_finetune_distr, 12ep), no gate.
#
# KNOWN GAP: train_loop.batch_size below falls back to configs/v8_source/<ds>/<bb>/train.yaml's
# own value, since the resolved config the original run read it from (in a non-anonymized
# sibling repository) is not part of this artifact -- see README.md "Known limitations". Set
# BATCH explicitly if you know the original value.
#
# Usage: bash scripts/run_fewshot_distrep.sh <ds>/<bb> "<seeds>" "<ks>"
set -euo pipefail
RV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RV"
SETTING="$1"; SEEDS="$2"; KS="$3"
ds="${SETTING%/*}"; bb="${SETTING#*/}"
# CR_SUBSET varies how much clean data the repair is allowed to see (the access-budget axis);
# OUT_TAG keeps such a sweep out of the recorded results tree. Defaults reproduce the original run.
CR_SUBSET="${CR_SUBSET:-2048}"; OUT_TAG="${OUT_TAG:-}"
# EPOCHS added 2026-07-29: `ours` trains 40 epochs while this was hardcoded to 12, so every
# comparison so far was budget-mismatched in our favour (see note/PITFALLS.md). Default 12 keeps
# the recorded runs bit-identical; pair EPOCHS=40 with OUT_TAG so it lands in its own tree.
EPOCHS="${EPOCHS:-12}"
# CLEAN_DIR (added 2026-07-29): must mirror run_fewshot_safepatch.sh. Giving ours the site-local
# clean set while this arm keeps the global one would be an unfair comparison in our favour --
# the same class of bias as the 40-vs-12 epoch mismatch found the same day. Empty = global.
CLEAN_DIR="${CLEAN_DIR:-}"
BATCH="${BATCH:-}"
PY="${PY:-uv run python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_ALLOC_CONF=expandable_segments:True
src="configs/v8_source/${ds}/${bb}/train.yaml"
[ -f "$src" ] || { echo "no config for ${ds}/${bb}: $src" >&2; exit 1; }
train_bs="${BATCH:-$($PY -c "from omegaconf import OmegaConf;print(int(OmegaConf.load('$src').train_loop.batch_size))")}"

for seed in $SEEDS; do
  d="$RV/artifacts/bug_sets/v8_splits_seed${seed}/${ds}_${bb}"
  for k in $KS; do
    # k=full means the whole bug_train pool; only the sub-k points have subsample files.
    if [ "$k" = "full" ]; then sub="${d}/${ds}_bug_train_indices.json"
    else sub="$RV/outputs/fewshot_v8_splits/s${seed}/${ds}_${bb}_k${k}/${ds}_bug_train_indices.json"; fi
    [ -f "$sub" ] || { echo "[skip missing sub] s${seed} k${k}"; continue; }
    cd_="${d}"
    if [ -n "$CLEAN_DIR" ]; then
      cd_="$RV/$(printf '%s' "$CLEAN_DIR" | sed "s/{seed}/${seed}/g; s/{k}/${k}/g; s/{ds}/${ds}/g; s/{bb}/${bb}/g")"
      [ -f "${cd_}/${ds}_clean_test_indices.json" ] || {
        echo "[skip missing clean] s${seed} k${k} ${ds}/${bb} -> ${cd_}"; continue; }
    fi
    out="$RV/outputs/fewshot_distrep${OUT_TAG}_v8_s${seed}_k${k}/${ds}/${bb}"
    if [ -f "$out/predictions/repair_holdout_unseen_predictions.csv" ]; then
      echo "[skip done] s${seed} k${k}"; continue; fi
    extra=()
    [ "$ds" = tt100k_signs ] && [ "$bb" = vgg16 ] && \
      extra+=("model.checkpoint_path=outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt")
    echo "=== [distrep] s${seed} k${k} ${ds}/${bb} @ $(date -Is) ==="
    $PY scripts/train_head_repair_baseline.py \
      --config "$src" --output-root "$out" \
      --mode full_finetune_distr --lr 1e-4 --clean-replay-weight 1.0 --clean-replay-objective ce \
      --epochs "$EPOCHS" --num-workers 8 --clean-replay-subset-size "$CR_SUBSET" --clean-replay-batch-limit "$train_bs" \
      --overrides \
        "data.bug_indices_path=${d}/${ds}_bug_indices.json" \
        "data.bug_eval_indices_path=${d}/${ds}_bug_eval_indices.json" \
        "data.bug_train_indices_path=${sub}" \
        "data.bug_val_indices_path=${sub}" \
        "data.clean_eval_indices_path=${cd_}/${ds}_clean_test_indices.json" \
        "runtime.device=cuda:0" "${extra[@]}" >/dev/null 2>&1 \
      && echo "  [done] s${seed} k${k}" || echo "  [FAIL] s${seed} k${k}"
  done
done
echo "DISTREP FEWSHOT COMPLETE ${SETTING} @ $(date -Is)"
