#!/usr/bin/env bash
# run_fewshot_headrepair.sh -- the SAME-SCENARIO rival we never ran.
#
# DistRep(PSO) (run_distrep_pso.py) edits every weight, so it plays a different game: it is a
# development-time upper bound, not a rival under frozen deployment. `head_only` is the cheap
# repair that IS available when the backbone is frozen, and it is the honest test of whether a
# conditioned hypernet patch earns its complexity.
#
# Runs entirely LOCALLY (no external checkout needed): every data path already lives in this
# repo (artifacts/bug_sets/, configs/shuffled_split_source/). Hyperparameters mirror the DistRep column
# (12 epochs, ce clean replay @1.0) so the two baselines differ only in WHICH weights they are
# allowed to touch.
#
# Usage: bash scripts/run_fewshot_headrepair.sh <ds>/<bb> "<seeds>" "<ks>" head_only
#
# Env overrides (same convention as run_fewshot_safepatch.sh). Defaults reproduce the shipped
# baseline exactly, so an unset environment leaves every existing result bit-identical:
#   CR_WEIGHT  clean-replay weight        (default 1.0; 0.0 removes the preservation term)
#   EPOCHS     training budget            (default 12)
#   OUT_TAG    suffix on the output tree  (default empty -- ALWAYS set it for an ablation,
#              otherwise the run overwrites the mainline baseline)
#   BATCH      overrides train_loop.batch_size. KNOWN GAP: the original run read this from a
#              DistRep resolved-config file produced by an earlier training run in a
#              (non-anonymized) sibling repository, which is not part of this artifact and whose
#              exact value could not be recovered. Default here falls back to
#              configs/shuffled_split_source/<ds>/<bb>/train.yaml's own train_loop.batch_size -- this is a
#              real, in-repo number, but is NOT verified to be bit-identical to the batch size
#              used for the paper's shipped baseline numbers. Set BATCH explicitly if you know
#              the original value.
set -euo pipefail
RV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RV"
SETTING="$1"; SEEDS="$2"; KS="$3"; MODE="${4:-head_only}"
CR_WEIGHT="${CR_WEIGHT:-1.0}"; EPOCHS="${EPOCHS:-12}"; OUT_TAG="${OUT_TAG:-}"
BATCH="${BATCH:-}"
case "$MODE" in
  head_only) TAG="headonly" ;;
  *) echo "unsupported mode: $MODE (use head_only)" >&2; exit 2 ;;
esac
ds="${SETTING%/*}"; bb="${SETTING#*/}"
PY="${PY:-uv run python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_ALLOC_CONF=expandable_segments:True
src="configs/shuffled_split_source/${ds}/${bb}/train.yaml"
[ -f "$src" ] || { echo "no config for ${ds}/${bb}: $src" >&2; exit 1; }
train_bs="${BATCH:-$($PY -c "from omegaconf import OmegaConf;print(int(OmegaConf.load('$src').train_loop.batch_size))")}"

for seed in $SEEDS; do
  d="$RV/artifacts/bug_sets/shuffled_split_seed${seed}/${ds}_${bb}"
  for k in $KS; do
    if [ "$k" = "full" ]; then sub="${d}/${ds}_bug_train_indices.json"
    else sub="$RV/outputs/fewshot_shuffled_splits/s${seed}/${ds}_${bb}_k${k}/${ds}_bug_train_indices.json"; fi
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
