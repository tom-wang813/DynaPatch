#!/usr/bin/env bash
# run_fewshot_arachne.sh -- "TopKSearch": greedy top-k last-layer weight search, few-shot. Runs
# entirely LOCALLY (no external checkout needed): every data path already lives in this repo.
#
# WARNING: THIS IS NOT ARACHNE. The file name and the output tree keep the old name only so
# existing results stay addressable. src/baselines/arachne.py calls itself "Arachne-STYLE" and
# shares neither of Arachne's two core components: fault localisation is |grad| top-k rather
# than Arachne's bidirectional gradient-loss x forward-impact criterion, and the repair search
# is greedy coordinate descent (3 rounds) rather than Differential Evolution; it is also
# confined to the final linear layer. Its measured RR is 0.000-0.100 across all 12 settings.
# Arachne is by Sohn/Kang/Yoo (KAIST, TOSEM 2022, https://github.com/coinse/arachne); this
# column must never be labelled "Arachne". The real re-implementation is
# scripts/run_arachne_de.py / src/baselines/arachne_de.py (Differential Evolution).
#
# Usage: bash scripts/run_fewshot_arachne.sh <ds>/<bb> "<seeds>" "<ks>"
set -euo pipefail
RV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RV"
SETTING="$1"; SEEDS="$2"; KS="$3"; ds="${SETTING%/*}"; bb="${SETTING#*/}"
PY="${PY:-uv run python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_ALLOC_CONF=expandable_segments:True
src="configs/v8_source/${ds}/${bb}/train.yaml"
[ -f "$src" ] || { echo "no config for ${ds}/${bb}: $src" >&2; exit 1; }

for seed in $SEEDS; do
  d="$RV/artifacts/bug_sets/v8_splits_seed${seed}/${ds}_${bb}"
  for k in $KS; do
    # k=full means the whole bug_train pool; only the sub-k points have subsample files.
    if [ "$k" = "full" ]; then sub="${d}/${ds}_bug_train_indices.json"
    else sub="$RV/outputs/fewshot_v8_splits/s${seed}/${ds}_${bb}_k${k}/${ds}_bug_train_indices.json"; fi
    [ -f "$sub" ] || { echo "[skip missing sub] s${seed} k${k}"; continue; }
    out="$RV/outputs/fewshot_arachne_v8_s${seed}_k${k}/${ds}/${bb}"
    if [ -f "$out/predictions/repair_holdout_unseen_predictions.csv" ]; then
      echo "[skip done] s${seed} k${k}"; continue; fi
    extra=()
    [ "$ds" = tt100k_signs ] && [ "$bb" = vgg16 ] && \
      extra+=("model.checkpoint_path=outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt")
    echo "=== [topksearch] s${seed} k${k} ${ds}/${bb} @ $(date -Is) ==="
    # The search seed was hard-coded to 11, so the three "seeds" only varied the data split and
    # the search itself was deterministic across them -- its variance was never measured. Tie it
    # to the experiment seed; SEARCH_SEED=11 reproduces the archived runs exactly.
    $PY scripts/train_arachne_baseline.py --config "$src" --output-root "$out" \
      --seed "${SEARCH_SEED:-$seed}" \
      --top-k 64 --rounds 3 --step-scale 0.5 --clean-tradeoff 0.25 \
      --clean-replay-subset-size 2048 --num-workers 8 \
      --overrides \
        "data.bug_indices_path=${d}/${ds}_bug_indices.json" \
        "data.bug_eval_indices_path=${d}/${ds}_bug_eval_indices.json" \
        "data.bug_train_indices_path=${sub}" \
        "data.bug_val_indices_path=${sub}" \
        "data.clean_eval_indices_path=${d}/${ds}_clean_test_indices.json" \
        "runtime.device=cuda:0" "${extra[@]}" >/dev/null 2>&1 \
      && echo "  [done] s${seed} k${k}" || echo "  [FAIL] s${seed} k${k}"
  done
done
echo "TOPKSEARCH FEWSHOT COMPLETE ${SETTING} @ $(date -Is)"
