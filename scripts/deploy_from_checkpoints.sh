#!/usr/bin/env bash
# Runs DynaPatch deploy-eval across all 12 settings x 3 seeds, driven entirely by
# artifacts/checkpoints/manifest.json's `experiment_checkpoints` list. Needs the 36 repair
# checkpoints obtained separately (not shipped as binaries in this repo -- see README.md
# "Using checkpoints directly"), placed at the paths manifest.json names.
#
# Skips a cell if its checkpoint is missing (so you can run this with a partial set of
# checkpoints) or if its predictions already exist (safe to re-run/resume).
#
# Also runs a second "calib" deploy-eval pass per cell (bug_eval_indices_path -> bug_val) and,
# at the end, fits DPGate over everything dumped -- same two-pass layout reproduce_all.sh uses,
# so this checkpoint-only path gets the RQ3/RQ4 gate numbers too, not just RQ1/RQ2's ungated ones.
#
# Usage: bash scripts/deploy_from_checkpoints.sh [outdir]
#   uv run python scripts/table_from_checkpoints.py   # after this, to render the summary table
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-outputs/from_checkpoints}"
RUN="${RUN:-uv run python}"
DEVICE="${DEVICE:-cuda:0}"
DUMP_TAG="${DUMP_TAG:-}"

$RUN -c "
import json
m = json.load(open('artifacts/checkpoints/manifest.json'))
for e in m['experiment_checkpoints']:
    print(e['dataset'], e['backbone'], e['seed'], e['checkpoint_file'])
" | while read -r ds bb seed ckpt; do
  out="${OUT}/${ds}/${bb}/s${seed}"
  if [ -f "${out}/deploy/predictions/clean_eval_predictions.csv" ]; then
    echo "[skip done] ${ds}/${bb} s${seed}"; continue
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[skip missing checkpoint] ${ds}/${bb} s${seed} -> ${ckpt}"; continue
  fi
  split_dir="artifacts/bug_sets/shuffled_split_seed${seed}/${ds}_${bb}"
  extra=()
  [ "$ds" = tt100k_signs ] && [ "$bb" = vgg16 ] && \
    extra+=("model.checkpoint_path=artifacts/checkpoints/backbones/tt100k_signs_vgg16/backbone_last.pt")
  gate_root="outputs/effect_dump${DUMP_TAG}_v8_s${seed}/${ds}/${bb}"
  echo "=== [deploy] ${ds}/${bb} s${seed} @ $(date -Is) ==="
  $RUN scripts/run_resolved_experiment.py \
    --config "configs/shuffled_split_source/${ds}/${bb}/deploy.yaml" \
    --output-root "${out}/deploy" \
    --deployment-checkpoint-path "$ckpt" \
    --overrides \
      "data.bug_indices_path=${split_dir}/${ds}_bug_indices.json" \
      "data.bug_train_indices_path=${split_dir}/${ds}_bug_train_indices.json" \
      "data.bug_eval_indices_path=${split_dir}/${ds}_bug_eval_indices.json" \
      "data.clean_eval_indices_path=${split_dir}/${ds}_clean_eval_indices.json" \
      "loss.lambda_robust=0.0" "loss.lambda_field=0.0" "loss.lambda_clean_replay=1.0" \
      "runtime.device=${DEVICE}" "${extra[@]}" \
    && echo "  [done] ${ds}/${bb} s${seed}" || echo "  [FAIL] ${ds}/${bb} s${seed}"
  echo "  [gate dumps] ${ds}/${bb} s${seed} -> ${gate_root}/{deploy_direct,deploy_direct_calib}"
  $RUN scripts/run_resolved_experiment.py \
    --config "configs/shuffled_split_source/${ds}/${bb}/deploy.yaml" \
    --output-root "${gate_root}/deploy_direct" \
    --deployment-checkpoint-path "$ckpt" \
    --overrides \
      "data.bug_indices_path=${split_dir}/${ds}_bug_indices.json" \
      "data.bug_train_indices_path=${split_dir}/${ds}_bug_train_indices.json" \
      "data.bug_eval_indices_path=${split_dir}/${ds}_bug_eval_indices.json" \
      "data.clean_eval_indices_path=${split_dir}/${ds}_clean_eval_indices.json" \
      "loss.lambda_robust=0.0" "loss.lambda_field=0.0" "loss.lambda_clean_replay=1.0" \
      "runtime.device=${DEVICE}" "deployment.save_route_features=true" "${extra[@]}"
  $RUN scripts/run_resolved_experiment.py \
    --config "configs/shuffled_split_source/${ds}/${bb}/deploy.yaml" \
    --output-root "${gate_root}/deploy_direct_calib" \
    --deployment-checkpoint-path "$ckpt" \
    --overrides \
      "data.bug_indices_path=${split_dir}/${ds}_bug_indices.json" \
      "data.bug_train_indices_path=${split_dir}/${ds}_bug_train_indices.json" \
      "data.bug_eval_indices_path=${split_dir}/${ds}_bug_val_indices.json" \
      "data.clean_eval_indices_path=${split_dir}/${ds}_clean_eval_indices.json" \
      "loss.lambda_robust=0.0" "loss.lambda_field=0.0" "loss.lambda_clean_replay=1.0" \
      "runtime.device=${DEVICE}" "deployment.save_route_features=true" "${extra[@]}"
done
echo "  [gate fit] fitting DPGate over every cell dumped under outputs/effect_dump${DUMP_TAG}_v8_s*/"
DUMP_TAG="$DUMP_TAG" $RUN scripts/gate_protocol_b.py --min-pos 1
echo "DEPLOY-FROM-CHECKPOINTS COMPLETE @ $(date -Is)"
