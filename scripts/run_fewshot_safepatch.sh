#!/usr/bin/env bash
# run_fewshot_safepatch.sh -- MAIN LINE: no-gate safe patch, few-shot.
# Trains the patch under the cr10 recipe (noaux + clean_replay=1.0 -> low CReg by construction),
# then deploys AlwaysPatch (NO gate) on S_held + S_clean^test. No feature dump (18M/run).
# The patched model IS the deployed model; safety comes from training, not from a gate.
# Usage: bash scripts/run_fewshot_safepatch.sh <ds>/<bb> "<seeds>" "<ks>"
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=".venv/bin/python"
SETTING="$1"; SEEDS="$2"; KS="$3"; ds="${SETTING%/*}"; bb="${SETTING#*/}"
CR10="loss.lambda_robust=0.0 loss.lambda_field=0.0 loss.lambda_clean_replay=1.0"
# EPOCHS overrides the training budget; OUT_TAG selects a separate output tree so a
# re-run at a different budget does not overwrite the 12-epoch results of record.
# Phase I showed epochs=12 underfits every VGG-16 setting (and tt100k/densenet121).
# EXTRA_OV appends arbitrary space-separated overrides (e.g. model.hypernet_hidden_dim=512).
# Always pair it with OUT_TAG so the ablation lands in its own output tree.
EPOCHS="${EPOCHS:-}"; OUT_TAG="${OUT_TAG:-}"; EXTRA_OV="${EXTRA_OV:-}"
# CLEAN_DIR (added 2026-07-29): where the clean calib/eval index files come from. Empty = the
# global split dir, which reproduces every recorded run bit-identically. Set it to a per-site
# directory to give a deployment site its OWN clean traffic: with the global set, a site-local
# repair is charged for disturbing images that site never sees, which holds every site-local
# method to a global safety standard when deployment only requires a local one.
CLEAN_DIR="${CLEAN_DIR:-}"
[ -n "$EPOCHS" ] && CR10="$CR10 train_loop.epochs=${EPOCHS}"
[ -n "$EXTRA_OV" ] && CR10="$CR10 $EXTRA_OV"

for seed in $SEEDS; do
  d="artifacts/bug_sets/v8_splits_seed${seed}/${ds}_${bb}"
  extra=()
  [ "$ds" = tt100k_signs ] && [ "$bb" = vgg16 ] && \
    extra+=("model.checkpoint_path=outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt")
  for k in $KS; do
    if [ "$k" = "full" ]; then sub="${d}/${ds}_bug_train_indices.json"
    else
      sub="outputs/fewshot_v8_splits/s${seed}/${ds}_${bb}_k${k}/${ds}_bug_train_indices.json"
      if [ ! -f "$sub" ]; then mkdir -p "$(dirname "$sub")"
        $PY -c "import json,random;idx=json.load(open('${d}/${ds}_bug_train_indices.json'))['indices'];o=list(idx);random.Random(${seed}).shuffle(o);json.dump({'indices':sorted(o[:${k}])},open('${sub}','w'))"
      fi
    fi
    # cd = where the clean calib/eval indices come from. Defaults to the global split dir `$d`.
    cd_="${d}"
    if [ -n "$CLEAN_DIR" ]; then
      cd_="$(printf '%s' "$CLEAN_DIR" | sed "s/{seed}/${seed}/g; s/{k}/${k}/g; s/{ds}/${ds}/g; s/{bb}/${bb}/g")"
      [ -f "${cd_}/${ds}_clean_eval_indices.json" ] || {
        echo "[skip missing clean] s${seed} k${k} ${ds}/${bb} -> ${cd_}"; continue; }
    fi
    out="outputs/fewshot_safepatch${OUT_TAG}_v8_s${seed}_k${k}/${ds}/${bb}"
    # Resume only when EVERY prediction file a metric needs is present. Testing just the
    # holdout CSV let a cell that died after writing it be marked done, which silently turns a
    # 3-seed mean into a 2-seed one (gtsrb/resnet50 k16 bank, 2026-07-29). Same failure family
    # as the baseline k=full skip fixed on 2026-07-25.
    done_cell=1
    for f in repair_holdout_unseen_predictions.csv clean_eval_predictions.csv \
             repair_support_seen_predictions.csv; do
      [ -f "$out/deploy/predictions/$f" ] || { done_cell=0; break; }
    done
    if [ "$done_cell" = 1 ]; then
      echo "[skip done] s${seed} k${k}"; continue; fi
    echo "=== [safepatch TRAIN] s${seed} k${k} ${ds}/${bb} @ $(date -Is) ==="
    $PY scripts/run_resolved_experiment.py --config "configs/v8_source/${ds}/${bb}/train.yaml" \
      --output-root "${out}/train" --overrides \
      "data.bug_indices_path=${d}/${ds}_bug_indices.json" "data.bug_eval_indices_path=${d}/${ds}_bug_eval_indices.json" \
      "data.bug_train_indices_path=${sub}" "data.bug_val_indices_path=${sub}" \
      "data.clean_eval_indices_path=${cd_}/${ds}_clean_calib_indices.json" \
      "train_loop.early_stop_metric=heldout_repaired" "runtime.device=cuda:0" $CR10 "${extra[@]}" >/dev/null 2>&1
    $PY scripts/run_resolved_experiment.py --config "configs/v8_source/${ds}/${bb}/deploy.yaml" \
      --output-root "${out}/deploy" --deployment-checkpoint-path "${out}/train/checkpoints/repair_best.pt" --overrides \
      "data.bug_indices_path=${d}/${ds}_bug_indices.json" "data.bug_train_indices_path=${d}/${ds}_bug_train_indices.json" \
      "data.bug_eval_indices_path=${d}/${ds}_bug_eval_indices.json" "data.clean_eval_indices_path=${cd_}/${ds}_clean_eval_indices.json" \
      "runtime.device=cuda:0" $CR10 "${extra[@]}" >/dev/null 2>&1 \
      && echo "  [done] s${seed} k${k}" || echo "  [FAIL] s${seed} k${k}"
  done
done
echo "SAFEPATCH FEWSHOT COMPLETE ${SETTING} @ $(date -Is)"
