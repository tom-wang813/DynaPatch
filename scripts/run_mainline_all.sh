#!/usr/bin/env bash
# run_mainline_all.sh -- MAIN LINE overnight sweep for ONE dataset (so 3 datasets map to 3 GPUs).
# For each of the 4 backbones x per-setting k-grid x seeds, runs:
#   safepatch (ours, no-gate cr10)
# Resumable (sub-drivers skip done). Respects external CUDA_VISIBLE_DEVICES.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_mainline_all.sh gtsrb "101 202 303"
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS="${1:?usage: run_mainline_all.sh <dataset> \"<seeds>\"}"; SEEDS="${2:-101 202 303}"
METHODS="${METHODS:-safepatch}"

kgrid() {  # per-setting k-grid capped at bug_train size
  case "$1" in
    lisa_signs/convnext_tiny|lisa_signs/densenet121) echo "1 2 4 8 full";;
    lisa_signs/resnet50|lisa_signs/vgg16)            echo "1 2 4 8 16 full";;
    *)                                                echo "1 2 4 8 16 32 full";;
  esac
}

for bb in resnet50 convnext_tiny densenet121 vgg16; do
  setting="${DS}/${bb}"; ks="$(kgrid "$setting")"
  echo "######## $setting  ks=[$ks]  @ $(date -Is) ########"
  for m in $METHODS; do
    case "$m" in
      safepatch) bash scripts/run_fewshot_safepatch.sh "$setting" "$SEEDS" "$ks" || echo "[ERR safepatch] $setting";;
    esac
  done
done
echo "======== MAINLINE COMPLETE ${DS} @ $(date -Is) ========"
