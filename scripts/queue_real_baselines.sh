#!/usr/bin/env bash
# The three "real" baseline runs, in one SEQUENTIAL queue.
#
# Sequential on purpose. Running DistRep's PSO alongside the two dataloader-heavy jobs took a
# gtsrb/resnet50 cell from 116 s (measured 2026-08-01, box idle) to >10 min: the 64-core box is
# saturated by ~25 dataloader workers per fine-tuning job, and PSO is CPU-bound. Do not
# parallelise these without re-measuring.
#
# Stage 1  DistRep(PSO), seeds 202+303 at the SAME reduced budget as the archived s101 tree
#          (3x15x15, clean-cap 512). Matching s101 is the point: a three-seed column must be
#          one budget. The full-budget run is a separate tree, not this one.
# Stage 2  Weighted Retraining at 40 epochs (the archived tree is 12; head_only and last_delta
#          were fixed to 40 in 2026-07-29, weighted_retraining was missed).
# Stage 3  TopKSearch step_scale grid. step_scale=0.5 -- the shipped value, never swept -- is
#          what produces the 0.046/0.035 row. On gtsrb/resnet50 s101: RR_held 0.042 (0.5) ->
#          0.125 (2) -> 0.354 (8) -> 0.417 (32) -> 0.417 (128, but Reg 5x worse) -> 0.312 (512,
#          collapses). Grid {8,32,128} brackets the optimum; 0.5 already exists on disk.
#          top_k is NOT swept here: 64 vs 1024 gave bit-identical results (same 2 accepted steps).
#
# Usage: bash scripts/queue_real_baselines.sh [stages]     e.g. "1 2 3", default all
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RV="$PWD"; ORIG="${DYNAPATCH_BASELINE_REPO:?Set DYNAPATCH_BASELINE_REPO -- see README.md (HeadFT/FullFT/DistRep-old/TopKSearch stages of this queue require an external checkout not shipped in this repo; Arachne(DE) and DistRep(PSO), the two baselines with self-contained code here, do not need it)}"
STAGES="${1:-1 2 3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SETTINGS="gtsrb/resnet50 gtsrb/convnext_tiny gtsrb/densenet121 gtsrb/vgg16 \
tt100k_signs/resnet50 tt100k_signs/convnext_tiny tt100k_signs/densenet121 tt100k_signs/vgg16 \
lisa_signs/resnet50 lisa_signs/convnext_tiny lisa_signs/densenet121 lisa_signs/vgg16"

t0=$SECONDS
for stage in $STAGES; do
case "$stage" in
1)
  echo "########## STAGE 1: DistRep(PSO) s202 s303 @ $(date -Is)"
  SEEDS="202 303" bash scripts/queue_distrep_pso.sh
  ;;
2)
  echo "########## STAGE 2: Weighted Retraining ep40 @ $(date -Is)"
  EPOCHS=40 bash scripts/queue_wr_ep40.sh "101 202 303"
  ;;
3)
  echo "########## STAGE 3: TopKSearch step_scale grid @ $(date -Is)"
  cd "$ORIG"
  for ss in 8.0 32.0 128.0; do
    for seed in 101 202 303; do
      for st in $SETTINGS; do
        ds="${st%/*}"; bb="${st#*/}"
        d="$RV/artifacts/bug_sets/v8_splits_seed${seed}/${ds}_${bb}"
        src="outputs/baselines_lsr_v8_s101/${ds}/${bb}/arachne_style/config_resolved.yaml"
        [ -f "$src" ] || { echo "[skip no config] ${st}"; continue; }
        out="$RV/outputs/topksearch_ss${ss}_v8_s${seed}/${ds}/${bb}"
        [ -f "$out/predictions/repair_holdout_unseen_predictions.csv" ] && \
          { echo "[skip done] ss${ss} s${seed} ${st}"; continue; }
        extra=()
        [ "$ds" = tt100k_signs ] && [ "$bb" = vgg16 ] && \
          extra+=("model.checkpoint_path=outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt")
        echo "=== [topk ss=${ss}] s${seed} ${st} @ $(date -Is) ==="; ts=$SECONDS
        .venv/bin/python scripts/train_arachne_baseline.py --config "$src" --output-root "$out" \
          --seed "$seed" --top-k 1024 --rounds 5 --step-scale "$ss" \
          --clean-tradeoff 0.25 --clean-replay-subset-size 2048 --num-workers 8 \
          --overrides \
            "data.bug_indices_path=${d}/${ds}_bug_indices.json" \
            "data.bug_eval_indices_path=${d}/${ds}_bug_eval_indices.json" \
            "data.bug_train_indices_path=${d}/${ds}_bug_train_indices.json" \
            "data.bug_val_indices_path=${d}/${ds}_bug_train_indices.json" \
            "data.clean_eval_indices_path=${d}/${ds}_clean_test_indices.json" \
            "runtime.device=cuda:0" "${extra[@]}" >/dev/null 2>&1 \
          && echo "  [done] in $((SECONDS-ts))s" || echo "  !!! FAIL ss${ss} s${seed} ${st}"
      done
    done
  done
  cd "$RV"
  ;;
esac
done
echo "########## ALL STAGES COMPLETE in $((SECONDS-t0))s @ $(date -Is)"
