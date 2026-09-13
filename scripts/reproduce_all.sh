#!/usr/bin/env bash
# One entrypoint for the full-retrain path: backbone training, DynaPatch (DPGen+DPGate)
# train+deploy, and all 6 of the paper's baselines, for ONE (dataset, backbone, seed) setting
# at a time. Everything below runs LOCALLY -- no external checkout is needed any more.
#
# Usage:
#   uv sync
#   bash scripts/reproduce_all.sh <dataset> <backbone> <seed> [outdir]
#
# Example:
#   bash scripts/reproduce_all.sh gtsrb resnet50 101 outputs/repro_run
#
# Datasets:  gtsrb | tt100k_signs | lisa_signs
# Backbones: resnet50 | convnext_tiny | densenet121 | vgg16   (Hydra model group: resnet50 |
#            convnext | densenet121 | vgg16 -- "convnext_tiny" vs "convnext" naming differs
#            between the backbone-training config group and the v8_source setting directories;
#            this script translates it, see BACKBONE_MODEL_GROUP below)
# Seeds used in the paper: 101 202 303
#
# What this script covers, and what it does not:
#   [0] Backbone training (frozen base classifier) -- OPTIONAL, only runs if the checkpoint
#       configs/v8_source/<ds>/<bb>/{train,deploy}.yaml expects is missing. Uses seed=42
#       (the fixed backbone seed baked into every v8_source config's model.checkpoint_path;
#       backbones are trained ONCE per (dataset, backbone), not per repair seed).
#   [1] DynaPatch (DPGen+DPGate): train + deploy-eval.
#   [2] Arachne(DE) and DistRep(PSO) -- the two baselines with a faithful, self-contained
#       re-implementation in this repo (src/baselines/{arachne_de,distrep_pso}.py).
#   [3] HeadFT, FullFT, DistRep-original (full_finetune_distr) -- all three are
#       scripts/train_head_repair_baseline.py with a different --mode; see that script's
#       --help. KNOWN GAP: the train_loop.batch_size these were originally run at came from a
#       resolved config in a separate (non-anonymized) sibling repository not included in this
#       artifact; this script falls back to configs/v8_source/<ds>/<bb>/train.yaml's own
#       train_loop.batch_size (an in-repo, real number, but not verified bit-identical to the
#       paper's shipped runs for this one hyperparameter) -- see README.md "Known limitations".
#       Set BATCH to override.
#   [4] TopKSearch (mislabeled "Arachne" in some old output-tree names -- see
#       scripts/run_fewshot_arachne.sh's header for why it must never be reported as Arachne).
#   [5] NNPatch / PatchNAS (the two prior-art patch baselines) -- printed as a follow-up command,
#       not run automatically here (they operate on cached features across ALL settings at once
#       via scripts/dump_prior_features.py + scripts/baseline_prior_patches.py, not per-setting).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DS="${1:?usage: reproduce_all.sh <dataset> <backbone> <seed> [outdir]}"
BB="${2:?usage: reproduce_all.sh <dataset> <backbone> <seed> [outdir]}"
SEED="${3:?usage: reproduce_all.sh <dataset> <backbone> <seed> [outdir]}"
OUT="${4:-outputs/repro_run/${DS}_${BB}_s${SEED}}"
RUN="${RUN:-uv run python}"
DEVICE="${DEVICE:-cuda:0}"

TRAIN_CFG="configs/v8_source/${DS}/${BB}/train.yaml"
DEPLOY_CFG="configs/v8_source/${DS}/${BB}/deploy.yaml"
SPLIT_DIR="artifacts/bug_sets/v8_splits_seed${SEED}/${DS}_${BB}"

if [ ! -f "$TRAIN_CFG" ] || [ ! -f "$DEPLOY_CFG" ]; then
  echo "ERROR: no config for ${DS}/${BB} (expected ${TRAIN_CFG} and ${DEPLOY_CFG})" >&2
  exit 1
fi
if [ ! -d "$SPLIT_DIR" ]; then
  echo "ERROR: no split manifest dir for ${DS}/${BB} seed ${SEED} (expected ${SPLIT_DIR})" >&2
  exit 1
fi

# Same dot-path overrides run_fewshot_safepatch.sh applies -- the v8_source/*/train.yaml files
# carry stale default split paths (an old v7 output tree) that only ever worked because every
# real invocation overrides them explicitly; see STRUCTURE.md.
#
# TRAIN vs DEPLOY must NOT share one override set. src/data/factory.py's
# build_bug_repair_dataloaders() builds dataloaders["bug_eval"] from data.bug_val_indices_path
# WHEN IT IS SET, falling back to data.bug_eval_indices_path only if bug_val is absent (see
# note/PITFALLS.md, "Arachne 的 RR_held 一直是在证据集上重测一遍" -- the same silent-fallback
# bug, previously found in a baseline script, caught here in DynaPatch's own deploy step by a
# CPU/GPU smoke test: with bug_val_indices_path set, deploy-eval's "repair_holdout_unseen" split
# silently evaluated on the 27-sample bug_train set instead of the true 34-sample held-out set,
# producing a non-comparable, inflated RR). scripts/run_repairbench_v8.sh (the script that
# actually produced the paper's numbers) never sets bug_val_indices_path for its deploy calls --
# only for train, where it is intentional (early-stopping must not see the gate-calibration
# slice). Mirror that split here: DEPLOY_OVERRIDES omits bug_val_indices_path entirely and
# points clean_eval_indices_path at the true test split (clean_eval), not the train-time
# calibration split (clean_calib) that TRAIN_OVERRIDES uses.
TRAIN_OVERRIDES=(
  "data.bug_indices_path=${SPLIT_DIR}/${DS}_bug_indices.json"
  "data.bug_eval_indices_path=${SPLIT_DIR}/${DS}_bug_eval_indices.json"
  "data.bug_train_indices_path=${SPLIT_DIR}/${DS}_bug_train_indices.json"
  "data.bug_val_indices_path=${SPLIT_DIR}/${DS}_bug_train_indices.json"
  "data.clean_eval_indices_path=${SPLIT_DIR}/${DS}_clean_calib_indices.json"
)
DEPLOY_OVERRIDES=(
  "data.bug_indices_path=${SPLIT_DIR}/${DS}_bug_indices.json"
  "data.bug_train_indices_path=${SPLIT_DIR}/${DS}_bug_train_indices.json"
  "data.bug_eval_indices_path=${SPLIT_DIR}/${DS}_bug_eval_indices.json"
  "data.clean_eval_indices_path=${SPLIT_DIR}/${DS}_clean_eval_indices.json"
)
# The 6 baselines' own driver scripts (train_head_repair_baseline.py, train_arachne_baseline.py,
# run_arachne_de.py, run_distrep_pso.py) do not go through build_bug_repair_dataloaders' val/eval
# fallback -- they were checked against this same run's shipped predictions (34/34 samples,
# indices matching data/*_bug_eval_indices.json) and are unaffected, so they keep using
# TRAIN_OVERRIDES (renamed from the old shared SPLIT_OVERRIDES; same content) for both their
# training and their own held-out dump.
SPLIT_OVERRIDES=("${TRAIN_OVERRIDES[@]}")
CR10=(loss.lambda_robust=0.0 loss.lambda_field=0.0 loss.lambda_clean_replay=1.0)
EXTRA_MODEL_CKPT=()
[ "$DS" = tt100k_signs ] && [ "$BB" = vgg16 ] && \
  EXTRA_MODEL_CKPT+=("model.checkpoint_path=outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt")

echo "[assets] checking frozen-backbone / checkpoint inventory ..."
$RUN scripts/validate_assets.py || true

echo "######## [0/5] Backbone (frozen base classifier): ${DS}/${BB} ########"
BACKBONE_CKPT="$($RUN -c "
from omegaconf import OmegaConf
print(OmegaConf.load('${TRAIN_CFG}').model.checkpoint_path)
")"
if [ -f "$BACKBONE_CKPT" ]; then
  echo "  [skip] backbone checkpoint already present: ${BACKBONE_CKPT}"
else
  # BACKBONE_MODEL_GROUP: the Hydra `model=` group under configs/model/ names ConvNeXt-tiny
  # "convnext" (configs/model/convnext.yaml), not "convnext_tiny" as v8_source's directory
  # names and the paper do.
  case "$BB" in
    convnext_tiny) MODEL_GROUP=convnext ;;
    *) MODEL_GROUP="$BB" ;;
  esac
  BACKBONE_ROOT="$(dirname "$(dirname "$BACKBONE_CKPT")")"
  EXP_ID="$(basename "$BACKBONE_ROOT")"
  echo "  training backbone -> ${BACKBONE_ROOT} (this is SHARED across all 3 repair seeds for ${DS}/${BB})"
  # BATCH overrides the batch size for every stage below (backbone included), not just
  # HeadFT/FullFT/DistRep-original -- useful on a shared GPU where free VRAM varies with what
  # other jobs are currently running (see README.md "GPU setup"). Unset -> configs/train/
  # backbone_finetune.yaml's own default (64).
  BACKBONE_BATCH_OVERRIDE=()
  [ -n "${BATCH:-}" ] && BACKBONE_BATCH_OVERRIDE=("train.batch_size=${BATCH}")
  $RUN scripts/train_backbone.py dataset="$DS" model="$MODEL_GROUP" train=backbone_finetune runtime=local_gpu \
    +experiment.id="$EXP_ID" +experiment.name="$EXP_ID" +experiment.stage=backbone_finetune \
    +artifacts.root="$BACKBONE_ROOT" runtime.device="$DEVICE" "${BACKBONE_BATCH_OVERRIDE[@]}"
fi

echo "######## [1/5] DynaPatch train (DPGen+DPGate): ${DS}/${BB} seed=${SEED} ########"
$RUN scripts/run_resolved_experiment.py --config "$TRAIN_CFG" \
  --output-root "${OUT}/dynapatch/train" --seed "$SEED" \
  --overrides "${TRAIN_OVERRIDES[@]}" "${CR10[@]}" "${EXTRA_MODEL_CKPT[@]}" \
    "train_loop.early_stop_metric=heldout_repaired" "runtime.device=${DEVICE}"

echo "  DynaPatch deploy-eval: ${DS}/${BB} seed=${SEED}"
$RUN scripts/run_resolved_experiment.py --config "$DEPLOY_CFG" \
  --output-root "${OUT}/dynapatch/deploy" --seed "$SEED" \
  --deployment-checkpoint-path "${OUT}/dynapatch/train/checkpoints/repair_best.pt" \
  --overrides "${DEPLOY_OVERRIDES[@]}" "${CR10[@]}" "${EXTRA_MODEL_CKPT[@]}" "runtime.device=${DEVICE}"

echo "######## [2/5] Arachne(DE) and DistRep(PSO) -- self-contained real re-implementations ########"
echo "  -- Arachne(DE) --"
$RUN scripts/run_arachne_de.py --setting "${DS}/${BB}" --seed "$SEED" --k full \
  --output-root "${OUT}/arachne_de" --device "$DEVICE"
echo "  -- DistRep(PSO) --"
$RUN scripts/run_distrep_pso.py --setting "${DS}/${BB}" --seed "$SEED" --k full \
  --output-root "${OUT}/distrep_pso" --device "$DEVICE"

echo "######## [3/5] HeadFT, FullFT, DistRep-original (full_finetune_distr) ########"
for mode_pair in "head_only:headft" "full_finetune:fullft" "full_finetune_distr:distrep_original"; do
  MODE="${mode_pair%%:*}"; LABEL="${mode_pair##*:}"
  echo "  -- ${LABEL} (--mode ${MODE}) --"
  BATCH="${BATCH:-}"
  train_bs="${BATCH:-$($RUN -c "from omegaconf import OmegaConf;print(int(OmegaConf.load('${TRAIN_CFG}').train_loop.batch_size))")}"
  $RUN scripts/train_head_repair_baseline.py --config "$TRAIN_CFG" --output-root "${OUT}/${LABEL}" \
    --mode "$MODE" --lr 1e-4 --clean-replay-weight 1.0 --clean-replay-objective ce \
    --epochs 12 --num-workers 8 --clean-replay-subset-size 2048 --clean-replay-batch-limit "$train_bs" \
    --overrides "${SPLIT_OVERRIDES[@]}" "${EXTRA_MODEL_CKPT[@]}" "runtime.device=${DEVICE}"
done

echo "######## [4/5] TopKSearch (mislabeled 'Arachne' in some old trees -- NOT Arachne, see run_fewshot_arachne.sh) ########"
$RUN scripts/train_arachne_baseline.py --config "$TRAIN_CFG" --output-root "${OUT}/topksearch" \
  --seed "$SEED" --top-k 64 --rounds 3 --step-scale 0.5 --clean-tradeoff 0.25 \
  --clean-replay-subset-size 2048 --num-workers 8 \
  --overrides "${SPLIT_OVERRIDES[@]}" "${EXTRA_MODEL_CKPT[@]}" "runtime.device=${DEVICE}"

echo
echo "Done: ${OUT}/{dynapatch/{train,deploy},arachne_de,distrep_pso,headft,fullft,distrep_original,topksearch}"
echo "######## [5/5] NOT run automatically here: NNPatch / PatchNAS ########"
echo "These operate on cached features across settings, not per-setting training. Run:"
echo "  $RUN scripts/dump_prior_features.py --setting ${DS}/${BB} --seed ${SEED} --device ${DEVICE}"
echo "  $RUN scripts/baseline_prior_patches.py --settings ${DS}/${BB}"
