#!/usr/bin/env bash
# One entrypoint for the full-retrain path: DynaPatch (DPGen+DPGate) train+deploy, plus the two
# baselines that are fully self-contained in this repository (Arachne(DE), DistRep(PSO)), for
# ONE (dataset, backbone, seed) setting at a time.
#
# What this script does NOT cover, and why (see README.md "Known limitations" for the full
# explanation -- this is not a bug in this script, it is an honest boundary of what is shipped):
#   - Training the 4 frozen backbones from scratch. No backbone-training driver exists anywhere
#     in the source project this artifact was extracted from; backbones must be supplied as
#     checkpoints at outputs/exp_<dataset>_<backbone>_backbone_public_*/checkpoints/backbone_last.pt
#     (see artifacts/checkpoints/MANIFEST.md) before this script's DynaPatch/Arachne/DistRep steps
#     can run.
#   - HeadFT, FullFT, and the (non-headline) "TopKSearch" baseline. Their training drivers live
#     only in a separate, non-anonymized sibling repository and are not included here.
#
# Usage:
#   uv sync
#   bash scripts/reproduce_all.sh <dataset> <backbone> <seed> [outdir]
#
# Example:
#   bash scripts/reproduce_all.sh gtsrb resnet50 101 outputs/repro_run
#
# Datasets:  gtsrb | tt100k_signs | lisa_signs
# Backbones: resnet50 | convnext_tiny | densenet121 | vgg16
# Seeds used in the paper: 101 202 303
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
BACKBONE_CKPT_CHECK="artifacts/checkpoints/manifest.json"

if [ ! -f "$TRAIN_CFG" ] || [ ! -f "$DEPLOY_CFG" ]; then
  echo "ERROR: no config for ${DS}/${BB} (expected ${TRAIN_CFG} and ${DEPLOY_CFG})" >&2
  exit 1
fi

echo "[assets] checking frozen-backbone / checkpoint inventory against ${BACKBONE_CKPT_CHECK} ..."
$RUN scripts/validate_assets.py || {
  echo "See artifacts/checkpoints/MANIFEST.md -- backbone checkpoints are not shipped in this repo." >&2
}

echo "######## [1/3] DynaPatch train (DPGen+DPGate): ${DS}/${BB} seed=${SEED} ########"
$RUN scripts/run_resolved_experiment.py --config "$TRAIN_CFG" \
  --output-root "${OUT}/dynapatch/train" --seed "$SEED" \
  --overrides "runtime.device=${DEVICE}"

echo "######## [2/3] DynaPatch deploy-eval: ${DS}/${BB} seed=${SEED} ########"
$RUN scripts/run_resolved_experiment.py --config "$DEPLOY_CFG" \
  --output-root "${OUT}/dynapatch/deploy" --seed "$SEED" \
  --deployment-checkpoint-path "${OUT}/dynapatch/train/checkpoints/backbone_last.pt" \
  --overrides "runtime.device=${DEVICE}"

echo "######## [3/3] Baselines self-contained in this repo: Arachne(DE), DistRep(PSO) ########"
echo "  -- Arachne(DE) --"
$RUN scripts/run_arachne_de.py --setting "${DS}/${BB}" --seed "$SEED" --k full \
  --output-root "${OUT}/arachne_de" --device "$DEVICE"
echo "  -- DistRep(PSO) --"
$RUN scripts/run_distrep_pso.py --setting "${DS}/${BB}" --seed "$SEED" --k full \
  --output-root "${OUT}/distrep_pso" --device "$DEVICE"

echo
echo "Done: ${OUT}/{dynapatch/{train,deploy},arachne_de,distrep_pso}"
echo "NOT run by this script (see README.md 'Known limitations'): HeadFT, FullFT, TopKSearch"
echo "  (require an external, non-anonymized sibling repository via \$DYNAPATCH_BASELINE_REPO)."
echo "For NNPatch / PatchNAS (the two prior-art patch baselines), run separately:"
echo "  $RUN scripts/dump_prior_features.py --setting ${DS}/${BB} --seed ${SEED} --device ${DEVICE}"
echo "  $RUN scripts/baseline_prior_patches.py --settings ${DS}/${BB}"
