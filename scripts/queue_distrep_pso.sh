#!/usr/bin/env bash
# Experiment C of note/PREREG_20260801.md: the REAL DistrRep (Li Calsi et al., ICST 2023).
#
# The column previously labelled `DistRep` was `--mode full_finetune_distr`, i.e. plain full
# fine-tuning (note/PITFALLS.md 2026-08-01). This runs the actual three-phase PSO method, whose
# faithful re-implementation had been sitting unused in src/baselines/distrep_pso.py.
#
# TWO DELIBERATE WEAKENINGS, both pre-registered and both to be stated in the paper:
#   clean-cap 512 (Arachne uses 2048) -- two PSO worker threads each hold a copy of the model and
#     the clean set; 2048 OOMs on this single card while another queue is running.
#   reduced PSO budget + ONE seed -- at the paper's default (5 partitions x 40 particles x 40
#     iterations) a single cell exceeded 10 minutes on the SMALLEST setting, because DistrRep
#     localises over all layers and cannot evaluate on cached last-layer features the way Arachne
#     can. Reduced to 3 x 15 x 15; measured 115 s on gtsrb/resnet50.
# => The numbers this produces are a LOWER BOUND on the method. The table must say "reduced PSO
#    budget, single seed", and no claim about DistrRep's capability may rest on them.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_ALLOC_CONF=expandable_segments:True

SEEDS="${SEEDS:-101}"
SETTINGS="${SETTINGS:-gtsrb/resnet50 gtsrb/densenet121 gtsrb/convnext_tiny gtsrb/vgg16 \
tt100k_signs/resnet50 tt100k_signs/densenet121 tt100k_signs/convnext_tiny tt100k_signs/vgg16 \
lisa_signs/resnet50 lisa_signs/densenet121 lisa_signs/convnext_tiny lisa_signs/vgg16}"

for s in $SEEDS; do
  for st in $SETTINGS; do
    out="outputs/fewshot_distrepPSO_v8_s${s}_kfull/${st}"
    if [ -f "${out}/predictions/repair_holdout_unseen_predictions.csv" ]; then
      echo "=== [skip done] ${st} s${s}"; continue
    fi
    echo "=== [distrepPSO] ${st} s${s} @ $(date -Is) ==="
    mkdir -p "$out"
    .venv/bin/python scripts/run_distrep_pso.py \
      --setting "$st" --seed "$s" --k full --clean-cap 512 \
      --n-partitions 3 --particles 15 --iterations 15 \
      --output-root "$out" || echo "!!! FAIL ${st} s${s}"
  done
done
echo "### distrepPSO queue complete @ $(date -Is)"
