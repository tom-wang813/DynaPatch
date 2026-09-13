#!/usr/bin/env bash
# Arachne (our PyTorch re-implementation) across all 12 settings x 3 seeds -- the column that has
# been missing from every comparison table.
#
# WHY IT WAS MISSING. Not because it fails: `run_arachne_de.py` runs, converges, and writes standard
# prediction CSVs. It had simply never been queued -- one pilot cell existed (gtsrb/resnet50 s101)
# and nothing else. At 32 s/cell this whole sweep is ~20 minutes, so "too expensive" was never the
# reason either.
#
# A DIAGNOSIS THAT WAS WRONG, RECORDED SO IT IS NOT REPEATED. The pilot's metrics.json shows
# `num_places: 8` against the runner default of 64, which looked like a crippled search budget. It
# is not: `num_places` CAPS the Pareto front produced by localisation, and on this data the front
# only contains 7-8 places, so cap=8 and cap=64 give bit-identical results (RR_held 0.132 both
# ways, verified 2026-07-31). Arachne's low score here is real, not a budget artefact.
#
# WHAT THE LOW SCORE DOES AND DOES NOT LICENSE. It licenses reporting the number. It does NOT
# license the claim "Arachne is weak" -- this is a re-implementation of a TensorFlow/Keras artefact,
# and the localisation step producing only ~7 places is exactly where a re-implementation would
# diverge. Label the column "Arachne (our PyTorch re-implementation)" and say so.
#
# TO LAUNCH:  nohup bash scripts/queue_arachne_de.sh > outputs/queue_arachne_de.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=".venv/bin/python"

SETTINGS="${SETTINGS:-gtsrb/resnet50 gtsrb/convnext_tiny gtsrb/densenet121 gtsrb/vgg16 \
tt100k_signs/resnet50 tt100k_signs/convnext_tiny tt100k_signs/densenet121 tt100k_signs/vgg16 \
lisa_signs/resnet50 lisa_signs/convnext_tiny lisa_signs/densenet121 lisa_signs/vgg16}"
SEEDS="${SEEDS:-101 202 303}"
# 2026-07-31 RE-RUN under TAG=_fix. The first sweep's held-out column was invalid: the script
# overrode data.bug_val_indices_path with the evidence set, and build_repair_dataloaders prefers
# that key over bug_eval_indices_path, so `loaders["bug_eval"]` returned the samples the repair was
# fitted on. Fixed in run_arachne_de.py. The old tree is kept, unread, so the two can be diffed;
# nothing under outputs/ is deleted.
TAG="${TAG:-_fix}"

echo "### arachne_de full sweep${TAG} @ $(date -Is)"
for st in $SETTINGS; do
  ds="${st%/*}"; bb="${st#*/}"
  for seed in $SEEDS; do
    out="outputs/fewshot_arachnede${TAG}_v8_s${seed}_kfull/${ds}/${bb}"
    if [ -f "$out/predictions/repair_holdout_unseen_predictions.csv" ]; then
      echo "[skip done] s${seed} ${st}"; continue
    fi
    echo "=== [arachne_de] s${seed} ${st} @ $(date -Is) ==="
    $PY scripts/run_arachne_de.py --setting "$st" --seed "$seed" --k full \
        --output-root "$out" 2>&1 | grep -E "^\[arachne\]" || echo "  [FAILED] s${seed} ${st}"
  done
done
echo "### arachne_de sweep complete @ $(date -Is)"
