#!/usr/bin/env bash
# queue_mainline_baselines.sh -- run the remaining DistRep+Arachne baseline sweeps back to back
# on ONE GPU. gtsrb is normally already running when this is launched, so we first wait for that
# process to exit, then take the remaining datasets in turn.
#
# run_mainline_all.sh is resumable (finished cells are skipped), so re-running this is safe.
# Usage: WAIT_PID=<pid of the running gtsrb sweep> bash scripts/queue_mainline_baselines.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_PID="${WAIT_PID:-}"
SEEDS="${SEEDS:-101 202 303}"
DATASETS="${DATASETS:-tt100k_signs lisa_signs}"

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for pid ${WAIT_PID} (running sweep) @ $(date -Is)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[queue] pid ${WAIT_PID} exited @ $(date -Is)"
fi

for ds in $DATASETS; do
  echo "[queue] === starting ${ds} @ $(date -Is) ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" METHODS="distrep arachne" \
    bash scripts/run_mainline_all.sh "$ds" "$SEEDS" \
    > "outputs/mainline_baselines_${ds}.log" 2>&1 \
    || echo "[queue] [ERR] ${ds} driver returned nonzero"
  echo "[queue] === finished ${ds} @ $(date -Is) ==="
done
echo "[queue] ALL BASELINE SWEEPS DONE @ $(date -Is)"
