#!/usr/bin/env bash
# DistRep(PSO), seeds 202+303, at the SAME reduced budget as the archived s101 tree
# (3x15x15, clean-cap 512). Matching s101 is the point: a three-seed column must be one budget.
# The full-budget run is a separate tree, not this one.
#
# Usage: bash scripts/queue_real_baselines.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "DistRep(PSO) s202 s303 @ $(date -Is)"
SEEDS="202 303" bash scripts/queue_distrep_pso.sh
echo "DONE @ $(date -Is)"
