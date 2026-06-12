#!/bin/bash
# Server launcher — AUG-RCM (5 seeds), flanking 100 & 500 in parallel on 2 GPUs.
# Usage: bash scripts/server/aug_rcm.sh [SEEDS]
set -u
cd "$(dirname "$0")/../.." || exit 1
SEEDS="${1:-1 2 3 4 5}"
EXP_LOG_SUFFIX="_fb100" bash scripts/exp/aug_rcm.sh 0 "$SEEDS" "100" &
EXP_LOG_SUFFIX="_fb500" bash scripts/exp/aug_rcm.sh 1 "$SEEDS" "500" &
wait
echo "[aug_rcm] done — aggregate: python analysis/evaluate_rcm_aux.py"
