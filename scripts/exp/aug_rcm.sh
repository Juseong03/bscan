#!/bin/bash
# AUG-RCM — RCM auxiliary-branch study (FM vs FM+RCM at flanking 100/500).
# Usage:  bash scripts/exp/aug_rcm.sh [GPU] [SEEDS] [FLANKS]
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
SEEDS="${2:-1 2 3 4 5 6 7 8 9 10}"
FLANKS="${3:-100 500}"
mkdir -p logs/exp
LOG="logs/exp/aug_rcm_gpu${GPU}.log"

echo "[$(date +%T)] aug_rcm START gpu=$GPU flanks=$FLANKS seeds=$SEEDS" | tee -a "$LOG"
if bash scripts/run_rcm_aux.sh "$FLANKS" "$GPU" "$SEEDS" >> "$LOG" 2>&1; then
  echo "[$(date +%T)] EXP_DONE aug_rcm" | tee -a "$LOG"
else
  echo "[$(date +%T)] EXP_FAIL aug_rcm" | tee -a "$LOG"
fi
