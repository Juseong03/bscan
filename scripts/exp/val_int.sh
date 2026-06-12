#!/bin/bash
# VAL-INT — internal transcript-split comparison (FM 4 + onehot 3 + 10 baselines).
# Usage:  bash scripts/exp/val_int.sh [GPU] [SEEDS]
#   e.g.  bash scripts/exp/val_int.sh 0 "42 777 2025 9001"
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
read -r -a SEEDS <<< "${2:-1 2 3 4 5 6 7 8 9 10}"
mkdir -p logs/exp
LOG="logs/exp/val_int_gpu${GPU}${EXP_LOG_SUFFIX:-}.log"

echo "[$(date +%T)] val_int START gpu=$GPU seeds=${SEEDS[*]}" | tee -a "$LOG"
for S in "${SEEDS[@]}"; do
  echo "[$(date +%T)] val_int seed=$S ..." | tee -a "$LOG"
  if bash scripts/run_all_experiments.sh train "$GPU" "$S" >> "$LOG" 2>&1; then
    echo "[$(date +%T)] EXP_DONE val_int seed=$S" | tee -a "$LOG"
  else
    echo "[$(date +%T)] EXP_FAIL val_int seed=$S" | tee -a "$LOG"
  fi
done
