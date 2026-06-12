#!/bin/bash
# MECH-HN — hard-negative 3-tier probe + augmented training.
# Usage:  bash scripts/exp/mech_hardneg.sh [GPU] [SEEDS]
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
read -r -a SEEDS <<< "${2:-42 123 315 777 1004 2024 2025 2026 3407 9001}"
mkdir -p logs/exp
LOG="logs/exp/mech_hardneg_gpu${GPU}.log"

echo "[$(date +%T)] mech_hardneg START gpu=$GPU seeds=${SEEDS[*]}" | tee -a "$LOG"
for S in "${SEEDS[@]}"; do
  echo "[$(date +%T)] mech_hardneg seed=$S ..." | tee -a "$LOG"
  if bash scripts/run_all_experiments.sh hardneg "$GPU" "$S" >> "$LOG" 2>&1; then
    echo "[$(date +%T)] EXP_DONE mech_hardneg seed=$S" | tee -a "$LOG"
  else
    echo "[$(date +%T)] EXP_FAIL mech_hardneg seed=$S" | tee -a "$LOG"
  fi
done
