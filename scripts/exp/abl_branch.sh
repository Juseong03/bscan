#!/bin/bash
# ABL-BRANCH — branch ablation (7 FM variants: full/cnnonly/.../noattn).
# Usage:  bash scripts/exp/abl_branch.sh [GPU] [SEEDS]
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
read -r -a SEEDS <<< "${2:-1 2 3 4 5 6 7 8 9 10}"
mkdir -p logs/exp
LOG="logs/exp/abl_branch_gpu${GPU}.log"

echo "[$(date +%T)] abl_branch START gpu=$GPU seeds=${SEEDS[*]}" | tee -a "$LOG"
for S in "${SEEDS[@]}"; do
  echo "[$(date +%T)] abl_branch seed=$S ..." | tee -a "$LOG"
  if bash scripts/run_all_experiments.sh ablation "$GPU" "$S" >> "$LOG" 2>&1; then
    echo "[$(date +%T)] EXP_DONE abl_branch seed=$S" | tee -a "$LOG"
  else
    echo "[$(date +%T)] EXP_FAIL abl_branch seed=$S" | tee -a "$LOG"
  fi
done
