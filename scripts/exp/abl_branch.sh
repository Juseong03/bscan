#!/bin/bash
# ABL-BRANCH — branch ablation (7 FM variants: full/cnnonly/.../noattn).
# Usage:  bash scripts/exp/abl_branch.sh [GPU] [SEEDS]
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
read -r -a SEEDS <<< "${2:-42 123 315 777 1004 2024 2025 2026 3407 9001}"
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
