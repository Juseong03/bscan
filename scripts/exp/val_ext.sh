#!/bin/bash
# VAL-EXT — external validation (baselines + headline FM) + leakage controls.
# Evaluates all saved checkpoints, so run AFTER val_int. Not seed-looped.
# Usage:  bash scripts/exp/val_ext.sh [GPU] [SEEDS]
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
SEEDS="${2:-42 123 315 777 1004 2024 2025 2026 3407 9001}"
mkdir -p logs/exp
LOG="logs/exp/val_ext.log"

echo "[$(date +%T)] val_ext START gpu=$GPU" | tee -a "$LOG"
if bash scripts/run_all_experiments.sh external "$GPU" "$SEEDS" >> "$LOG" 2>&1; then
  echo "[$(date +%T)] EXP_DONE val_ext" | tee -a "$LOG"
else
  echo "[$(date +%T)] EXP_FAIL val_ext" | tee -a "$LOG"
fi
