#!/bin/bash
# ANALYSIS — masking / ALU / duplex / statistics on saved checkpoints.
# Inference-only; run AFTER val_int. Not seed-looped.
# Usage:  bash scripts/exp/analysis.sh [GPU]
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
mkdir -p logs/exp
LOG="logs/exp/analysis.log"

echo "[$(date +%T)] analysis START gpu=$GPU" | tee -a "$LOG"
if bash scripts/run_all_experiments.sh analysis "$GPU" "42 123 315" >> "$LOG" 2>&1; then
  echo "[$(date +%T)] EXP_DONE analysis" | tee -a "$LOG"
else
  echo "[$(date +%T)] EXP_FAIL analysis" | tee -a "$LOG"
fi
