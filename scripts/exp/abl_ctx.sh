#!/bin/bash
# ABL-CTX — context-window sweep (re-extract FM embeddings at wider windows).
# Usage:  bash scripts/exp/abl_ctx.sh [GPU] [SEEDS] [WINDOWS] [ENC]
#   e.g.  bash scripts/exp/abl_ctx.sh 1 "1 2 3" "250" rnafm
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
SEEDS="${2:-1 2 3 4 5}"
WINDOWS="${3:-250}"
ENC="${4:-rnafm}"
mkdir -p logs/exp
LOG="logs/exp/abl_ctx_gpu${GPU}${EXP_LOG_SUFFIX:-}.log"

echo "[$(date +%T)] abl_ctx START gpu=$GPU windows=$WINDOWS enc=$ENC seeds=$SEEDS" | tee -a "$LOG"
if bash scripts/run_context_window_sweep.sh "$WINDOWS" "$ENC" "$GPU" "$SEEDS" >> "$LOG" 2>&1; then
  echo "[$(date +%T)] EXP_DONE abl_ctx windows=$WINDOWS" | tee -a "$LOG"
else
  echo "[$(date +%T)] EXP_FAIL abl_ctx windows=$WINDOWS" | tee -a "$LOG"
fi
