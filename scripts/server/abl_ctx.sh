#!/bin/bash
# Server launcher — ABL-CTX (5 seeds), windows 250 & 500 in parallel on 2 GPUs.
# Usage: bash scripts/server/abl_ctx.sh [SEEDS]
set -u
cd "$(dirname "$0")/../.." || exit 1
SEEDS="${1:-1 2 3 4 5}"
EXP_LOG_SUFFIX="_jb250" bash scripts/exp/abl_ctx.sh 1 "$SEEDS" "250" rnafm &
EXP_LOG_SUFFIX="_jb500" bash scripts/exp/abl_ctx.sh 2 "$SEEDS" "500" rnafm &
wait
echo "[abl_ctx] done"
