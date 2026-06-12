#!/bin/bash
# Server launcher — one-time prep (rcm_scores + external seq_dict/embeddings).
# Run ONCE before the other launchers.  Usage: bash scripts/server/prep.sh [GPU]
set -u
cd "$(dirname "$0")/../.." || exit 1
bash scripts/exp/prep.sh "${1:-0}"
