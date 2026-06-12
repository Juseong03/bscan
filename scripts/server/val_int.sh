#!/bin/bash
# Server launcher — val_int across all GPUs (10 seeds, packed).
# Usage:  bash scripts/server/val_int.sh [JOBS_PER_GPU] [SEEDS]
#   e.g.  nohup bash scripts/server/val_int.sh 2 > logs/exp/val_int.run 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/server/_lib.sh
JPG="${1:-2}"
read -r -a SEEDS <<< "${2:-1 2 3 4 5 6 7 8 9 10}"
distribute val_int "$JPG" "${SEEDS[@]}"
