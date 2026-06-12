#!/bin/bash
# Server launcher — abl_branch across all GPUs (10 seeds, packed).
# Usage:  bash scripts/server/abl_branch.sh [JOBS_PER_GPU] [SEEDS]
#   e.g.  nohup bash scripts/server/abl_branch.sh 2 > logs/exp/abl_branch.run 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/server/_lib.sh
JPG="${1:-2}"
read -r -a SEEDS <<< "${2:-1 2 3 4 5 6 7 8 9 10}"
distribute abl_branch "$JPG" "${SEEDS[@]}"
