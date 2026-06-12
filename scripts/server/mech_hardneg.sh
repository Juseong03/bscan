#!/bin/bash
# Server launcher — mech_hardneg across all GPUs (10 seeds, packed).
# Usage:  bash scripts/server/mech_hardneg.sh [JOBS_PER_GPU] [SEEDS]
#   e.g.  nohup bash scripts/server/mech_hardneg.sh 2 > logs/exp/mech_hardneg.run 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/server/_lib.sh
JPG="${1:-2}"
read -r -a SEEDS <<< "${2:-1 2 3 4 5 6 7 8 9 10}"
distribute mech_hardneg "$JPG" "${SEEDS[@]}"
