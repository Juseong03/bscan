#!/bin/bash
# Server launcher — masking / ALU / duplex / stats. Run AFTER val_int.
# Usage: bash scripts/server/analysis.sh [GPU]
set -u
cd "$(dirname "$0")/../.." || exit 1
bash scripts/exp/analysis.sh "${1:-0}"
