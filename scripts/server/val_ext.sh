#!/bin/bash
# Server launcher — external validation (baselines + headline FM) + leakage.
# Inference-only; run AFTER val_int.  Usage: bash scripts/server/val_ext.sh [GPU] [SEEDS]
set -u
cd "$(dirname "$0")/../.." || exit 1
bash scripts/exp/val_ext.sh "${1:-0}" "${2:-1 2 3 4 5 6 7 8 9 10}"
