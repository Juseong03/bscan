#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-42 123 315}"
MODELS="${MODELS:-bscan_seq_lite circcnn circcnndouble circdc jedi}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NEGATIVE_MODE="${NEGATIVE_MODE:-lower_intron}"

cmd=(
  python evaluate_hard_negative_pairing.py
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --negative-mode "$NEGATIVE_MODE"
  --seeds $SEEDS
  --models $MODELS
)

if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi

"${cmd[@]}"
