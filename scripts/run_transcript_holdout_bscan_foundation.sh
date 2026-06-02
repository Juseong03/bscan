#!/bin/bash
set -euo pipefail

# Transcript-holdout validation for BSCAN representation variants.
# These runs are substantially slower than the core one-hot/public models.
# Usage:
#   DEVICE=0 bash run_transcript_holdout_bscan_foundation.sh

if [[ -n "${TRANSCRIPT_HOLDOUT_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "$TRANSCRIPT_HOLDOUT_SEEDS"
else
  SEEDS=(42 123 315 777 1004)
fi
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
EARLYSTOP="${EARLYSTOP:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUT_DIR="${OUT_DIR:-research_results}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODELS=(
  bscan_unified_onehot
  bscan_unified_ernie
  bscan_unified_bert
  bscan_unified_fm
  bscan_unified_msm
  bscan_embedonly_ernie
  bscan_embedonly_bert
  bscan_embedonly_fm
  bscan_embedonly_msm
)

mkdir -p "$OUT_DIR" logs

for SEED in "${SEEDS[@]}"; do
  TAG="transcript_holdout_bscan_foundation_seed_${SEED}"
  LOG="logs/${TAG}.log"
  echo "[$(date)] Running transcript-holdout BSCAN representation variants, seed=${SEED}, batch_size=${BATCH_SIZE}" | tee "$LOG"
  python run_model_comparison.py \
    --models "${MODELS[@]}" \
    --epochs "$EPOCHS" \
    --earlystop "$EARLYSTOP" \
    --batch_size "$BATCH_SIZE" \
    --batch_size_pretrained "$BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --split_strategy transcript \
    --tag "$TAG" \
    --out_dir "$OUT_DIR" 2>&1 | tee -a "$LOG"
done

python summarize_transcript_holdout.py \
  --pattern "${OUT_DIR}/model_comparison_transcript_holdout_bscan_foundation_seed_*.csv" \
  --out "${OUT_DIR}/transcript_holdout_bscan_foundation_summary.csv"
