#!/bin/bash
# PREP — one-time setup shared by the per-experiment scripts.
# Builds rcm_scores/ (for circcnntri) + circAtlas external seq_dict + external
# FM embeddings. Run ONCE before launching experiments on multiple GPUs
# (avoids 3 jobs racing to build rcm_scores/).
#
# Usage:  bash scripts/exp/prep.sh [GPU]
set -u
cd "$(dirname "$0")/../.." || exit 1
GPU="${1:-0}"
mkdir -p logs/exp
LOG="logs/exp/prep.log"

echo "[$(date +%T)] prep START (gpu=$GPU)" | tee -a "$LOG"

# 1) RCM scores (full coverage; needed by circcnntri)
if [ -z "$(ls rcm_scores 2>/dev/null)" ]; then
  echo ">>> rcm_scores" | tee -a "$LOG"
  python pipeline/generate_rcm_scores_subset.py \
      --junction_bps 100 --flanking_bps 100 --max_samples 100000 >> "$LOG" 2>&1
fi

# 2) circAtlas external controls + seq_dict
EXT_TSV="external_data/circatlas/exon_controls/circatlas_exon_external_controls.tsv"
EXT_SEQ="external_data/circatlas/exon_controls/seq_dict/junction.json"
[ -f "$EXT_TSV" ] || { echo ">>> make_circatlas_exon_controls" | tee -a "$LOG"; python pipeline/make_circatlas_exon_controls.py >> "$LOG" 2>&1; }
[ -f "$EXT_SEQ" ] || { echo ">>> build_circatlas_seq_dict"      | tee -a "$LOG"; python pipeline/build_circatlas_seq_dict.py >> "$LOG" 2>&1; }

# 3) external FM embeddings (resumable; skips existing)
if [ -f "$EXT_SEQ" ] && [ ! -d "external_data/circatlas/exon_controls/fm_embeddings/rnafm" ]; then
  echo ">>> external FM embeddings" | tee -a "$LOG"
  bash scripts/extract_all_fm_embeddings.sh "$GPU" "rnafm rnabert rnaernie rnamsm" external 256 >> "$LOG" 2>&1
fi

echo "[$(date +%T)] EXP_DONE prep" | tee -a "$LOG"
