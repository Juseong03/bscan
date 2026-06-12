#!/bin/bash
# ABL-CTX — Context-window study (docs/DESIGN_new_experiments.md)
# "Does giving BSCAN a wider intronic window improve external generalization,
#  or is exon-bias structural?"  All models see the SAME junction_bps (fair).
#
# For each junction_bps: (1) rebuild junction seqs from genome, (2) re-extract FM
# embeddings into fm_embeddings/<enc>_jb<N>/, (3) train models on that window.
#
# Usage (run from repo root):
#   bash scripts/run_context_window_sweep.sh [WINDOWS] [ENC] [DEVICE] [SEEDS]
#     WINDOWS  junction_bps list (default: "100 250 500")
#     ENC      FM encoder for the FM model (default: rnafm)
#     DEVICE   GPU id (default: 0)
#     SEEDS    quoted seeds (default: "1 2 3 4 5")
#
# Pilot first:  bash scripts/run_context_window_sweep.sh "250" rnafm 0 "42"
#
# Requires data/hg19_seq_dict.json (genome) for windows >100 (rebuilds seq_dict +
# embeddings). FM length cap ~1024 tokens → junction_bps<=500 (1000 tokens).
set -u
cd "$(dirname "$0")/.." || exit 1

WINDOWS="${1:-100 250 500}"
ENC="${2:-rnafm}"
DEVICE="${3:-0}"
SEEDS="${4:-1 2 3 4 5}"
FM_MODEL="bscan_unified_fm"            # encoder selected by ENC via experiment.py mapping
[ "$ENC" = "rnaernie" ] && FM_MODEL="bscan_unified_ernie"
[ "$ENC" = "rnabert" ]  && FM_MODEL="bscan_unified_bert"
[ "$ENC" = "rnamsm" ]   && FM_MODEL="bscan_unified_msm"
CMP_MODELS="$FM_MODEL bscan circcnn"   # FM + onehot-arch control + CNN baseline (same window)
EPOCHS=100; EARLYSTOP=30

echo "ABL-CTX sweep | windows=$WINDOWS enc=$ENC models=$CMP_MODELS device=cuda:$DEVICE seeds=$SEEDS"

for JB in $WINDOWS; do
  echo ""; echo "=== [$(date +%H:%M:%S)] junction_bps=$JB ==="

  # 1) embeddings for this window (jb=100 reuses existing dir; >100 → _jb<N>)
  echo ">>> FM embeddings (enc=$ENC, jb=$JB)"
  python pipeline/extract_fm_embeddings.py --enc_type "$ENC" --device "$DEVICE" --batch_size 256 --junction_bps "$JB"

  # 2) train FM + controls at this window
  for SEED in $SEEDS; do
    echo ">>> train (jb=$JB, seed=$SEED): $CMP_MODELS"
    python pipeline/run_model_comparison.py \
        --models $CMP_MODELS \
        --junction_bps "$JB" --split_strategy transcript \
        --epochs $EPOCHS --earlystop $EARLYSTOP --device "$DEVICE" --seed "$SEED" \
        --tag "ablctx_jb${JB}_seed_${SEED}" --out_dir research_results
  done
done

echo ""
echo "=== ABL-CTX done ($(date +%H:%M:%S)) ==="
echo "Results: research_results/model_comparison_ablctx_jb*_seed_*.csv"
echo "NOTE: external eval at each window needs the circAtlas set rebuilt per junction_bps"
echo "      (make_circatlas_exon_controls + extract_external_fm_embeddings --junction_bps)."
