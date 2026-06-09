#!/bin/bash
# AUG-RCM — RCM auxiliary-branch study (docs/DESIGN_new_experiments.md §2)
# "If we explicitly feed the model the flanking reverse-complement-match (RCM)
#  signal the FM never sees, does external generalization improve, or is the
#  exon-bias structural?"  Compares bscan_unified_fm vs bscan_unified_fm_rcm
#  at the SAME 200nt junction window (only the aux feature differs).
#
# For each flanking width: (1) extract RCM k-mer scores into rcm_scores/,
# (2) train FM baseline + FM+RCM across seeds.
#
# Usage (run from repo root):
#   bash scripts/run_rcm_aux.sh [FLANKS] [DEVICE] [SEEDS]
#     FLANKS   flanking_bps list (default: "100 500")
#     DEVICE   GPU id (default: 0)
#     SEEDS    quoted seeds (default: "42 123 315")
#
# Pilot first:  bash scripts/run_rcm_aux.sh "100" 0 "42"
#
# flanking>100 rebuilds flanking seqs from data/hg19_seq_dict.json (genome).
# RCM extraction uses the full sample set (--max_samples 100000).
set -u
cd "$(dirname "$0")/.." || exit 1

FLANKS="${1:-100 500}"
DEVICE="${2:-0}"
SEEDS="${3:-42 123 315}"
EPOCHS=100; EARLYSTOP=30
OUT=research_results

echo "AUG-RCM | flanks=$FLANKS device=cuda:$DEVICE seeds=$SEEDS"

for FB in $FLANKS; do
  echo ""; echo "=== [$(date +%H:%M:%S)] flanking_bps=$FB ==="

  # 1) RCM k-mer scores for this flanking width (resumable: skips existing files)
  echo ">>> RCM scores (flanking=$FB)"
  python pipeline/generate_rcm_scores_subset.py \
      --junction_bps 100 --flanking_bps "$FB" --max_samples 100000

  # 2) train FM baseline + FM+RCM at this flanking width
  for SEED in $SEEDS; do
    echo ">>> train (flanking=$FB, seed=$SEED): bscan_unified_fm bscan_unified_fm_rcm"
    python pipeline/run_model_comparison.py \
        --models bscan_unified_fm bscan_unified_fm_rcm \
        --junction_bps 100 --flanking_bps "$FB" --split_strategy transcript \
        --epochs $EPOCHS --earlystop $EARLYSTOP --device "$DEVICE" --seed "$SEED" \
        --tag "augrcm_fb${FB}_seed_${SEED}" --out_dir "$OUT"
  done
done

echo ""
echo "=== AUG-RCM done ($(date +%H:%M:%S)) ==="
echo "Results: research_results/model_comparison_augrcm_fb*_seed_*.csv"
echo "Aggregate: python analysis/evaluate_rcm_aux.py"
echo "NOTE: external/Tier2-3 eval reuses the standard circAtlas + hard-negative"
echo "      pipelines on the saved bscan_unified_fm_rcm checkpoints."
