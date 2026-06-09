#!/bin/bash
# Generate RCM scores (and the matching flanking_{N}.json) for several flanking
# window sizes, then optionally train an RCM model at each size.
#
# RCM captures reverse-complementary (ALU-like) matches in the intronic flanks —
# a longer-range signal, so larger windows (250–500+) capture more than 100bp.
#
# Requires the hg19 genome (data/hg19_seq_dict.json) because flanking_{N>100}.json
# is rebuilt from it. (On a fresh server: tar -xzf bscan_genome.tar.gz first.)
#
# Usage (run from repo root):
#   bash scripts/run_rcm_flanking_sweep.sh [FLANKINGS] [MODEL] [DEVICE] [SEED]
#     FLANKINGS  quoted list (default: "100 200 300 400 500")
#     MODEL      circcnntri | circcnnrcm | none  (default: none = only build RCM)
#     DEVICE     GPU id for training (default: 0)
#     SEED       seed (default: 42)
#
# Examples:
#   bash scripts/run_rcm_flanking_sweep.sh                                  # build RCM for 100..500
#   bash scripts/run_rcm_flanking_sweep.sh "100 250 500" circcnntri 0 42    # build + train circcnntri

set -u
cd "$(dirname "$0")/.." || exit 1   # repo root

FLANKINGS="${1:-100 200 300 400 500}"
MODEL="${2:-none}"
DEVICE="${3:-0}"
SEED="${4:-42}"
MAXS=100000   # large → no subsampling → all keys

if [ ! -f data/hg19_seq_dict.json ] && ! ls data/seq_dict/100/flanking_*.json >/dev/null 2>&1; then
    echo "WARNING: data/hg19_seq_dict.json not found and only flanking_100 may exist."
    echo "         Larger flanking sizes need the genome (tar -xzf bscan_genome.tar.gz)."
fi

echo "=================================================="
echo "RCM flanking sweep"
echo "  flankings: ${FLANKINGS}"
echo "  model    : ${MODEL}   device: cuda:${DEVICE}   seed: ${SEED}"
echo "=================================================="

for F in ${FLANKINGS}; do
    echo ""
    echo ">>> [RCM] flanking_bps=${F}  ($(date +%H:%M:%S))"
    # builds data/seq_dict/100/flanking_${F}.json (from genome if missing) + rcm_scores/*_${F}_bps_*
    python pipeline/generate_rcm_scores_subset.py \
        --max_samples "${MAXS}" --flanking_bps "${F}" --seed "${SEED}" --out_dir ./rcm_scores

    if [ "${MODEL}" != "none" ]; then
        echo ">>> [train] ${MODEL}  flanking_bps=${F}  ($(date +%H:%M:%S))"
        python pipeline/experiment.py --model_name "${MODEL}" \
            --flanking_bps "${F}" --split_strategy transcript \
            --device "${DEVICE}" --seed "${SEED}"
    fi
done

echo ""
echo "=== Done ($(date +%H:%M:%S)) ==="
echo "RCM scores → ./rcm_scores/{flanking,upper,lower}_<F>_bps_<k>mer_scores.json"
