#!/bin/bash

# Final BS/LS comparison: our main model versus the actual baselines we keep in the paper.
# `bscan_seq_lite` is the implementation code name for the model we now present as BSCAN.

SEEDS=(42 123 315)
DEVICE=1
TAG="publication_classification_comparison"
OUT_DIR="research_results"

MODELS_TO_COMPARE=(
    "bscan_seq_lite"
    "deepcirccode"
    "circdeep"
    "jedi"
    "circdc"
    "circcnn"
    "circcnndouble"
    "circcnntri"
)

echo "Starting publication classification comparison..."
echo "Models: ${MODELS_TO_COMPARE[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Device: ${DEVICE}"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Seed: $SEED"
    echo "=========================================="

    python pipeline/run_model_comparison.py \
        --models ${MODELS_TO_COMPARE[*]} \
        --epochs 100 \
        --earlystop 10 \
        --batch_size 128 \
        --batch_size_pretrained 8 \
        --device $DEVICE \
        --seed $SEED \
        --tag "${TAG}_seed_${SEED}" \
        --out_dir ${OUT_DIR}
done

echo ""
echo "Publication comparison complete."
