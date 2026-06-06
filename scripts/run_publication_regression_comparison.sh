#!/bin/bash

# Publication regression comparison.
# The implementation name `bscan_seq_lite_regression` is the model we now
# present as BSCAN in the paper.
#
# Supported regression baselines in this repo:
#   - deepcirccode_regression
#   - circcnn_regression

SEEDS=(42 123 315)
DEVICE=1
TAG="publication_regression_comparison"
OUT_DIR="research_results"

MODELS_TO_COMPARE=(
    "bscan_seq_lite_regression"
    "deepcirccode_regression"
    "circcnn_regression"
)

echo "Starting publication regression comparison..."
echo "Models: ${MODELS_TO_COMPARE[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Device: ${DEVICE}"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Seed: $SEED"
    echo "=========================================="

    python pipeline/experiment_regression.py \
        --model_name bscan_seq_lite_regression \
        --epochs 200 \
        --earlystop 10 \
        --batch_size 64 \
        --loss smoothl1 \
        --corr_loss_weight 0.02 \
        --early_metric composite \
        --device $DEVICE \
        --seed $SEED \
        | tee "logs/${TAG}_bscan_seq_lite_regression_seed_${SEED}.log"

    python pipeline/experiment_regression.py \
        --model_name deepcirccode_regression \
        --epochs 200 \
        --earlystop 10 \
        --batch_size 64 \
        --loss smoothl1 \
        --corr_loss_weight 0.0 \
        --early_metric mae \
        --device $DEVICE \
        --seed $SEED \
        | tee "logs/${TAG}_deepcirccode_regression_seed_${SEED}.log"

    python pipeline/experiment_regression.py \
        --model_name circcnn_regression \
        --epochs 200 \
        --earlystop 10 \
        --batch_size 64 \
        --loss smoothl1 \
        --corr_loss_weight 0.0 \
        --early_metric mae \
        --device $DEVICE \
        --seed $SEED \
        | tee "logs/${TAG}_circcnn_regression_seed_${SEED}.log"
done

python pipeline/summarize_publication_regression.py
