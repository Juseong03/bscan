SEEDS=(2024 2025 2026 3407 9001)
for SEED in "${SEEDS[@]}"; do
    python pipeline/run_model_comparison.py --epochs 50 --earlystop 10 --batch_size 64 --device 1 --seed $SEED --models bscan_unified_ernie bscan_unified_msm --tag "bscan_unified_final_seed_${SEED}" --out_dir "research_results"
done
