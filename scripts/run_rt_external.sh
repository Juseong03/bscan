#!/bin/bash
# Second external benchmark: rt_circRNA (read-through circRNAs) — independent source
# from circAtlas. Mirrors the circAtlas external phase: extract FM embeddings on the
# rt_circRNA matched-control set, then evaluate baselines + headline FM models with
# the trained seed checkpoints. Confirms the generalization finding on a 2nd dataset.
#
# Usage (from repo root):  bash scripts/run_rt_external.sh [GPU] [SEEDS]
set -u
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

GPU="${1:-0}"
SEEDS="${2:-1 2 3 4 5 6 7 8 9 10}"
RT=external_data/rt_circrna/matched_controls
SEQ="$RT/seq_dict/junction.json"
EMB="$RT/fm_embeddings"
ENCS="rnafm rnabert rnaernie rnamsm"

[ -f "$SEQ" ] || { echo "[error] $SEQ missing (rt_circRNA seq_dict)"; exit 1; }
echo "rt_circRNA external | seq=$SEQ | gpu=$GPU | seeds=$SEEDS"

# 1) FM embeddings for rt_circRNA (resumable; skips existing .pt)
for ENC in $ENCS; do
  if [ ! -d "$EMB/$ENC" ] || [ -z "$(ls "$EMB/$ENC"/*.pt 2>/dev/null)" ]; then
    echo ">>> extract FM embeddings: $ENC"
    python pipeline/extract_external_fm_embeddings.py \
        --junction_json "$SEQ" --model "$ENC" --out_dir "$EMB" \
        --device "cuda:$GPU" --batch_size 256
  fi
done

# 2) baselines + embed-only FM (features computed live; no cached emb needed)
echo ">>> baselines external eval"
python pipeline/evaluate_circatlas_all_baselines.py \
    --seq_json "$SEQ" --out_dir "$RT" --device "cuda:$GPU" --seeds $SEEDS

# 3) headline cached-FM models (+ ablation/head variants) on rt_circRNA embeddings
echo ">>> FM external eval"
python analysis/evaluate_circatlas_fm_unified.py \
    --seq_json "$SEQ" --out_dir "$RT" --fm_emb_dir "$EMB" \
    --device "cuda:$GPU" --seeds $SEEDS \
    --models bscan_unified_fm bscan_unified_ernie bscan_unified_bert bscan_unified_msm

echo ""
echo "=== rt_circRNA external summary ==="
cat "$RT/all_model_external_control_summary.csv" 2>/dev/null | head
cat "$RT/all_fm_external_control_summary.csv" 2>/dev/null | head
echo ""
echo "Done. Compare to circAtlas: FM should stay high (~0.8), baselines collapse."
