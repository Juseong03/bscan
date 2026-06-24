#!/bin/bash
# Head-comparison study: which downstream on frozen RNA-FM generalizes?
#   1) FM + CNN  + Stem          (bscan_unified_fm_noattn)    -- drop Attn
#   2) FM + Mamba + Stem         (bscan_unified_fm_mambastem) -- local/sequential SSM
#   3) FM + CNN  + Stem + Attn   (bscan_unified_fm_fulltr)    -- reference (full)
# Trains each on the transcript split (internal) and evaluates on the circAtlas
# external set, over N seeds. Hypothesis: (1) ~ (2) >> attention-heavy on external.
#
# Usage (from repo root):  bash scripts/run_head_comparison.sh [GPU] [SEEDS]
#   e.g.  bash scripts/run_head_comparison.sh 0 "1 2 3 4 5"
set -u
cd "$(dirname "$0")/.." || exit 1
# CPU thread cap (Mamba/CNN both CPU-sensitive when concurrent)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

GPU="${1:-0}"
SEEDS="${2:-1 2 3 4 5}"
MODELS="bscan_unified_fm_noattn bscan_unified_fm_mambastem bscan_unified_fm_fulltr"
OUT=research_results
mkdir -p logs/head "$OUT"

echo "Head-comparison | models: $MODELS | gpu=$GPU | seeds: $SEEDS"

# 1) internal training (writes model_comparison_head_seed_<S>.csv with test_auc)
for S in $SEEDS; do
  echo ">>> [$(date +%H:%M:%S)] internal train seed=$S"
  python pipeline/run_model_comparison.py --models $MODELS \
      --split_strategy transcript --epochs 100 --earlystop 30 --batch_size 128 \
      --device "$GPU" --seed "$S" --tag "head_seed_${S}" --out_dir "$OUT" \
      > "logs/head/internal_seed_${S}.log" 2>&1
done

# 2) external evaluation on the saved checkpoints (all seeds at once)
echo ">>> [$(date +%H:%M:%S)] external eval (circAtlas)"
python analysis/evaluate_circatlas_fm_unified.py --models $MODELS \
    --seeds $SEEDS --device "cuda:$GPU" > "logs/head/external.log" 2>&1

# 3) quick aggregate
echo ""
echo "=== INTERNAL (mean test_auc over seeds) ==="
python3 - "$OUT" "$SEEDS" <<'PY'
import sys, glob, csv, statistics as st
from collections import defaultdict
out=sys.argv[1]; seeds=sys.argv[2].split()
d=defaultdict(list)
for s in seeds:
    f=f"{out}/model_comparison_head_seed_{s}.csv"
    try:
        for r in csv.DictReader(open(f)):
            try: d[r["model"]].append(float(r["test_auc"]))
            except: pass
    except FileNotFoundError: pass
for m,v in d.items():
    if v: print(f"  {m:<28} {st.mean(v):.4f} ± {st.pstdev(v):.4f}  (n={len(v)})")
PY
echo ""
echo "=== EXTERNAL (circAtlas) ==="
cat external_data/circatlas/exon_controls/all_fm_external_control_summary.csv 2>/dev/null \
  | awk -F, 'NR==1||/noattn|mambastem|fulltr/{printf "  %-30s %s\n",$1,$3}'
echo ""
echo "Done. Logs in logs/head/. Hypothesis: noattn ~ mambastem (CNN/Mamba+Stem) on external."
