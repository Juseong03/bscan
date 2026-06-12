#!/bin/bash
# ============================================================================
# BSCAN — 3-GPU parallel dispatcher
# Splits the reproduction workload across 3 GPUs along collision-free axes:
#   emb   : by ENCODER   (4 FM encoders → 3 GPUs; each writes its own dir)
#   main  : by SEED      (round-robin across GPUs; saved_models/<m>/<seed>/ +
#                         model_comparison_<tag-with-seed>.csv are seed-disjoint)
#   newexp: by EXPERIMENT (AUG-RCM | ABL-CTX jb250 | ABL-CTX jb500)
# external + analysis are inference-only and light → run once on GPU 0.
#
# Usage (run from repo root):
#   bash scripts/run_multi_gpu.sh [STAGE] [SEEDS] [JOBS_PER_GPU]
#     STAGE        emb | main | external | newexp | all   (default: all)
#     SEEDS        quoted seed list (default: the paper's 10 seeds).
#     JOBS_PER_GPU concurrent training jobs per GPU (default 2). On a 40GB GPU
#                  the models are tiny, so use 3-4 to pack it (watch nvidia-smi).
#             Fast pilot : bash scripts/run_multi_gpu.sh main "1 2 3"
#             Paper, pack: bash scripts/run_multi_gpu.sh main "" 4     (10 seeds, 4/GPU)
#
# Long-running — launch under tmux/nohup so it survives disconnects:
#   tmux new -s bscan 'bash scripts/run_multi_gpu.sh all'
#   # or:  nohup bash scripts/run_multi_gpu.sh all > logs/multigpu/run.log 2>&1 &
#
# Per-GPU logs land in logs/multigpu/. Stages run sequentially (emb → main →
# external → newexp); within a stage the 3 GPUs run concurrently then `wait`.
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

STAGE="${1:-all}"
# Seeds: quoted 2nd arg overrides. Default = the paper's 10 seeds
# Default 1-10. For a fast pilot pass e.g. "1 2 3".
read -r -a SEEDS <<< "${2:-1 2 3 4 5 6 7 8 9 10}"
# JOBS_PER_GPU: concurrent training jobs packed onto each GPU (3rd arg). BSCAN
# models are small (~2M params, cached FM ~3-4GB/job), so a 40GB GPU is mostly
# idle at 1/GPU. On 40GB use 3-4 (watch nvidia-smi; bump if mem <50%).
JOBS_PER_GPU="${3:-2}"
GPUS=(0 1 2)
NG=${#GPUS[@]}
ENC0="rnafm"; ENC1="rnabert rnaernie"; ENC2="rnamsm"   # encoder split for emb
BATCH=256
LOG="logs/multigpu"
mkdir -p "$LOG"
SEEDLIST="${SEEDS[*]}"        # space-joined, for passing to sub-scripts

stage(){ [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }
hdr(){ echo ""; echo "############ [$(date +%H:%M:%S)] $* ############"; }

# ---------------------------------------------------------------------------
# emb — FM embeddings, split by encoder (one-time prerequisite for FM models)
# ---------------------------------------------------------------------------
EXT_SEQ="external_data/circatlas/exon_controls/seq_dict/junction.json"
EXT_TSV="external_data/circatlas/exon_controls/circatlas_exon_external_controls.tsv"

if stage emb; then
  hdr "emb [internal]: FM embeddings (encoder split: gpu0=$ENC0 gpu1=$ENC1 gpu2=$ENC2)"
  bash scripts/extract_all_fm_embeddings.sh 0 "$ENC0" internal "$BATCH" > "$LOG/emb_int_gpu0.log" 2>&1 &
  bash scripts/extract_all_fm_embeddings.sh 1 "$ENC1" internal "$BATCH" > "$LOG/emb_int_gpu1.log" 2>&1 &
  bash scripts/extract_all_fm_embeddings.sh 2 "$ENC2" internal "$BATCH" > "$LOG/emb_int_gpu2.log" 2>&1 &
  wait
  echo "emb[internal] done → $LOG/emb_int_gpu*.log"

  hdr "emb [external]: build circAtlas seq_dict, then extract (encoder split)"
  [ -f "$EXT_TSV" ] || python pipeline/make_circatlas_exon_controls.py
  [ -f "$EXT_SEQ" ] || python pipeline/build_circatlas_seq_dict.py
  if [ -f "$EXT_SEQ" ]; then
    bash scripts/extract_all_fm_embeddings.sh 0 "$ENC0" external "$BATCH" > "$LOG/emb_ext_gpu0.log" 2>&1 &
    bash scripts/extract_all_fm_embeddings.sh 1 "$ENC1" external "$BATCH" > "$LOG/emb_ext_gpu1.log" 2>&1 &
    bash scripts/extract_all_fm_embeddings.sh 2 "$ENC2" external "$BATCH" > "$LOG/emb_ext_gpu2.log" 2>&1 &
    wait
    echo "emb[external] done → $LOG/emb_ext_gpu*.log"
  else
    echo "[note] $EXT_SEQ missing (genome required) — skipped external embeddings."
  fi
fi

# ---------------------------------------------------------------------------
# main — internal train + branch ablation + hard-negative, one SEED per GPU
# ---------------------------------------------------------------------------
if stage main; then
  hdr "main: train+ablation+hardneg | ${#SEEDS[@]} seeds round-robin over ${NG} GPUs"
  echo "  seeds: $SEEDLIST"
  # Pre-generate rcm_scores ONCE before the parallel split. Otherwise the parallel
  # train jobs would race to build rcm_scores/ for circcnntri and corrupt it.
  # Seed-independent (k-mer counts on sequences), so a single build serves all.
  if [ -z "$(ls rcm_scores 2>/dev/null)" ]; then
    echo "[prep] generating rcm_scores (once, full coverage) ..."
    python pipeline/generate_rcm_scores_subset.py \
        --junction_bps 100 --flanking_bps 100 --max_samples 100000 > "$LOG/prep_rcm.log" 2>&1
  fi
  # Worker pool: NWORKERS = NG * JOBS_PER_GPU. Worker wi runs on GPU[wi % NG],
  # so JOBS_PER_GPU workers share each physical GPU concurrently (they target
  # the same --device; (model,seed) checkpoint dirs are disjoint → no collision).
  # Seeds are round-robined across workers.
  NWORKERS=$(( NG * JOBS_PER_GPU ))
  echo "  ${#SEEDS[@]} seeds | ${NG} GPUs x ${JOBS_PER_GPU} jobs = ${NWORKERS} workers"
  for (( wi=0; wi<NWORKERS; wi++ )); do
    G="${GPUS[$(( wi % NG ))]}"
    mine=()
    for si in "${!SEEDS[@]}"; do
      (( si % NWORKERS == wi )) && mine+=("${SEEDS[$si]}")
    done
    [ "${#mine[@]}" -eq 0 ] && continue
    echo "  worker $wi → GPU $G ← seeds: ${mine[*]}"
    (
      for S in "${mine[@]}"; do
        bash scripts/run_all_experiments.sh train    "$G" "$S"
        bash scripts/run_all_experiments.sh ablation "$G" "$S"
        bash scripts/run_all_experiments.sh hardneg  "$G" "$S"
      done
    ) > "$LOG/main_w${wi}_gpu${G}.log" 2>&1 &
  done
  wait
  echo "main done → see $LOG/main_w*_gpu*.log"
fi

# ---------------------------------------------------------------------------
# external + analysis — inference-only, run once on GPU 0
# ---------------------------------------------------------------------------
if stage external; then
  hdr "external + analysis (single GPU 0)"
  bash scripts/run_all_experiments.sh external "${GPUS[0]}" "$SEEDLIST" > "$LOG/external.log" 2>&1
  bash scripts/run_all_experiments.sh analysis "${GPUS[0]}" "$SEEDLIST" > "$LOG/analysis.log" 2>&1
  echo "external+analysis done → see $LOG/external.log $LOG/analysis.log"
fi

# ---------------------------------------------------------------------------
# newexp — new experiments split by experiment across the 3 GPUs
# ---------------------------------------------------------------------------
if stage newexp; then
  hdr "newexp: AUG-RCM (gpu0) | ABL-CTX jb250 (gpu1) | ABL-CTX jb500 (gpu2)"
  bash scripts/run_rcm_aux.sh            "100 500" 0 "$SEEDLIST" > "$LOG/newexp_augrcm_gpu0.log"  2>&1 &
  bash scripts/run_context_window_sweep.sh "250"  rnafm 1 "$SEEDLIST" > "$LOG/newexp_ablctx250_gpu1.log" 2>&1 &
  bash scripts/run_context_window_sweep.sh "500"  rnafm 2 "$SEEDLIST" > "$LOG/newexp_ablctx500_gpu2.log" 2>&1 &
  wait
  echo "newexp done → see $LOG/newexp_*.log"
fi

hdr "ALL DONE (stage=$STAGE). Logs in $LOG/  |  results in research_results/ & results/"
