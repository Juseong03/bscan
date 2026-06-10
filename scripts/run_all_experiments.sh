#!/bin/bash
# ============================================================================
# BSCAN — full experiment reproduction orchestrator
# Reproduces the completed registry experiments (docs/EXPERIMENTS.md) in
# dependency order. Run from repo root.
#
# Usage:
#   bash scripts/run_all_experiments.sh [PHASE] [DEVICE] [SEEDS]
#     PHASE   all | emb | train | external | ablation | hardneg | analysis
#             (default: all)
#     DEVICE  GPU id (default: 0)
#     SEEDS   quoted seed list (default: "42 123 315")
#
# Examples:
#   bash scripts/run_all_experiments.sh emb 0                 # only FM embeddings
#   bash scripts/run_all_experiments.sh train 0 "42 123 315"  # only training
#   bash scripts/run_all_experiments.sh all 0 "42 123 315"    # everything
#
# Prerequisites (see docs/SETUP_NEW_SERVER.md):
#   data/seq_dict/100/        (required)
#   data/hg19_seq_dict.json   (only if embeddings/seq_dict need rebuilding)
#   data/rmsk_hg19.txt.gz      (only for ALU analysis; UCSC)
#   external_data/.../seq_dict (only for external eval; or build it)
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1   # repo root

PHASE="${1:-all}"
DEVICE="${2:-0}"
SEEDS="${3:-42 123 315}"
ENCODERS="rnafm rnabert rnaernie rnamsm"
FM_MODELS="bscan_unified_fm bscan_unified_ernie bscan_unified_bert bscan_unified_msm"
ONEHOT_MODELS="bscan_unified_onehot bscan"
BASELINES="circcnn circcnnsingle circcnndouble deepcirccode circdc jedi circdeep"
EPOCHS=100
EARLYSTOP=30
OUT=research_results

run(){ echo ">>> $*"; "$@"; }
phase(){ [ "$PHASE" = "all" ] || [ "$PHASE" = "$1" ]; }
log(){ echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="; }

echo "PHASE=$PHASE  DEVICE=$DEVICE  SEEDS=$SEEDS"

# ---------------------------------------------------------------------------
# Phase 1 — FM embeddings (VAL/ABL prerequisite for FM models)
# ---------------------------------------------------------------------------
EXT_SEQ="external_data/circatlas/exon_controls/seq_dict/junction.json"
EXT_TSV="external_data/circatlas/exon_controls/circatlas_exon_external_controls.tsv"

if phase emb; then
  log "Phase 1: FM embeddings (internal + external)"
  bash scripts/extract_all_fm_embeddings.sh "$DEVICE" "$ENCODERS" internal 256
  # External embeddings need the circAtlas seq_dict (junction.json) to exist
  # first. Build the coordinate-control TSVs here so an `all` run is correctly
  # ordered; note that turning the TSVs into seq_dict/junction.json (sequence
  # extraction from the genome) is a separate step not bundled here.
  [ -f "$EXT_TSV" ] || run python pipeline/make_circatlas_exon_controls.py
  if [ -f "$EXT_SEQ" ]; then
    bash scripts/extract_all_fm_embeddings.sh "$DEVICE" "$ENCODERS" external 256
  else
    echo "[note] $EXT_SEQ not found — skipping external embedding extraction."
    echo "       Build the external seq_dict from $EXT_TSV, then re-run: bash scripts/run_all_experiments.sh emb"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 2 — Internal training: VAL-INT (all models, transcript split)
# ---------------------------------------------------------------------------
if phase train; then
  log "Phase 2: internal training (VAL-INT)"
  for SEED in $SEEDS; do
    run python pipeline/run_model_comparison.py \
        --models $FM_MODELS $ONEHOT_MODELS $BASELINES \
        --split_strategy transcript --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device "$DEVICE" --seed "$SEED" \
        --tag "valint_seed_${SEED}" --out_dir "$OUT"
  done
fi

# ---------------------------------------------------------------------------
# Phase 3 — External validation + leakage: VAL-EXT, VAL-LEAK
# ---------------------------------------------------------------------------
if phase external; then
  log "Phase 3: external validation (VAL-EXT)"
  # evaluate_circatlas reads the external seq_dict directly; make_circatlas only
  # produces the coordinate TSVs, so require the seq_dict explicitly here.
  if [ ! -f "$EXT_SEQ" ]; then
    echo "[error] $EXT_SEQ missing — build the circAtlas external seq_dict first"
    echo "        (pipeline/make_circatlas_exon_controls.py → extract sequences → seq_dict/junction.json)."
    echo "        Skipping external validation."
  else
    run python pipeline/evaluate_circatlas_all_baselines.py --device "$DEVICE"
  fi
  log "Phase 3b: leakage controls (VAL-LEAK)"
  run python analysis/analyze_external_b_disjoint.py
  run python analysis/make_external_b_hostgene.py   # needs pyliftover
fi

# ---------------------------------------------------------------------------
# Phase 4 — Branch ablation: ABL-BRANCH
# ---------------------------------------------------------------------------
if phase ablation; then
  log "Phase 4: branch ablation (ABL-BRANCH)"
  ABL="bscan_unified_fm_fulltr bscan_unified_fm_cnnonly bscan_unified_fm_stemonly \
       bscan_unified_fm_attnonly bscan_unified_fm_nocnn bscan_unified_fm_nostem bscan_unified_fm_noattn"
  for M in $ABL; do for SEED in $SEEDS; do
    [ -f "saved_models/$M/$SEED/model.pth" ] && { echo "[skip] $M $SEED"; continue; }
    run python pipeline/experiment.py --model_name "$M" --split_strategy transcript \
        --epochs $EPOCHS --earlystop $EARLYSTOP --device "$DEVICE" --seed "$SEED"
  done; done
  run python analysis/evaluate_ablation.py
fi

# ---------------------------------------------------------------------------
# Phase 5 — Hard negative: MECH-HN3, MECH-HNAUG
# ---------------------------------------------------------------------------
if phase hardneg; then
  log "Phase 5: hard-negative 3-tier probe (MECH-HN3)"
  for MODE in lower_intron ls_lower_intron upper_intron both_introns; do
    run python pipeline/evaluate_hard_negative_pairing.py \
        --models bscan circcnn circcnndouble circdc jedi circcnnsingle deepcirccode \
                 $FM_MODELS \
        --negative-mode "$MODE" --seeds $SEEDS --device "cuda:$DEVICE" --out-dir "$OUT"
  done
  log "Phase 5b: hard-negative augmented training (MECH-HNAUG)"
  run python pipeline/train_hard_negative_augmented.py --models bscan circcnn --seeds $SEEDS
  run python pipeline/train_hard_negative_augmented_fm.py --enc-type rnafm --seeds $SEEDS --device "$DEVICE"
fi

# ---------------------------------------------------------------------------
# Phase 6 — Mechanism/enhancement analyses (inference-only on checkpoints)
# ---------------------------------------------------------------------------
if phase analysis; then
  log "Phase 6: masking / ALU / duplex / stats"
  run python analysis/analyze_masking.py
  if [ -f data/rmsk_hg19.txt.gz ]; then
    run python analysis/analyze_alu_repeats.py
    run python analysis/analyze_alu_multiscale.py
    run python analysis/analyze_alu_matched_tier2.py
  else
    echo "[skip] ALU: data/rmsk_hg19.txt.gz missing (wget from UCSC)"
  fi
  run python analysis/analyze_duplex_alpha.py
  run python analysis/analyze_statistics.py
fi

log "DONE (phase=$PHASE). Results in results/ and $OUT/. See docs/EXPERIMENTS.md"
