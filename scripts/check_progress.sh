#!/bin/bash
# Snapshot of a running (or finished) run_multi_gpu.sh job.
# Shows GPU activity, what each GPU log is doing now, and how many
# checkpoints / result CSVs exist so far.
#
# Usage (run from repo root, any time during/after a run):
#   bash scripts/check_progress.sh
#   watch -n 30 bash scripts/check_progress.sh     # auto-refresh every 30s
set -u
cd "$(dirname "$0")/.." || exit 1

LOG="logs/multigpu"
SEEDS_DEFAULT="1 2 3 4 5 6 7 8 9 10"
read -r -a SEEDS <<< "${1:-$SEEDS_DEFAULT}"

echo "================ BSCAN run progress  ($(date '+%F %T')) ================"

# 1) GPU activity ------------------------------------------------------------
echo ""; echo "── GPUs ──"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader,nounits |
    awk -F', ' '{printf "  GPU %s: util %3s%%  mem %5s/%5s MiB\n",$1,$2,$3,$4}'
else
  echo "  (nvidia-smi not available)"
fi

# 2) What each GPU log is doing now ------------------------------------------
echo ""; echo "── live log tails ($LOG) ──"
if ls "$LOG"/*.log >/dev/null 2>&1; then
  for f in "$LOG"/*.log; do
    last=$(grep -aE '^\[run\]|^>>>|^===|^####|Test \||Epoch' "$f" 2>/dev/null | tail -1)
    [ -z "$last" ] && last=$(tail -1 "$f" 2>/dev/null)
    printf "  %-26s %s\n" "$(basename "$f"):" "${last:0:90}"
  done
else
  echo "  (no logs yet in $LOG)"
fi

# 3) Internal training progress (valint comparison CSVs) ---------------------
echo ""; echo "── internal training (model_comparison_valint_seed_*.csv) ──"
done_seeds=0
for S in "${SEEDS[@]}"; do
  csv="research_results/model_comparison_valint_seed_${S}.csv"
  if [ -f "$csv" ]; then
    n=$(($(wc -l < "$csv") - 1))   # minus header
    printf "  seed %-5s ✅ %2d models\n" "$S" "$n"
    done_seeds=$((done_seeds+1))
  else
    printf "  seed %-5s ⏳ pending\n" "$S"
  fi
done
echo "  → ${done_seeds}/${#SEEDS[@]} seeds have a comparison CSV"

# 4) Checkpoints on disk -----------------------------------------------------
echo ""; echo "── checkpoints (saved_models/<model>/<seed>/model.pth) ──"
total=$(find -L saved_models -name 'model.pth' 2>/dev/null | wc -l | tr -d ' ')
echo "  total model.pth: $total"
for S in "${SEEDS[@]}"; do
  c=$(find -L saved_models -path "*/$S/model.pth" 2>/dev/null | wc -l | tr -d ' ')
  [ "$c" -gt 0 ] && printf "    seed %-5s : %2d models\n" "$S" "$c"
done

# 5) External + new experiments ---------------------------------------------
echo ""; echo "── external / new experiments ──"
fe="external_data/circatlas/exon_controls/all_fm_external_control_summary.csv"
be="external_data/circatlas/exon_controls/all_model_external_control_summary.csv"
[ -f "$be" ] && echo "  ✅ external baselines summary ($(($(wc -l < "$be")-1)) models)" || echo "  ⏳ external baselines: pending"
[ -f "$fe" ] && echo "  ✅ external FM summary ($(($(wc -l < "$fe")-1)) models)"        || echo "  ⏳ external FM: pending"
na=$(ls research_results/model_comparison_augrcm_*.csv 2>/dev/null | wc -l | tr -d ' ')
nc=$(ls research_results/model_comparison_ablctx_*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "  AUG-RCM result CSVs : $na    ABL-CTX result CSVs : $nc"
[ -f results/paper_table_master.csv ] && echo "  ✅ results/paper_table_master.csv present (final aggregation)"

echo ""; echo "=========================================================================="
