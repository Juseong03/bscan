#!/bin/bash
# At-a-glance health check for a run_multi_gpu.sh / run_all_experiments.sh run.
# Prints a single VERDICT (running / stalled / idle) + the few numbers that matter.
#
# Usage:  bash scripts/check_progress.sh
#         watch -n 60 bash scripts/check_progress.sh
set -u
cd "$(dirname "$0")/.." || exit 1
read -r -a SEEDS <<< "${1:-1 2 3 4 5 6 7 8 9 10}"
NSEED=${#SEEDS[@]}
STAMP="/tmp/.bscan_progress_stamp"

# --- gather numbers --------------------------------------------------------
procs=$(pgrep -fc "run_model_comparison|pipeline/experiment.py|evaluate_ablation|evaluate_hard_negative|run_context_window|run_rcm_aux" 2>/dev/null)
procs=${procs:-0}
cores=$(nproc 2>/dev/null || echo 1)
load1=$(awk '{printf "%.0f", $1}' /proc/loadavg 2>/dev/null || echo 0)
ckpt=$(find -L saved_models -name model.pth 2>/dev/null | grep -cE "/($(IFS='|'; echo "${SEEDS[*]}"))/model.pth$")
# "moving" = new CHECKPOINTS in last 5 min (a failed run still writes empty CSVs,
# so counting any file would falsely look healthy).
recent=$(find -L saved_models -name model.pth -newermt "5 min ago" 2>/dev/null | wc -l | tr -d ' ')
valint=0; for S in "${SEEDS[@]}"; do [ -f "research_results/model_comparison_valint_seed_${S}.csv" ] && valint=$((valint+1)); done
errs=$(grep -rliE "traceback|out of memory|MemoryError" logs/multigpu logs/exp 2>/dev/null | wc -l | tr -d ' ')

# delta vs previous run of this script
prev=0; prevt=0
[ -f "$STAMP" ] && read -r prev prevt < "$STAMP"
now=$(date +%s); echo "$ckpt $now" > "$STAMP"
delta=$((ckpt - prev)); mins=$(( (now - prevt) / 60 ))

# --- verdict ---------------------------------------------------------------
if [ "$procs" -eq 0 ]; then
  verdict="[IDLE]  no job running (finished / stopped / not started yet)"
elif [ "$recent" -gt 0 ] || { [ "$prevt" -gt 0 ] && [ "$delta" -gt 0 ]; }; then
  verdict="[OK]  RUNNING (healthy)"
elif [ "$load1" -gt "$cores" ]; then
  verdict="[!!]  STALL? CPU OVERLOAD  (load $load1 > cores $cores = thrashing)"
else
  verdict="[!!]  STALL? no new files in last 5 min (check)"
fi

echo "============ BSCAN progress  $(date '+%m-%d %H:%M') ============"
echo "  $verdict"
echo "------------------------------------------------------"
printf "  processes running : %s\n" "$procs"
printf "  CPU load          : %s / %s cores   %s\n" "$load1" "$cores" \
       "$([ "$load1" -le "$cores" ] && echo '[OK]' || echo '[OVERLOAD]')"
if [ "$prevt" -gt 0 ]; then
  printf "  checkpoints       : %s   (%+d since last check, %s min ago)\n" "$ckpt" "$delta" "$mins"
else
  printf "  checkpoints       : %s   (run again to see the increase)\n" "$ckpt"
fi
printf "  new files (5 min) : %s   %s\n" "$recent" "$([ "$recent" -gt 0 ] && echo '<- moving' || echo '<- no change')"
printf "  val_int done      : %s / %s seeds\n" "$valint" "$NSEED"
printf "  errors            : %s\n" "$([ "$errs" -eq 0 ] && echo 'none [OK]' || echo "$errs log file(s) [!!]  grep -ri traceback logs/")"
echo "------------------------------------------------------"
echo "  detail: tail -f logs/multigpu/main_w0_gpu0.log"
echo "======================================================"
