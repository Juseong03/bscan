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
recent=$(find -L saved_models research_results -newermt "5 min ago" -type f 2>/dev/null | wc -l | tr -d ' ')
valint=0; for S in "${SEEDS[@]}"; do [ -f "research_results/model_comparison_valint_seed_${S}.csv" ] && valint=$((valint+1)); done
errs=$(grep -rliE "traceback|out of memory|MemoryError" logs/multigpu logs/exp 2>/dev/null | wc -l | tr -d ' ')

# delta vs previous run of this script
prev=0; prevt=0
[ -f "$STAMP" ] && read -r prev prevt < "$STAMP"
now=$(date +%s); echo "$ckpt $now" > "$STAMP"
delta=$((ckpt - prev)); mins=$(( (now - prevt) / 60 ))

# --- verdict ---------------------------------------------------------------
if [ "$procs" -eq 0 ]; then
  verdict="⏸  실행 중인 작업 없음 (완료/중단되었거나 아직 시작 안 함)"
elif [ "$recent" -gt 0 ] || { [ "$prevt" -gt 0 ] && [ "$delta" -gt 0 ]; }; then
  verdict="✅  진행 중 (정상)"
elif [ "$load1" -gt "$cores" ]; then
  verdict="⚠️  정체 의심 — CPU 과부하 (load $load1 > cores $cores = thrashing)"
else
  verdict="⚠️  멈춘 듯 — 최근 5분간 새 파일 없음 (확인 필요)"
fi

echo "════════════ BSCAN 진행 상황  $(date '+%m-%d %H:%M') ════════════"
echo "  $verdict"
echo "────────────────────────────────────────────────────"
printf "  실행 프로세스   : %s개\n" "$procs"
printf "  CPU 부하        : load %s / %s cores  %s\n" "$load1" "$cores" \
       "$([ "$load1" -le "$cores" ] && echo '✓ 정상' || echo '✗ 과부하')"
if [ "$prevt" -gt 0 ]; then
  printf "  체크포인트      : %s개  (직전 확인 대비 %+d, %s분 전)\n" "$ckpt" "$delta" "$mins"
else
  printf "  체크포인트      : %s개  (다시 실행하면 증가량 표시)\n" "$ckpt"
fi
printf "  최근 5분 새 파일: %s개  %s\n" "$recent" "$([ "$recent" -gt 0 ] && echo '← 움직이는 중' || echo '← 변화 없음')"
printf "  val_int 완료    : %s/%s 시드\n" "$valint" "$NSEED"
printf "  에러 로그       : %s\n" "$([ "$errs" -eq 0 ] && echo '없음 ✓' || echo "$errs개 파일 ⚠️  grep -ri traceback logs/")"
echo "────────────────────────────────────────────────────"
echo "  자세히: tail -f logs/multigpu/main_w0_gpu0.log"
echo "════════════════════════════════════════════════════"
