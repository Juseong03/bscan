#!/bin/bash
# Completion counter for the per-experiment scripts (scripts/exp/*).
# Counts EXP_DONE / EXP_FAIL markers across logs/exp/*.log.
#
# Usage:  bash scripts/exp_status.sh
#         watch -n 30 bash scripts/exp_status.sh
set -u
cd "$(dirname "$0")/.." || exit 1
LOGDIR="logs/exp"

echo "===== experiment status  ($(date '+%F %T')) ====="
if ! ls "$LOGDIR"/*.log >/dev/null 2>&1; then
  echo "no logs yet in $LOGDIR/  (run scripts/exp/*.sh first)"; exit 0
fi

# Per-experiment done/fail counts (a seed-looped exp emits one DONE per seed).
echo ""
printf "%-16s %6s %6s   %s\n" "experiment" "done" "fail" "seeds done"
for name in prep val_int abl_branch mech_hardneg val_ext analysis aug_rcm abl_ctx; do
  done=$(grep -rhoE "EXP_DONE $name( seed=[0-9]+)?" "$LOGDIR" 2>/dev/null | wc -l | tr -d ' ')
  fail=$(grep -rhoE "EXP_FAIL $name( seed=[0-9]+)?" "$LOGDIR" 2>/dev/null | wc -l | tr -d ' ')
  seeds=$(grep -rhoE "EXP_DONE $name seed=[0-9]+" "$LOGDIR" 2>/dev/null | grep -oE "[0-9]+$" | sort -un | paste -sd' ' -)
  [ "$done" = "0" ] && [ "$fail" = "0" ] && continue
  printf "%-16s %6s %6s   %s\n" "$name" "$done" "$fail" "$seeds"
done

# Currently running (START without matching DONE/FAIL in same log) — quick hint.
echo ""
echo "── currently running (latest line per log) ──"
for f in "$LOGDIR"/*.log; do
  last=$(tail -1 "$f" 2>/dev/null)
  printf "  %-26s %s\n" "$(basename "$f"):" "${last:0:80}"
done

echo ""
echo "tip: failures →  grep -rn EXP_FAIL $LOGDIR"
