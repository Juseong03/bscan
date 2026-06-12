#!/bin/bash
# Shared helper for the per-experiment server launchers (scripts/server/*).
# distribute <exp> <jobs_per_gpu> <seed...> : round-robins the seeds across
# GPUS with JOBS_PER_GPU workers per GPU, each calling scripts/exp/<exp>.sh.
# Workers on the same GPU get a unique log suffix so logs don't interleave.

GPUS=(${BSCAN_GPUS:-0 1 2})     # override with BSCAN_GPUS="0 1" for a 2-GPU box

distribute() {
  local exp="$1" jpg="$2"; shift 2
  local seeds=("$@")
  local ng=${#GPUS[@]}
  local nworkers=$(( ng * jpg ))
  mkdir -p logs/exp
  echo "[$exp] ${#seeds[@]} seeds | ${ng} GPUs x ${jpg} jobs = ${nworkers} workers"
  for (( wi=0; wi<nworkers; wi++ )); do
    local g="${GPUS[$(( wi % ng ))]}"
    local mine=()
    for si in "${!seeds[@]}"; do (( si % nworkers == wi )) && mine+=("${seeds[$si]}"); done
    [ "${#mine[@]}" -eq 0 ] && continue
    echo "  worker $wi → GPU $g ← seeds: ${mine[*]}"
    EXP_LOG_SUFFIX="_w${wi}" bash "scripts/exp/${exp}.sh" "$g" "${mine[*]}" &
  done
  wait
  echo "[$exp] all workers finished — check: bash scripts/exp_status.sh"
}
