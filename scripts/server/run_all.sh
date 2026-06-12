#!/bin/bash
# Server master — runs every experiment in dependency order, one at a time.
# Each step uses all GPUs; steps are sequential so they never oversubscribe and
# checkpoint dependencies are respected (val_ext/analysis/mech_hardneg need the
# val_int checkpoints; everything needs prep).
#
# Fire once and walk away (under tmux/nohup):
#   tmux new -s bscan 'bash scripts/server/run_all.sh 2'
#   # or: nohup bash scripts/server/run_all.sh 2 > logs/exp/run_all.run 2>&1 &
#
# Arg 1 = JOBS_PER_GPU for the seed-parallel steps (default 2; 3-4 on 40GB).
# Monitor: bash scripts/exp_status.sh   (watch -n 30 ...)
set -u
cd "$(dirname "$0")/../.." || exit 1
JPG="${1:-2}"

step() { echo ""; echo "############ [$(date '+%F %T')] $* ############"; }

step "0/8 prep (rcm_scores + external seq_dict/embeddings)"
bash scripts/server/prep.sh 0            || { echo "prep FAILED — abort"; exit 1; }

step "1/8 val_int (10 seeds, ${JPG}/GPU)"        ; bash scripts/server/val_int.sh "$JPG"
step "2/8 abl_branch (10 seeds, ${JPG}/GPU)"     ; bash scripts/server/abl_branch.sh "$JPG"
step "3/8 mech_hardneg (10 seeds, ${JPG}/GPU)"   ; bash scripts/server/mech_hardneg.sh "$JPG"
step "4/8 val_ext (after val_int)"               ; bash scripts/server/val_ext.sh 0
step "5/8 analysis (after val_int)"              ; bash scripts/server/analysis.sh 0
step "6/8 aug_rcm (5 seeds)"                     ; bash scripts/server/aug_rcm.sh
step "7/8 abl_ctx (5 seeds)"                     ; bash scripts/server/abl_ctx.sh

step "8/8 DONE — summary"
bash scripts/exp_status.sh || true
echo ""
echo "Results: research_results/ , results/ , external_data/circatlas/exon_controls/"
