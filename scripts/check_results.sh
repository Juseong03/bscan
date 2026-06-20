#!/bin/bash
# Report experiment completion by RESULT FILES: which seeds are done vs missing,
# per experiment, so you know exactly what (if anything) to re-run.
# English/ASCII only. Usage:  bash scripts/check_results.sh
set -u
cd "$(dirname "$0")/.." || exit 1
SEEDS="1 2 3 4 5 6 7 8 9 10"
RR=research_results

# print "<label>  N/10 done   MISSING: <seeds>" for a per-seed file pattern.
# $2 is a printf format with one %s for the seed.
cov() {
  local label="$1" pat="$2" have=0 miss=""
  for s in $SEEDS; do
    if [ -f "$(printf "$pat" "$s")" ]; then have=$((have+1)); else miss="$miss $s"; fi
  done
  printf "  %-24s %2d/10" "$label" "$have"
  [ -n "$miss" ] && printf "   MISSING:%s" "$miss"
  echo
}

echo "=========== BSCAN results status  $(date '+%m-%d %H:%M') ==========="
echo "[ headline experiments  (want 10/10 seeds, seeds 1-10) ]"
cov "VAL-INT internal"   "$RR/model_comparison_valint_seed_%s.csv"
cov "ABL-BRANCH (fulltr)" "saved_models/bscan_unified_fm_fulltr/%s/model.pth"
echo
echo "[ auxiliary experiments ]"
cov "AUG-RCM  fb100"      "$RR/model_comparison_augrcm_fb100_seed_%s.csv"
cov "AUG-RCM  fb500"      "$RR/model_comparison_augrcm_fb500_seed_%s.csv"
cov "ABL-CTX  jb250"      "$RR/model_comparison_ablctx_jb250_seed_%s.csv"
cov "ABL-CTX  jb500"      "$RR/model_comparison_ablctx_jb500_seed_%s.csv"
echo
echo "[ aggregated outputs (exist?) ]"
chk() { local f="$1"; shift; [ -f "$f" ] && echo "  [OK]      $* ($f)" || echo "  [MISSING] $* ($f)"; }
chk "$RR/ablation_results.csv"                                                     "ABL-BRANCH"
chk "results/rcm_aux_summary.csv"                                                  "AUG-RCM"
chk "external_data/circatlas/exon_controls/all_fm_external_control_summary.csv"    "VAL-EXT FM"
chk "external_data/circatlas/exon_controls/all_model_external_control_summary.csv" "VAL-EXT baselines"
echo
echo "[ mechanism / leakage / analysis (other main-paper experiments) ]"
chk "$RR/hard_negative_pairing_lower_intron_results.csv"              "MECH-HN3 probe"
chk "$RR/hard_negative_augmented/hard_negative_augmented_results.csv" "MECH-HNAUG"
chk "$RR/external_b_sequence_disjoint.csv"                            "VAL-LEAK seq-disjoint"
chk "$RR/external_b_hostgene_disjoint.csv"                            "VAL-LEAK host-disjoint"
chk "$RR/masking_analysis.csv"                                       "MECH-MASK"
chk "$RR/alu_coverage.csv"                                           "MECH-ALU"
chk "$RR/duplex_alpha_sensitivity.csv"                               "AUG-DUPLEX"
echo
echo "[ failures ]"
fails=$(grep -rl -iE "Unknown failure|Traceback" "$RR"/*.json logs/multigpu logs/exp 2>/dev/null | wc -l | tr -d ' ')
if [ "$fails" -eq 0 ]; then echo "  none [OK]"; else
  echo "  $fails file(s) with failures:"
  grep -rl -iE "Unknown failure|Traceback" "$RR"/*.json logs/multigpu logs/exp 2>/dev/null | sed 's/^/    /' | head
fi
echo "=================================================================="
echo "Re-run ONLY: rows with MISSING seeds, or files with failures."
echo "Completed runs are valid even if they were slow -- do NOT re-run those."
