#!/bin/bash
# Verify FM embedding extraction completeness (internal + external).
# Compares the number of cached .pt files per encoder against the expected key
# count (junction.json), and reports how many are still pending.
#
# Usage (run from repo root):
#   bash scripts/check_embeddings.sh
#
# Exit code 0 if everything complete; 1 if any encoder has pending files.
set -u
cd "$(dirname "$0")/.." || exit 1

ENCODERS="${1:-rnafm rnabert rnaernie rnamsm}"
INT_SEQ="data/seq_dict/100/junction.json"
EXT_SEQ="external_data/circatlas/exon_controls/seq_dict/junction.json"

# count keys in a junction.json (top-level object keys) via python (robust)
keycount() {
  [ -f "$1" ] || { echo 0; return; }
  python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0
}

report() {  # $1=label  $2=expected  $3=base_dir
  local label="$1" exp="$2" base="$3" rc=0
  echo "── $label  (expected $exp keys per encoder) ──"
  if [ "$exp" = "0" ]; then echo "  [skip] no junction.json found"; return 0; fi
  for ENC in $ENCODERS; do
    local dir="$base/$ENC"
    if [ ! -d "$dir" ]; then
      printf "  %-9s  MISSING dir (%s)\n" "$ENC" "$dir"; rc=1; continue
    fi
    local n; n=$(find "$dir" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
    local pend=$(( exp - n ))
    if [ "$n" -ge "$exp" ]; then
      printf "  %-9s  %6d / %-6d  ✅ complete\n" "$ENC" "$n" "$exp"
    else
      printf "  %-9s  %6d / %-6d  ⏳ %d pending\n" "$ENC" "$n" "$exp" "$pend"; rc=1
    fi
  done
  return $rc
}

echo "FM embedding completeness check"
echo "================================"
INT_EXP=$(keycount "$INT_SEQ")
EXT_EXP=$(keycount "$EXT_SEQ")

overall=0
report "INTERNAL  fm_embeddings/" "$INT_EXP" "fm_embeddings" || overall=1
echo ""
report "EXTERNAL  external_data/circatlas/exon_controls/fm_embeddings/" \
       "$EXT_EXP" "external_data/circatlas/exon_controls/fm_embeddings" || overall=1

echo ""
if [ "$overall" = "0" ]; then
  echo "✅ ALL COMPLETE"
else
  echo "⏳ Some pending — re-run extraction (it skips existing .pt, so it only fills gaps):"
  echo "   bash scripts/extract_all_fm_embeddings.sh <DEVICE> \"$ENCODERS\" both 256"
fi
exit $overall
