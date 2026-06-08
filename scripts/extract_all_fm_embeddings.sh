#!/bin/bash
# Generate FM embeddings for all (or selected) encoders on this server.
#
# Usage (run from repo root):
#   bash scripts/extract_all_fm_embeddings.sh [DEVICE] [ENCODERS] [TARGET]
#     DEVICE    GPU id (default: 0)
#     ENCODERS  space-separated, quoted (default: "rnafm rnabert rnaernie rnamsm")
#     TARGET    internal | external | both   (default: internal)
#
# Examples:
#   bash scripts/extract_all_fm_embeddings.sh                       # internal, all 4, GPU0
#   bash scripts/extract_all_fm_embeddings.sh 1 "rnabert rnamsm"    # GPU1, 2 encoders
#   bash scripts/extract_all_fm_embeddings.sh 0 "rnafm" both        # internal + external
#
# Notes:
#   - Internal embeddings load from data/seq_dict/100/ (709MB); the 2.9GB genome is
#     only needed if seq_dict/ is absent. Both extractors skip already-cached .pt
#     (safe to re-run after interruption).
#   - External (circAtlas) requires external_data/circatlas/exon_controls/seq_dict/junction.json
#     (transfer it or build with pipeline/make_circatlas_exon_controls.py first).

set -u
cd "$(dirname "$0")/.." || exit 1   # repo root

DEVICE="${1:-0}"
ENCODERS="${2:-rnafm rnabert rnaernie rnamsm}"
TARGET="${3:-internal}"

EXT_JSON="external_data/circatlas/exon_controls/seq_dict/junction.json"
EXT_OUT="external_data/circatlas/exon_controls/fm_embeddings"

echo "=================================================="
echo "FM embedding extraction"
echo "  device   : cuda:${DEVICE}"
echo "  encoders : ${ENCODERS}"
echo "  target   : ${TARGET}"
echo "=================================================="

for ENC in ${ENCODERS}; do
    case "${TARGET}" in
        internal|both)
            echo ""
            echo ">>> [internal] ${ENC}  ($(date +%H:%M:%S))"
            python pipeline/extract_fm_embeddings.py --enc_type "${ENC}" --device "${DEVICE}"
            ;;
    esac

    case "${TARGET}" in
        external|both)
            if [ ! -f "${EXT_JSON}" ]; then
                echo ">>> [external] SKIP ${ENC}: ${EXT_JSON} not found"
                echo "    (transfer external_data/ or run pipeline/make_circatlas_exon_controls.py)"
            else
                echo ""
                echo ">>> [external] ${ENC}  ($(date +%H:%M:%S))"
                python pipeline/extract_external_fm_embeddings.py \
                    --junction_json "${EXT_JSON}" \
                    --model "${ENC}" \
                    --out_dir "${EXT_OUT}" \
                    --device "cuda:${DEVICE}" \
                    --batch_size 32
            fi
            ;;
    esac
done

echo ""
echo "=== Done ($(date +%H:%M:%S)) ==="
echo "internal → fm_embeddings/<enc>/   external → ${EXT_OUT}/<enc>/"
