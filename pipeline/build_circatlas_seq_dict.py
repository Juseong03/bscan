#!/usr/bin/env python
"""Build the circAtlas external seq_dict (junction.json + flanking) from controls.

`make_circatlas_exon_controls.py` produces only the coordinate TSV
(circatlas_exon_external_controls.tsv). The external-validation and external-FM
extraction steps need the junction *sequences* in the exact training-time
DataSetPrep convention. This script bridges that gap:

    controls TSV (coords) + hg19 genome  →  seq_dict/junction.json (+ flanking_<L>.json)

It uses DataSetPrep.get_junction_intron_seq() directly — the same code path that
builds the internal training sequences — so upper/lower intron-exon arrangement
and minus-strand handling match exactly (see analysis/rebuild_external_b.py).

Usage (run from repo root, after make_circatlas_exon_controls.py):
    python pipeline/build_circatlas_seq_dict.py
    python pipeline/build_circatlas_seq_dict.py --junction_bps 100 --flanking_bps 100
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import argparse, json, warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

from dataloader import DataSetPrep


def main():
    ap = argparse.ArgumentParser(description="Build circAtlas external seq_dict from controls TSV")
    ap.add_argument("--controls", type=Path,
                    default=Path("external_data/circatlas/exon_controls/circatlas_exon_external_controls.tsv"),
                    help="coordinate TSV from make_circatlas_exon_controls.py")
    ap.add_argument("--genome", type=Path, default=Path("data/hg19_seq_dict.json"),
                    help="hg19 sequence dict (needed to extract sequences)")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("external_data/circatlas/exon_controls/seq_dict"),
                    help="output dir for junction.json / flanking_<L>.json")
    ap.add_argument("--junction_bps", type=int, default=100)
    ap.add_argument("--flanking_bps", type=int, default=100)
    args = ap.parse_args()

    if not args.controls.exists():
        raise SystemExit(
            f"{args.controls} not found — run pipeline/make_circatlas_exon_controls.py first."
        )
    if not args.genome.exists():
        raise SystemExit(
            f"{args.genome} not found — the hg19 genome dict is required to extract sequences."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    L = args.junction_bps

    print(f"Instantiating DataSetPrep on {args.controls} (jb={L}, fb={args.flanking_bps})...")
    data = DataSetPrep(
        coord_path=str(args.controls),
        seq_dict_path=str(args.genome),
        junction_bps=L,
        flanking_bps=args.flanking_bps,
        seed=42,
    )

    print("Extracting junction sequences (DataSetPrep convention)...")
    junction, flanking = data.get_junction_intron_seq()
    print(f"  Extracted {len(junction)} junctions (after N/length filtering)")
    print(f"  Label distribution: {dict(Counter(v['label'] for v in junction.values()))}")

    jpath = args.out_dir / "junction.json"
    fpath = args.out_dir / f"flanking_{args.flanking_bps}.json"
    with open(jpath, "w") as f:
        json.dump(junction, f)
    with open(fpath, "w") as f:
        json.dump(flanking, f)
    print(f"  Saved: {jpath}")
    print(f"  Saved: {fpath}")

    # Convention sanity check — canonical GT-AG splice dinucleotides on BS junctions
    bs = [v for v in junction.values() if v["label"] == "BS"][:200]
    canonical = sum(
        1 for v in bs
        if v["upper_intron"][-2:] == "AG" and v["lower_intron"][:2] in ("GT", "GC")
    )
    if bs:
        print(f"  Canonical (AG...GT/GC): {canonical}/{len(bs)} BS junctions checked")
    print("Done.")


if __name__ == "__main__":
    main()
