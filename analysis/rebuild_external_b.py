#!/usr/bin/env python
"""Rebuild External-B junction sequences using the EXACT DataSetPrep convention.

The previous build used a custom extract_junction_seq() whose upper/lower
intron-exon arrangement did not match the training-time DataSetPrep convention,
causing all models to score below chance (AUC 0.16-0.46).

This version uses DataSetPrep.get_junction_intron_seq() directly on the
External-B coordinate file, guaranteeing identical sequence convention.

Convention (from DataSetPrep):
  upper_seq = upper_intron + upper_exon   (intron first)
  lower_seq = lower_exon  + lower_intron  (exon first)
  minus strand: reverse-complement + swap upper/lower introns
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import json, os, statistics, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dataloader import DataSetPrep

COORD_PATH = "external_data/circbase_b/coordinates.csv"
GENOME_PATH = "data/hg19_seq_dict.json"
OUT_DIR = Path("external_data/circbase_b/seq_dict")
L = 100


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Instantiating DataSetPrep with External-B coordinates...")
    data = DataSetPrep(
        coord_path=COORD_PATH,
        seq_dict_path=GENOME_PATH,
        junction_bps=L,
        flanking_bps=L,
        seed=42,
    )

    print("Extracting junction sequences (DataSetPrep convention)...")
    junction, flanking = data.get_junction_intron_seq()
    print(f"  Extracted {len(junction)} junctions (after N/length filtering)")

    # Label distribution
    from collections import Counter
    labels = Counter(v["label"] for v in junction.values())
    print(f"  Label distribution: {dict(labels)}")

    # Save junction.json
    with open(OUT_DIR / "junction.json", "w") as f:
        json.dump(junction, f)
    print(f"  Saved: {OUT_DIR / 'junction.json'}")

    with open(OUT_DIR / f"flanking_{L}.json", "w") as f:
        json.dump(flanking, f)
    print(f"  Saved: {OUT_DIR / f'flanking_{L}.json'}")

    # ── Verification: canonical splice site dinucleotides ──────────────────
    print("\nVerification — canonical GT-AG splice sites (BS junctions):")
    bs = [(k, v) for k, v in junction.items() if v["label"] == "BS"]
    canonical = 0
    for k, v in bs[:200]:
        # upper_intron ends at 3'SS (should end with AG)
        # lower_intron starts at 5'SS (should start with GT/GC)
        up_end = v["upper_intron"][-2:]
        lo_start = v["lower_intron"][:2]
        if up_end == "AG" and lo_start in ("GT", "GC"):
            canonical += 1
    print(f"  Canonical (AG...GT): {canonical}/200 BS junctions checked")

    # Compare convention with internal data
    print("\nConvention cross-check vs internal training data:")
    internal = DataSetPrep("data/BS_LS_coordinates_final.csv", GENOME_PATH,
                           junction_bps=L, flanking_bps=L, seed=42)
    internal.load_junction_flanking_seq()
    int_canon = 0
    int_bs = [(k, v) for k, v in internal.junction_seq.items() if v["label"] == "BS"][:200]
    for k, v in int_bs:
        up_end = v["upper_intron"][-2:]
        lo_start = v["lower_intron"][:2]
        if up_end == "AG" and lo_start in ("GT", "GC"):
            int_canon += 1
    print(f"  Internal canonical: {int_canon}/200")
    print(f"  External-B canonical: {canonical}/200")
    print(f"  → Convention {'MATCHES' if abs(canonical-int_canon) < 60 else 'STILL DIFFERS'}")

    print("\nDone. External-B rebuilt with correct convention.")


if __name__ == "__main__":
    main()
