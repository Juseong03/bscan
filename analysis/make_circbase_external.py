#!/usr/bin/env python
"""Build External-B validation set from circBase (hg19) + RJunBase LS junctions.

External-B is independent of both:
  - Internal training data (BS_LS_coordinates_final.csv)
  - External-A (circAtlas, hg38)

Positives:  circBase hg19 circRNA junctions NOT in internal training
Negatives:  RJunBase linear-splice junctions, exon-length matched (±50%)
Sequences:  Extracted from hg19_seq_dict.json (full genome)
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv, json, os, random, statistics
from pathlib import Path
from collections import defaultdict
import numpy as np

CIRCBASE_PATH = Path("/workspace/volume/circRNA_delta/data/circBase/hsa_hg19_circRNA.bed")
RJUNBASE_PATH = Path("/workspace/volume/circRNA_delta/Data/LinearRJUNBase/RJunBase_hg19_exonic.csv")
COORD_PATH    = Path("data/BS_LS_coordinates_final.csv")
GENOME_PATH   = Path("data/hg19_seq_dict.json")
OUT_DIR       = Path("external_data/circbase_b")
L = 100  # junction_bps
MAX_POSITIVES = 3000
MAX_RATIO_DIFF = 0.5
SEED = 42


def extract_junction_seq(genome: dict, chrom: str, start: int, end: int, strand: str) -> dict | None:
    """Extract 200-nt junction sequences (same as DataSetPrep logic)."""
    if chrom not in genome:
        return None
    seq = genome[chrom]
    chr_len = len(seq)

    # Upper: [start-L : start+L] (junction start)
    u_s, u_e = start - L, start + L
    if u_s < 0 or u_e > chr_len:
        return None
    upper_raw = seq[u_s:u_e]

    # Lower: [end-L : end+L] (junction end)
    l_s, l_e = end - L, end + L
    if l_s < 0 or l_e > chr_len:
        return None
    lower_raw = seq[l_s:l_e]

    # Reverse complement if minus strand
    def rc(s):
        comp = str.maketrans("ACGTacgt", "TGCAtgca")
        return s.translate(comp)[::-1]

    if strand == "-":
        upper_seq = rc(lower_raw)  # lower in genomic → upper in transcript
        lower_seq = rc(upper_raw)
    else:
        upper_seq = upper_raw
        lower_seq = lower_raw

    # Validate: no ambiguous
    valid = set("ACGTacgt")
    if any(c not in valid for c in upper_seq + lower_seq):
        return None

    upper_seq = upper_seq.upper()
    lower_seq = lower_seq.upper()

    return {
        "upper_seq":    upper_seq,
        "lower_seq":    lower_seq,
        "upper_intron": upper_seq[:L],
        "upper_exon":   upper_seq[L:],
        "lower_exon":   lower_seq[:L],
        "lower_intron": lower_seq[L:],
        "lower_seq_rc": lower_seq[::-1].translate(str.maketrans("ACGT", "TGCA")),
        "label": "BS",
        "junction_bps": L,
    }


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load internal training coords (for exclusion) ──────────────────────
    print("Loading internal training junctions...")
    internal_coords = set()
    with open(COORD_PATH) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            internal_coords.add((r["chr"], r["start"], r["end"]))
    print(f"  Internal: {len(internal_coords):,} junctions")

    # ── Load genome ────────────────────────────────────────────────────────
    print("Loading hg19 genome...")
    genome = json.load(open(GENOME_PATH))
    print(f"  Chromosomes: {len(genome)}")

    # ── Load circBase positives ────────────────────────────────────────────
    print("Loading circBase positives...")
    positives = []
    with open(CIRCBASE_PATH) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 6:
                continue
            chrom, start, end, name, _, strand = p[:6]
            if (chrom, start, end) in internal_coords:
                continue
            try:
                s, e = int(start), int(end)
            except ValueError:
                continue
            if e - s > 500000 or e - s < 50:  # filter very large/small
                continue
            positives.append({"chr": chrom, "start": s, "end": e, "strand": strand,
                               "length": e - s, "name": name})

    rng.shuffle(positives)
    print(f"  Non-overlapping circBase entries: {len(positives):,}")

    # ── Load RJunBase negatives ────────────────────────────────────────────
    print("Loading RJunBase negatives...")
    ls_junctions = []
    with open(RJUNBASE_PATH) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pos = r.get("hg19_LS_Position", "")
            if not pos or "|" not in pos:
                continue
            parts = pos.split("|")
            if len(parts) < 4:
                continue
            chrom, start_s, end_s, strand = parts[:4]
            try:
                s, e = int(start_s), int(end_s)
            except ValueError:
                continue
            length = e - s
            if length < 50:
                continue
            ls_junctions.append({"chr": chrom, "start": s, "end": e, "strand": strand,
                                  "length": length})

    # Index LS by length bucket for fast matching
    ls_by_bucket = defaultdict(list)
    for ls in ls_junctions:
        bucket = ls["length"] // 1000
        ls_by_bucket[bucket].append(ls)
    print(f"  RJunBase LS entries: {len(ls_junctions):,}")

    # ── Build matched pairs ────────────────────────────────────────────────
    print(f"Building External-B (target: {MAX_POSITIVES} pairs)...")
    junction_data = {}
    coord_list = []
    n_pos = n_neg = 0

    for pos in positives:
        if n_pos >= MAX_POSITIVES:
            break

        # Extract positive sequence
        rec = extract_junction_seq(genome, pos["chr"], pos["start"], pos["end"], pos["strand"])
        if rec is None:
            continue

        key = f"{pos['chr']}|{pos['start']}|{pos['end']}|{pos['strand']}"
        junction_data[key] = rec
        coord_list.append({"chr": pos["chr"], "start": pos["start"], "end": pos["end"],
                           "strand": pos["strand"], "Splicing_type": "BS",
                           "Transcript": pos["name"], "exon_pair_len": pos["length"]})
        n_pos += 1

    # Match negatives by length
    used_neg = set()
    for pos in positives[:n_pos]:
        target_len = pos["length"]
        bucket = target_len // 1000
        # Search nearby buckets
        candidates = []
        for b in range(max(0, bucket - 3), bucket + 4):
            for ls in ls_by_bucket.get(b, []):
                ratio_diff = abs(ls["length"] - target_len) / max(target_len, 1)
                if ratio_diff <= MAX_RATIO_DIFF:
                    ls_id = f"{ls['chr']}|{ls['start']}|{ls['end']}|{ls['strand']}"
                    if ls_id not in used_neg:
                        candidates.append(ls)
        if not candidates:
            continue
        neg = rng.choice(candidates)
        neg_id = f"{neg['chr']}|{neg['start']}|{neg['end']}|{neg['strand']}"
        used_neg.add(neg_id)

        rec = extract_junction_seq(genome, neg["chr"], neg["start"], neg["end"], neg["strand"])
        if rec is None:
            continue
        rec["label"] = "LS"

        junction_data[neg_id] = rec
        coord_list.append({"chr": neg["chr"], "start": neg["start"], "end": neg["end"],
                           "strand": neg["strand"], "Splicing_type": "LS",
                           "Transcript": neg_id, "exon_pair_len": neg["length"]})
        n_neg += 1

    print(f"  Built: {n_pos} positives, {n_neg} negatives ({n_pos+n_neg} total)")

    # ── Save ─────────────────────────────────────────────────────────────────
    seq_dir = OUT_DIR / "seq_dict"
    seq_dir.mkdir(exist_ok=True)

    with open(seq_dir / "junction.json", "w") as f:
        json.dump(junction_data, f)
    print(f"  Saved: {seq_dir / 'junction.json'} ({len(junction_data)} entries)")

    coord_out = OUT_DIR / "coordinates.csv"
    with open(coord_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(coord_list[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(coord_list)
    print(f"  Saved: {coord_out}")

    # Summary
    bs_lens = [r["exon_pair_len"] for r in coord_list if r["Splicing_type"] == "BS"]
    ls_lens = [r["exon_pair_len"] for r in coord_list if r["Splicing_type"] == "LS"]
    print(f"\nExternal-B summary:")
    print(f"  BS: n={len(bs_lens)}, length = {statistics.mean(bs_lens):.0f} ± {statistics.stdev(bs_lens):.0f}")
    print(f"  LS: n={len(ls_lens)}, length = {statistics.mean(ls_lens):.0f} ± {statistics.stdev(ls_lens):.0f}")
    print(f"  Source: circBase (positives), RJunBase (negatives)")
    print(f"  Genome: hg19 (independent of circAtlas hg38)")
    print("Done.")


if __name__ == "__main__":
    main()
