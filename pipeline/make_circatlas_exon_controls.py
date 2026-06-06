#!/usr/bin/env python
"""Create exon-aware controls for circAtlas-style circRNA coordinates."""

from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class Transcript(NamedTuple):
    chrom: str
    start: int
    end: int
    transcript: str
    strand: str
    exons: tuple[tuple[int, int], ...]


def parse_exons(row: pd.Series) -> tuple[tuple[int, int], ...]:
    tx_start = int(row.iloc[1])
    sizes = [int(x) for x in str(row.iloc[10]).strip(",").split(",") if x]
    starts = [int(x) for x in str(row.iloc[11]).strip(",").split(",") if x]
    return tuple((tx_start + s, tx_start + s + size) for s, size in zip(starts, sizes))


def load_transcripts(path: Path) -> dict[tuple[str, str], list[Transcript]]:
    bed = pd.read_csv(path, sep="\t", header=None)
    by_key: dict[tuple[str, str], list[Transcript]] = {}
    for _, row in bed.iterrows():
        chrom = str(row.iloc[0])
        strand = str(row.iloc[5])
        if not chrom.startswith("chr"):
            continue
        tx = Transcript(
            chrom=chrom,
            start=int(row.iloc[1]),
            end=int(row.iloc[2]),
            transcript=str(row.iloc[3]),
            strand=strand,
            exons=parse_exons(row),
        )
        by_key.setdefault((chrom, strand), []).append(tx)
    return by_key


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def candidate_exon_pairs(tx: Transcript, target_len: int, max_ratio_diff: float) -> list[tuple[float, int, int]]:
    candidates = []
    exons = sorted(tx.exons)
    for i in range(len(exons)):
        for j in range(i, len(exons)):
            start = exons[i][0]
            end = exons[j][1]
            length = end - start
            if length <= 0:
                continue
            ratio_diff = abs(length - target_len) / max(target_len, 1)
            if ratio_diff <= max_ratio_diff:
                candidates.append((ratio_diff, start, end))
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circatlas", type=Path, default=Path("data/human_bed_v3.0/human_bed_v3.0.txt"))
    parser.add_argument("--exon_bed", type=Path, default=Path("data/hg38_exon.bed"))
    parser.add_argument("--out_dir", type=Path, default=Path("external_data/circatlas/exon_controls"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_length", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=500000)
    parser.add_argument("--max_ratio_diff", type=float, default=0.5)
    parser.add_argument("--max_positives", type=int, default=5000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    circ = pd.read_csv(args.circatlas, sep="\t")
    circ = circ.rename(columns={"Chro": "chr", "Start": "start", "End": "end", "Strand": "strand"})
    circ = circ[circ["chr"].astype(str).str.startswith("chr") & circ["strand"].isin(["+", "-"])].copy()
    circ["start"] = circ["start"].astype(int)
    circ["end"] = circ["end"].astype(int)
    circ["length"] = circ["end"] - circ["start"]
    circ = circ[(circ["length"] >= args.min_length) & (circ["length"] <= args.max_length)].copy()
    circ = circ.sample(frac=1.0, random_state=args.seed).head(args.max_positives).copy()

    transcripts = load_transcripts(args.exon_bed)
    known_positive = set(zip(circ["chr"], circ["start"], circ["end"], circ["strand"]))
    pos_rows = []
    neg_rows = []
    for row in circ.itertuples(index=False):
        txs = [
            tx for tx in transcripts.get((row.chr, row.strand), [])
            if overlap(int(row.start), int(row.end), tx.start, tx.end)
        ]
        candidates = []
        for tx in txs:
            for ratio_diff, start, end in candidate_exon_pairs(tx, int(row.length), args.max_ratio_diff):
                if (row.chr, start, end, row.strand) in known_positive:
                    continue
                if start == int(row.start) and end == int(row.end):
                    continue
                candidates.append((ratio_diff, start, end, tx.transcript))
        if not candidates:
            continue
        best_diff = candidates[0][0]
        best = [c for c in candidates if c[0] <= best_diff + 1e-9]
        ratio_diff, start, end, tx_id = rng.choice(best)
        pos_id = str(row.circAltas_ID)
        pos_rows.append(
            {
                "chr": row.chr,
                "strand": row.strand,
                "start": int(row.start),
                "end": int(row.end),
                "Transcript": pos_id,
                "Splicing_type": "BS",
                "exon_pair_len": int(row.length),
            }
        )
        neg_rows.append(
            {
                "chr": row.chr,
                "strand": row.strand,
                "start": int(start),
                "end": int(end),
                "Transcript": f"circatlas_exon_control_{len(neg_rows)}",
                "Splicing_type": "LS",
                "exon_pair_len": int(end - start),
                "matched_positive": pos_id,
                "control_transcript": tx_id,
                "length_ratio_diff": float(ratio_diff),
            }
        )

    pos = pd.DataFrame(pos_rows)
    neg = pd.DataFrame(neg_rows)
    combined = pd.concat(
        [
            pos[["chr", "strand", "start", "end", "Transcript", "Splicing_type", "exon_pair_len"]],
            neg[["chr", "strand", "start", "end", "Transcript", "Splicing_type", "exon_pair_len"]],
        ],
        ignore_index=True,
    )
    pos.to_csv(args.out_dir / "circatlas_positive_coordinates.tsv", sep="\t", index=False)
    neg.to_csv(args.out_dir / "circatlas_exon_aware_pseudonegatives.tsv", sep="\t", index=False)
    combined.to_csv(args.out_dir / "circatlas_exon_external_controls.tsv", sep="\t", index=False)
    summary = {
        "source_candidates_after_filter": int(len(circ)),
        "positives_with_controls": int(len(pos)),
        "pseudo_negatives": int(len(neg)),
        "max_ratio_diff": args.max_ratio_diff,
        "max_positives": args.max_positives,
        "control_definition": "circAtlas positives matched to real exon-pair intervals from overlapping same-strand transcripts",
    }
    (args.out_dir / "circatlas_exon_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
