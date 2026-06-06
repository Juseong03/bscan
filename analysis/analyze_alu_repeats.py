#!/usr/bin/env python
"""P5: RepeatMasker-based ALU/SINE analysis.

Requires: data/rmsk_hg19.txt.gz (UCSC hg19 RepeatMasker track)

Analyses:
1. ALU/SINE coverage in BS vs LS intronic flanks
2. ALU coverage in Tier2 real BS vs synthetic negatives
3. Correlation of model score with ALU coverage
4. ALU-density distribution comparison
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv, gzip, json, os, sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import statistics

warnings.filterwarnings("ignore")

RMSK_PATH = Path("data/rmsk_hg19.txt.gz")
L = 100  # intronic flank length

# UCSC rmsk.txt columns (0-indexed):
# 5=genoName, 6=genoStart, 7=genoEnd, 8=strand, 10=repClass, 11=repFamily, 12=repName
RMSK_CHR    = 5
RMSK_START  = 6
RMSK_END    = 7
RMSK_STRAND = 9
RMSK_NAME   = 10   # repName: (CCCTAA)n, AluSx, L1MC etc.
RMSK_CLASS  = 11   # repClass: SINE, LINE, Simple_repeat etc.
RMSK_FAMILY = 12   # repFamily: Alu, L1 etc.


def build_rmsk_index(rmsk_path: Path, chroms: set[str]) -> dict[str, dict]:
    """Parse rmsk.txt.gz and build per-chromosome sorted interval arrays.

    Returns: {chrom: {'alu': (starts, ends), 'sine': (starts, ends)}}
    Uses sorted arrays + bisect for O(log n) overlap queries.
    """
    import bisect
    print(f"Parsing RepeatMasker track: {rmsk_path}...")
    raw = defaultdict(lambda: {"alu": [], "sine": []})
    n_alu = n_sine = 0
    with gzip.open(rmsk_path, "rt") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 13:
                continue
            chrom = parts[RMSK_CHR]
            if chrom not in chroms:
                continue
            rep_class = parts[RMSK_CLASS]
            rep_name  = parts[RMSK_NAME].strip()
            try:
                start = int(parts[RMSK_START])
                end   = int(parts[RMSK_END])
            except ValueError:
                continue
            if rep_class == "SINE":
                raw[chrom]["sine"].append((start, end))
                n_sine += 1
                if "Alu" in rep_name:
                    raw[chrom]["alu"].append((start, end))
                    n_alu += 1

    # Sort and convert to parallel arrays for fast bisect
    index = {}
    for chrom, d in raw.items():
        index[chrom] = {}
        for key in ("alu", "sine"):
            ivs = sorted(d[key])
            index[chrom][key] = (
                np.array([s for s, e in ivs], dtype=np.int64),
                np.array([e for s, e in ivs], dtype=np.int64),
            )
    print(f"  ALU: {n_alu:,}  SINE: {n_sine:,}  chromosomes: {len(index)}")
    return index


def _interval_coverage(starts: np.ndarray, ends: np.ndarray,
                       region_start: int, region_end: int) -> float:
    """Fraction of [region_start, region_end) covered by sorted intervals."""
    import bisect
    length = region_end - region_start
    if length <= 0 or len(starts) == 0:
        return 0.0
    # Candidates: all intervals whose end > region_start and start < region_end
    lo = int(np.searchsorted(ends, region_start, side="right"))
    hi = int(np.searchsorted(starts, region_end, side="left"))
    if lo >= hi:
        return 0.0
    covered = 0
    for i in range(lo, hi):
        ol = min(ends[i], region_end) - max(starts[i], region_start)
        if ol > 0:
            covered += ol
    return min(covered, length) / length


def alu_coverage(chrom, region_start, region_end, rmsk_index) -> float:
    if chrom not in rmsk_index or "alu" not in rmsk_index[chrom]:
        return 0.0
    starts, ends = rmsk_index[chrom]["alu"]
    return _interval_coverage(starts, ends, region_start, region_end)


def sine_coverage(chrom, region_start, region_end, rmsk_index) -> float:
    if chrom not in rmsk_index or "sine" not in rmsk_index[chrom]:
        return 0.0
    starts, ends = rmsk_index[chrom]["sine"]
    return _interval_coverage(starts, ends, region_start, region_end)


def main():
    if not RMSK_PATH.exists():
        print(f"ERROR: {RMSK_PATH} not found. Run download first.")
        sys.exit(1)

    from dataloader import DataSetPrep

    print("Loading junction sequences...")
    data = DataSetPrep(
        "data/BS_LS_coordinates_final.csv",
        "data/hg19_seq_dict.json",
        junction_bps=L, flanking_bps=L, seed=42
    )
    data.load_junction_flanking_seq()

    # Get all chromosomes for efficient filtering
    chroms = set()
    for k in data.junction_seq:
        parts = k.replace("|", ":").split(":")
        chroms.add(parts[0])

    rmsk_index = build_rmsk_index(RMSK_PATH, chroms)

    # Build chr/start/end from junction keys
    # Key format: chr1_12345_67890_+ or chr1|12345|67890|+
    def parse_key(k):
        # Key format: chr2|130831107|130878181|-
        parts = k.split("|")
        if len(parts) == 4:
            try:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                strand = parts[3]
                return chrom, start, end, strand
            except (ValueError, IndexError):
                pass
        return None, None, None, None

    print("\nComputing ALU coverage for all junctions...")
    results = []
    for key, rec in data.junction_seq.items():
        chrom, junc_start, junc_end, strand = parse_key(key)
        if chrom is None:
            continue
        label = rec["label"]

        # Upper intron: junc_start - L to junc_start (L nt upstream of junction start)
        upper_int_start = junc_start - L
        upper_int_end = junc_start
        # Lower intron: junc_end to junc_end + L
        lower_int_start = junc_end
        lower_int_end = junc_end + L

        alu_upper  = alu_coverage(chrom, upper_int_start, upper_int_end, rmsk_index)
        alu_lower  = alu_coverage(chrom, lower_int_start, lower_int_end, rmsk_index)
        sine_upper = sine_coverage(chrom, upper_int_start, upper_int_end, rmsk_index)
        sine_lower = sine_coverage(chrom, lower_int_start, lower_int_end, rmsk_index)
        has_alu_upper = int(alu_upper > 0)
        has_alu_lower = int(alu_lower > 0)
        inverted_pair = int(has_alu_upper and has_alu_lower)

        results.append({
            "key": key,
            "label": label,
            "chrom": chrom,
            "alu_upper": round(alu_upper, 4),
            "alu_lower": round(alu_lower, 4),
            "alu_mean": round((alu_upper + alu_lower) / 2, 4),
            "sine_upper": round(sine_upper, 4),
            "sine_lower": round(sine_lower, 4),
            "has_alu_upper": has_alu_upper,
            "has_alu_lower": has_alu_lower,
            "inverted_alu_pair": inverted_pair,
        })

    # Analysis 1: BS vs LS ALU coverage
    print("\n" + "=" * 60)
    print("Analysis 1: ALU coverage — BS vs LS intronic flanks (100nt)")
    print("=" * 60)
    bs = [r for r in results if r["label"] == "BS"]
    ls = [r for r in results if r["label"] == "LS"]

    for label, group in [("BS", bs), ("LS", ls)]:
        alu_means = [r["alu_mean"] for r in group]
        has_pair = [r["inverted_alu_pair"] for r in group]
        print(f"\n{label} (n={len(group)})")
        print(f"  ALU coverage (mean ± std): {statistics.mean(alu_means):.4f} ± {statistics.stdev(alu_means):.4f}")
        print(f"  Has ALU in both flanks (%): {sum(has_pair)/len(has_pair)*100:.1f}%")
        print(f"  Median ALU coverage: {statistics.median(alu_means):.4f}")

    # Statistical test: BS vs LS ALU
    from scipy import stats as sp_stats
    bs_alu = [r["alu_mean"] for r in bs]
    ls_alu = [r["alu_mean"] for r in ls]
    t_stat, p_val = sp_stats.mannwhitneyu(bs_alu, ls_alu, alternative="greater")
    print(f"\nMann-Whitney U (BS > LS ALU): U={t_stat:.1f}, p={p_val:.2e}")

    # Analysis 2: ALU breakdown
    print("\n" + "=" * 60)
    print("Analysis 2: ALU presence rates")
    print("=" * 60)
    for label, group in [("BS", bs), ("LS", ls)]:
        has_any = sum(1 for r in group if r["alu_upper"] > 0 or r["alu_lower"] > 0)
        has_both = sum(r["inverted_alu_pair"] for r in group)
        print(f"{label}: has_any_ALU={has_any/len(group)*100:.1f}%  has_pair={has_both/len(group)*100:.1f}%")

    # Save
    os.makedirs("research_results", exist_ok=True)
    out = Path("research_results/alu_coverage.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out}")

    # Summary stats
    summary_out = Path("research_results/alu_summary.csv")
    with open(summary_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "n", "alu_mean", "alu_std", "alu_median",
                         "pct_has_upper", "pct_has_lower", "pct_has_pair"])
        for label, group in [("BS", bs), ("LS", ls)]:
            alu_m = [r["alu_mean"] for r in group]
            writer.writerow([
                label, len(group),
                round(statistics.mean(alu_m), 4),
                round(statistics.stdev(alu_m), 4),
                round(statistics.median(alu_m), 4),
                round(sum(r["has_alu_upper"] for r in group)/len(group)*100, 1),
                round(sum(r["has_alu_lower"] for r in group)/len(group)*100, 1),
                round(sum(r["inverted_alu_pair"] for r in group)/len(group)*100, 1),
            ])
    print(f"Saved: {summary_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
