#!/usr/bin/env python
"""Multi-scale ALU/SINE analysis at 100, 250, 500 nt intronic flanks.

Uses hg19 RepeatMasker + junction coordinates from BS_LS_coordinates_final.csv.
No model retraining needed — pure bioinformatics.
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv, gzip, os, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

RMSK_PATH = Path("data/rmsk_hg19.txt.gz")
COORD_PATH = Path("data/BS_LS_coordinates_final.csv")
WINDOWS = [100, 250, 500]

RMSK_CHR = 5; RMSK_START = 6; RMSK_END = 7
RMSK_NAME = 10; RMSK_CLASS = 11; RMSK_FAMILY = 12


def build_index(chroms):
    raw = defaultdict(lambda: {"alu": [], "sine": []})
    with gzip.open(RMSK_PATH, "rt") as f:
        for line in f:
            p = line.split("\t")
            if len(p) < 13: continue
            chrom = p[RMSK_CHR]
            if chrom not in chroms: continue
            rep_class, rep_name = p[RMSK_CLASS], p[RMSK_NAME].strip()
            try:
                s, e = int(p[RMSK_START]), int(p[RMSK_END])
            except ValueError:
                continue
            if rep_class == "SINE":
                raw[chrom]["sine"].append((s, e))
                if "Alu" in rep_name:
                    raw[chrom]["alu"].append((s, e))
    index = {}
    for chrom, d in raw.items():
        index[chrom] = {}
        for key in ("alu", "sine"):
            ivs = sorted(d[key])
            index[chrom][key] = (
                np.array([s for s, e in ivs], dtype=np.int64),
                np.array([e for s, e in ivs], dtype=np.int64),
            )
    return index


def cov(starts, ends, r_s, r_e):
    length = r_e - r_s
    if length <= 0 or len(starts) == 0: return 0.0
    lo = int(np.searchsorted(ends, r_s, side="right"))
    hi = int(np.searchsorted(starts, r_e, side="left"))
    covered = sum(max(0, min(ends[i], r_e) - max(starts[i], r_s)) for i in range(lo, hi))
    return min(covered, length) / length


def main():
    # Load coordinates
    junctions = []
    with open(COORD_PATH) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            junctions.append({
                "key": f"{row['chr']}|{row['start']}|{row['end']}|{row['strand']}",
                "chr": row["chr"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "strand": row["strand"],
                "label": "BS" if row["Splicing_type"] == "BS" else "LS",
            })
    chroms = set(j["chr"] for j in junctions)

    print(f"Loaded {len(junctions)} junctions. Building RepeatMasker index...")
    index = build_index(chroms)
    print("Index ready.\n")

    results = []
    for junc in junctions:
        chrom = junc["chr"]
        junc_start, junc_end = junc["start"], junc["end"]
        row = {"key": junc["key"], "label": junc["label"]}

        for w in WINDOWS:
            # Upper intron: w nt upstream of junc_start
            upper_s = junc_start - w
            upper_e = junc_start
            # Lower intron: w nt downstream of junc_end
            lower_s = junc_end
            lower_e = junc_end + w

            for rep in ("alu", "sine"):
                if chrom in index and rep in index[chrom]:
                    st, en = index[chrom][rep]
                    u_cov = cov(st, en, upper_s, upper_e)
                    l_cov = cov(st, en, lower_s, lower_e)
                else:
                    u_cov = l_cov = 0.0
                row[f"{rep}_upper_{w}"] = round(u_cov, 4)
                row[f"{rep}_lower_{w}"] = round(l_cov, 4)
                row[f"{rep}_mean_{w}"] = round((u_cov + l_cov) / 2, 4)
                row[f"has_{rep}_{w}"] = int(u_cov > 0 or l_cov > 0)
                row[f"inv_{rep}_{w}"] = int(u_cov > 0 and l_cov > 0)

        results.append(row)

    # Save
    os.makedirs("research_results", exist_ok=True)
    out = Path("research_results/alu_multiscale.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)

    # Summary table
    print("=" * 70)
    print(f"{'Window':>8} {'Rep':>6} {'Label':>5} {'Has any%':>10} {'Inv pair%':>10} {'Mean cov':>10} {'p (MW)':>12}")
    print("=" * 70)
    bs = [r for r in results if r["label"] == "BS"]
    ls = [r for r in results if r["label"] == "LS"]

    summary_rows = []
    for w in WINDOWS:
        for rep in ("alu", "sine"):
            col = f"{rep}_mean_{w}"
            has_col = f"has_{rep}_{w}"
            inv_col = f"inv_{rep}_{w}"
            bs_vals = [r[col] for r in bs]
            ls_vals = [r[col] for r in ls]
            _, p = sp_stats.mannwhitneyu(bs_vals, ls_vals, alternative="greater")
            bs_has = sum(r[has_col] for r in bs) / len(bs) * 100
            ls_has = sum(r[has_col] for r in ls) / len(ls) * 100
            bs_inv = sum(r[inv_col] for r in bs) / len(bs) * 100
            ls_inv = sum(r[inv_col] for r in ls) / len(ls) * 100
            bs_mean = statistics.mean(bs_vals)
            ls_mean = statistics.mean(ls_vals)
            print(f"{w:>8}nt {rep:>6}  BS  {bs_has:>9.1f}% {bs_inv:>9.2f}% {bs_mean:>10.4f} {p:>12.2e}")
            print(f"{w:>8}nt {rep:>6}  LS  {ls_has:>9.1f}% {ls_inv:>9.2f}% {ls_mean:>10.4f}")
            print()
            summary_rows.append({
                "window": w, "rep": rep,
                "bs_has_pct": round(bs_has, 2), "ls_has_pct": round(ls_has, 2),
                "bs_inv_pct": round(bs_inv, 3), "ls_inv_pct": round(ls_inv, 3),
                "bs_mean_cov": round(bs_mean, 4), "ls_mean_cov": round(ls_mean, 4),
                "mw_p": f"{p:.2e}",
            })

    sum_out = Path("research_results/alu_multiscale_summary.csv")
    with open(sum_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)

    print(f"Saved: {out}")
    print(f"Saved: {sum_out}")


if __name__ == "__main__":
    main()
