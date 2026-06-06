#!/usr/bin/env python
"""External-B: host-locus-disjoint subset of circAtlas (External-A).

Builds a stricter external set by removing circAtlas samples whose genomic
locus overlaps any internal training junction. Because internal coords are hg19
and circAtlas is hg38, we liftOver internal hg19 spans to hg38, then exclude
External-A samples that overlap (same chr/strand, any span overlap) an internal
locus — i.e. host-locus-disjoint.

Re-evaluates all models on this disjoint subset using existing per-key
prediction CSVs (no re-inference). If AUCs hold, generalization is not driven
by host-gene overlap.

Run from repo root: python analysis/make_external_b_hostgene.py
"""
from __future__ import annotations

import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import csv, json, statistics
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from pyliftover import LiftOver

EXT_DIR = Path("external_data/circatlas/exon_controls")
COORD = "data/BS_LS_coordinates_final.csv"
PAD = 100  # extend each locus by junction window when checking overlap
SEEDS = [42, 123, 315, 777, 1004, 2024, 2025, 2026, 3407, 9001]
FM_MODELS = ["bscan_unified_fm", "bscan_unified_ernie", "bscan_unified_msm", "bscan_unified_bert"]
BASE_MODELS = ["bscan", "circcnn", "circcnnsingle", "circcnntri", "jedi", "bscan_unified_onehot"]


def liftover_internal():
    """Lift internal hg19 BS/LS spans to hg38. Return per-(chr,strand) sorted intervals."""
    lo = LiftOver("hg19", "hg38")
    intervals = defaultdict(list)  # (chr,strand) -> [(start,end)]
    n_ok = n_fail = 0
    with open(COORD) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            chrom, strand = r["chr"], r["strand"]
            s, e = int(r["start"]), int(r["end"])
            cs = lo.convert_coordinate(chrom, s)
            ce = lo.convert_coordinate(chrom, e)
            if not cs or not ce:
                n_fail += 1; continue
            c1, p1 = cs[0][0], cs[0][1]
            c2, p2 = ce[0][0], ce[0][1]
            if c1 != c2 or c1 != chrom:
                n_fail += 1; continue
            lo_s, hi_s = min(p1, p2), max(p1, p2)
            intervals[(chrom, strand)].append((lo_s, hi_s))
            n_ok += 1
    # sort each
    for k in intervals:
        intervals[k] = sorted(intervals[k])
    print(f"  liftOver internal: {n_ok} ok, {n_fail} failed")
    return intervals


def overlaps(intervals, chrom, strand, s, e):
    """True if [s-PAD, e+PAD] overlaps any internal interval on same chr/strand."""
    arr = intervals.get((chrom, strand))
    if not arr:
        return False
    qs, qe = s - PAD, e + PAD
    # linear scan (a few thousand per chr) — fine for one-off
    for (a, b) in arr:
        if b < qs:
            continue
        if a > qe:
            break
        return True  # overlap
    return False


def main():
    print("Lifting internal hg19 → hg38...")
    internal = liftover_internal()

    extA = json.load(open(EXT_DIR / "seq_dict/junction.json"))
    disjoint = set()
    for k, v in extA.items():
        chrom, s, e, strand = k.split("|")
        if not overlaps(internal, chrom, strand, int(s), int(e)):
            disjoint.add(k)
    n_pos = sum(1 for k in disjoint if extA[k]["label"] == "BS")
    n_neg = len(disjoint) - n_pos
    print(f"External-A: {len(extA)} → host-locus-disjoint: {len(disjoint)} "
          f"({len(disjoint)/len(extA)*100:.1f}%)  [BS={n_pos}, LS={n_neg}]")

    # Re-evaluate from per-key prediction CSVs
    rows = []
    for model in FM_MODELS + BASE_MODELS:
        full_a, disj_a = [], []
        for seed in SEEDS:
            pf = EXT_DIR / f"predictions_{model}_{seed}.csv"
            if not pf.exists():
                continue
            with open(pf) as f:
                pr = list(csv.DictReader(f))
            keys = [r["key"] for r in pr]
            lab = np.array([int(r["label"]) for r in pr])
            prob = np.array([float(r["prob_bs"]) for r in pr])
            full_a.append(roc_auc_score(lab, prob))
            mask = np.array([k in disjoint for k in keys])
            if mask.sum() > 20 and len(set(lab[mask])) == 2:
                disj_a.append(roc_auc_score(lab[mask], prob[mask]))
        if full_a and disj_a:
            fm, dm = statistics.mean(full_a), statistics.mean(disj_a)
            rows.append({"model": model, "n_seeds": len(full_a),
                         "full_auc": round(fm, 4), "hostdisjoint_auc": round(dm, 4),
                         "delta": round(dm - fm, 4),
                         "n_disjoint": len(disjoint)})
            print(f"  {model:<22} full={fm:.4f}  host-disjoint={dm:.4f}  Δ={dm-fm:+.4f}")

    out = Path("research_results/external_b_hostgene_disjoint.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out}")
    print("Interpretation: Δ≈0 → generalization not driven by host-locus overlap with training.")


if __name__ == "__main__":
    main()
