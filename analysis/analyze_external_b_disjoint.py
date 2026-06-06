#!/usr/bin/env python
"""External-B: sequence-disjoint subset of circAtlas (External-A).

Addresses the reviewer's leakage concern by re-evaluating all models on the
subset of circAtlas samples whose junction sequences do NOT appear in the
internal training set. Uses existing per-key prediction CSVs (no re-inference).

If model AUCs on the disjoint subset match the full External-A AUCs, this
confirms that the external generalization results are not driven by leakage.
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv, glob, json, os, statistics, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
from dataloader import DataSetPrep

EXT_DIR = Path("external_data/circatlas/exon_controls")
FM_MODELS = ["bscan_unified_fm", "bscan_unified_ernie", "bscan_unified_msm", "bscan_unified_bert"]
BASE_MODELS = ["bscan", "circcnnsingle", "circcnntri", "jedi", "bscan_unified_onehot"]
SEEDS = [42, 123, 315, 777, 1004, 2024, 2025, 2026, 3407, 9001]


def main():
    # ── Build internal sequence set ────────────────────────────────────────
    print("Loading internal training sequences...")
    d = DataSetPrep("data/BS_LS_coordinates_final.csv", "data/hg19_seq_dict.json",
                    junction_bps=100, flanking_bps=100, seed=42)
    d.load_junction_flanking_seq()
    internal_seqs = set()
    for v in d.junction_seq.values():
        internal_seqs.add(v["upper_seq"] + v["lower_seq"])

    # ── External-A sequences → identify disjoint keys ──────────────────────
    extA = json.load(open(EXT_DIR / "seq_dict/junction.json"))
    disjoint_keys = set()
    for k, v in extA.items():
        if (v["upper_seq"] + v["lower_seq"]) not in internal_seqs:
            disjoint_keys.add(k)
    print(f"External-A: {len(extA)} total, {len(disjoint_keys)} sequence-disjoint "
          f"({len(disjoint_keys)/len(extA)*100:.1f}%)")

    # ── Recompute AUC per model on full vs disjoint ────────────────────────
    rows = []
    all_models = FM_MODELS + BASE_MODELS
    for model in all_models:
        full_aucs, disj_aucs = [], []
        for seed in SEEDS:
            pred_file = EXT_DIR / f"predictions_{model}_{seed}.csv"
            if not pred_file.exists():
                continue
            with open(pred_file) as f:
                pred = list(csv.DictReader(f))
            labels = np.array([int(r["label"]) for r in pred])
            probs = np.array([float(r["prob_bs"]) for r in pred])
            keys = [r["key"] for r in pred]

            # Full
            full_aucs.append(roc_auc_score(labels, probs))
            # Disjoint subset
            mask = np.array([k in disjoint_keys for k in keys])
            if mask.sum() > 10 and len(set(labels[mask])) == 2:
                disj_aucs.append(roc_auc_score(labels[mask], probs[mask]))

        if full_aucs and disj_aucs:
            fm = statistics.mean(full_aucs)
            dm = statistics.mean(disj_aucs)
            rows.append({
                "model": model,
                "n_seeds": len(full_aucs),
                "full_auc": round(fm, 4),
                "full_std": round(statistics.stdev(full_aucs) if len(full_aucs) > 1 else 0, 4),
                "disjoint_auc": round(dm, 4),
                "disjoint_std": round(statistics.stdev(disj_aucs) if len(disj_aucs) > 1 else 0, 4),
                "delta": round(dm - fm, 4),
            })
            print(f"  {model:<25} full={fm:.4f}  disjoint={dm:.4f}  Δ={dm-fm:+.4f}")

    out = Path("research_results/external_b_sequence_disjoint.csv")
    os.makedirs("research_results", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")
    print(f"\nInterpretation: if Δ ≈ 0, external results are NOT driven by leakage.")


if __name__ == "__main__":
    main()
