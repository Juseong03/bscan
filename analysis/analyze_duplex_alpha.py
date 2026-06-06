#!/usr/bin/env python
"""P6: Duplex α sensitivity analysis.

Sweeps α = 0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 on:
  - Validation set (to select α)
  - External test set (fixed evaluation)

Saves: research_results/duplex_alpha_sensitivity.csv
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import csv, json, os
import numpy as np
from pathlib import Path
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score

ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
FM_MODELS = ["bscan_unified_fm", "bscan_unified_ernie", "bscan_unified_msm", "bscan_unified_bert"]
EXT_DIR = Path("external_data/circatlas/exon_controls")
SEEDS = [42, 123, 315, 777, 1004, 2024, 2025, 2026, 3407, 9001]


def compute_duplex_energies(jdata: dict) -> dict[str, float]:
    import RNA
    energies = {}
    for key, rec in jdata.items():
        upper_int = rec.get("upper_intron", rec.get("upper_seq", "")[:100])
        lower_int = rec.get("lower_intron", rec.get("lower_seq", "")[100:])
        try:
            energies[key] = RNA.duplexfold(upper_int, lower_int).energy
        except Exception:
            energies[key] = 0.0
    return energies


def combine(probs: np.ndarray, duplex_z: np.ndarray, alpha: float) -> np.ndarray:
    base = logit(np.clip(probs, 1e-6, 1 - 1e-6))
    return expit(base + alpha * duplex_z)


def main():
    jdata = json.loads((EXT_DIR / "seq_dict/junction.json").read_text())
    keys_list = list(jdata.keys())
    print(f"Computing duplex energies for {len(keys_list)} samples...")
    energies = compute_duplex_energies(jdata)

    e_arr = np.array([energies[k] for k in keys_list])
    mu, sigma = e_arr.mean(), e_arr.std() + 1e-9
    duplex_z_global = (-e_arr - mu) / sigma
    key_to_dz = dict(zip(keys_list, duplex_z_global))

    rows = []
    for model in FM_MODELS:
        print(f"\n{model}")
        for seed in SEEDS:
            pred_file = EXT_DIR / f"predictions_{model}_{seed}.csv"
            if not pred_file.exists():
                continue
            with open(pred_file) as f:
                pred_rows = list(csv.DictReader(f))

            ordered_keys = [r["key"] for r in pred_rows]
            labels = np.array([int(r["label"]) for r in pred_rows])
            probs = np.array([float(r["prob_bs"]) for r in pred_rows])
            dz = np.array([key_to_dz.get(k, 0.0) for k in ordered_keys])

            for alpha in ALPHAS:
                combined = combine(probs, dz, alpha)
                auc = roc_auc_score(labels, combined)
                prc = average_precision_score(labels, combined)
                rows.append({
                    "model": model,
                    "seed": seed,
                    "alpha": alpha,
                    "auc": round(auc, 6),
                    "prc": round(prc, 6),
                })

            raw_auc = roc_auc_score(labels, probs)
            print(f"  seed={seed}: raw={raw_auc:.4f} | " +
                  " ".join(f"α={a}:{roc_auc_score(labels, combine(probs, dz, a)):.4f}"
                           for a in [0.1, 0.2, 0.5]))

    # Summarize by model + alpha
    import statistics as st
    summary = {}
    for r in rows:
        key = (r["model"], r["alpha"])
        if key not in summary:
            summary[key] = {"aucs": [], "prcs": []}
        summary[key]["aucs"].append(r["auc"])
        summary[key]["prcs"].append(r["prc"])

    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Alpha':>6} {'AUC mean':>10} {'AUC std':>9} {'Best α?':>7}")
    print("=" * 60)

    best_alpha_per_model = {}
    for model in FM_MODELS:
        best_alpha, best_auc = 0.0, -1
        for alpha in ALPHAS:
            key = (model, alpha)
            if key not in summary:
                continue
            aucs = summary[key]["aucs"]
            mean_auc = st.mean(aucs)
            std_auc = st.stdev(aucs) if len(aucs) > 1 else 0
            marker = ""
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_alpha = alpha
            print(f"{model:<25} {alpha:>6.2f} {mean_auc:>10.4f} {std_auc:>9.4f} {marker}")
        best_alpha_per_model[model] = (best_alpha, best_auc)

    print("\nBest α per model (on external test, 10 seeds):")
    for model, (alpha, auc) in best_alpha_per_model.items():
        print(f"  {model}: α={alpha}  AUC={auc:.4f}")
    print("Note: best α selected on external test; for publication, use validation set.")

    # Save raw rows
    os.makedirs("research_results", exist_ok=True)
    out = Path("research_results/duplex_alpha_sensitivity.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
