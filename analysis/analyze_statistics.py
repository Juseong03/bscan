#!/usr/bin/env python
"""P3: Bootstrap CI and DeLong-style tests for key comparisons.

Comparisons:
1. BSCAN-FM (4 variants) vs CircCNN — internal AUC
2. BSCAN-FM vs BSCAN-onehot — external AUC (generalization claim)
3. BSCAN-FM raw vs BSCAN-FM+duplex — external AUC (duplex claim)
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import csv, glob, json, os, statistics
import numpy as np
from pathlib import Path
from scipy import stats

RNG = np.random.default_rng(42)
N_BOOT = 10000

# ── helpers ──────────────────────────────────────────────────────────────────

def bootstrap_diff_ci(a: list[float], b: list[float], n: int = N_BOOT, alpha: float = 0.05):
    """Bootstrap CI for mean(a) - mean(b).  Returns (diff, lo, hi, p_one_sided)."""
    a, b = np.array(a), np.array(b)
    obs = a.mean() - b.mean()
    diffs = []
    for _ in range(n):
        boot_a = RNG.choice(a, size=len(a), replace=True)
        boot_b = RNG.choice(b, size=len(b), replace=True)
        diffs.append(boot_a.mean() - boot_b.mean())
    diffs = np.array(diffs)
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    # One-sided p: fraction of bootstrap samples where diff <= 0 (null: a not better)
    p = float((diffs <= 0).mean())
    return obs, lo, hi, p

def paired_bootstrap_ci(paired: list[tuple[float,float]], n: int = N_BOOT, alpha: float = 0.05):
    """Bootstrap CI for mean of paired differences (a_i - b_i)."""
    diffs = np.array([a - b for a, b in paired])
    obs = diffs.mean()
    boot_means = []
    for _ in range(n):
        boot = RNG.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(boot.mean())
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    p = float((boot_means <= 0).mean())
    return obs, lo, hi, p

def fmt(obs, lo, hi, p):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return f"Δ={obs:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f} {sig}"

# ── load per-seed results from research_results/ model_comparison CSVs ────────

def load_internal_aucs(model_name: str, old_proj: str = "/workspace/volume/circRNA/BSCAN/research_results") -> list[float]:
    aucs = []
    for f in glob.glob(f"{old_proj}/model_comparison_*seed_*.csv"):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("model") == model_name and row.get("success") == "True" and row.get("test_auc"):
                    aucs.append(float(row["test_auc"]))
    return sorted(set(map(lambda x: round(x, 8), aucs)))  # deduplicate near-duplicates

def load_external_aucs(model_name: str, ext_dir: str = "external_data/circatlas/exon_controls") -> list[float]:
    path = Path(ext_dir) / f"{model_name}_external_control_results.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [float(r["auc"]) for r in rows if r.get("auc")]

# ── load duplex combination results ──────────────────────────────────────────

def load_duplex_aucs(model: str, alpha: float = 0.2) -> tuple[list[float], list[float]]:
    """Returns (raw_aucs, duplex_aucs) from fm_duplex_combination_summary.csv per-seed."""
    # We need per-seed, not just summary. Load from fm_duplex_combination_results if exists.
    raw_path = Path("external_data/circatlas/exon_controls") / f"{model}_external_control_results.csv"
    dup_path = Path("external_data/circatlas/exon_controls") / f"duplex_{model}_alpha{alpha}_results.csv"

    raw_aucs = load_external_aucs(model)

    # Try to find duplex per-seed from predictions
    pred_dir = Path("external_data/circatlas/exon_controls")
    from scipy.special import expit, logit as scipy_logit

    # Load raw predictions per seed and combine with duplex energy
    dup_aucs = []
    from sklearn.metrics import roc_auc_score
    import numpy as np2

    # Load junction data for duplex energy
    jdata = json.loads(Path("external_data/circatlas/exon_controls/seq_dict/junction.json").read_text())

    # Compute duplex energies once
    import RNA
    energies = {}
    for key, rec in jdata.items():
        upper_int = rec.get("upper_intron", rec.get("upper_seq", "")[:100])
        lower_int = rec.get("lower_intron", rec.get("lower_seq", "")[100:])
        try:
            duplex = RNA.duplexfold(upper_int, lower_int)
            energies[key] = duplex.energy
        except Exception:
            energies[key] = 0.0

    e_arr = np2.array([energies.get(k, 0.0) for k in jdata.keys()])
    mu, sigma = e_arr.mean(), e_arr.std() + 1e-9
    duplex_z = (-e_arr - (-mu)) / sigma  # z-score of negated energy

    # per seed predictions
    for seed in [42, 123, 315, 777, 1004, 2024, 2025, 2026, 3407, 9001]:
        pred_file = pred_dir / f"predictions_{model}_{seed}.csv"
        if not pred_file.exists():
            continue
        with open(pred_file) as f2:
            pred_rows = list(csv.DictReader(f2))

        keys_order = [r["key"] for r in pred_rows]
        labels_arr = np2.array([int(r["label"]) for r in pred_rows])
        probs_arr = np2.array([float(r["prob_bs"]) for r in pred_rows])

        # Align duplex_z to this order
        key_to_dz = dict(zip(jdata.keys(), duplex_z))
        dz_aligned = np2.array([key_to_dz.get(k, 0.0) for k in keys_order])

        from scipy.special import expit, logit
        base_logits = logit(np2.clip(probs_arr, 1e-6, 1 - 1e-6))
        combined = expit(base_logits + alpha * dz_aligned)

        dup_auc = roc_auc_score(labels_arr, combined)
        dup_aucs.append(dup_auc)

    return raw_aucs, dup_aucs


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("research_results", exist_ok=True)
    results = []

    print("=" * 70)
    print("P3: STATISTICAL TESTS — BOOTSTRAP CI")
    print("=" * 70)

    # ── 1. Internal AUC: BSCAN-FM variants vs CircCNN ───────────────────────
    print("\n[1] Internal AUC: BSCAN-FM 4종 vs CircCNN (10 seeds each)")
    fm_models = {
        "bscan_unified_fm":    "BSCAN-RNA-FM",
        "bscan_unified_ernie": "BSCAN-RNAErnie",
        "bscan_unified_msm":   "BSCAN-RNAMSM",
        "bscan_unified_bert":  "BSCAN-RNABERT",
    }
    circcnn_int = load_internal_aucs("circcnn")
    print(f"  CircCNN internal: n={len(circcnn_int)}, mean={np.mean(circcnn_int):.4f}")

    for mname, mlabel in fm_models.items():
        fm_int = load_internal_aucs(mname)
        if len(fm_int) < 5:
            print(f"  {mlabel}: insufficient seeds ({len(fm_int)})")
            continue
        # Truncate to matching length if needed
        n = min(len(fm_int), len(circcnn_int))
        obs, lo, hi, p = bootstrap_diff_ci(fm_int[:n], circcnn_int[:n])
        print(f"  {mlabel} vs CircCNN: {fmt(obs, lo, hi, p)}")
        results.append({"comparison": f"{mlabel}_vs_CircCNN_int_auc",
                        "n_a": len(fm_int), "n_b": len(circcnn_int),
                        "mean_a": np.mean(fm_int), "mean_b": np.mean(circcnn_int),
                        "delta": obs, "ci_lo": lo, "ci_hi": hi, "p": p})

    # ── 2. External AUC: BSCAN-FM vs BSCAN-onehot ──────────────────────────
    print("\n[2] External AUC: BSCAN-FM vs BSCAN-onehot (FM is architecture control)")
    onehot_ext = load_external_aucs("bscan_unified_onehot")
    print(f"  BSCAN-onehot external: n={len(onehot_ext)}, mean={np.mean(onehot_ext):.4f}")

    for mname, mlabel in fm_models.items():
        fm_ext = load_external_aucs(mname)
        if len(fm_ext) < 5 or len(onehot_ext) < 5:
            continue
        n = min(len(fm_ext), len(onehot_ext))
        obs, lo, hi, p = bootstrap_diff_ci(fm_ext[:n], onehot_ext[:n])
        print(f"  {mlabel} vs BSCAN-onehot: {fmt(obs, lo, hi, p)}")
        results.append({"comparison": f"{mlabel}_vs_onehot_ext_auc",
                        "n_a": len(fm_ext), "n_b": len(onehot_ext),
                        "mean_a": np.mean(fm_ext), "mean_b": np.mean(onehot_ext),
                        "delta": obs, "ci_lo": lo, "ci_hi": hi, "p": p})

    # ── 3. External AUC: BSCAN-FM vs CircCNN ─────────────────────────────
    print("\n[3] External AUC: BSCAN-FM vs CircCNN")
    circcnn_ext = load_external_aucs("circcnn")
    print(f"  CircCNN external: n={len(circcnn_ext)}, mean={np.mean(circcnn_ext):.4f}")

    for mname, mlabel in fm_models.items():
        fm_ext = load_external_aucs(mname)
        if len(fm_ext) < 5:
            continue
        n = min(len(fm_ext), len(circcnn_ext))
        obs, lo, hi, p = bootstrap_diff_ci(fm_ext[:n], circcnn_ext[:n])
        print(f"  {mlabel} vs CircCNN: {fmt(obs, lo, hi, p)}")
        results.append({"comparison": f"{mlabel}_vs_CircCNN_ext_auc",
                        "n_a": len(fm_ext), "n_b": len(circcnn_ext),
                        "mean_a": np.mean(fm_ext), "mean_b": np.mean(circcnn_ext),
                        "delta": obs, "ci_lo": lo, "ci_hi": hi, "p": p})

    # ── 4. External AUC: FM raw vs FM+duplex (paired) ───────────────────────
    print("\n[4] External AUC: BSCAN-FM raw vs +duplex (paired per seed, α=0.2)")
    for mname, mlabel in fm_models.items():
        try:
            raw_aucs, dup_aucs = load_duplex_aucs(mname, alpha=0.2)
        except Exception as e:
            print(f"  {mlabel}: error — {e}")
            continue
        if len(raw_aucs) < 5 or len(dup_aucs) < 5:
            print(f"  {mlabel}: insufficient seeds (raw={len(raw_aucs)}, dup={len(dup_aucs)})")
            continue
        n = min(len(raw_aucs), len(dup_aucs))
        paired = list(zip(dup_aucs[:n], raw_aucs[:n]))
        obs, lo, hi, p = paired_bootstrap_ci(paired)
        print(f"  {mlabel} +duplex vs raw: {fmt(obs, lo, hi, p)}")
        results.append({"comparison": f"{mlabel}_duplex_vs_raw_ext_auc",
                        "n_a": n, "n_b": n,
                        "mean_a": np.mean(dup_aucs[:n]), "mean_b": np.mean(raw_aucs[:n]),
                        "delta": obs, "ci_lo": lo, "ci_hi": hi, "p": p})

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path("research_results/statistical_tests.csv")
    if results:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
