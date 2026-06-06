#!/usr/bin/env python
"""Update paper_table_master.csv with BSCAN-FM+Mamba results after training completes."""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv
import statistics
from pathlib import Path


def load_summary(path: Path, model_name: str) -> dict | None:
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["model"] == model_name:
                return row
    return None


def load_external_summary(model_name: str) -> dict | None:
    path = Path("external_data/circatlas/exon_controls") / f"{model_name}_external_control_summary.csv"
    if not path.exists():
        print(f"[skip] External summary not found: {path}")
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def load_internal_summary(model_name: str) -> dict | None:
    """Read internal AUC from research_results or old project results."""
    # Try research_results first
    for pattern in [
        f"research_results/model_comparison_*{model_name}*.csv",
    ]:
        import glob
        files = sorted(glob.glob(str(pattern)))
        if files:
            aucs = []
            for f in files:
                with open(f) as fh:
                    for row in csv.DictReader(fh):
                        if row.get("model") == model_name and row.get("success") == "True" and row.get("test_auc"):
                            aucs.append(float(row["test_auc"]))
            if aucs:
                return {"auc_mean": statistics.mean(aucs), "auc_std": statistics.stdev(aucs) if len(aucs) > 1 else 0.0, "n": len(aucs)}
    return None


def main():
    model_name = "bscan_unified_fm_mambaadapter"
    master_path = Path("results/paper_table_master.csv")

    print(f"Collecting results for {model_name}...")

    # Internal AUC from training logs (research_results csvs)
    import glob
    int_aucs, int_prcs, int_mccs = [], [], []
    for f in sorted(glob.glob("saved_models/bscan_unified_fm_mambaadapter/*/training_log.csv")):
        pass  # not available in standard setup

    # Try reading from research_results model_comparison CSVs
    for f in sorted(glob.glob("/workspace/volume/circRNA/BSCAN/research_results/model_comparison_bscan_unified_ablation_seed_*.csv")):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("model") == "bscan_unified_fm_mambaadapter" and row.get("success") == "True":
                    if row.get("test_auc"):
                        int_aucs.append(float(row["test_auc"]))
                    if row.get("test_prc"):
                        int_prcs.append(float(row["test_prc"]))
                    if row.get("test_mcc"):
                        int_mccs.append(float(row["test_mcc"]))

    if not int_aucs:
        print("[warn] No internal AUC data found — check research_results CSVs")
    else:
        print(f"  Internal AUC: {statistics.mean(int_aucs):.4f}±{statistics.stdev(int_aucs) if len(int_aucs)>1 else 0:.4f} (n={len(int_aucs)})")

    # External AUC
    ext_path = Path("external_data/circatlas/exon_controls/bscan_unified_fm_mambaadapter_external_control_summary.csv")
    ext_data = None
    if ext_path.exists():
        with open(ext_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            ext_data = rows[0]
            print(f"  External AUC: {float(ext_data['auc_mean']):.4f}±{float(ext_data['auc_std']):.4f}")

    # Tier2
    t2_path = Path("research_results/hard_negative_pairing_ls_lower_intron_summary.csv")
    t2_data = None
    if t2_path.exists():
        with open(t2_path) as f:
            for row in csv.DictReader(f):
                if row["model"] == model_name:
                    t2_data = row
                    print(f"  Tier2 AUC: {float(t2_data['auc_mean']):.4f}±{float(t2_data['auc_std']):.4f}")
                    break

    # Tier3
    t3_path = Path("research_results/hard_negative_pairing_lower_intron_summary.csv")
    t3_data = None
    if t3_path.exists():
        with open(t3_path) as f:
            for row in csv.DictReader(f):
                if row["model"] == model_name:
                    t3_data = row
                    print(f"  Tier3 AUC: {float(t3_data['auc_mean']):.4f}±{float(t3_data['auc_std']):.4f}")
                    break

    # Build row
    if not (int_aucs and ext_data):
        print("[warn] Insufficient data to update paper table. Exiting.")
        return

    int_auc_mean = statistics.mean(int_aucs)
    int_auc_std = statistics.stdev(int_aucs) if len(int_aucs) > 1 else 0.0
    ext_auc_mean = float(ext_data["auc_mean"])
    ext_auc_std = float(ext_data["auc_std"])
    ext_prc = float(ext_data["prc_mean"])
    ext_mcc = float(ext_data["mcc_mean"])
    drop_pct = (int_auc_mean - ext_auc_mean) / int_auc_mean * 100

    new_row = {
        "display": "BSCAN-FM+Mamba",
        "n_seeds": len(int_aucs),
        "params": "",
        "int_auc": round(int_auc_mean, 4),
        "int_auc_std": round(int_auc_std, 4),
        "int_prc": round(statistics.mean(int_prcs), 4) if int_prcs else "",
        "int_mcc": round(statistics.mean(int_mccs), 4) if int_mccs else "",
        "ext_auc": round(ext_auc_mean, 4),
        "ext_auc_std": round(ext_auc_std, 4),
        "ext_prc": round(ext_prc, 4),
        "ext_mcc": round(ext_mcc, 4),
        "drop_pct": round(drop_pct, 4),
        "t2_auc": round(float(t2_data["auc_mean"]), 4) if t2_data else "",
        "t2_auc_std": round(float(t2_data["auc_std"]), 4) if t2_data else "",
        "t3_auc": round(float(t3_data["auc_mean"]), 4) if t3_data else "",
        "t3_auc_std": round(float(t3_data["auc_std"]), 4) if t3_data else "",
        "t3_dup_auc": "",
    }

    # Read existing master and insert after BSCAN-FM+CNN row
    with open(master_path) as f:
        rows = list(csv.DictReader(f))

    # Remove any existing Mamba row
    rows = [r for r in rows if r["display"] != "BSCAN-FM+Mamba"]

    # Insert after BSCAN-FM+CNN
    insert_idx = next((i for i, r in enumerate(rows) if r["display"] == "BSCAN-FM+CNN(ada)"), 4) + 1
    rows.insert(insert_idx, new_row)

    fieldnames = list(rows[0].keys())
    with open(master_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Updated {master_path} with BSCAN-FM+Mamba row.")
    print(f"   Internal AUC: {int_auc_mean:.4f}, External AUC: {ext_auc_mean:.4f}, Drop: {drop_pct:.1f}%")


if __name__ == "__main__":
    main()
