#!/usr/bin/env python
"""Aggregate AUG-RCM results (docs/DESIGN_new_experiments.md §2).

Reads the model_comparison CSVs produced by scripts/run_rcm_aux.sh
(research_results/model_comparison_augrcm_fb<F>_seed_<S>.csv) and summarises
internal test AUC/PRC for the FM baseline vs FM+RCM at each flanking width.

  rows: (bscan_unified_fm, bscan_unified_fm_rcm) x flanking width
  cols: n_seeds, test_auc mean±std, test_prc mean±std, Δauc vs baseline

External-A (circAtlas) and Tier2/3 are produced by the standard
evaluate_*/hard_negative pipelines on the saved bscan_unified_fm_rcm
checkpoints — this script only covers the internal split.

Usage (from repo root):  python analysis/evaluate_rcm_aux.py
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import csv, glob, re, statistics
from collections import defaultdict

PATTERN = "research_results/model_comparison_augrcm_fb*_seed_*.csv"
FB_RE = re.compile(r"_fb(\d+)_seed_(\d+)\.csv$")
MODELS = ["bscan_unified_fm", "bscan_unified_fm_rcm"]
LABELS = {"bscan_unified_fm": "FM baseline", "bscan_unified_fm_rcm": "FM + RCM aux"}
OUT = "results/rcm_aux_summary.csv"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        print(f"No files match {PATTERN}. Run scripts/run_rcm_aux.sh first.")
        return

    # (flanking, model) -> {"auc": [...], "prc": [...]}
    acc = defaultdict(lambda: {"auc": [], "prc": []})
    for path in files:
        m = FB_RE.search(path)
        if not m:
            continue
        fb = int(m.group(1))
        with open(path) as fh:
            for row in csv.DictReader(fh):
                model = row.get("model")
                if model not in MODELS:
                    continue
                auc, prc = _f(row.get("test_auc")), _f(row.get("test_prc"))
                if auc is not None:
                    acc[(fb, model)]["auc"].append(auc)
                if prc is not None:
                    acc[(fb, model)]["prc"].append(prc)

    def ms(v):
        if not v:
            return (None, None)
        return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)

    flanks = sorted({fb for (fb, _) in acc})
    rows = []
    for fb in flanks:
        base_auc = ms(acc[(fb, "bscan_unified_fm")]["auc"])[0]
        for model in MODELS:
            a_m, a_s = ms(acc[(fb, model)]["auc"])
            p_m, p_s = ms(acc[(fb, model)]["prc"])
            n = len(acc[(fb, model)]["auc"])
            delta = (a_m - base_auc) if (a_m is not None and base_auc is not None) else None
            rows.append({
                "flanking_bps": fb,
                "model": model,
                "label": LABELS[model],
                "n_seeds": n,
                "test_auc_mean": None if a_m is None else round(a_m, 4),
                "test_auc_std": None if a_s is None else round(a_s, 4),
                "test_prc_mean": None if p_m is None else round(p_m, 4),
                "test_prc_std": None if p_s is None else round(p_s, 4),
                "delta_auc_vs_fm": None if delta is None else round(delta, 4),
            })

    _os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # console table
    print(f"\nAUG-RCM internal-split summary  ({len(files)} run files)\n")
    print(f"{'flank':>6} {'model':<26} {'n':>2} {'AUC':>16} {'PRC':>16} {'Δauc':>8}")
    for r in rows:
        auc = "—" if r["test_auc_mean"] is None else f"{r['test_auc_mean']:.4f}±{r['test_auc_std']:.4f}"
        prc = "—" if r["test_prc_mean"] is None else f"{r['test_prc_mean']:.4f}±{r['test_prc_std']:.4f}"
        d = "" if r["delta_auc_vs_fm"] is None else f"{r['delta_auc_vs_fm']:+.4f}"
        print(f"{r['flanking_bps']:>6} {r['label']:<26} {r['n_seeds']:>2} {auc:>16} {prc:>16} {d:>8}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
