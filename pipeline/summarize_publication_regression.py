"""
Summarize publication regression comparison logs.

Maps implementation names to paper-facing labels:
  - bscan_seq_lite_regression -> BSCAN
  - deepcirccode_regression -> DeepCircCode
  - circcnn_regression -> CircCNN
"""

from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import glob
import os
import re

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "bscan_seq_lite_regression": "BSCAN",
    "bscan_seq_rcattn_regression": "BSCAN-RCAttn",
    "bscan_plus_regression": "BSCAN+",
    "deepcirccode_regression": "DeepCircCode",
    "circcnn_regression": "CircCNN",
}


DL_PATTERN = re.compile(
    r"\| Test\s*\| Loss: ([\d\.]+) \| MAE: ([\d\.]+) \| RMSE: ([\d\.]+) \| "
    r"Pearson r: ([\-\d\.]+) \|(?: R2: ([\-\d\.]+) \|)?"
)


def parse_one_log(path: str):
    basename = os.path.basename(path).replace(".log", "")
    match = re.match(r"publication_regression_comparison_(.+)_seed_(\d+)$", basename)
    if not match:
        return None
    model_tag = match.group(1)
    seed = int(match.group(2))

    with open(path, "r") as f:
        content = f.read()

    found = DL_PATTERN.search(content)
    if not found:
        return None

    loss, mae, rmse, pearson = map(float, found.groups()[:4])
    r2 = float(found.group(5)) if found.group(5) is not None else np.nan
    model = model_tag
    if model.endswith("_seed"):
        model = model[:-5]
    if model == "deepcode":
        model = "deepcirccode_regression"

    return {
        "model": model,
        "seed": seed,
        "test_loss": loss,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_pearson": pearson,
        "test_r2": r2,
    }


def main() -> None:
    pattern = "logs/publication_regression_comparison_*.log"
    files = glob.glob(pattern)
    if not files:
        print(f"No log files found for pattern: {pattern}")
        return

    rows = []
    for path in files:
        row = parse_one_log(path)
        if row is not None:
            rows.append(row)

    if not rows:
        print("Could not parse any regression results.")
        return

    df = pd.DataFrame(rows)
    summary = df.groupby("model")[["test_mae", "test_rmse", "test_pearson", "test_r2"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["model"] + [f"{m}_{s}" for m in ["test_mae", "test_rmse", "test_pearson", "test_r2"] for s in ["mean", "std"]]
    summary["display_name"] = summary["model"].map(MODEL_LABELS).fillna(summary["model"])
    summary = summary.sort_values("test_pearson_mean", ascending=False)

    print("\n" + "=" * 88)
    print("Publication regression summary")
    print("=" * 88)
    for _, row in summary.iterrows():
        print(
            f"{row['display_name']:<14} "
            f"MAE {row['test_mae_mean']:.4f} ± {row['test_mae_std']:.4f}  "
            f"RMSE {row['test_rmse_mean']:.4f} ± {row['test_rmse_std']:.4f}  "
            f"Pearson {row['test_pearson_mean']:.4f} ± {row['test_pearson_std']:.4f}  "
            f"R2 {row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}"
        )

    out_path = "research_results/publication_regression_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
