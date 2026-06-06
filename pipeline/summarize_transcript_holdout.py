from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import argparse
import glob
import os
import re

import pandas as pd


METRICS = ["test_auc", "test_prc", "test_acc", "test_mcc", "seconds"]


def infer_seed(path: str) -> int | None:
    match = re.search(r"seed[_-](\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frames = []
    for path in sorted(glob.glob(args.pattern)):
        df = pd.read_csv(path)
        df["source_file"] = path
        df["seed"] = infer_seed(path)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No files matched: {args.pattern}")

    raw = pd.concat(frames, ignore_index=True)
    if {"model", "seed", "success"}.issubset(raw.columns):
        raw["_source_mtime"] = raw["source_file"].map(lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0)
        raw["_success_rank"] = raw["success"].map(lambda x: 1 if bool(x) else 0)
        raw = (
            raw.sort_values(["model", "seed", "_success_rank", "_source_mtime"])
            .drop_duplicates(["model", "seed"], keep="last")
            .drop(columns=["_success_rank", "_source_mtime"])
        )
    raw_out = args.out.replace(".csv", "_raw.csv")
    raw.to_csv(raw_out, index=False)

    ok = raw[raw["success"] == True].copy()
    rows = []
    for model, sub in ok.groupby("model"):
        row = {"model": model, "n_seeds": sub["seed"].nunique()}
        for metric in METRICS:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    if "test_auc_mean" in summary.columns:
        summary = summary.sort_values("test_auc_mean", ascending=False)
    summary.to_csv(args.out, index=False)

    print(f"Wrote:\n- {args.out}\n- {raw_out}")
    if not summary.empty:
        cols = [c for c in ["model", "n_seeds", "test_auc_mean", "test_prc_mean", "test_acc_mean", "test_mcc_mean"] if c in summary]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
