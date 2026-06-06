#!/usr/bin/env python
"""
Run `experiment.py` across multiple models and summarize test metrics.

Why this script?
- Existing comparison scripts in this repo are ad-hoc / partially broken.
- We want a reproducible way to compare models with identical settings.

Features:
- Runs each model via subprocess, captures stdout/stderr
- Parses the final "| Test |" metrics line from stdout
- Writes results to CSV + JSON
- If RCM models are requested and `rcm_scores/` is missing, auto-generates a *subset*
  compatible with `--max_samples` using `generate_rcm_scores_subset.py`
"""

from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional


# Match the trainer's summary line, e.g. "| Test | 2/3 | 0.50 | ... |"
TEST_LINE_RE = re.compile(r"\|\s*Test\s*\|")


@dataclass
class RunResult:
    model: str
    success: bool
    exit_code: int
    seconds: float
    # metrics (may be None if parse fails)
    test_acc: Optional[float] = None
    test_macro_f1: Optional[float] = None
    test_precision_macro: Optional[float] = None
    test_recall_macro: Optional[float] = None
    test_mcc: Optional[float] = None
    test_auc: Optional[float] = None
    test_prc: Optional[float] = None
    # misc
    error: Optional[str] = None
    stdout_tail: Optional[str] = None


def parse_test_metrics(stdout: str) -> Dict[str, float]:
    """
    Parse the last "| Test |" line printed by trainer.
    Expected format (columns):
      | Test | epoch | acc | f1_macro | precision_macro | recall_macro | mcc | auc | prc |
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    test_lines = [ln for ln in lines if TEST_LINE_RE.search(ln)]
    if not test_lines:
        raise ValueError("No test line found in stdout")

    # pick last occurrence
    ln = test_lines[-1]
    # split by '|' and strip
    parts = [p.strip() for p in ln.split("|") if p.strip()]
    # Example parts:
    # ['Test', '2/2', '0.4375', '0.3043', '0.2188', '0.5000', '0.0000', '0.5000', '0.4375']
    if len(parts) < 9:
        raise ValueError(f"Unexpected test line format: {ln}")

    # parts[0] == 'Test'
    # parts[1] == epoch_str
    metrics = {
        "test_acc": float(parts[2]),
        "test_macro_f1": float(parts[3]),
        "test_precision_macro": float(parts[4]),
        "test_recall_macro": float(parts[5]),
        "test_mcc": float(parts[6]),
        "test_auc": float(parts[7]),
        "test_prc": float(parts[8]),
    }
    return metrics


def ensure_rcm_scores(
    *,
    repo_dir: str,
    junction_bps: int,
    flanking_bps: int,
    max_samples: Optional[int],
    seed: int,
    kmers: List[int],
) -> None:
    rcm_dir = os.path.join(repo_dir, "rcm_scores")
    # For reproducible comparisons with `--max_samples`, the required keys depend on seed and the
    # exact subsampling sequence. It's safest to regenerate a matching subset each run.
    if os.path.isdir(rcm_dir) and max_samples is not None:
        # remove existing subset to avoid missing-key errors
        for fn in os.listdir(rcm_dir):
            try:
                os.remove(os.path.join(rcm_dir, fn))
            except OSError:
                pass
    elif os.path.isdir(rcm_dir) and max_samples is None:
        return

    if max_samples is None:
        raise RuntimeError(
            "rcm_scores/ is missing and max_samples is None; generating full rcm_scores may be very slow. "
            "Please re-run with --max_samples N (recommended) or generate rcm_scores manually."
        )

    cmd = [
        "python",
        "pipeline/generate_rcm_scores_subset.py",
        "--junction_bps",
        str(junction_bps),
        "--flanking_bps",
        str(flanking_bps),
        "--max_samples",
        str(max_samples),
        "--seed",
        str(seed),
        "--kmers",
        *[str(k) for k in kmers],
    ]
    print(f"[rcm] rcm_scores/ missing -> generating subset with: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_dir, check=True)


def run_one_model(
    *,
    repo_dir: str,
    model: str,
    junction_bps: int,
    flanking_bps: int,
    epochs: int,
    earlystop: int,
    batch_size: int,
    lr: float,
    optimizer: str,
    device: int,
    seed: int,
    max_samples: Optional[int],
    split_strategy: str,
    extra_args: List[str],
) -> RunResult:
    cmd = [
        "python",
        "pipeline/experiment.py",
        "--model_name",
        model,
        "--junction_bps",
        str(junction_bps),
        "--flanking_bps",
        str(flanking_bps),
        "--epochs",
        str(epochs),
        "--earlystop",
        str(earlystop),
        "--batch_size",
        str(batch_size),
        "--optimizer",
        optimizer,
        "--lr",
        str(lr),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--split_strategy",
        split_strategy,
    ]
    if max_samples is not None and max_samples > 0:
        cmd += ["--max_samples", str(max_samples)]

    # verbose prints progress; keep it on for parsing.
    cmd += ["--verbose"]
    cmd += extra_args

    start = time.time()
    p = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    seconds = time.time() - start

    res = RunResult(model=model, success=(p.returncode == 0), exit_code=p.returncode, seconds=seconds)
    # keep last ~2000 chars for debugging
    out_tail = (p.stdout or "")[-2000:]
    err_tail = (p.stderr or "")[-2000:]
    res.stdout_tail = out_tail

    if p.returncode != 0:
        res.error = err_tail or out_tail or "Unknown failure"
        return res

    try:
        metrics = parse_test_metrics(p.stdout or "")
        for k, v in metrics.items():
            setattr(res, k, v)
    except Exception as e:
        res.success = False
        res.error = f"Metric parse failed: {e}"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Model names to run (default: common set)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--earlystop", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--batch_size_pretrained", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", type=str, default="adamw")
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--junction_bps", type=int, default=100)
    ap.add_argument("--flanking_bps", type=int, default=100)
    ap.add_argument("--max_samples", type=int, default=None, help="Max samples per split (default: None for full run)")
    ap.add_argument(
        "--split_strategy",
        type=str,
        default="sample",
        choices=["sample", "transcript", "chromosome"],
        help="sample: original split; transcript: hold transcript IDs out across train/valid/test; chromosome: hold chromosomes out across partitions.",
    )
    ap.add_argument("--out_dir", type=str, default="logs")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--include_rcm", action="store_true", help="Include circcnnrcm/circcnntri (requires rcm_scores/).")
    args = ap.parse_args()

    repo_dir = _ROOT  # repo root (this file lives in pipeline/)
    os.makedirs(os.path.join(repo_dir, args.out_dir), exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    default_models = [
        "deepcirccode",
        "circcnnsingle",
        "circcnn",
        "circcnndouble",
        "circnet",
        "circdeep",
        "jedi",
        "circdc",
        # pretrained-style
        "circcnnatt",
        "circattrcm",
        "circmamba",
        "circfusion",
        "circalignmap",
        "circunified",
        "circattrcm_scratch",
        "circalignmap_scratch",
        "circunified_scratch",
    ]
    if args.include_rcm:
        default_models += ["circcnnrcm", "circcnntri"]

    # Our new interpretable motif model
    default_models += ["circmotif", "circstem", "circstemv2", "circbialign"]

    models = args.models if args.models is not None and len(args.models) > 0 else default_models

    # Ensure rcm_scores if needed
    if any(m in {"circcnnrcm", "circcnntri"} for m in models):
        ensure_rcm_scores(
            repo_dir=repo_dir,
            junction_bps=args.junction_bps,
            flanking_bps=args.flanking_bps,
            max_samples=args.max_samples,
            seed=args.seed,
            kmers=[5, 7, 9, 11, 13],
        )

    results: List[RunResult] = []
    for m in models:
        is_pretrained = m in {"circcnnatt", "bscan_v2", "bscan_seq", "bscan_seq_lite", "bscan_seq_lite_xattn", "bscan_seq_rcaug", "bscan_seq_rcattn", "bscan_seq_mamba_aux", "bscan_plus", "circattrcm", "circmamba", "circfusion", "circalignmap"}
        bs = args.batch_size_pretrained if is_pretrained else args.batch_size
        print(f"[run] {m} (batch_size={bs}) ...")
        rr = run_one_model(
            repo_dir=repo_dir,
            model=m,
            junction_bps=args.junction_bps,
            flanking_bps=args.flanking_bps,
            epochs=args.epochs,
            earlystop=args.earlystop,
            batch_size=bs,
            lr=args.lr,
            optimizer=args.optimizer,
            device=args.device,
            seed=args.seed,
            max_samples=args.max_samples,
            split_strategy=args.split_strategy,
            extra_args=[],
        )
        results.append(rr)
        status = "OK" if rr.success else "FAIL"
        print(f"[{status}] {m}  acc={rr.test_acc} auc={rr.test_auc} prc={rr.test_prc}  ({rr.seconds:.1f}s)")
        if not rr.success and rr.error:
            print(f"      error: {rr.error.splitlines()[-1]}")

    # Write JSON
    out_json = os.path.join(repo_dir, args.out_dir, f"model_comparison_{tag}.json")
    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Write CSV
    out_csv = os.path.join(repo_dir, args.out_dir, f"model_comparison_{tag}.csv")
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    # Print a short leaderboard
    ok = [r for r in results if r.success and r.test_auc is not None]
    ok.sort(key=lambda r: (r.test_auc or 0.0), reverse=True)
    print("\n=== Leaderboard (by test_auc) ===")
    for r in ok:
        print(f"{r.model:12s}  auc={r.test_auc:.4f}  acc={r.test_acc:.4f}  prc={r.test_prc:.4f}  sec={r.seconds:.1f}")
    print(f"\nWrote:\n- {out_csv}\n- {out_json}")


if __name__ == "__main__":
    main()
