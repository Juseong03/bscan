"""
Generate *subset* rcm_scores/ files compatible with `dataloader.DataSetPrep.seq_to_tensor_w_rcm`.

Why subset?
- Full CircCNNs-style RCM computation across many flanking lengths and all samples can be very slow.
- For quick debugging / mini experiments (`experiment.py --max_samples N`), we only need RCM scores
  for the keys that will be used in those subsampled splits.

This script replicates the same split + subsampling logic in `experiment.py` and then computes RCM
score distributions for the union of train/valid/test keys.

Outputs:
  rcm_scores/{flanking|upper|lower}_{flanking_bps}_bps_{k}mer_scores.json
where each file is a dict: key -> [25 ints] (5x5 interval counts).
"""

from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List

from tqdm import tqdm

from dataloader import DataSetPrep
from RCSFinder import RCSFinder


def _subsample(keys: List[str], n: int, rng: random.Random) -> List[str]:
    if len(keys) <= n:
        return keys
    return rng.sample(keys, n)


def compute_scores_for_keys(
    flanking_seq: Dict[str, dict],
    keys: Iterable[str],
    *,
    kmers: int,
    rcm_type: str,
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for key in tqdm(list(keys), desc=f"RCM {rcm_type} k={kmers}"):
        value = flanking_seq[key]
        if rcm_type == "flanking":
            _, dist = RCSFinder(
                key=key,
                upper_seq=value["upper_flanking"],
                lower_seq=value["lower_flanking"],
                is_flanking_introns=True,
                kmers=kmers,
                is_upper_intron=False,
                seq_fraction_of_spacer=0,
                allowed_seed_mismatch=0,
            ).subseq_validity_check()
        elif rcm_type == "upper":
            _, dist = RCSFinder(
                key=key,
                upper_seq=value["upper_flanking"],
                lower_seq=value["lower_flanking"],
                is_flanking_introns=False,
                kmers=kmers,
                is_upper_intron=True,
                seq_fraction_of_spacer=0,
                allowed_seed_mismatch=0,
            ).subseq_validity_check()
        elif rcm_type == "lower":
            _, dist = RCSFinder(
                key=key,
                upper_seq=value["upper_flanking"],
                lower_seq=value["lower_flanking"],
                is_flanking_introns=False,
                kmers=kmers,
                is_upper_intron=False,
                seq_fraction_of_spacer=0,
                allowed_seed_mismatch=0,
            ).subseq_validity_check()
        else:
            raise ValueError(f"Invalid rcm_type: {rcm_type}")
        # Ensure plain Python ints for JSON serialization (avoid numpy.int64)
        out[key] = [int(x) for x in dist]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--junction_bps", type=int, default=100)
    ap.add_argument("--flanking_bps", type=int, default=100)
    ap.add_argument("--max_samples", type=int, default=64, help="Match experiment.py --max_samples.")
    ap.add_argument("--seed", type=int, default=42, help="Match experiment.py --seed.")
    ap.add_argument("--kmers", type=int, nargs="*", default=[5, 7, 9, 11, 13])
    ap.add_argument("--out_dir", type=str, default="./rcm_scores")
    args = ap.parse_args()

    data = DataSetPrep(
        coord_path="./data/BS_LS_coordinates_final.csv",
        seq_dict_path="./data/hg19_seq_dict.json",
        junction_bps=args.junction_bps,
        flanking_bps=args.flanking_bps,
        seed=args.seed,
        use_full_intron=False,
    )

    try:
        data.load_junction_flanking_seq()
    except Exception:
        data.get_junction_intron_seq()
        data.save_junction_flanking_seq()

    keys_train, keys_valid, keys_test = data.split_data()
    # Match `experiment.py` behavior: use a single RNG instance across splits so
    # sampling is deterministic and consistent with the experiment run.
    rng = random.Random(args.seed)
    keys_train = _subsample(keys_train, args.max_samples, rng=rng)
    keys_valid = _subsample(keys_valid, args.max_samples, rng=rng)
    keys_test = _subsample(keys_test, args.max_samples, rng=rng)

    keys_union = sorted(set(keys_train) | set(keys_valid) | set(keys_test))
    print(
        f"Keys for subset RCM: train={len(keys_train)} valid={len(keys_valid)} test={len(keys_test)} union={len(keys_union)}"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    for rcm_type in ["flanking", "upper", "lower"]:
        for k in args.kmers:
            scores = compute_scores_for_keys(data.flanking_seq, keys_union, kmers=k, rcm_type=rcm_type)
            out_path = os.path.join(args.out_dir, f"{rcm_type}_{args.flanking_bps}_bps_{k}mer_scores.json")
            with open(out_path, "w") as f:
                json.dump(scores, f)
            print(f"Wrote {out_path} ({len(scores)} keys)")


if __name__ == "__main__":
    main()


