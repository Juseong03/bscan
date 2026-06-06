"""
Lightweight smoke test for model wiring (data -> model forward -> loss).

Goal: quickly verify that each model can run a single forward pass on a small batch
without doing full training. This catches shape/return-type/embedding-index issues early.
"""

from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import random
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from dataloader import DataSetPrep, circData_single, circData_double, circData_rcm, circData_triple
from trainer import Trainer
from utils import seed_everything, get_device


@dataclass
class SmokeResult:
    model_name: str
    ok: bool
    batch_size: int = 0
    logits_shape: Optional[Tuple[int, ...]] = None
    error: Optional[str] = None


def _pick_keys(keys: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if len(keys) <= n:
        return list(keys)
    return rng.sample(keys, n)


def _one_batch_forward(trainer: Trainer, loader: torch.utils.data.DataLoader) -> Tuple[int, Tuple[int, ...]]:
    batch = next(iter(loader))
    # Keep everything no_grad; we're just checking forward & loss wiring.
    with torch.no_grad():
        _, _, label, pred = trainer.forward(batch)
        if isinstance(pred, (tuple, list)):
            # Defensive: some model variants may return (logits, aux)
            pred = pred[0]
        # Ensure loss is computable
        _ = trainer.loss_fn(pred, label)
    return int(label.shape[0]), tuple(pred.shape)


def run_smoke(
    models: List[str],
    *,
    seed: int = 42,
    device_num: int = -1,
    junction_bps: int = 100,
    flanking_bps: int = 100,
    sample_n: int = 32,
    batch_size: int = 8,
) -> List[SmokeResult]:
    seed_everything(seed)
    device = get_device(device_num)

    data = DataSetPrep(
        coord_path="./data/BS_LS_coordinates_final.csv",
        seq_dict_path="./data/hg19_seq_dict.json",
        junction_bps=junction_bps,
        flanking_bps=flanking_bps,
        seed=seed,
    )

    try:
        data.load_junction_flanking_seq()
    except Exception:
        data.get_junction_intron_seq()
        data.save_junction_flanking_seq()

    keys_train, _, _ = data.split_data()
    keys = _pick_keys(keys_train, sample_n, seed=seed)

    # Use a temporary save dir to avoid touching real experiment artifacts.
    trainer = Trainer(seed=seed, device=device, dir_save="/tmp/circRNA_smoke_saved_models", dir_log="/tmp/circRNA_smoke_logs")

    results: List[SmokeResult] = []

    # RCM models need precomputed jsons.
    has_rcm_scores = False
    try:
        import os

        has_rcm_scores = os.path.isdir("./rcm_scores")
    except Exception:
        has_rcm_scores = False

    for model_name in models:
        if model_name in {"circcnnrcm", "circcnntri"} and not has_rcm_scores:
            results.append(
                SmokeResult(
                    model_name=model_name,
                    ok=False,
                    error="rcm_scores/ is missing (generate with calculate_rcm.py / RCSFinder.py before testing this model).",
                )
            )
            continue

        try:
            # Build a small dataset matching the model’s expected inputs (mirrors experiment.py).
            if model_name in ["deepcirccode", "circcnnsingle", "circnet"]:
                seq_tensor, _, label_tensor = data.seq_to_tensor(keys, is_concat=True)
                ds = circData_single(seq_tensor, label_tensor)

            elif model_name in ["circcnn", "circcnndouble", "circdc"]:
                upper_tensor, lower_tensor, label_tensor = data.seq_to_tensor(keys)
                ds = circData_double(upper_tensor, lower_tensor, label_tensor)

            elif model_name in ["circattrcm", "circcnnatt", "bscan_v2", "circmamba", "circfusion", "circalignmap", "circunified"]:
                # Important: keep sequence length aligned with junction-derived lengths (e.g. 200),
                # so strip special tokens.
                upper, lower, lower_rc, label_tensor = data.tensor_for_pretrained(
                    keys, rc=True, tokenizer="rnaernie", special_tokens=False
                )
                ds = circData_triple(upper, lower, lower_rc, label_tensor)

            elif model_name in ["bscan_seq", "bscan_seq_lite", "bscan_seq_lite_xattn", "bscan_seq_rcaug", "bscan_seq_rcattn", "bscan_seq_mamba_aux", "bscan_plus"]:
                upper, lower, _, label_tensor = data.tensor_for_pretrained(
                    keys, rc=False, tokenizer="rnaernie", special_tokens=False
                )
                ds = circData_double(upper, lower, label_tensor)

            elif model_name in ["circcnnrcm"]:
                rcm_flanking_list = [flanking_bps]
                rcm_kmer_list = [5, 7, 9, 11, 13]
                upper, lower, flanking_rcm, upper_rcm, lower_rcm, label_tensor = data.seq_to_tensor_w_rcm(
                    keys, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
                )
                ds = circData_triple(flanking_rcm, upper_rcm, lower_rcm, label_tensor)

            elif model_name in ["circcnntri"]:
                rcm_flanking_list = [flanking_bps]
                rcm_kmer_list = [5, 7, 9, 11, 13]
                upper, lower, flanking_rcm, upper_rcm, lower_rcm, label_tensor = data.seq_to_tensor_w_rcm(
                    keys, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
                )
                ds = circData_rcm(upper, lower, flanking_rcm, upper_rcm, lower_rcm, label_tensor)

            elif model_name in ["circdeep", "jedi"]:
                # kmer indexing models
                kmer = 3
                if model_name == "circdeep":
                    seqs, _, _, _, label_tensor = data.seq_to_index(keys, kmer=kmer)
                    ds = circData_single(seqs, label_tensor)
                else:
                    _, upper, lower, _, label_tensor = data.seq_to_index(keys, kmer=kmer)
                    ds = circData_double(upper, lower, label_tensor)

            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            loader = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False)

            if model_name in ["circcnnrcm", "circcnntri"]:
                n_rcm_features = 1 * 5  # flanking_list=[flanking_bps], kmer_list=[5,7,9,11,13]
                trainer.define_model(model_name, n_rcm_features=n_rcm_features)
            else:
                trainer.define_model(model_name)
            bsz, shape = _one_batch_forward(trainer, loader)
            results.append(SmokeResult(model_name=model_name, ok=True, batch_size=bsz, logits_shape=shape))

        except Exception:
            results.append(
                SmokeResult(
                    model_name=model_name,
                    ok=False,
                    error=traceback.format_exc(limit=8),
                )
            )

    return results


def main() -> None:
    models = [
        # baselines / non-pretrained
        "deepcirccode",
        "circcnnsingle",
        "circcnn",
        "circcnndouble",
        "circnet",
        "circdeep",
        "jedi",
        "circdc",
        # pretrained-based
        "circcnnatt",
        "bscan_v2",
        "bscan_seq",
        "bscan_seq_lite",
        "bscan_seq_rcaug",
        "bscan_seq_rcattn",
        "bscan_seq_mamba_aux",
        "bscan_plus",
        "circattrcm",
        "circmamba",
        "circfusion",
        "circalignmap",
        "circunified",
        # rcm-based (will be skipped if rcm_scores missing)
        "circcnnrcm",
        "circcnntri",
    ]

    results = run_smoke(models, device_num=-1, junction_bps=100, flanking_bps=100, sample_n=32, batch_size=8)

    print("\n=== SMOKE RESULTS ===")
    any_fail = False
    for r in results:
        if r.ok:
            print(f"OK  {r.model_name:12s}  batch={r.batch_size:2d}  logits={r.logits_shape}")
        else:
            any_fail = True
            print(f"FAIL {r.model_name:12s}  {r.error.splitlines()[-1] if r.error else ''}")

    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
