#!/usr/bin/env python
"""Evaluate BSCANUnified FM adapter models on circAtlas exon-aware external set.

Uses pre-extracted FM embeddings from external_data/circatlas/exon_controls/fm_embeddings/
and saved checkpoints from saved_models/{model_name}/{seed}/model.pth.

Outputs results in the same format as evaluate_circatlas_all_baselines.py.
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score
from torch.utils.data import DataLoader, Dataset

L = 100  # junction_bps

NUC_OH = {"A": 0, "G": 1, "C": 2, "T": 3, "U": 3}

# FM adapter configs: model_name -> (encoder_type, adapter_type, adapter_layers)
FM_ADAPTER_MODELS = {
    "bscan_unified_fm_cnnadapter":   ("rnafm",    "cnn",   2),
    "bscan_unified_fm_mambaadapter": ("rnafm",    "mamba", 1),
    "bscan_unified_ernie_cnnadapter":  ("rnaernie", "cnn",   2),
    "bscan_unified_ernie_mambaadapter":("rnaernie", "mamba", 1),
}

SEEDS = [42, 123, 315, 777, 1004, 2024, 2025, 2026, 3407, 9001]


def seq_to_onehot(seq: str, length: int) -> np.ndarray:
    arr = np.zeros((4, length), dtype=np.float32)
    for j, base in enumerate(seq[:length].upper()):
        ch = NUC_OH.get(base)
        if ch is not None:
            arr[ch, j] = 1.0
    return arr


class CircAtlasFMDataset(Dataset):
    """Loads pre-cached FM embeddings + one-hot stems for each sample."""

    def __init__(self, junction: dict, fm_emb_dir: Path):
        self.keys = list(junction.keys())
        self.junction = junction
        self.fm_emb_dir = fm_emb_dir

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        rec = self.junction[key]

        # Load pre-extracted FM embeddings
        pt_path = self.fm_emb_dir / f"{key.replace('|', '_')}.pt"
        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        upper_emb = emb["upper"].float()   # [seq_len, d_fm]
        lower_emb = emb["lower"].float()
        lower_rc_emb = emb["lower_rc"].float()

        # Compute one-hot for stem branch
        upper_seq = rec["upper_seq"]
        lower_seq = rec["lower_seq"]

        # upper_oh: upper intron [4, L] = upper_seq[:L]
        upper_oh = torch.from_numpy(seq_to_onehot(upper_seq[:L], L))

        # lower_rc_oh: RC of lower intron [4, L] = lower_seq[L:] reversed complemented
        lower_intron = lower_seq[L:2*L]
        lower_intron_oh = seq_to_onehot(lower_intron, L)  # [4, L]
        # reverse complement: swap A<->T (0<->3), C<->G (1<->2), then reverse
        lower_rc_arr = lower_intron_oh[[3, 2, 1, 0], ::-1].copy()
        lower_rc_oh = torch.from_numpy(lower_rc_arr)

        raw_label = rec.get("label", "BS")
        label = 1 if str(raw_label).upper() == "BS" else 0

        return key, upper_emb, lower_emb, lower_rc_emb, upper_oh, lower_rc_oh, label


def collate_fn(batch):
    keys = [b[0] for b in batch]
    upper_embs   = [b[1] for b in batch]
    lower_embs   = [b[2] for b in batch]
    lower_rc_embs = [b[3] for b in batch]
    upper_ohs    = torch.stack([b[4] for b in batch])
    lower_rc_ohs = torch.stack([b[5] for b in batch])
    labels       = torch.tensor([b[6] for b in batch], dtype=torch.long)

    # Pad FM embeddings to same length
    def pad_embs(emb_list):
        max_len = max(e.shape[0] for e in emb_list)
        padded = torch.zeros(len(emb_list), max_len, emb_list[0].shape[1])
        for i, e in enumerate(emb_list):
            padded[i, :e.shape[0]] = e
        return padded

    return (keys, pad_embs(upper_embs), pad_embs(lower_embs),
            pad_embs(lower_rc_embs), upper_ohs, lower_rc_ohs, labels)


def load_model(model_name: str, seed: int, device: str) -> torch.nn.Module | None:
    from models.bscan_unified import BSCANUnified

    enc_type, adapter_type, adapter_layers = FM_ADAPTER_MODELS[model_name]
    ckpt = Path("saved_models") / model_name / str(seed) / "model.pth"
    if not ckpt.exists():
        print(f"  [skip] {ckpt} not found")
        return None

    model = BSCANUnified(
        encoder_type=enc_type,
        use_cached=True,
        adapter_type=adapter_type,
        adapter_layers=adapter_layers,
    )
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    all_labels, all_probs = [], []
    for (keys, upper, lower, lower_rc, upper_oh, lower_rc_oh, labels) in loader:
        upper     = upper.to(device)
        lower     = lower.to(device)
        lower_rc  = lower_rc.to(device)
        upper_oh  = upper_oh.to(device)
        lower_rc_oh = lower_rc_oh.to(device)

        # BSCANUnified.forward(upper, lower, lower_rc, upper_oh, lower_rc_oh)
        logits = model(upper, lower, lower_rc, upper_oh, lower_rc_oh)
        probs = torch.softmax(logits.float(), dim=-1)[:, 1]
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def evaluate_model(
    model_name: str,
    junction: dict,
    out_dir: Path,
    device: str,
    batch_size: int,
    seeds: list[int],
):
    enc_type = FM_ADAPTER_MODELS[model_name][0]
    fm_emb_dir = Path("external_data/circatlas/exon_controls/fm_embeddings") / enc_type

    if not fm_emb_dir.exists():
        print(f"[error] FM embedding dir not found: {fm_emb_dir}")
        return None

    dataset = CircAtlasFMDataset(junction, fm_emb_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    all_rows = []
    for seed in seeds:
        model = load_model(model_name, seed, device)
        if model is None:
            continue

        labels, probs = run_inference(model, loader, device)
        del model
        torch.cuda.empty_cache()

        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(labels, probs)
        prc = average_precision_score(labels, probs)
        mcc = matthews_corrcoef(labels, preds)
        acc = accuracy_score(labels, preds)

        row = {"model": model_name, "seed": seed, "auc": auc, "prc": prc,
               "mcc": mcc, "acc": acc,
               "pos_score": float(probs[labels == 1].mean()),
               "neg_score": float(probs[labels == 0].mean())}
        all_rows.append(row)
        print(f"  seed={seed}  AUC={auc:.4f}  PRC={prc:.4f}  MCC={mcc:.4f}")

    if not all_rows:
        print(f"  [warn] No results for {model_name}")
        return None

    import csv
    result_path = out_dir / f"{model_name}_external_control_results.csv"
    fieldnames = list(all_rows[0].keys())
    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    aucs = [r["auc"] for r in all_rows]
    prcs = [r["prc"] for r in all_rows]
    mccs = [r["mcc"] for r in all_rows]
    n = len(all_rows)
    summary = {
        "model": model_name,
        "n_seeds": n,
        "auc_mean": statistics.mean(aucs),
        "auc_std": statistics.stdev(aucs) if n > 1 else 0.0,
        "prc_mean": statistics.mean(prcs),
        "prc_std": statistics.stdev(prcs) if n > 1 else 0.0,
        "mcc_mean": statistics.mean(mccs),
        "mcc_std": statistics.stdev(mccs) if n > 1 else 0.0,
    }

    summary_path = out_dir / f"{model_name}_external_control_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"\n  → AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}  "
          f"PRC={summary['prc_mean']:.4f}  MCC={summary['mcc_mean']:.4f}")
    print(f"  Saved: {result_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["bscan_unified_fm_cnnadapter"])
    parser.add_argument("--out_dir", type=Path,
                        default=Path("external_data/circatlas/exon_controls"))
    parser.add_argument("--seq_json", type=Path,
                        default=Path("external_data/circatlas/exon_controls/seq_dict/junction.json"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    junction = json.loads(args.seq_json.read_text())
    print(f"Loaded {len(junction)} external examples | device={args.device}")

    for model_name in args.models:
        if model_name not in FM_ADAPTER_MODELS:
            print(f"[skip] Unknown FM adapter model: {model_name}")
            continue
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        evaluate_model(model_name, junction, args.out_dir, args.device,
                       args.batch_size, args.seeds)


if __name__ == "__main__":
    main()
