#!/usr/bin/env python
"""P4: Exon / Intron masking analysis.

For each model and masking condition, run inference on the internal test set
using existing checkpoints (no retraining).

Masking conditions:
  full          - original input (baseline)
  exon_masked   - exon positions replaced with N (→ zero one-hot / mask token)
  intron_masked - intron positions replaced with N
  upper_intron_masked - upper intron only
  lower_intron_masked - lower intron only

Models: BSCAN-FM (rnafm), BSCAN-onehot, BSCAN-base, CircCNN, BSCAN-hnaug
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv, os, sys, warnings
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import statistics

warnings.filterwarnings("ignore")

from dataloader import DataSetPrep, circData_cached_fm
from utils import get_device

DEVICE = get_device(0)
L = 100  # junction_bps
_RC_PERM = [3, 2, 1, 0]
SEEDS = [42, 123, 315]  # 3 seeds for speed


def mask_sequence(seq_tensor: torch.Tensor, region: str, L: int) -> torch.Tensor:
    """Zero-out one-hot for specified region. seq_tensor: [N, 4, 2L]."""
    t = seq_tensor.clone()
    if region == "upper_intron":      # positions 0:L in upper ([:, :, 0:L])
        t[:, :, :L] = 0
    elif region == "upper_exon":      # positions L:2L in upper
        t[:, :, L:] = 0
    elif region == "lower_exon":      # positions 0:L in lower
        t[:, :, :L] = 0
    elif region == "lower_intron":    # positions L:2L in lower
        t[:, :, L:] = 0
    elif region == "exon":            # upper_exon + lower_exon
        t[:, :, L:] = 0   # upper exon (second half of upper)
        t[:, :, :L] = 0   # lower exon (first half of lower) — applied to lower below
    elif region == "intron":          # upper_intron + lower_intron
        t[:, :, :L] = 0   # upper intron
        t[:, :, L:] = 0   # lower intron — applied to lower below
    return t


def mask_upper_lower(upper: torch.Tensor, lower: torch.Tensor,
                     condition: str, L: int):
    """Return masked (upper, lower) pair for a given condition."""
    u, l = upper.clone(), lower.clone()
    if condition == "full":
        pass
    elif condition == "exon_masked":
        u[:, :, L:] = 0   # upper exon = second half of upper seq
        l[:, :, :L] = 0   # lower exon = first half of lower seq
    elif condition == "intron_masked":
        u[:, :, :L] = 0   # upper intron = first half of upper seq
        l[:, :, L:] = 0   # lower intron = second half of lower seq
    elif condition == "upper_intron_masked":
        u[:, :, :L] = 0
    elif condition == "lower_intron_masked":
        l[:, :, L:] = 0
    return u, l


CONDITIONS = ["full", "exon_masked", "intron_masked", "upper_intron_masked", "lower_intron_masked"]


def eval_onehot_model(model, upper_tensor, lower_tensor, label_tensor, condition, batch_size=128):
    u, l = mask_upper_lower(upper_tensor, lower_tensor, condition, L)
    loader = DataLoader(TensorDataset(u, l, label_tensor), batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for ub, lb, lbl in loader:
            logits = model(ub.to(DEVICE), lb.to(DEVICE))
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            labels.extend(lbl.numpy())
    return roc_auc_score(labels, probs)


def eval_fm_model(model, keys, data, label_tensor, condition, enc_type, batch_size=64):
    """For FM model: mask the one-hot stem inputs; FM embeddings are unchanged."""
    _u_oh, _l_oh, _ = data.seq_to_tensor(keys)
    lower_rc_oh = _l_oh[:, _RC_PERM, L:].flip(dims=[2])
    upper_oh = _u_oh[:, :, :L]

    # Apply masking to one-hot (stem branch only)
    u_oh_masked, l_rc_oh_masked = upper_oh.clone(), lower_rc_oh.clone()
    if condition == "exon_masked":
        # Exon in upper = upper_seq[L:], exon in lower = lower_seq[:L]
        # For stem: upper_oh = upper intron, lower_rc_oh = lower intron RC
        # Exon masking doesn't affect stem (which uses introns) — effect via CNN only
        pass  # Can't directly mask exon from cached FM embeddings
    elif condition == "intron_masked":
        u_oh_masked[:] = 0   # zero upper intron one-hot
        l_rc_oh_masked[:] = 0  # zero lower intron RC one-hot

    ds = circData_cached_fm(keys, label_tensor, enc_type,
                             upper_oh=u_oh_masked, lower_rc_oh=l_rc_oh_masked)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            u_emb, l_emb, lr_emb, uo, lro, lbl = batch
            # For FM model: we can zero the FM embedding to simulate full masking
            if condition == "exon_masked":
                # Zero second half of upper embedding (exon region) and first half of lower
                u_emb = u_emb.clone(); u_emb[:, L:, :] = 0
                l_emb = l_emb.clone(); l_emb[:, :L, :] = 0
            elif condition == "intron_masked":
                u_emb = u_emb.clone(); u_emb[:, :L, :] = 0
                l_emb = l_emb.clone(); l_emb[:, L:, :] = 0
            elif condition == "upper_intron_masked":
                u_emb = u_emb.clone(); u_emb[:, :L, :] = 0
                uo = torch.zeros_like(uo)
            elif condition == "lower_intron_masked":
                l_emb = l_emb.clone(); l_emb[:, L:, :] = 0
                lro = torch.zeros_like(lro)

            logits = model(u_emb.to(DEVICE), l_emb.to(DEVICE), lr_emb.to(DEVICE),
                          uo.to(DEVICE), lro.to(DEVICE))
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            labels_list.extend(lbl.numpy())
    return roc_auc_score(labels_list, probs)


def main():
    os.makedirs("research_results", exist_ok=True)
    rows = []

    for seed in SEEDS:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

        data = DataSetPrep(
            "data/BS_LS_coordinates_final.csv",
            "data/hg19_seq_dict.json",
            junction_bps=L, flanking_bps=L, seed=seed
        )
        data.load_junction_flanking_seq()
        _, _, test_keys = data.split_data_grouped(group_by="transcript")
        upper_tensor, lower_tensor, label_tensor = data.seq_to_tensor(test_keys)

        # ── 1. BSCAN-FM (rnafm) ──────────────────────────────────────────
        print("\n[BSCAN-FM]")
        from models.bscan_unified import BSCANUnified
        ckpt = Path(f"saved_models/bscan_unified_fm/{seed}/model.pth")
        if ckpt.exists():
            model = BSCANUnified(encoder_type="rnafm", use_cached=True)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
            model.to(DEVICE).eval()
            for cond in CONDITIONS:
                try:
                    auc = eval_fm_model(model, test_keys, data, label_tensor, cond, "rnafm")
                    print(f"  {cond}: AUC={auc:.4f}")
                    rows.append({"model": "bscan_unified_fm", "seed": seed, "condition": cond, "auc": round(auc, 6)})
                except Exception as e:
                    print(f"  {cond}: ERROR {e}")
            del model; torch.cuda.empty_cache()

        # ── 2. BSCAN-onehot ──────────────────────────────────────────────
        print("\n[BSCAN-onehot]")
        ckpt = Path(f"saved_models/bscan_unified_onehot/{seed}/model.pth")
        if ckpt.exists():
            model = BSCANUnified(encoder_type="onehot", use_cached=False)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
            model.to(DEVICE).eval()
            for cond in CONDITIONS:
                try:
                    if cond == "full":
                        u, l = upper_tensor, lower_tensor
                    else:
                        u, l = mask_upper_lower(upper_tensor, lower_tensor, cond, L)
                    loader = DataLoader(TensorDataset(u, l, label_tensor), batch_size=128, shuffle=False)
                    probs, labels_list = [], []
                    with torch.no_grad():
                        for ub, lb, lbl in loader:
                            from dataloader import DataSetPrep as _
                            # onehot model uses token indices (vocab-indexed) not one-hot
                            break
                    # Use seq_to_tensor which gives one-hot directly
                    auc = eval_onehot_model(model, upper_tensor, lower_tensor, label_tensor, cond)
                    print(f"  {cond}: AUC={auc:.4f}")
                    rows.append({"model": "bscan_unified_onehot", "seed": seed, "condition": cond, "auc": round(auc, 6)})
                except Exception as e:
                    print(f"  {cond}: ERROR {e}")
            del model; torch.cuda.empty_cache()

        # ── 3. BSCAN-base (CircCombine) ───────────────────────────────────
        print("\n[BSCAN-base]")
        ckpt = Path(f"saved_models/bscan/{seed}/model.pth")
        if ckpt.exists():
            from trainer import Trainer
            tr = Trainer(seed=seed, device=DEVICE)
            tr.define_model("bscan", junction_bps=L)
            tr.model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
            tr.model.eval()
            for cond in CONDITIONS:
                try:
                    auc = eval_onehot_model(tr.model, upper_tensor, lower_tensor, label_tensor, cond)
                    print(f"  {cond}: AUC={auc:.4f}")
                    rows.append({"model": "bscan_base", "seed": seed, "condition": cond, "auc": round(auc, 6)})
                except Exception as e:
                    print(f"  {cond}: ERROR {e}")
            del tr; torch.cuda.empty_cache()

        # ── 4. CircCNN ────────────────────────────────────────────────────
        print("\n[CircCNN]")
        ckpt = Path(f"saved_models/circcnn/{seed}/model.pth")
        if ckpt.exists():
            tr = Trainer(seed=seed, device=DEVICE)
            tr.define_model("circcnn", junction_bps=L)
            tr.model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
            tr.model.eval()
            for cond in CONDITIONS:
                try:
                    auc = eval_onehot_model(tr.model, upper_tensor, lower_tensor, label_tensor, cond)
                    print(f"  {cond}: AUC={auc:.4f}")
                    rows.append({"model": "circcnn", "seed": seed, "condition": cond, "auc": round(auc, 6)})
                except Exception as e:
                    print(f"  {cond}: ERROR {e}")
            del tr; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Model':<25} {'Condition':<22} {'AUC (mean±std)':>20}")
    print("=" * 65)
    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        summary[(r["model"], r["condition"])].append(r["auc"])
    for model in ["bscan_unified_fm", "bscan_unified_onehot", "bscan_base", "circcnn"]:
        for cond in CONDITIONS:
            aucs = summary[(model, cond)]
            if aucs:
                mean = statistics.mean(aucs)
                std = statistics.stdev(aucs) if len(aucs) > 1 else 0
                marker = " ← FULL" if cond == "full" else ""
                print(f"{model:<25} {cond:<22} {mean:.4f} ± {std:.4f}{marker}")
        print()

    out = Path("research_results/masking_analysis.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "condition", "auc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
