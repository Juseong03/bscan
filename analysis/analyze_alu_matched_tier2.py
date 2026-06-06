#!/usr/bin/env python
"""ALU-density-matched Tier 2 analysis.

Standard Tier 2: replaces lower intron with ANY LS junction's intron.
ALU-matched Tier 2: replaces lower intron with an LS junction that has
  SIMILAR ALU coverage (within ±0.05) to the real BS junction's lower intron.

If ALU density drives Tier 2 discrimination, ALU-matched Tier 2 should show
  LOWER AUC than standard Tier 2 (the model can't use ALU to distinguish).
If other features drive Tier 2, AUC should remain similar.
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv, os, random, sys, warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch
import statistics

warnings.filterwarnings("ignore")

from dataloader import DataSetPrep
from evaluate_hard_negative_pairing import (
    seqs_to_onehot, seqs_to_tokens, seqs_to_kmer_index,
    metrics, FM_UNIFIED_MODELS, load_fm_components, extract_fm_embeddings,
    load_bscan_unified
)
from trainer import Trainer
from utils import get_device

L = 100
DEVICE = get_device(0)
SEEDS = [42, 123, 315]
ALU_TOL = 0.05  # ±5% ALU coverage tolerance for matching


def load_alu_coverage() -> dict[str, float]:
    """Load per-junction ALU lower intron coverage from alu_coverage.csv."""
    path = Path("research_results/alu_coverage.csv")
    if not path.exists():
        return {}
    cov = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cov[row["key"]] = float(row["alu_lower"])
    return cov


def build_alu_matched_negatives(data, bs_keys, ls_keys, alu_cov, rng, seed):
    """Build Tier2 negatives where LS donor is matched by ALU coverage."""
    upper, lower, labels = [], [], []

    # Group LS keys by ALU coverage bucket
    ls_by_alu = {}
    for k in ls_keys:
        bucket = round(alu_cov.get(k, 0.0) / ALU_TOL) * ALU_TOL
        ls_by_alu.setdefault(bucket, []).append(k)

    unmatched = 0
    for k in bs_keys:
        value = data.junction_seq[k]
        upper.append(value["upper_seq"])
        lower.append(value["lower_seq"])
        labels.append(1)

    for k in bs_keys:
        value = data.junction_seq[k]
        bs_alu = alu_cov.get(k, 0.0)

        # Find LS donors with similar ALU coverage
        best_donors = []
        for delta in [0, 1, 2, 3]:
            target = round((bs_alu + delta * ALU_TOL) / ALU_TOL) * ALU_TOL
            target_neg = round((bs_alu - delta * ALU_TOL) / ALU_TOL) * ALU_TOL
            for t in set([target, target_neg]):
                best_donors.extend(ls_by_alu.get(t, []))
            if best_donors:
                break

        if best_donors:
            donor_k = rng.choice(best_donors)
        else:
            donor_k = rng.choice(ls_keys)
            unmatched += 1

        donor = data.junction_seq[donor_k]
        neg_upper = value["upper_seq"]
        neg_lower = value["lower_exon"] + donor["lower_intron"]
        upper.append(neg_upper)
        lower.append(neg_lower)
        labels.append(0)

    print(f"  seed={seed}: {unmatched}/{len(bs_keys)} could not be ALU-matched (used random)")
    return upper, lower, labels


def predict_twoinput(trainer, upper_seqs, lower_seqs, labels_list):
    u = seqs_to_onehot(upper_seqs, 2 * L)
    l = seqs_to_onehot(lower_seqs, 2 * L)
    lbl = torch.tensor(labels_list, dtype=torch.long)
    loader = DataLoader(TensorDataset(u, l, lbl), batch_size=256, shuffle=False)
    probs = []
    trainer.model.eval()
    with torch.no_grad():
        for ub, lb, _ in loader:
            logits = trainer.model(ub.to(DEVICE), lb.to(DEVICE))
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.array(probs)


def main():
    alu_cov = load_alu_coverage()
    if not alu_cov:
        print("ERROR: alu_coverage.csv not found. Run analyze_alu_repeats.py first.")
        return

    rows = []

    for seed in SEEDS:
        rng = random.Random(seed)
        data = DataSetPrep(
            "data/BS_LS_coordinates_final.csv",
            "data/hg19_seq_dict.json",
            junction_bps=L, flanking_bps=L, seed=seed,
        )
        data.load_junction_flanking_seq()
        _, _, test_keys = data.split_data_grouped(group_by="transcript")

        bs_keys = [k for k in test_keys if data.junction_seq[k]["label"] == "BS"]
        ls_keys = [k for k in test_keys if data.junction_seq[k]["label"] == "LS"]
        print(f"\nSeed {seed}: BS={len(bs_keys)}, LS={len(ls_keys)}")

        # ── Standard Tier2 (for comparison) ───────────────────────────────
        ls_shuffled = ls_keys.copy(); rng.shuffle(ls_shuffled)
        ls_cycle = [ls_shuffled[i % len(ls_shuffled)] for i in range(len(bs_keys))]
        std_upper, std_lower, std_labels = [], [], []
        for k in bs_keys:
            v = data.junction_seq[k]
            std_upper.append(v["upper_seq"]); std_lower.append(v["lower_seq"]); std_labels.append(1)
        for i, k in enumerate(bs_keys):
            v = data.junction_seq[k]; donor = data.junction_seq[ls_cycle[i]]
            std_upper.append(v["upper_seq"]); std_lower.append(v["lower_exon"] + donor["lower_intron"]); std_labels.append(0)

        # ── ALU-matched Tier2 ─────────────────────────────────────────────
        mat_upper, mat_lower, mat_labels = build_alu_matched_negatives(
            data, bs_keys, ls_keys, alu_cov, rng, seed
        )

        # ── Evaluate models ───────────────────────────────────────────────
        for model_name, (std_u, std_l, std_lab), (mat_u, mat_l, mat_lab) in [
            ("bscan",   (std_upper, std_lower, std_labels), (mat_upper, mat_lower, mat_labels)),
            ("circcnn", (std_upper, std_lower, std_labels), (mat_upper, mat_lower, mat_labels)),
        ]:
            ckpt = Path(f"saved_models/{model_name}/{seed}/model.pth")
            if not ckpt.exists():
                print(f"  [skip] {model_name} seed={seed}")
                continue
            tr = Trainer(seed=seed, device=DEVICE)
            tr.define_model(model_name, junction_bps=L)
            tr.model.load_state_dict(
                torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
            tr.model.eval()

            probs_std = predict_twoinput(tr, std_u, std_l, std_lab)
            probs_mat = predict_twoinput(tr, mat_u, mat_l, mat_lab)

            auc_std = roc_auc_score(std_lab, probs_std)
            auc_mat = roc_auc_score(mat_lab, probs_mat)
            print(f"  {model_name} seed={seed}: std_tier2={auc_std:.4f}  alu_matched={auc_mat:.4f}  Δ={auc_mat-auc_std:+.4f}")

            rows.append({"model": model_name, "seed": seed,
                         "tier2_std": round(auc_std, 4), "tier2_alu_matched": round(auc_mat, 4),
                         "delta": round(auc_mat - auc_std, 4)})
            del tr; torch.cuda.empty_cache()

    # FM models (bscan_unified_fm)
    fm_model, tokenizer = load_fm_components("rnafm", str(DEVICE))
    for seed in SEEDS:
        rng = random.Random(seed)
        data = DataSetPrep("data/BS_LS_coordinates_final.csv", "data/hg19_seq_dict.json",
                           junction_bps=L, flanking_bps=L, seed=seed)
        data.load_junction_flanking_seq()
        _, _, test_keys = data.split_data_grouped(group_by="transcript")
        bs_keys = [k for k in test_keys if data.junction_seq[k]["label"] == "BS"]
        ls_keys = [k for k in test_keys if data.junction_seq[k]["label"] == "LS"]

        ls_shuffled = ls_keys.copy(); rng.shuffle(ls_shuffled)
        ls_cycle = [ls_shuffled[i % len(ls_shuffled)] for i in range(len(bs_keys))]
        std_upper, std_lower, std_labels = [], [], []
        for k in bs_keys:
            v = data.junction_seq[k]
            std_upper.append(v["upper_seq"]); std_lower.append(v["lower_seq"]); std_labels.append(1)
        for i, k in enumerate(bs_keys):
            v = data.junction_seq[k]; donor = data.junction_seq[ls_cycle[i]]
            std_upper.append(v["upper_seq"]); std_lower.append(v["lower_exon"] + donor["lower_intron"]); std_labels.append(0)

        mat_upper, mat_lower, mat_labels = build_alu_matched_negatives(
            data, bs_keys, ls_keys, alu_cov, rng, seed)

        ckpt = Path(f"saved_models/bscan_unified_fm/{seed}/model.pth")
        if not ckpt.exists(): continue
        bscan = load_bscan_unified("bscan_unified_fm", "rnafm", seed, str(DEVICE))
        if bscan is None: continue

        def predict_fm(upper_s, lower_s, labels_l):
            from evaluate_hard_negative_pairing import seqs_to_onehot as s2oh
            NUC_OH = {"A": 0, "G": 1, "C": 2, "T": 3}
            def oh4(seqs, length):
                arr = np.zeros((len(seqs), 4, length), dtype=np.float32)
                for i, seq in enumerate(seqs):
                    for j, b in enumerate(seq[:length]):
                        c = NUC_OH.get(b);
                        if c is not None: arr[i, c, j] = 1.0
                return torch.from_numpy(arr)
            rc_perm = [3, 2, 1, 0]
            probs_all = []
            BS = 64
            for i in range(0, len(upper_s), BS):
                u_s = upper_s[i:i+BS]; l_s = lower_s[i:i+BS]
                u_emb = extract_fm_embeddings(u_s, fm_model, tokenizer, str(DEVICE))
                l_emb = extract_fm_embeddings(l_s, fm_model, tokenizer, str(DEVICE))
                lr_emb = extract_fm_embeddings([s[::-1] for s in l_s], fm_model, tokenizer, str(DEVICE))
                u_oh = oh4([s[:L] for s in u_s], L).to(DEVICE)
                l_oh_int = oh4([s[L:] for s in l_s], L)
                lr_oh = l_oh_int[:, rc_perm, :].flip(-1).to(DEVICE)
                with torch.no_grad():
                    logits = bscan(u_emb.to(DEVICE), l_emb.to(DEVICE), lr_emb.to(DEVICE), u_oh, lr_oh)
                probs_all.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            return np.array(probs_all)

        probs_std = predict_fm(std_upper, std_lower, std_labels)
        probs_mat = predict_fm(mat_upper, mat_lower, mat_labels)
        auc_std = roc_auc_score(std_labels, probs_std)
        auc_mat = roc_auc_score(mat_labels, probs_mat)
        print(f"  bscan_unified_fm seed={seed}: std={auc_std:.4f}  alu_matched={auc_mat:.4f}  Δ={auc_mat-auc_std:+.4f}")
        rows.append({"model": "bscan_unified_fm", "seed": seed,
                     "tier2_std": round(auc_std, 4), "tier2_alu_matched": round(auc_mat, 4),
                     "delta": round(auc_mat - auc_std, 4)})

    # Summary
    print("\n=== SUMMARY ===")
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for model, mrs in sorted(by_model.items()):
        stds = [r["tier2_std"] for r in mrs]
        mats = [r["tier2_alu_matched"] for r in mrs]
        deltas = [r["delta"] for r in mrs]
        print(f"{model}: std={statistics.mean(stds):.4f}  alu_matched={statistics.mean(mats):.4f}  Δ={statistics.mean(deltas):+.4f}")

    out = Path("research_results/alu_matched_tier2.csv")
    os.makedirs("research_results", exist_ok=True)
    with open(out, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
