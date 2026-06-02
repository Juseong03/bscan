#!/usr/bin/env python
"""Train BSCANUnified (FM-based) with hard negative augmentation.

Strategy:
  - Positive (BS) + standard negative (LS): use pre-cached FM embeddings (disk)
  - Augmented hard negative (BS-exon + swapped intron, label=0):
      FM embeddings computed once before training, kept in memory

Early stopping: 0.5 × valid_AUC + 0.5 × valid_HN_AUC  (same as CNN hnaug)
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from dataloader import DataSetPrep, circData_cached_fm
from evaluate_hard_negative_pairing import (
    build_hard_negative_pairs, extract_fm_embeddings, load_fm_components,
)
from models.bscan_unified import BSCANUnified
from utils import get_device, seed_everything


# ── Custom mixed dataset ───────────────────────────────────────────────────────

def _fix_len(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Crop or zero-pad a [seq_len, dim] tensor to target_len on dim 0."""
    if t.shape[0] >= target_len:
        return t[:target_len]
    pad = torch.zeros(target_len - t.shape[0], t.shape[1])
    return torch.cat([t, pad], dim=0)


class MixedFMDataset(Dataset):
    """Combines cached-FM standard samples with in-memory augmented negatives."""

    def __init__(
        self,
        cached_ds: circData_cached_fm,         # standard BS+LS samples
        aug_u_emb: torch.Tensor,               # [N_aug, 2L, dim]
        aug_l_emb: torch.Tensor,               # [N_aug, 2L, dim]
        aug_l_rc_emb: torch.Tensor,            # [N_aug, 2L, dim] (lower RC)
        aug_upper_oh: torch.Tensor,            # [N_aug, 4, L]
        aug_lower_rc_oh: torch.Tensor,         # [N_aug, 4, L]
        seq_len: int = 200,
    ):
        self.cached = cached_ds
        self.n_cached = len(cached_ds)
        self.aug_u_emb    = aug_u_emb
        self.aug_l_emb    = aug_l_emb
        self.aug_l_rc_emb = aug_l_rc_emb
        self.aug_upper_oh    = aug_upper_oh
        self.aug_lower_rc_oh = aug_lower_rc_oh
        self.aug_labels = torch.zeros(len(aug_u_emb), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return self.n_cached + len(self.aug_u_emb)

    def __getitem__(self, idx):
        if idx < self.n_cached:
            u_emb, l_emb, l_rc_emb, u_oh, l_rc_oh, label = self.cached[idx]
            # Ensure consistent length (cached embeddings can vary)
            u_emb    = _fix_len(u_emb,    self.seq_len)
            l_emb    = _fix_len(l_emb,    self.seq_len)
            l_rc_emb = _fix_len(l_rc_emb, self.seq_len)
            return u_emb, l_emb, l_rc_emb, u_oh, l_rc_oh, label
        i = idx - self.n_cached
        return (
            self.aug_u_emb[i],
            self.aug_l_emb[i],
            self.aug_l_rc_emb[i],
            self.aug_upper_oh[i],
            self.aug_lower_rc_oh[i],
            self.aug_labels[i],
        )


# ── FM embedding helpers ───────────────────────────────────────────────────────

def embed_seqs(seqs: list[str], fm_model, tokenizer, device: str,
               L: int, batch_size: int = 32) -> torch.Tensor:
    """Return [N, 2L, dim] FM embeddings, padded/cropped to 2L."""
    all_emb = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        emb = extract_fm_embeddings(batch, fm_model, tokenizer, device)  # [B, seq_len, dim]
        emb = emb[:, :2 * L, :]
        if emb.shape[1] < 2 * L:
            pad = torch.zeros(emb.shape[0], 2 * L - emb.shape[1], emb.shape[2], device=emb.device)
            emb = torch.cat([emb, pad], dim=1)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)   # [N, 2L, dim]


def seqs_to_oh(seqs: list[str], length: int) -> torch.Tensor:
    mapping = {"A": 0, "G": 1, "C": 2, "T": 3}
    arr = np.zeros((len(seqs), 4, length), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, b in enumerate(seq[:length].upper()):
            ch = mapping.get(b)
            if ch is not None:
                arr[i, ch, j] = 1.0
    return torch.from_numpy(arr)


def rc_oh(oh: torch.Tensor) -> torch.Tensor:
    """Reverse complement one-hot [N, 4, L]."""
    return torch.stack([oh[:, 3], oh[:, 2], oh[:, 1], oh[:, 0]], dim=1).flip(-1)


# ── Build augmented FM tensors ─────────────────────────────────────────────────

def build_fm_augmented_tensors(
    data: DataSetPrep, keys: list[str], seed: int, L: int,
    fm_model, tokenizer, device: str, batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (u_emb, l_emb, l_rc_emb, upper_oh, lower_rc_oh) for augmented negatives."""
    pair = build_hard_negative_pairs(data, list(keys), seed, None, "lower_intron")
    n = len(pair.label) // 2
    aug_upper = pair.upper[n:]   # BS upper seqs (same as real)
    aug_lower = pair.lower[n:]   # BS lower_exon + DIFFERENT lower_intron

    print(f"  Computing FM embeddings for {len(aug_upper)} augmented negatives...")
    u_emb    = embed_seqs(aug_upper, fm_model, tokenizer, device, L, batch_size)
    l_emb    = embed_seqs(aug_lower, fm_model, tokenizer, device, L, batch_size)

    # RC of lower for lower_rc branch
    aug_lower_intron_seqs = [s[L:] for s in aug_lower]   # just the intron portion
    lower_rc_seqs = [s[::-1].translate(str.maketrans("ACGT","TGCA")) for s in aug_lower_intron_seqs]
    l_rc_emb = embed_seqs(lower_rc_seqs, fm_model, tokenizer, device, L, batch_size)

    # One-hot: upper intron [N,4,L] and lower intron RC [N,4,L]
    aug_upper_intron = [s[:L] for s in aug_upper]
    aug_lower_intron = [s[L:] for s in aug_lower]
    upper_oh_t   = seqs_to_oh(aug_upper_intron, L)
    lower_int_oh = seqs_to_oh(aug_lower_intron, L)
    lower_rc_oh_t = rc_oh(lower_int_oh)

    return u_emb, l_emb, l_rc_emb, upper_oh_t, lower_rc_oh_t


# ── Evaluation helpers ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model: str
    seed: int
    epoch: int
    split: str
    auc: float
    prc: float
    acc: float
    f1: float
    mcc: float
    real_bs_score: float | None = None
    mismatch_score: float | None = None
    score_gap: float | None = None


def run_model(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            u_emb, l_emb, l_rc_emb, u_oh, l_rc_oh, lbl = [b.to(device) for b in batch]
            logits = model(u_emb, l_emb, l_rc_emb, u_oh, l_rc_oh)
            all_probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
            all_labels.append(lbl.cpu())
    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()


def score(labels, probs) -> dict:
    pred = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "prc": average_precision_score(labels, probs),
        "acc": accuracy_score(labels, pred),
        "f1":  f1_score(labels, pred, zero_division=0),
        "mcc": matthews_corrcoef(labels, pred),
    }


def eval_standard(model, dataset, batch_size, device) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels, probs = run_model(model, loader, device)
    return score(labels, probs)


def eval_hard_neg(model, data, keys, seed, L, fm_model, tokenizer, device, batch_size,
                  negative_mode="lower_intron") -> dict:
    from evaluate_hard_negative_pairing import build_hard_negative_pairs
    pair = build_hard_negative_pairs(data, list(keys), seed, None, negative_mode)
    n = len(pair.label) // 2
    upper_seqs = pair.upper
    lower_seqs = pair.lower
    labels_np = np.array(pair.label)

    u_emb  = embed_seqs(upper_seqs, fm_model, tokenizer, device, L, batch_size)
    l_emb  = embed_seqs(lower_seqs, fm_model, tokenizer, device, L, batch_size)

    lower_intron_seqs = [s[L:] for s in lower_seqs]
    lower_rc_seqs = [s[::-1].translate(str.maketrans("ACGT","TGCA")) for s in lower_intron_seqs]
    l_rc_emb = embed_seqs(lower_rc_seqs, fm_model, tokenizer, device, L, batch_size)

    upper_oh_t  = seqs_to_oh([s[:L] for s in upper_seqs], L)
    lower_int   = seqs_to_oh(lower_intron_seqs, L)
    lower_rc_oh_t = rc_oh(lower_int)

    loader = DataLoader(
        torch.utils.data.TensorDataset(u_emb, l_emb, l_rc_emb, upper_oh_t, lower_rc_oh_t,
                                        torch.tensor(labels_np, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    labels_out, probs = run_model(model, loader, device)
    metrics = score(labels_out, probs)
    metrics["real_bs_score"]   = float(probs[labels_out == 1].mean())
    metrics["mismatch_score"]  = float(probs[labels_out == 0].mean())
    metrics["score_gap"]       = metrics["real_bs_score"] - metrics["mismatch_score"]
    return metrics


# ── Main training ──────────────────────────────────────────────────────────────

def train_one(args, enc_type: str, seed: int, fm_model, tokenizer) -> list[EvalResult]:
    seed_everything(seed)
    device = get_device(args.device)
    L = args.junction_bps

    data = DataSetPrep(
        coord_path="./data/BS_LS_coordinates_final.csv",
        seq_dict_path="./data/hg19_seq_dict.json",
        junction_bps=L, flanking_bps=L, seed=seed,
    )
    data.load_junction_flanking_seq()
    keys_train, keys_valid, keys_test = data.split_data()

    # Standard one-hot for stem branch
    u_oh_tr, l_oh_tr, labels_tr = data.seq_to_tensor(keys_train)
    u_oh_va, l_oh_va, labels_va = data.seq_to_tensor(keys_valid)
    u_oh_te, l_oh_te, labels_te = data.seq_to_tensor(keys_test)
    _RC = [3, 2, 1, 0]
    get_lrc = lambda l_oh: l_oh[:, :, L:][:, _RC, :].flip(dims=[2])

    # Standard cached datasets
    std_train = circData_cached_fm(keys_train, labels_tr, enc_type,
                                   upper_oh=u_oh_tr[:, :, :L], lower_rc_oh=get_lrc(l_oh_tr))
    std_valid = circData_cached_fm(keys_valid, labels_va, enc_type,
                                   upper_oh=u_oh_va[:, :, :L], lower_rc_oh=get_lrc(l_oh_va))
    std_test  = circData_cached_fm(keys_test,  labels_te, enc_type,
                                   upper_oh=u_oh_te[:, :, :L], lower_rc_oh=get_lrc(l_oh_te))

    # Pre-compute ALL augmented FM tensors once (train + val hard neg for all modes)
    print(f"[seed={seed}] Pre-computing augmented FM tensors (train)...")
    au_emb, al_emb, alrc_emb, aoh, alrc_oh = build_fm_augmented_tensors(
        data, keys_train, seed, L, fm_model, tokenizer, device, args.batch_size
    )
    train_ds = MixedFMDataset(std_train, au_emb, al_emb, alrc_emb, aoh, alrc_oh, seq_len=2*L)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Pre-compute validation/test hard neg tensors for fast per-epoch eval
    print(f"[seed={seed}] Pre-computing hard neg FM tensors (valid/test)...")
    precomp: dict[str, dict] = {}   # mode → {split → (loader, labels)}
    for neg_mode in ["lower_intron"] + args.eval_negative_modes:
        neg_mode = neg_mode  # deduplicate below
        if neg_mode in precomp:
            continue
        precomp[neg_mode] = {}
        for split_name, split_keys in [("valid", list(keys_valid)), ("test", list(keys_test))]:
            pair = build_hard_negative_pairs(data, split_keys, seed, None, neg_mode)
            upper_seqs, lower_seqs = pair.upper, pair.lower
            labels_np = np.array(pair.label)
            u_e  = embed_seqs(upper_seqs, fm_model, tokenizer, device, L, args.batch_size)
            l_e  = embed_seqs(lower_seqs, fm_model, tokenizer, device, L, args.batch_size)
            li_seqs = [s[L:] for s in lower_seqs]
            lrc_seqs = [s[::-1].translate(str.maketrans("ACGT","TGCA")) for s in li_seqs]
            l_rc_e = embed_seqs(lrc_seqs, fm_model, tokenizer, device, L, args.batch_size)
            u_oh_t  = seqs_to_oh([s[:L] for s in upper_seqs], L)
            l_oh_t  = seqs_to_oh(li_seqs, L)
            l_rc_oh_t = rc_oh(l_oh_t)
            lbl_t = torch.tensor(labels_np, dtype=torch.long)
            ds = torch.utils.data.TensorDataset(u_e, l_e, l_rc_e, u_oh_t, l_rc_oh_t, lbl_t)
            precomp[neg_mode][split_name] = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        print(f"  {neg_mode}: done")

    # Model (fresh init of non-FM layers)
    model_name = f"bscan_unified_{enc_type.replace('rna','')}_hnaug"
    model = BSCANUnified(encoder_type=enc_type, use_cached=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    save_dir = os.path.join(args.save_dir, model_name, str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.pth")

    best_score = -np.inf
    best_epoch = -1
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            u_emb, l_emb, l_rc_emb, u_oh_b, l_rc_oh_b, lbl = [b.to(device) for b in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(u_emb, l_emb, l_rc_emb, u_oh_b, l_rc_oh_b)
            loss = loss_fn(logits, lbl.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        valid_std = eval_standard(model, std_valid, args.batch_size, device)
        # Use pre-computed validation hard neg (fast, no FM inference per epoch)
        hn_labels, hn_probs = run_model(model, precomp["lower_intron"]["valid"], device)
        valid_hn_auc = roc_auc_score(hn_labels, hn_probs)
        sel = 0.5 * valid_std["auc"] + 0.5 * valid_hn_auc
        print(
            f"{model_name} seed={seed} ep={epoch}: "
            f"loss={np.mean(losses):.4f} std={valid_std['auc']:.4f} "
            f"hn={valid_hn_auc:.4f} sel={sel:.4f}"
        )

        if sel > best_score:
            best_score, best_epoch, patience = sel, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience > args.earlystop:
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    rows: list[EvalResult] = []
    # Standard eval
    for split_name, ds in [("valid_standard", std_valid), ("test_standard", std_test)]:
        s = eval_standard(model, ds, args.batch_size, device)
        rows.append(EvalResult(model=model_name, seed=seed, epoch=best_epoch, split=split_name, **s))
    # Hard neg eval using pre-computed loaders
    for neg_mode in args.eval_negative_modes:
        for split_name in ["valid", "test"]:
            if neg_mode not in precomp or split_name not in precomp[neg_mode]:
                continue
            loader = precomp[neg_mode][split_name]
            lbl_np, probs = run_model(model, loader, device)
            s = score(lbl_np, probs)
            s["real_bs_score"]  = float(probs[lbl_np == 1].mean())
            s["mismatch_score"] = float(probs[lbl_np == 0].mean())
            s["score_gap"]      = s["real_bs_score"] - s["mismatch_score"]
            rows.append(EvalResult(
                model=model_name, seed=seed, epoch=best_epoch,
                split=f"{split_name}_{neg_mode}_hardneg", **s,
            ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-type", default="rnafm",
                        choices=["rnafm", "rnaernie", "rnabert", "rnamsm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 315])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--earlystop", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--junction-bps", type=int, default=100)
    parser.add_argument(
        "--eval-negative-modes", nargs="+",
        default=["lower_intron", "ls_lower_intron"],
        choices=["lower_intron", "upper_intron", "both_introns",
                 "full_lower", "ls_lower_intron", "ls_upper_intron", "ls_both_introns"],
    )
    parser.add_argument("--save-dir", default="saved_models_hnaug")
    parser.add_argument("--out-dir", default="research_results/hard_negative_augmented")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)

    print(f"Loading FM: {args.enc_type}...")
    fm_model, tokenizer = load_fm_components(args.enc_type, device)

    rows: list[EvalResult] = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"=== {args.enc_type} hnaug  seed={seed} ===")
        rows.extend(train_one(args, args.enc_type, seed, fm_model, tokenizer))

    df = pd.DataFrame([r.__dict__ for r in rows])
    tag = args.enc_type.replace("rna", "")
    result_path = os.path.join(args.out_dir, f"hnaug_fm_{tag}_results.csv")
    summary_path = os.path.join(args.out_dir, f"hnaug_fm_{tag}_summary.csv")
    df.to_csv(result_path, index=False)

    summary = df.groupby(["model", "split"]).agg(["mean", "std"])
    summary.to_csv(summary_path)
    print("\n=== Summary ===")
    key_splits = ["test_standard", "test_lower_intron_hardneg", "test_ls_lower_intron_hardneg"]
    key_df = df[df["split"].isin(key_splits)].groupby(["model","split"])[["auc","prc","mcc"]].agg(["mean","std"])
    print(key_df.round(4).to_string())
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
