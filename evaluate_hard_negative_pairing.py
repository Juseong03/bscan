#!/usr/bin/env python
"""
Hard-negative pairing evaluation for BS/LS models.

This task asks whether a trained classifier can distinguish real BS examples
from synthetic hard negatives made by mismatching specific BS sequence regions.
The default mode changes only the lower intron while keeping both exons fixed,
which targets intronic pairing more directly than replacing a full lower side.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from dataloader import DataSetPrep
from trainer import Trainer


TOKEN_IDS = {"A": 6, "C": 7, "G": 8, "T": 9, "N": 0}
TWO_INPUT_MODELS = {
    "bscan_seq_lite",
    "bscan_seq_lite_xattn",
    "bscan_seq",
    "bscan_seq_rcattn",
    "bscan_seq_rcaug",
    "bscan_seq_mamba_aux",
    "bscan_plus",
    "bscan_region_stem",
    "bscan",
    "circcnn",
    "circcnndouble",
    "circcnndoubleshare",
    "circdc",
    "jedi",
}
SINGLE_INPUT_MODELS = {"circcnnsingle", "deepcirccode"}
RCM_INPUT_MODELS = {"circcnntri"}
TOKEN_MODELS = {
    "bscan_seq_lite",
    "bscan_seq_lite_xattn",
    "bscan_seq",
    "bscan_seq_rcattn",
    "bscan_seq_rcaug",
    "bscan_seq_mamba_aux",
    "bscan_plus",
}
KMER_MODELS = {"jedi"}
RCM_KMERS = [5, 7, 9, 11, 13]

# FM-based unified models: model_name → encoder_type
FM_UNIFIED_MODELS = {
    "bscan_unified_bert":  "rnabert",
    "bscan_unified_fm":    "rnafm",
    "bscan_unified_ernie": "rnaernie",
    "bscan_unified_msm":   "rnamsm",
    # adapter ablation variants
    "bscan_unified_fm_cnnadapter":   "rnafm",
    "bscan_unified_fm_mambaadapter": "rnafm",
}

# adapter kwargs per model name (for load_bscan_unified)
_FM_ADAPTER_KWARGS: dict[str, dict] = {
    "bscan_unified_fm_cnnadapter":   {"adapter_type": "cnn",   "adapter_layers": 2},
    "bscan_unified_fm_mambaadapter": {"adapter_type": "mamba", "adapter_layers": 1},
}


@dataclass
class PairBatch:
    upper: list[str]
    lower: list[str]
    label: list[int]


def seqs_to_tokens(seqs: list[str], expected_len: int) -> torch.Tensor:
    arr = np.zeros((len(seqs), expected_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        clipped = seq[:expected_len]
        arr[i, : len(clipped)] = [TOKEN_IDS.get(base, 0) for base in clipped]
    return torch.from_numpy(arr)


def seqs_to_onehot(seqs: list[str], expected_len: int) -> torch.Tensor:
    # Match DataSetPrep.seq_to_matrix channel order used during training.
    mapping = {"A": 0, "G": 1, "C": 2, "T": 3}
    arr = np.zeros((len(seqs), 4, expected_len), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq[:expected_len]):
            channel = mapping.get(base)
            if channel is not None:
                arr[i, channel, j] = 1.0
    return torch.from_numpy(arr)


def seqs_to_kmer_index(seqs: list[str], kmer: int = 3) -> torch.Tensor:
    """Match DataSetPrep.seq_to_index() for JEDI-style k-mer inputs."""
    nucleotides = ["A", "C", "G", "T"]
    index_mapping = {"".join(chars): idx for idx, chars in enumerate(itertools.product(nucleotides, repeat=kmer))}

    rows = []
    for seq in seqs:
        rows.append([index_mapping.get(seq[i : i + kmer], 0) for i in range(len(seq) - kmer + 1)])
    return torch.tensor(rows, dtype=torch.long)


def build_hard_negative_pairs(
    data: DataSetPrep,
    keys: list[str],
    seed: int,
    max_samples: int | None,
    negative_mode: str,
) -> PairBatch:
    bs_keys = [key for key in keys if data.junction_seq[key]["label"] == "BS"]
    ls_keys = [key for key in keys if data.junction_seq[key]["label"] == "LS"]
    if max_samples is not None:
        bs_keys = bs_keys[:max_samples]
    if len(bs_keys) < 2:
        raise ValueError("Need at least two BS test examples to build hard negatives.")

    rng = random.Random(seed)

    # BS-intron donor pool (for BS-swap modes)
    donor_keys = bs_keys.copy()
    rng.shuffle(donor_keys)
    if any(a == b for a, b in zip(bs_keys, donor_keys)):
        donor_keys = donor_keys[1:] + donor_keys[:1]

    # LS-intron donor pool (for ls_lower_intron / ls_upper_intron modes)
    ls_donor_keys: list[str] = []
    if negative_mode in ("ls_lower_intron", "ls_upper_intron", "ls_both_introns"):
        if len(ls_keys) < 1:
            raise ValueError("No LS keys in test split for ls_* negative modes.")
        ls_shuffled = ls_keys.copy()
        rng.shuffle(ls_shuffled)
        # Cycle LS donors to match bs_keys length
        ls_donor_keys = [ls_shuffled[i % len(ls_shuffled)] for i in range(len(bs_keys))]

    upper: list[str] = []
    lower: list[str] = []
    labels: list[int] = []

    for key in bs_keys:
        value = data.junction_seq[key]
        upper.append(value["upper_seq"])
        lower.append(value["lower_seq"])
        labels.append(1)

    for i, key in enumerate(bs_keys):
        value = data.junction_seq[key]
        donor = data.junction_seq[donor_keys[i]]

        if negative_mode == "full_lower":
            neg_upper = value["upper_seq"]
            neg_lower = donor["lower_seq"]
        elif negative_mode == "lower_intron":
            neg_upper = value["upper_seq"]
            neg_lower = value["lower_exon"] + donor["lower_intron"]
        elif negative_mode == "upper_intron":
            neg_upper = donor["upper_intron"] + value["upper_exon"]
            neg_lower = value["lower_seq"]
        elif negative_mode == "both_introns":
            neg_upper = donor["upper_intron"] + value["upper_exon"]
            neg_lower = value["lower_exon"] + donor["lower_intron"]
        # ── LS-intron modes (intermediate difficulty) ──────────────────────
        # Donor is an LS junction; only the intron portion is replaced.
        # Exon sequences are preserved from the real BS junction.
        # Tests: can the model distinguish BS-type (ALU-rich) from LS-type introns?
        elif negative_mode == "ls_lower_intron":
            ls_donor = data.junction_seq[ls_donor_keys[i]]
            neg_upper = value["upper_seq"]
            neg_lower = value["lower_exon"] + ls_donor["lower_intron"]
        elif negative_mode == "ls_upper_intron":
            ls_donor = data.junction_seq[ls_donor_keys[i]]
            neg_upper = ls_donor["upper_intron"] + value["upper_exon"]
            neg_lower = value["lower_seq"]
        elif negative_mode == "ls_both_introns":
            ls_donor = data.junction_seq[ls_donor_keys[i]]
            neg_upper = ls_donor["upper_intron"] + value["upper_exon"]
            neg_lower = value["lower_exon"] + ls_donor["lower_intron"]
        else:
            raise ValueError(f"Unknown negative_mode: {negative_mode}")
        upper.append(neg_upper)
        lower.append(neg_lower)
        labels.append(0)

    return PairBatch(upper=upper, lower=lower, label=labels)


def model_kwargs(model_name: str, junction_bps: int) -> dict:
    if model_name in TOKEN_MODELS:
        return {"junction_bps": junction_bps, "length_seq": 2 * junction_bps}
    if model_name in {"circcnndouble", "circcnndoubleshare"}:
        return {"length_seq": 2 * junction_bps}
    if model_name == "circcnntri":
        return {"length_seq": 2 * junction_bps, "n_rcm_features": 5}
    if model_name in {"circcnn", "bscan", "bscan_region_stem"}:
        return {"junction_bps": junction_bps}
    return {}


# ── FM-unified model helpers ───────────────────────────────────────────────────

def load_fm_components(enc_type: str, device: str):
    """Load (fm_model, tokenizer) for a given encoder_type. FM weights are frozen."""
    from multimolecule import RnaTokenizer, RnaErnieModel, RnaBertModel, RnaFmModel, RnaMsmModel
    _fm_cls = {"rnaernie": RnaErnieModel, "rnabert": RnaBertModel,
               "rnafm": RnaFmModel, "rnamsm": RnaMsmModel}
    _tok_name = {"rnaernie": "multimolecule/rnaernie", "rnabert": "multimolecule/rnabert",
                 "rnafm": "multimolecule/rnafm", "rnamsm": "multimolecule/rnamsm"}
    tok = RnaTokenizer.from_pretrained(_tok_name[enc_type])
    fm = _fm_cls[enc_type].from_pretrained(_tok_name[enc_type]).to(device).eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    return fm, tok


def extract_fm_embeddings(seqs: list[str], fm_model, tokenizer, device: str) -> torch.Tensor:
    """Tokenize sequences → run FM → return hidden states [B, L, dim] (CLS/SEP cropped)."""
    enc = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = fm_model(input_ids=input_ids)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    # Crop CLS (pos 0) and SEP (last pos); result length = input_len - 2
    return hidden[:, 1:-1, :]   # [B, seq_len, dim]


def load_bscan_unified(model_name: str, enc_type: str, seed: int, device: str):
    """Load BSCANUnified with use_cached=True (no internal FM); load checkpoint weights."""
    from models.bscan_unified import BSCANUnified
    ckpt = os.path.join("saved_models", model_name, str(seed), "model.pth")
    if not os.path.exists(ckpt):
        print(f"[skip] Missing checkpoint: {ckpt}")
        return None
    adapter_kwargs = _FM_ADAPTER_KWARGS.get(model_name, {})
    model = BSCANUnified(encoder_type=enc_type, use_cached=True, **adapter_kwargs)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def predict_fm(
    bscan_model,
    fm_model,
    tokenizer,
    pair_batch: PairBatch,
    batch_size: int,
    junction_bps: int,
    device: str,
) -> np.ndarray:
    """Run inference for a BSCANUnified FM model on hard-negative pair batch."""
    NUC_OH = {"A": 0, "G": 1, "C": 2, "T": 3}

    def _rc_oh(oh):
        """Reverse complement a one-hot tensor [B, 4, L]."""
        rc = oh.flip(-1)  # reverse
        # swap A↔T (ch 0,3) and C↔G (ch 2,1)
        return torch.stack([rc[:, 3], rc[:, 2], rc[:, 1], rc[:, 0]], dim=1)

    def seqs_to_oh(seqs, length):
        arr = np.zeros((len(seqs), 4, length), dtype=np.float32)
        for i, seq in enumerate(seqs):
            for j, b in enumerate(seq[:length].upper()):
                ch = NUC_OH.get(b)
                if ch is not None:
                    arr[i, ch, j] = 1.0
        return torch.from_numpy(arr)

    L = junction_bps
    n = len(pair_batch.label)
    probs_all = []

    for start in range(0, n, batch_size):
        upper_seqs = pair_batch.upper[start:start + batch_size]
        lower_seqs = pair_batch.lower[start:start + batch_size]

        # FM embeddings [B, 2L, dim]
        u_emb = extract_fm_embeddings(upper_seqs, fm_model, tokenizer, device)
        l_emb = extract_fm_embeddings(lower_seqs, fm_model, tokenizer, device)
        # Pad/crop to exactly 2L
        u_emb = u_emb[:, :2*L, :]
        l_emb = l_emb[:, :2*L, :]

        # One-hot for stem branch: upper_intron [B, 4, L] and lower_rc_intron [B, 4, L]
        upper_oh = seqs_to_oh([s[:L] for s in upper_seqs], L).to(device)
        lower_oh_int = seqs_to_oh([s[L:] for s in lower_seqs], L).to(device)
        lower_rc_oh = _rc_oh(lower_oh_int)

        with torch.no_grad():
            logits = bscan_model(u_emb, l_emb, upper_oh=upper_oh, lower_rc_oh=lower_rc_oh)
        probs_all.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())

    return torch.cat(probs_all).numpy()


# ── Duplex energy helpers ──────────────────────────────────────────────────────

def compute_duplex_energies(pair_batch: PairBatch, junction_bps: int) -> np.ndarray:
    """Compute ViennaRNA duplexfold energy for each (upper_intron, lower_intron) pair."""
    import RNA
    energies = []
    for upper_seq, lower_seq in zip(pair_batch.upper, pair_batch.lower):
        upper_intron = upper_seq[:junction_bps]
        lower_intron = lower_seq[junction_bps:]   # lower = lower_exon[L:] + lower_intron[:L]
        duplex = RNA.duplexfold(upper_intron, lower_intron)
        energies.append(duplex.energy)
    return np.array(energies, dtype=np.float64)


def combine_with_duplex(probs: np.ndarray, energies: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Combine model probabilities with duplex energy via logit-space addition."""
    # More negative energy = stronger pairing → higher score; zscore normalise
    scores = -energies
    mu, sigma = scores.mean(), scores.std()
    duplex_z = (scores - mu) / (sigma + 1e-9)
    base_logits = logit(np.clip(probs, 1e-6, 1 - 1e-6))
    return expit(base_logits + alpha * duplex_z)


def load_model(model_name: str, seed: int, device: str, junction_bps: int) -> Trainer | None:
    ckpt = os.path.join("saved_models", model_name, str(seed), "model.pth")
    if not os.path.exists(ckpt):
        print(f"[skip] Missing checkpoint: {ckpt}")
        return None

    trainer = Trainer(seed=seed, device=device)
    trainer.define_model(model_name, **model_kwargs(model_name, junction_bps))
    state = torch.load(ckpt, map_location=device, weights_only=True)
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.eval()
    return trainer


def predict(
    trainer: Trainer,
    pair_batch: PairBatch,
    batch_size: int,
    junction_bps: int,
    device: str,
) -> np.ndarray:
    expected_len = 2 * junction_bps
    if trainer.model_name in TOKEN_MODELS:
        upper = seqs_to_tokens(pair_batch.upper, expected_len)
        lower = seqs_to_tokens(pair_batch.lower, expected_len)
    elif trainer.model_name in KMER_MODELS:
        upper = seqs_to_kmer_index(pair_batch.upper)
        lower = seqs_to_kmer_index(pair_batch.lower)
    else:
        upper = seqs_to_onehot(pair_batch.upper, expected_len)
        lower = seqs_to_onehot(pair_batch.lower, expected_len)

    labels = torch.tensor(pair_batch.label, dtype=torch.long)
    loader = DataLoader(TensorDataset(upper, lower, labels), batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for upper_batch, lower_batch, _ in loader:
            logits = trainer.model(upper_batch.to(device), lower_batch.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())
    return torch.cat(probs).numpy()


def predict_single(
    trainer: Trainer,
    pair_batch: PairBatch,
    batch_size: int,
    junction_bps: int,
    device: str,
) -> np.ndarray:
    """Concatenate upper+lower one-hots into [B, 4, 4*L] for single-input models."""
    expected_len = 2 * junction_bps
    upper = seqs_to_onehot(pair_batch.upper, expected_len)   # [B, 4, 2L]
    lower = seqs_to_onehot(pair_batch.lower, expected_len)   # [B, 4, 2L]
    x = torch.cat([upper, lower], dim=2)                      # [B, 4, 4L]
    labels = torch.tensor(pair_batch.label, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, labels), batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = trainer.model(x_batch.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())
    return torch.cat(probs).numpy()


def compute_rcm_batch(
    pair_batch: PairBatch, junction_bps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute RCM feature arrays for all pairs. Returns three [N, 5, 25] float32 arrays."""
    from RCSFinder import RCSFinder

    def _rcm_entry(upper_intron: str, lower_intron: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entry: dict = {"flanking": {}, "upper": {}, "lower": {}}
        for k in RCM_KMERS:
            for rcm_type, is_flanking, is_upper in [
                ("flanking", True,  False),
                ("upper",    False, True),
                ("lower",    False, False),
            ]:
                _, dist = RCSFinder(
                    key="pair",
                    upper_seq=upper_intron,
                    lower_seq=lower_intron,
                    is_flanking_introns=is_flanking,
                    kmers=k,
                    is_upper_intron=is_upper,
                    seq_fraction_of_spacer=0,
                    allowed_seed_mismatch=0,
                ).subseq_validity_check()
                entry[rcm_type][str(k)] = [int(v) for v in dist]

        def build(rcm_type: str) -> np.ndarray:
            mats = [np.log(np.array(entry[rcm_type][str(k)]).reshape(5, 5) + 1) for k in RCM_KMERS]
            return np.concatenate(mats, axis=1).astype(np.float32)  # [5, 25]

        return build("flanking"), build("upper"), build("lower")

    flanking_list, upper_list, lower_list = [], [], []
    for upper_seq, lower_seq in zip(pair_batch.upper, pair_batch.lower):
        upper_intron = upper_seq[:junction_bps]
        lower_intron = lower_seq[junction_bps:]
        f, u, l = _rcm_entry(upper_intron, lower_intron)
        flanking_list.append(f)
        upper_list.append(u)
        lower_list.append(l)

    return (
        np.stack(flanking_list, axis=0),   # [N, 5, 25]
        np.stack(upper_list, axis=0),
        np.stack(lower_list, axis=0),
    )


def predict_rcm(
    trainer: Trainer,
    pair_batch: PairBatch,
    batch_size: int,
    junction_bps: int,
    device: str,
) -> np.ndarray:
    """Run inference for CircCNNtri which takes (upper, lower, rcm_flanking, rcm_upper, rcm_lower)."""
    expected_len = 2 * junction_bps
    upper_oh = seqs_to_onehot(pair_batch.upper, expected_len)   # [N, 4, 2L]
    lower_oh = seqs_to_onehot(pair_batch.lower, expected_len)   # [N, 4, 2L]

    print(f"    Computing RCM features for {len(pair_batch.label)} pairs...")
    rcm_f, rcm_u, rcm_l = compute_rcm_batch(pair_batch, junction_bps)
    rcm_f_t = torch.from_numpy(rcm_f)   # [N, 5, 25]
    rcm_u_t = torch.from_numpy(rcm_u)
    rcm_l_t = torch.from_numpy(rcm_l)

    labels = torch.tensor(pair_batch.label, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(upper_oh, lower_oh, rcm_f_t, rcm_u_t, rcm_l_t, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    probs = []
    with torch.no_grad():
        for ub, lb, rf, ru, rl, _ in loader:
            logits = trainer.model(ub.to(device), lb.to(device),
                                   rf.to(device), ru.to(device), rl.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())
    return torch.cat(probs).numpy()


def metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "prc": average_precision_score(labels, probs),
        "acc": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred, zero_division=0),
        "mcc": matthews_corrcoef(labels, pred),
        "real_bs_score": float(probs[labels == 1].mean()),
        "mismatch_score": float(probs[labels == 0].mean()),
        "score_gap": float(probs[labels == 1].mean() - probs[labels == 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["bscan_seq_lite", "circcnn", "circcnndouble", "circdc", "jedi"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 315])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--junction-bps", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="research_results")
    parser.add_argument("--split-strategy", choices=["sample", "transcript"], default="transcript")
    parser.add_argument(
        "--negative-mode",
        choices=[
            "full_lower", "lower_intron", "upper_intron", "both_introns",
            "ls_lower_intron", "ls_upper_intron", "ls_both_introns",
        ],
        default="lower_intron",
        help="Region mismatch used for synthetic negatives. "
             "ls_* modes replace introns with LS-junction introns (intermediate difficulty).",
    )
    parser.add_argument(
        "--with-duplex", action="store_true",
        help="For FM models, also evaluate FM+duplex combined score (alpha=0.2).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = DataSetPrep(
        "./data/BS_LS_coordinates_final.csv",
        "./data/hg19_seq_dict.json",
        junction_bps=args.junction_bps,
        flanking_bps=args.junction_bps,
        seed=42,
    )
    data.load_junction_flanking_seq()
    if args.split_strategy == "transcript":
        _, _, test_keys = data.split_data_grouped(group_by="transcript")
    else:
        _, _, test_keys = data.split_data()

    # Separate FM-unified models from standard models
    fm_models   = [m for m in args.models if m in FM_UNIFIED_MODELS]
    std_models  = [m for m in args.models if m not in FM_UNIFIED_MODELS]

    rows = []
    raw_rows = []

    # Pre-load FM components once per encoder type (shared across seeds)
    fm_components: dict[str, tuple] = {}  # enc_type → (fm_model, tokenizer)
    for model_name in fm_models:
        enc_type = FM_UNIFIED_MODELS[model_name]
        if enc_type not in fm_components:
            print(f"Loading FM components for {enc_type}...")
            fm_components[enc_type] = load_fm_components(enc_type, args.device)

    for seed in args.seeds:
        pair_batch = build_hard_negative_pairs(
            data,
            test_keys,
            seed=seed,
            max_samples=args.max_samples,
            negative_mode=args.negative_mode,
        )
        labels = np.array(pair_batch.label)

        # Compute duplex energies once per seed (shared across FM and std models)
        duplex_energies: np.ndarray | None = None
        if args.with_duplex:
            print(f"  Computing duplex energies (seed={seed})...")
            duplex_energies = compute_duplex_energies(pair_batch, args.junction_bps)

        # Standard models (two-input, single-input, and RCM)
        for model_name in std_models:
            if model_name not in TWO_INPUT_MODELS | SINGLE_INPUT_MODELS | RCM_INPUT_MODELS:
                print(f"[skip] {model_name}: unsupported input type for hard-negative pairing.")
                continue
            trainer = load_model(model_name, seed, args.device, args.junction_bps)
            if trainer is None:
                continue
            if model_name in SINGLE_INPUT_MODELS:
                probs = predict_single(trainer, pair_batch, args.batch_size, args.junction_bps, args.device)
            elif model_name in RCM_INPUT_MODELS:
                probs = predict_rcm(trainer, pair_batch, args.batch_size, args.junction_bps, args.device)
            else:
                probs = predict(trainer, pair_batch, args.batch_size, args.junction_bps, args.device)

            def _append_std(name, p):
                row = {
                    "model": name,
                    "seed": seed,
                    "negative_mode": args.negative_mode,
                    "n_real_bs": int((labels == 1).sum()),
                    "n_hard_neg": int((labels == 0).sum()),
                }
                row.update(metrics(labels, p))
                rows.append(row)
                for i, (label, prob) in enumerate(zip(labels, p)):
                    raw_rows.append({
                        "model": name, "seed": seed,
                        "negative_mode": args.negative_mode,
                        "sample_idx": i, "label": int(label), "prob_bs": float(prob),
                    })
                print(f"{name} seed={seed}: AUC={row['auc']:.4f}, gap={row['score_gap']:.4f}")

            _append_std(model_name, probs)

            # Optional duplex combination for standard models too
            if args.with_duplex and duplex_energies is not None:
                combined_probs = combine_with_duplex(probs, duplex_energies, alpha=0.2)
                _append_std(model_name + "+duplex", combined_probs)

        # FM-unified models
        for model_name in fm_models:
            enc_type = FM_UNIFIED_MODELS[model_name]
            fm_model, tokenizer = fm_components[enc_type]
            bscan_model = load_bscan_unified(model_name, enc_type, seed, args.device)
            if bscan_model is None:
                continue
            probs = predict_fm(bscan_model, fm_model, tokenizer, pair_batch,
                               args.batch_size, args.junction_bps, args.device)

            def _append(name, p):
                row = {
                    "model": name,
                    "seed": seed,
                    "negative_mode": args.negative_mode,
                    "n_real_bs": int((labels == 1).sum()),
                    "n_hard_neg": int((labels == 0).sum()),
                }
                row.update(metrics(labels, p))
                rows.append(row)
                for i, (label, prob) in enumerate(zip(labels, p)):
                    raw_rows.append({
                        "model": name, "seed": seed,
                        "negative_mode": args.negative_mode,
                        "sample_idx": i, "label": int(label), "prob_bs": float(prob),
                    })
                print(f"{name} seed={seed}: AUC={row['auc']:.4f}, gap={row['score_gap']:.4f}")

            _append(model_name, probs)

            # FM + duplex combined variant
            if args.with_duplex and duplex_energies is not None:
                combined_probs = combine_with_duplex(probs, duplex_energies, alpha=0.2)
                _append(model_name + "+duplex", combined_probs)

    result = pd.DataFrame(rows)
    result_path = os.path.join(args.out_dir, f"hard_negative_pairing_{args.negative_mode}_results.csv")

    # Merge with existing results (append new models, don't overwrite old ones)
    existing_path = result_path
    if os.path.exists(existing_path) and not result.empty:
        existing = pd.read_csv(existing_path)
        new_models = result["model"].unique()
        existing = existing[~existing["model"].isin(new_models)]
        result = pd.concat([existing, result], ignore_index=True)
    result.to_csv(result_path, index=False)

    raw_path = os.path.join(args.out_dir, f"hard_negative_pairing_{args.negative_mode}_raw.csv")
    raw_df = pd.DataFrame(raw_rows)
    if os.path.exists(raw_path) and not raw_df.empty:
        existing_raw = pd.read_csv(raw_path)
        new_models = raw_df["model"].unique()
        existing_raw = existing_raw[~existing_raw["model"].isin(new_models)]
        raw_df = pd.concat([existing_raw, raw_df], ignore_index=True)
    raw_df.to_csv(raw_path, index=False)

    if not result.empty:
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        summary = result.groupby("model")[numeric_cols].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary.insert(1, "negative_mode", args.negative_mode)
        summary_path = os.path.join(args.out_dir, f"hard_negative_pairing_{args.negative_mode}_summary.csv")
        summary.to_csv(summary_path, index=False)
        print("\nSummary:")
        print(summary[["model", "auc_mean", "auc_std", "score_gap_mean"]].to_string(index=False))
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
