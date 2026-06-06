#!/usr/bin/env python
"""Evaluate all missing baseline models on the circAtlas exon-aware external set.

Runs inference with saved checkpoints (no retraining).
Produces per-model result CSVs compatible with all_model_external_control_summary.csv.
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, accuracy_score
from torch.utils.data import DataLoader, Dataset

L = 100       # junction_bps
RCM_KMERS = [5, 7, 9, 11, 13]

# ── Tokenisation helpers ──────────────────────────────────────────────────────
NUC_OH = {"A": 0, "G": 1, "C": 2, "T": 3, "U": 3}   # matches DataSetPrep channel order
TOKEN   = {"A": 6, "C": 7, "G": 8, "T": 9, "U": 9}

_KMER3 = {"".join(c): i for i, c in enumerate(itertools.product("ACGT", repeat=3))}


def seq_to_onehot(seq: str, length: int) -> np.ndarray:
    arr = np.zeros((4, length), dtype=np.float32)
    for j, base in enumerate(seq[:length].upper()):
        ch = NUC_OH.get(base)
        if ch is not None:
            arr[ch, j] = 1.0
    return arr


def seq_to_tokens(seq: str, length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.int64)
    for j, base in enumerate(seq[:length].upper()):
        arr[j] = TOKEN.get(base, 0)
    return arr


def seq_to_kmer3(seq: str, length: int) -> np.ndarray:
    seq = seq[:length].upper()
    return np.array([_KMER3.get(seq[i:i+3], 0) for i in range(len(seq) - 2)], dtype=np.int64)


# ── Dataset ───────────────────────────────────────────────────────────────────
class CircAtlasDataset(Dataset):
    """Generic dataset; returns all pre-converted tensors per sample."""
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]


def collate_fn(batch):
    keys_out = [b["key"] for b in batch]
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    tensors  = {}
    for field in batch[0].keys():
        if field in ("key", "label"):
            continue
        tensors[field] = torch.from_numpy(np.stack([b[field] for b in batch]))
    return keys_out, labels, tensors


# ── RCM score computation (for circcnntri) ────────────────────────────────────
def compute_rcm_scores(junction: dict, cache_path: Path | None = None) -> dict:
    """Compute RCM scores for all keys in junction dict.

    Returns {key: {type: {kmer: [25 ints]}}} where type ∈ {flanking, upper, lower}.
    Caches to cache_path if given (JSON), reloads if cache exists.
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached RCM scores from {cache_path}")
        return json.loads(cache_path.read_text())

    from RCSFinder import RCSFinder
    from tqdm import tqdm

    result = {}
    for key, rec in tqdm(junction.items(), desc="Computing RCM scores"):
        upper_seq = rec["upper_intron"]   # 100 nt
        lower_seq = rec["lower_intron"]   # 100 nt
        entry: dict = {"flanking": {}, "upper": {}, "lower": {}}
        for k in RCM_KMERS:
            for rcm_type, is_flanking, is_upper in [
                ("flanking", True,  False),
                ("upper",    False, True),
                ("lower",    False, False),
            ]:
                _, dist = RCSFinder(
                    key=key,
                    upper_seq=upper_seq,
                    lower_seq=lower_seq,
                    is_flanking_introns=is_flanking,
                    kmers=k,
                    is_upper_intron=is_upper,
                    seq_fraction_of_spacer=0,
                    allowed_seed_mismatch=0,
                ).subseq_validity_check()
                entry[rcm_type][str(k)] = [int(x) for x in dist]
        result[key] = entry

    if cache_path:
        cache_path.write_text(json.dumps(result))
        print(f"  Saved RCM cache → {cache_path}")
    return result


def rcm_entry_to_arrays(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one RCM dict entry to three [5, 25] float32 arrays."""
    def build(rcm_type: str) -> np.ndarray:
        mats = [np.log(np.array(entry[rcm_type][str(k)]).reshape(5, 5) + 1) for k in RCM_KMERS]
        return np.concatenate(mats, axis=1).astype(np.float32)  # [5, 25]
    return build("flanking"), build("upper"), build("lower")


def build_records(junction: dict, input_kind: str,
                  rcm_scores: dict | None = None) -> list[dict]:
    records = []
    for key, rec in junction.items():
        upper = rec["upper_seq"]   # 200 nt (100 intron + 100 exon)
        lower = rec["lower_seq"]   # 200 nt (100 exon + 100 intron, already in model order)
        upper_intron = rec["upper_intron"]  # 100 nt
        lower_intron = rec["lower_intron"]  # 100 nt
        label = 1 if rec["label"] == "BS" else 0

        r: dict = {"key": key, "label": label}

        if input_kind == "concat_onehot":
            # [4, 4L] — upper then lower concatenated
            oh = np.concatenate([seq_to_onehot(upper, 2*L), seq_to_onehot(lower, 2*L)], axis=1)
            r["seq"] = oh
        elif input_kind == "double_onehot":
            r["upper"] = seq_to_onehot(upper, 2*L)
            r["lower"] = seq_to_onehot(lower, 2*L)
        elif input_kind == "concat_kmer3":
            # circdeep: concat upper+lower as k-mer index [4L-2]
            concat = (upper + lower)[:4*L]
            r["seq"] = seq_to_kmer3(concat, 4*L)
        elif input_kind == "double_kmer3":
            # jedi
            r["upper"] = seq_to_kmer3(upper, 2*L)
            r["lower"] = seq_to_kmer3(lower, 2*L)
        elif input_kind == "embedonly":
            # BSCANUnifiedEmbedOnly: token IDs + one-hot stems
            r["upper_tok"] = seq_to_tokens(upper, 2*L)
            r["lower_tok"] = seq_to_tokens(lower, 2*L)
            # Stem branch needs upper_intron one-hot and lower_intron RC one-hot
            u_oh = seq_to_onehot(upper_intron, L)
            l_oh = seq_to_onehot(lower_intron, L)
            RC = {0: 3, 1: 2, 2: 1, 3: 0}   # A↔T, C↔G
            l_rc_oh = np.zeros((4, L), dtype=np.float32)
            for ch in range(4):
                l_rc_oh[RC[ch], :] = l_oh[ch, ::-1]
            r["upper_oh"] = u_oh
            r["lower_rc_oh"] = l_rc_oh
        elif input_kind == "rcm":
            # circcnntri: double one-hot + RCM features
            r["upper"] = seq_to_onehot(upper, 2*L)
            r["lower"] = seq_to_onehot(lower, 2*L)
            if rcm_scores is None:
                raise ValueError("rcm_scores required for 'rcm' input_kind")
            flank_arr, upper_arr, lower_arr = rcm_entry_to_arrays(rcm_scores[key])
            r["rcm_flanking"] = flank_arr   # [5, 25]
            r["rcm_upper"]    = upper_arr
            r["rcm_lower"]    = lower_arr
        else:
            raise ValueError(f"Unknown input_kind: {input_kind}")

        records.append(r)
    return records


# ── Model loading ─────────────────────────────────────────────────────────────
def build_model(model_name: str):
    from models.circCNNSingle import CircCNNSingle
    from models.circCNNDouble import CircCNNDouble
    from models.circCNNDoubleShare import CircCNNDoubleShare
    from models.circCNN import CircCNN
    from models.circDC import CircDC
    from models.circNet import CircNet
    from models.circDeep import CircDeep
    from models.deepCircCode import DeepCircCode
    from models.jedi import JEDI
    from models.bscan_unified import BSCANUnifiedEmbedOnly

    EMBEDONLY_ENC = {
        "bscan_embedonly_ernie": "rnaernie",
        "bscan_embedonly_bert":  "rnabert",
        "bscan_embedonly_fm":    "rnafm",
        "bscan_embedonly_msm":   "rnamsm",
    }
    if model_name == "circcnntri":
        from models.circCNNtri import CircCNNtri
        return CircCNNtri(length_seq=2*L, n_rcm_features=5), "rcm"
    if model_name == "circcnnsingle":
        return CircCNNSingle(), "concat_onehot"
    if model_name == "circcnndouble":
        return CircCNNDouble(length_seq=2*L), "double_onehot"
    if model_name == "circcnndoubleshare":
        return CircCNNDoubleShare(length_seq=2*L), "double_onehot"
    if model_name == "deepcirccode":
        return DeepCircCode(), "concat_onehot"
    if model_name == "circnet":
        return CircNet(), "concat_onehot"
    if model_name == "circdeep":
        return CircDeep(), "concat_kmer3"
    if model_name == "jedi":
        return JEDI(), "double_kmer3"
    if model_name in EMBEDONLY_ENC:
        enc = EMBEDONLY_ENC[model_name]
        return BSCANUnifiedEmbedOnly(encoder_type=enc, junction_bps=L), "embedonly"
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def run_inference(model, tensors: dict, input_kind: str, device: str) -> np.ndarray:
    model.eval()
    if input_kind == "concat_onehot":
        seq = tensors["seq"].to(device)
        logits = model(seq)
    elif input_kind == "double_onehot":
        u = tensors["upper"].to(device)
        l = tensors["lower"].to(device)
        logits = model(u, l)
    elif input_kind == "concat_kmer3":
        seq = tensors["seq"].to(device)
        logits = model(seq)
    elif input_kind == "double_kmer3":
        u = tensors["upper"].to(device)
        l = tensors["lower"].to(device)
        logits = model(u, l)
    elif input_kind == "embedonly":
        u_tok    = tensors["upper_tok"].to(device)
        l_tok    = tensors["lower_tok"].to(device)
        u_oh     = tensors["upper_oh"].to(device)
        l_rc_oh  = tensors["lower_rc_oh"].to(device)
        logits = model(u_tok, l_tok, upper_oh=u_oh, lower_rc_oh=l_rc_oh)
    elif input_kind == "rcm":
        u  = tensors["upper"].to(device)
        l  = tensors["lower"].to(device)
        rf = tensors["rcm_flanking"].to(device)
        ru = tensors["rcm_upper"].to(device)
        rl = tensors["rcm_lower"].to(device)
        logits = model(u, l, rf, ru, rl)
    else:
        raise ValueError(f"Unknown input_kind: {input_kind}")

    probs = torch.softmax(logits.float(), dim=-1)[:, 1]
    return probs.cpu().numpy()


# ── Main evaluation ───────────────────────────────────────────────────────────
SEEDS = [42, 123, 315, 777, 1004, 2024, 2025, 2026, 3407, 9001]

MODELS = [
    "circcnnsingle",
    "circcnndouble",
    "circcnndoubleshare",
    "jedi",
    "deepcirccode",
    "circnet",
    "circdeep",
    "bscan_embedonly_bert",
    "bscan_embedonly_ernie",
    "bscan_embedonly_fm",
    "bscan_embedonly_msm",
    "circcnntri",
]


def evaluate_model(model_name: str, junction: dict, out_dir: Path, device: str, batch_size: int):
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    model_obj, input_kind = build_model(model_name)

    rcm_scores = None
    if input_kind == "rcm":
        rcm_cache = out_dir / "rcm_scores_cache.json"
        rcm_scores = compute_rcm_scores(junction, cache_path=rcm_cache)

    records = build_records(junction, input_kind, rcm_scores=rcm_scores)
    dataset = CircAtlasDataset(records)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_rows = []
    for seed in SEEDS:
        ckpt = Path("saved_models") / model_name / str(seed) / "model.pth"
        if not ckpt.exists():
            print(f"  [skip] {ckpt} not found")
            continue

        sd = torch.load(ckpt, map_location=device, weights_only=True)
        # EmbedOnly: embedding vocab size may differ between FM versions (26 vs 28)
        if "embedding.weight" in sd:
            ew = sd["embedding.weight"]
            cur = model_obj.embedding.weight
            if cur.shape != ew.shape:
                import torch.nn as _nn
                model_obj.embedding = _nn.Embedding(ew.shape[0], ew.shape[1])
        model_obj.load_state_dict(sd, strict=True)
        model_obj.to(device).eval()

        all_keys, all_labels, all_probs = [], [], []
        for keys_b, labels_b, tensors_b in loader:
            probs_b = run_inference(model_obj, tensors_b, input_kind, device)
            all_keys.extend(keys_b)
            all_labels.extend(labels_b.numpy().tolist())
            all_probs.extend(probs_b.tolist())

        labels_arr = np.array(all_labels)
        probs_arr  = np.array(all_probs)
        preds_arr  = (probs_arr >= 0.5).astype(int)

        auc = roc_auc_score(labels_arr, probs_arr)
        prc = average_precision_score(labels_arr, probs_arr)
        mcc = matthews_corrcoef(labels_arr, preds_arr)
        acc = accuracy_score(labels_arr, preds_arr)

        # Save per-seed prediction CSV (compatible with duplex combination scripts)
        pred_df = pd.DataFrame({"key": all_keys, "label": all_labels,
                                 "prob_bs": all_probs, "pred": preds_arr.tolist()})
        pred_path = out_dir / f"predictions_{model_name}_{seed}.csv"
        pred_df.to_csv(pred_path, index=False)

        row = {"model": model_name, "seed": seed, "auc": auc, "prc": prc,
               "mcc": mcc, "acc": acc,
               "pos_score": float(probs_arr[labels_arr == 1].mean()),
               "neg_score": float(probs_arr[labels_arr == 0].mean())}
        all_rows.append(row)
        print(f"  seed={seed}  AUC={auc:.4f}  PRC={prc:.4f}  MCC={mcc:.4f}")

    if not all_rows:
        print(f"  [warn] No results for {model_name}")
        return None

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / f"{model_name}_external_control_results.csv", index=False)

    summary = {
        "model":         model_name,
        "n_seeds":       len(all_rows),
        "auc_mean":      results_df["auc"].mean(),
        "auc_std":       results_df["auc"].std(),
        "prc_mean":      results_df["prc"].mean(),
        "prc_std":       results_df["prc"].std(),
        "mcc_mean":      results_df["mcc"].mean(),
        "mcc_std":       results_df["mcc"].std(),
        "pos_score_mean": results_df["pos_score"].mean(),
        "neg_score_mean": results_df["neg_score"].mean(),
    }
    print(f"  → AUC {summary['auc_mean']:.4f}±{summary['auc_std']:.4f}  "
          f"PRC {summary['prc_mean']:.4f}±{summary['prc_std']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--out_dir", type=Path,
                        default=Path("external_data/circatlas/exon_controls"))
    parser.add_argument("--seq_json", type=Path,
                        default=Path("external_data/circatlas/exon_controls/seq_dict/junction.json"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    junction = json.loads(args.seq_json.read_text())
    print(f"Loaded {len(junction)} external examples | device={args.device}")

    summaries = []
    for model_name in args.models:
        s = evaluate_model(model_name, junction, args.out_dir, args.device, args.batch_size)
        if s:
            summaries.append(s)

    if not summaries:
        print("No new results.")
        return

    # Merge with existing summary
    new_df = pd.DataFrame(summaries)
    summary_path = args.out_dir / "all_model_external_control_summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Remove any rows that will be replaced
        existing = existing[~existing["model"].isin(new_df["model"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values("auc_mean", ascending=False)
    combined.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print("Updated:", summary_path)
    print(combined[["model", "n_seeds", "auc_mean", "auc_std", "prc_mean", "prc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
