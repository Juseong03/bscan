#!/usr/bin/env python
"""Train one-hot BS/LS models with synthetic intron-mismatch hard negatives.

This is a focused probe for the intron-pairing limitation: it keeps the
standard BS/LS task but augments training with negatives made by preserving the
real BS exons and swapping only an intron from another BS locus.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from dataloader import DataSetPrep, circData_double
from evaluate_hard_negative_pairing import build_hard_negative_pairs, seqs_to_onehot
from trainer import Trainer
from utils import get_device, seed_everything


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


def make_standard_dataset(data: DataSetPrep, keys: list[str]) -> circData_double:
    upper, lower, labels = data.seq_to_tensor(keys)
    return circData_double(upper, lower, labels)


def make_augmented_dataset(
    data: DataSetPrep,
    keys: list[str],
    seed: int,
    junction_bps: int,
    hard_negative_ratio: float,
) -> circData_double:
    upper, lower, labels = data.seq_to_tensor(keys)
    pair_batch = build_hard_negative_pairs(
        data=data,
        keys=list(keys),
        seed=seed,
        max_samples=None,
        negative_mode="lower_intron",
    )
    n_real = len(pair_batch.label) // 2
    hard_upper = pair_batch.upper[n_real:]
    hard_lower = pair_batch.lower[n_real:]
    if hard_negative_ratio < 1.0:
        rng = random.Random(seed)
        keep_n = max(1, int(len(hard_upper) * hard_negative_ratio))
        keep_idx = sorted(rng.sample(range(len(hard_upper)), keep_n))
        hard_upper = [hard_upper[i] for i in keep_idx]
        hard_lower = [hard_lower[i] for i in keep_idx]

    expected_len = 2 * junction_bps
    hard_upper_tensor = seqs_to_onehot(hard_upper, expected_len)
    hard_lower_tensor = seqs_to_onehot(hard_lower, expected_len)
    hard_labels = torch.zeros(len(hard_upper), dtype=labels.dtype)

    return circData_double(
        torch.cat([upper, hard_upper_tensor], dim=0),
        torch.cat([lower, hard_lower_tensor], dim=0),
        torch.cat([labels, hard_labels], dim=0),
    )


def model_kwargs(model_name: str, junction_bps: int) -> dict:
    if model_name in {"bscan", "bscan_region_stem", "circcnn"}:
        return {"junction_bps": junction_bps}
    if model_name == "circcnndouble":
        return {"length_seq": 2 * junction_bps}
    raise ValueError(f"Only one-hot two-input models are supported here: {model_name}")


def predict(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for upper, lower, label in loader:
            logits = model(upper.to(device), lower.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())
            labels.append(label.long().detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def score_predictions(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "prc": average_precision_score(labels, probs),
        "acc": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred, zero_division=0),
        "mcc": matthews_corrcoef(labels, pred),
    }


def evaluate_standard(model: torch.nn.Module, dataset: circData_double, batch_size: int, device: str) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels, probs = predict(model, loader, device)
    return score_predictions(labels, probs)


def evaluate_hard_negative(
    model: torch.nn.Module,
    data: DataSetPrep,
    keys: list[str],
    seed: int,
    junction_bps: int,
    batch_size: int,
    device: str,
    negative_mode: str,
) -> dict[str, float]:
    pair_batch = build_hard_negative_pairs(data, keys, seed, None, negative_mode)
    expected_len = 2 * junction_bps
    upper = seqs_to_onehot(pair_batch.upper, expected_len)
    lower = seqs_to_onehot(pair_batch.lower, expected_len)
    labels = torch.tensor(pair_batch.label, dtype=torch.long)
    loader = DataLoader(TensorDataset(upper, lower, labels), batch_size=batch_size, shuffle=False)
    labels_np, probs = predict(model, loader, device)
    scores = score_predictions(labels_np, probs)
    scores["real_bs_score"] = float(probs[labels_np == 1].mean())
    scores["mismatch_score"] = float(probs[labels_np == 0].mean())
    scores["score_gap"] = scores["real_bs_score"] - scores["mismatch_score"]
    return scores


def train_one(args: argparse.Namespace, model_name: str, seed: int) -> list[EvalResult]:
    seed_everything(seed)
    device = get_device(args.device)
    data = DataSetPrep(
        coord_path="./data/BS_LS_coordinates_final.csv",
        seq_dict_path="./data/hg19_seq_dict.json",
        junction_bps=args.junction_bps,
        flanking_bps=args.flanking_bps,
        seed=seed,
    )
    data.load_junction_flanking_seq()

    if args.split_strategy == "transcript":
        keys_train, keys_valid, keys_test = data.split_data_grouped(group_by="transcript")
    elif args.split_strategy == "chromosome":
        keys_train, keys_valid, keys_test = data.split_data_grouped(group_by="chromosome")
    else:
        keys_train, keys_valid, keys_test = data.split_data()

    train_dataset = make_augmented_dataset(data, keys_train, seed, args.junction_bps, args.hard_negative_ratio)
    valid_dataset = make_standard_dataset(data, keys_valid)
    test_dataset = make_standard_dataset(data, keys_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    trainer = Trainer(seed=seed, device=device, dir_save=args.save_dir)
    trainer.define_model(model_name, **model_kwargs(model_name, args.junction_bps))
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_score = -np.inf
    best_epoch = -1
    patience = 0
    save_path = os.path.join(args.save_dir, f"{model_name}_hnaug", str(seed), "model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        trainer.model.train()
        losses = []
        for upper, lower, label in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = trainer.model(upper.to(device), lower.to(device))
            loss = loss_fn(logits, label.long().to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        valid_scores = evaluate_standard(trainer.model, valid_dataset, args.batch_size, device)
        valid_hn = evaluate_hard_negative(
            trainer.model, data, list(keys_valid), seed, args.junction_bps, args.batch_size, device, "lower_intron"
        )
        selection_score = 0.5 * valid_scores["auc"] + 0.5 * valid_hn["auc"]
        print(
            f"{model_name} seed={seed} epoch={epoch}: "
            f"loss={np.mean(losses):.4f} valid_auc={valid_scores['auc']:.4f} "
            f"valid_hn_auc={valid_hn['auc']:.4f} select={selection_score:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            patience = 0
            torch.save(trainer.model.state_dict(), save_path)
        else:
            patience += 1
            if patience > args.earlystop:
                break

    trainer.model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True), strict=False)
    rows: list[EvalResult] = []
    for split_name, dataset in [("valid_standard", valid_dataset), ("test_standard", test_dataset)]:
        scores = evaluate_standard(trainer.model, dataset, args.batch_size, device)
        rows.append(EvalResult(model=f"{model_name}_hnaug", seed=seed, epoch=best_epoch, split=split_name, **scores))
    for negative_mode in args.eval_negative_modes:
        for split_name, keys in [("valid", list(keys_valid)), ("test", list(keys_test))]:
            scores = evaluate_hard_negative(
                trainer.model,
                data,
                keys,
                seed,
                args.junction_bps,
                args.batch_size,
                device,
                negative_mode,
            )
            rows.append(
                EvalResult(
                    model=f"{model_name}_hnaug",
                    seed=seed,
                    epoch=best_epoch,
                    split=f"{split_name}_{negative_mode}_hardneg",
                    **scores,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["bscan", "bscan_region_stem"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 315])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--earlystop", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--junction-bps", type=int, default=100)
    parser.add_argument("--flanking-bps", type=int, default=100)
    parser.add_argument("--split-strategy", choices=["sample", "transcript", "chromosome"], default="transcript")
    parser.add_argument("--hard-negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--eval-negative-modes",
        nargs="+",
        default=["lower_intron", "upper_intron", "both_introns", "full_lower"],
        choices=["full_lower", "lower_intron", "upper_intron", "both_introns",
                 "ls_lower_intron", "ls_upper_intron", "ls_both_introns"],
    )
    parser.add_argument("--save-dir", default="saved_models_hnaug")
    parser.add_argument("--out-dir", default="research_results/hard_negative_augmented")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows: list[EvalResult] = []
    for model in args.models:
        for seed in args.seeds:
            rows.extend(train_one(args, model, seed))

    df = pd.DataFrame([row.__dict__ for row in rows])
    result_path = os.path.join(args.out_dir, "hard_negative_augmented_results.csv")
    summary_path = os.path.join(args.out_dir, "hard_negative_augmented_summary.csv")
    df.to_csv(result_path, index=False)
    summary = df.groupby(["model", "split"]).agg(["mean", "std"])
    summary.to_csv(summary_path)
    print("\nSummary:")
    print(summary)
    print(f"\nSaved: {result_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
