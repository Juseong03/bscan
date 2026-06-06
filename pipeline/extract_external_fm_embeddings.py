#!/usr/bin/env python
"""Extract foundation-model embeddings for an external junction sequence cache."""

from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import argparse
import json
from pathlib import Path

import torch
from multimolecule import RnaBertModel, RnaErnieModel, RnaFmModel, RnaMsmModel, RnaTokenizer
from tqdm import tqdm


MODEL_LOADERS = {
    "rnaernie": (RnaErnieModel, "multimolecule/rnaernie"),
    "rnabert": (RnaBertModel, "multimolecule/rnabert"),
    "rnafm": (RnaFmModel, "multimolecule/rnafm"),
    "rnamsm": (RnaMsmModel, "multimolecule/rnamsm"),
}


def safe_key(key: str) -> str:
    return key.replace("|", "_")


def load_model(model_name: str, device: torch.device):
    cls, hf_name = MODEL_LOADERS[model_name]
    model = cls.from_pretrained(hf_name).to(device)
    tokenizer = RnaTokenizer.from_pretrained(hf_name)
    model.eval()
    return model, tokenizer


def batch_embedding(model, tokenizer, seqs: list[str], device: torch.device) -> torch.Tensor:
    inputs = tokenizer(seqs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return emb.to(torch.float16).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junction_json", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_LOADERS), required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    junctions = json.loads(args.junction_json.read_text())
    keys = list(junctions)
    cache_dir = args.out_dir / args.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, device)

    pending = [key for key in keys if not (cache_dir / f"{safe_key(key)}.pt").exists()]
    print(f"{args.model}: total={len(keys)} pending={len(pending)} cache_dir={cache_dir}", flush=True)
    for i in tqdm(range(0, len(pending), args.batch_size)):
        batch_keys = pending[i : i + args.batch_size]
        upper = [junctions[k]["upper_seq"] for k in batch_keys]
        lower = [junctions[k]["lower_seq"] for k in batch_keys]
        lower_rc = [junctions[k]["lower_seq_rc"] for k in batch_keys]
        u_emb = batch_embedding(model, tokenizer, upper, device)
        l_emb = batch_embedding(model, tokenizer, lower, device)
        lr_emb = batch_embedding(model, tokenizer, lower_rc, device)
        for j, key in enumerate(batch_keys):
            torch.save(
                {
                    "upper": u_emb[j].clone(),
                    "lower": l_emb[j].clone(),
                    "lower_rc": lr_emb[j].clone(),
                },
                cache_dir / f"{safe_key(key)}.pt",
            )
    print(f"Saved external embeddings for {args.model}: {cache_dir}")


if __name__ == "__main__":
    main()
