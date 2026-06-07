
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from multimolecule import RnaErnieModel, RnaBertModel, RnaFmModel, RnaMsmModel, RnaTokenizer
from dataloader import DataSetPrep

def extract_and_cache_embeddings(model_name, device_id=0):
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    cache_dir = f"./fm_embeddings/{model_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"🚀 Loading {model_name} and data...")
    if model_name == 'rnaernie':
        model = RnaErnieModel.from_pretrained('multimolecule/rnaernie').to(device)
        tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
    elif model_name == 'rnabert':
        model = RnaBertModel.from_pretrained('multimolecule/rnabert').to(device)
        tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnabert')
    elif model_name == 'rnafm':
        model = RnaFmModel.from_pretrained('multimolecule/rnafm').to(device)
        tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnafm')
    elif model_name == 'rnamsm':
        model = RnaMsmModel.from_pretrained('multimolecule/rnamsm').to(device)
        tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnamsm')
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()

    # Load all sequences.
    # Prefer the pre-built junction dict (data/seq_dict/100/, ~709MB); fall back to
    # rebuilding from the full hg19 genome (data/hg19_seq_dict.json, ~2.9GB) only if absent.
    data = DataSetPrep(
        coord_path='./data/BS_LS_coordinates_final.csv',
        seq_dict_path='./data/hg19_seq_dict.json',
        junction_bps=100
    )
    try:
        junctions, _ = data.load_junction_flanking_seq()
        print("Loaded junctions from data/seq_dict/ (no genome needed)")
    except Exception:
        print("seq_dict not found — rebuilding from hg19 genome ...")
        junctions, _ = data.get_junction_intron_seq()
    keys = list(junctions.keys())

    # Resume: skip keys whose .pt already exists (safe to re-run after interruption)
    pending = [k for k in keys if not os.path.exists(os.path.join(cache_dir, f"{k.replace('|','_')}.pt"))]
    print(f"📦 {len(keys)} total, {len(pending)} pending → extracting embeddings...")
    keys = pending

    batch_size = 64 # Balanced for speed and memory
    for i in tqdm(range(0, len(keys), batch_size)):
        batch_keys = keys[i:i+batch_size]
        
        # We need embeddings for Upper, Lower, and Lower_RC
        u_seqs = [junctions[k]['upper_seq'] for k in batch_keys]
        l_seqs = [junctions[k]['lower_seq'] for k in batch_keys]
        l_rc_seqs = [junctions[k]['lower_seq_rc'] for k in batch_keys]
        
        def get_batch_emb(seq_list):
            inputs = tokenizer(seq_list, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
                if hasattr(out, 'last_hidden_state'):
                    emb = out.last_hidden_state
                else:
                    emb = out[0]
            # Convert to float16 to save 50% space and IO time, and use clone() to avoid storage leak
            return emb.to(torch.float16).cpu()

        u_emb = get_batch_emb(u_seqs)
        l_emb = get_batch_emb(l_seqs)
        l_rc_emb = get_batch_emb(l_rc_seqs)
        
        for j, key in enumerate(batch_keys):
            save_path = os.path.join(cache_dir, f"{key.replace('|', '_')}.pt")
            torch.save({
                'upper': u_emb[j].clone(),
                'lower': l_emb[j].clone(),
                'lower_rc': l_rc_emb[j].clone()
            }, save_path)

    print(f"✅ Finished caching for {model_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-extract & cache FM embeddings to fm_embeddings/<enc>/")
    # accept both --enc_type (docs) and --model (legacy) as aliases
    parser.add_argument('--enc_type', '--model', dest='enc_type', type=str, required=True,
                        choices=['rnafm', 'rnabert', 'rnaernie', 'rnamsm'],
                        help="RNA foundation model encoder to extract")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    extract_and_cache_embeddings(args.enc_type, args.device)
