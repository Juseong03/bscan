#!/usr/bin/env python
"""Evaluate branch-ablation models: internal AUC + External-A (circAtlas) + Tier2/3.

All ablation models use rnafm cached embeddings. Internal uses transcript split.
External-A reuses pre-extracted circAtlas rnafm embeddings.
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)


import csv, json, os, statistics, sys, warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from dataloader import DataSetPrep, circData_cached_fm
from models.bscan_unified import BSCANUnified
from utils import get_device

DEVICE = get_device(0)
L = 100
_RC_PERM = [3, 2, 1, 0]
SEEDS = [42, 123, 315]

ABLATION = {
    "bscan_unified_fm_fulltr":   dict(use_cnn=True,  use_stem=True,  use_attn=True),
    "bscan_unified_fm_cnnonly":  dict(use_cnn=True,  use_stem=False, use_attn=False),
    "bscan_unified_fm_stemonly": dict(use_cnn=False, use_stem=True,  use_attn=False),
    "bscan_unified_fm_attnonly": dict(use_cnn=False, use_stem=False, use_attn=True),
    "bscan_unified_fm_nocnn":    dict(use_cnn=False, use_stem=True,  use_attn=True),
    "bscan_unified_fm_nostem":   dict(use_cnn=True,  use_stem=False, use_attn=True),
    "bscan_unified_fm_noattn":   dict(use_cnn=True,  use_stem=True,  use_attn=False),
}

# Display labels and which branches are active
LABELS = {
    "bscan_unified_fm_fulltr":   "Full (CNN+Stem+Attn)",
    "bscan_unified_fm_cnnonly":  "FM+CNN",
    "bscan_unified_fm_stemonly": "FM+Stem",
    "bscan_unified_fm_attnonly": "FM+Attn",
    "bscan_unified_fm_nocnn":    "Full −CNN",
    "bscan_unified_fm_nostem":   "Full −Stem",
    "bscan_unified_fm_noattn":   "Full −Attn",
}


def load_model(name, seed):
    ckpt = Path(f"saved_models/{name}/{seed}/model.pth")
    if not ckpt.exists():
        return None
    m = BSCANUnified(encoder_type="rnafm", use_cached=True, **ABLATION[name])
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
    return m.to(DEVICE).eval()


def eval_internal(name, seed):
    data = DataSetPrep("data/BS_LS_coordinates_final.csv", "data/hg19_seq_dict.json",
                       junction_bps=L, flanking_bps=L, seed=seed)
    data.load_junction_flanking_seq()
    _, _, test_keys = data.split_data_grouped(group_by="transcript")
    u_oh, l_oh, label_tensor = data.seq_to_tensor(test_keys)
    lower_rc_oh = l_oh[:, _RC_PERM, L:].flip(dims=[2])
    upper_oh = u_oh[:, :, :L]
    ds = circData_cached_fm(test_keys, label_tensor, "rnafm", upper_oh=upper_oh, lower_rc_oh=lower_rc_oh)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model = load_model(name, seed)
    if model is None:
        return None
    probs, labels = [], []
    with torch.no_grad():
        for u, l, lr, uo, lro, lbl in loader:
            logits = model(u.to(DEVICE), l.to(DEVICE), lr.to(DEVICE), uo.to(DEVICE), lro.to(DEVICE))
            probs.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
            labels.extend(lbl.numpy())
    del model; torch.cuda.empty_cache()
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


def eval_external_a(name, seed):
    """Evaluate on circAtlas External-A using pre-extracted rnafm embeddings."""
    junction = json.load(open("external_data/circatlas/exon_controls/seq_dict/junction.json"))
    emb_dir = Path("external_data/circatlas/exon_controls/fm_embeddings/rnafm")
    keys = list(junction.keys())

    NUC = {"A": 0, "G": 1, "C": 2, "T": 3, "U": 3}
    def oh(seq, length):
        a = np.zeros((4, length), dtype=np.float32)
        for j, b in enumerate(seq[:length].upper()):
            c = NUC.get(b)
            if c is not None: a[c, j] = 1.0
        return a

    model = load_model(name, seed)
    if model is None:
        return None
    probs, labels = [], []
    BSZ = 64
    with torch.no_grad():
        for i in range(0, len(keys), BSZ):
            bk = keys[i:i+BSZ]
            u_embs, l_embs, lr_embs, u_ohs, lr_ohs, labs = [], [], [], [], [], []
            valid = []
            for k in bk:
                pt = emb_dir / f"{k.replace('|','_')}.pt"
                if not pt.exists():
                    continue
                e = torch.load(pt, map_location="cpu", weights_only=True)
                rec = junction[k]
                u_embs.append(e["upper"].float()); l_embs.append(e["lower"].float()); lr_embs.append(e["lower_rc"].float())
                u_ohs.append(torch.from_numpy(oh(rec["upper_seq"][:L], L)))
                li = rec["lower_seq"][L:2*L]
                lo = oh(li, L)
                lr_ohs.append(torch.from_numpy(lo[[3,2,1,0], ::-1].copy()))
                labs.append(1 if rec["label"] == "BS" else 0)
            if not valid and not labs:
                continue
            def pad(lst):
                mx = max(x.shape[0] for x in lst)
                out = torch.zeros(len(lst), mx, lst[0].shape[1])
                for j, x in enumerate(lst): out[j, :x.shape[0]] = x
                return out
            logits = model(pad(u_embs).to(DEVICE), pad(l_embs).to(DEVICE), pad(lr_embs).to(DEVICE),
                           torch.stack(u_ohs).to(DEVICE), torch.stack(lr_ohs).to(DEVICE))
            probs.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
            labels.extend(labs)
    del model; torch.cuda.empty_cache()
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


def main():
    rows = []
    for name in ABLATION:
        int_aucs, int_prcs, ext_aucs, ext_prcs = [], [], [], []
        for seed in SEEDS:
            r = eval_internal(name, seed)
            if r: int_aucs.append(r[0]); int_prcs.append(r[1])
            re = eval_external_a(name, seed)
            if re: ext_aucs.append(re[0]); ext_prcs.append(re[1])
        if not int_aucs:
            print(f"[skip] {name}: no checkpoints"); continue
        ia = statistics.mean(int_aucs); ist = statistics.stdev(int_aucs) if len(int_aucs)>1 else 0
        ea = statistics.mean(ext_aucs) if ext_aucs else 0; est = statistics.stdev(ext_aucs) if len(ext_aucs)>1 else 0
        drop = (ia - ea)/ia*100 if ea else None
        rows.append({
            "model": name, "label": LABELS[name], "n_seeds": len(int_aucs),
            "int_auc": round(ia,4), "int_auc_std": round(ist,4),
            "int_prc": round(statistics.mean(int_prcs),4),
            "ext_auc": round(ea,4) if ea else "", "ext_auc_std": round(est,4) if ea else "",
            "ext_prc": round(statistics.mean(ext_prcs),4) if ext_prcs else "",
            "drop_pct": round(drop,2) if drop else "",
        })
        print(f"{LABELS[name]:<14} int={ia:.4f}±{ist:.4f}  ext={ea:.4f}  drop={drop:.1f}%" if ea
              else f"{LABELS[name]:<14} int={ia:.4f}±{ist:.4f}  ext=N/A")

    out = Path("research_results/ablation_results.csv")
    os.makedirs("research_results", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
