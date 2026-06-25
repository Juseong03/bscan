#!/usr/bin/env python3
"""DeLong test for the difference between two correlated ROC AUCs.

Reads two prediction CSVs (each with a label column and a score/prob column,
evaluated on the SAME samples) and reports AUC_a, AUC_b, and the DeLong p-value.

Usage:
  python analysis/delong_test.py A.csv B.csv [--label label --score prob]
  # or aggregate the external prediction files written by the evaluators:
  python analysis/delong_test.py \
      external_data/circatlas/exon_controls/predictions_bscan_unified_fm_1.csv \
      external_data/circatlas/exon_controls/predictions_circcnn_1.csv
"""
import argparse, csv, sys
import numpy as np
from scipy import stats


def _read(path, label_col, score_col):
    labels, scores, keys = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        lc = label_col if label_col in cols else next((c for c in cols if c.lower() in ("label", "y", "true")), None)
        sc = score_col if score_col in cols else next((c for c in cols if c.lower() in ("prob", "score", "pred", "prob1")), None)
        kc = next((c for c in cols if c.lower() in ("key", "id")), None)
        for row in r:
            try:
                labels.append(int(float(row[lc]))); scores.append(float(row[sc]))
                keys.append(row[kc] if kc else None)
            except (ValueError, TypeError, KeyError):
                pass
    return np.array(labels), np.array(scores), keys


# --- fast DeLong (Sun & Xu 2014) ---
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def _fast_delong(preds_sorted, m):
    # preds_sorted: [2, n] rows = predictors; first m columns positive
    k, n = preds_sorted.shape; nneg = n - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    tx = np.array([_compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([_compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([_compute_midrank(preds_sorted[r]) for r in range(k)])
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / nneg
    v01 = (tz[:, :m] - tx) / nneg
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    s = sx / m + sy / nneg
    return auc, s

def delong_p(labels, score_a, score_b):
    order = np.argsort(-labels)  # positives first
    lab = labels[order]; m = int(lab.sum())
    preds = np.vstack([score_a[order], score_b[order]])
    auc, s = _fast_delong(preds, m)
    s = np.atleast_2d(s)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return float(auc[0]), float(auc[1]), 1.0
    z = (auc[0] - auc[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(auc[0]), float(auc[1]), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a"); ap.add_argument("file_b")
    ap.add_argument("--label", default="label"); ap.add_argument("--score", default="prob")
    a = ap.parse_args()
    la, sa, ka = _read(a.file_a, a.label, a.score)
    lb, sb, kb = _read(a.file_b, a.label, a.score)
    if ka[0] is not None and kb[0] is not None:
        # align by key
        db = {k: (l, s) for k, l, s in zip(kb, lb, sb)}
        rows = [(la[i], sa[i], db[ka[i]][1]) for i in range(len(ka)) if ka[i] in db]
        lab = np.array([r[0] for r in rows]); sa = np.array([r[1] for r in rows]); sb = np.array([r[2] for r in rows])
    else:
        n = min(len(la), len(lb)); lab = la[:n]; sa = sa[:n]; sb = sb[:n]
    auc_a, auc_b, p = delong_p(lab, sa, sb)
    print(f"AUC(A)={auc_a:.4f}  AUC(B)={auc_b:.4f}  ΔAUC={auc_a-auc_b:+.4f}  DeLong p={p:.2e}  (n={len(lab)})")


if __name__ == "__main__":
    main()
