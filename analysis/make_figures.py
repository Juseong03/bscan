#!/usr/bin/env python
"""Generate publication figures for the BSCAN paper.

Figures:
  Fig1  Internal vs External AUC across models (generalization gap)
  Fig2  Branch ablation (external AUC by branch composition)
  Fig3  Hard-negative 3-tier (Tier1/2/3 + hnaug)
  Fig4  ALU multi-scale enrichment + ALU-matched Tier2
  Fig5  Duplex alpha sensitivity

Outputs PNG (300 dpi) + PDF to figures/.
"""
from __future__ import annotations
import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import csv, os, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 300, "savefig.bbox": "tight",
})

# Color palette
C_FM   = "#2c7fb8"   # FM models (blue)
C_BASE = "#7fcdbb"   # BSCAN-base/CNN (teal)
C_BL   = "#bdbdbd"   # baselines (grey)
C_HN   = "#d95f02"   # hnaug (orange)
C_ACC  = "#e34a33"   # accent/red


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return None


# ─────────────────────────────────────────────────────────────────────────────
def fig1_generalization():
    rows = load("results/paper_table_master.csv")
    # Keep models with both int and ext
    data = []
    for r in rows:
        ia, ea = fnum(r["int_auc"]), fnum(r["ext_auc"])
        if ia is None or ea is None: continue
        data.append((r["display"], ia, ea, fnum(r["drop_pct"])))
    data.sort(key=lambda x: -x[2])  # by external AUC

    labels = [d[0] for d in data]
    ext = [d[2] for d in data]
    intl = [d[1] for d in data]

    def color(name):
        if name.startswith("BSCAN-RNA"): return C_FM
        if name in ("BSCAN-base", "BSCAN-onehot"): return C_BASE
        return C_BL

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios":[1.15,1]})

    # Left: paired internal→external (dumbbell)
    y = np.arange(len(labels))[::-1]
    for yi, name, ia, ea, dp in zip(y, labels, intl, ext, [d[3] for d in data]):
        ax1.plot([ia, ea], [yi, yi], color="#cccccc", lw=1.5, zorder=1)
        ax1.scatter(ia, yi, color="#555555", s=28, zorder=2)
        ax1.scatter(ea, yi, color=color(name), s=46, zorder=3)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.axvline(0.5, color="#999", ls=":", lw=0.8)
    ax1.set_xlabel("AUC"); ax1.set_xlim(0.44, 0.95)
    ax1.set_title("a  Internal → External generalization")
    ax1.legend(handles=[
        plt.Line2D([],[],marker="o",color="w",markerfacecolor="#555",markersize=7,label="Internal"),
        plt.Line2D([],[],marker="o",color="w",markerfacecolor=C_FM,markersize=8,label="External (FM)"),
        plt.Line2D([],[],marker="o",color="w",markerfacecolor=C_BASE,markersize=8,label="External (one-hot arch.)"),
        plt.Line2D([],[],marker="o",color="w",markerfacecolor=C_BL,markersize=8,label="External (baseline)"),
    ], fontsize=7.5, loc="lower right", frameon=False)

    # Right: drop% bar
    dporder = sorted(data, key=lambda x: x[3])
    dlabels = [d[0] for d in dporder]; drops = [d[3] for d in dporder]
    yb = np.arange(len(dlabels))[::-1]
    ax2.barh(yb, drops, color=[color(n) for n in dlabels], edgecolor="white")
    ax2.set_yticks(yb); ax2.set_yticklabels(dlabels, fontsize=8.5)
    ax2.set_xlabel("Internal→External AUC drop (%)")
    ax2.set_title("b  Generalization gap")
    for yi, d in zip(yb, drops):
        ax2.text(d+0.5, yi, f"{d:.0f}%", va="center", fontsize=7.5)
    ax2.set_xlim(0, 50)

    fig.tight_layout()
    fig.savefig(FIG/"Fig1_generalization.png"); fig.savefig(FIG/"Fig1_generalization.pdf")
    plt.close(fig); print("Fig1 done")


# ─────────────────────────────────────────────────────────────────────────────
def fig2_ablation():
    rows = load("results/ablation_summary.csv")
    order = ["Full (CNN+Stem+Attn)","Full −Attn","Full −Stem","FM+CNN","Full −CNN","FM+Stem","FM+Attn"]
    by = {r["label"]: r for r in rows}
    labels, ext, exterr, intl = [], [], [], []
    for lab in order:
        if lab not in by: continue
        labels.append(lab.replace("Full (CNN+Stem+Attn)","Full"))
        ext.append(fnum(by[lab]["ext_auc"])); exterr.append(fnum(by[lab]["ext_auc_std"]))
        intl.append(fnum(by[lab]["int_auc"]))

    x = np.arange(len(labels))
    # color: has CNN = blue, no CNN = red
    cols = [C_FM if ("CNN" in l or l=="Full" or "Attn" in l and "−" in l) else C_ACC for l in []]  # placeholder
    has_cnn = [("Full"==l) or ("−Attn" in l) or ("−Stem" in l) or (l=="FM+CNN") for l in labels]
    cols = [C_FM if h else C_ACC for h in has_cnn]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(x, ext, yerr=exterr, capsize=3, color=cols, edgecolor="white", width=0.66, zorder=3)
    ax.axhline(0.5, color="#999", ls=":", lw=0.8)
    for xi, e, er, h in zip(x, ext, exterr, has_cnn):
        ax.text(xi, e+er+0.012, f"{e:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("External AUC"); ax.set_ylim(0.5, 0.92)
    ax.set_title("Branch ablation: external generalization")
    ax.legend(handles=[Patch(facecolor=C_FM,label="CNN branch present"),
                       Patch(facecolor=C_ACC,label="CNN branch absent")],
              fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"Fig2_ablation.png"); fig.savefig(FIG/"Fig2_ablation.pdf")
    plt.close(fig); print("Fig2 done")


# ─────────────────────────────────────────────────────────────────────────────
def fig3_hardneg():
    master = {r["display"]: r for r in load("results/paper_table_master.csv")}
    # Models to show with tier1/2/3
    sel = ["BSCAN-base","CircCNN","CircCNN-tri","BSCAN-RNA-FM","BSCAN-RNAMSM"]
    t1, t2, t3, labels = [], [], [], []
    for m in sel:
        r = master.get(m)
        if not r: continue
        a1 = fnum(r["int_auc"]); a2 = fnum(r["t2_auc"]); a3 = fnum(r["t3_auc"])
        if a2 is None or a3 is None: continue
        labels.append(m); t1.append(a1); t2.append(a2); t3.append(a3)

    # hnaug from results/hard_neg_augmented_summary.csv -> use known values
    hn = {"BSCAN-hnaug":(0.872,0.901,0.843), "CircCNN-hnaug":(0.872,0.897,0.836), "BSCAN-FM-hnaug":(0.909,0.533,0.509)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios":[1.4,1]})

    # Panel A: standard training tier1/2/3 grouped
    x = np.arange(len(labels)); w = 0.26
    axA.bar(x-w, t1, w, label="Tier 1 (BS vs LS)", color="#2c7fb8")
    axA.bar(x,   t2, w, label="Tier 2 (LS-intron)", color="#7fcdbb")
    axA.bar(x+w, t3, w, label="Tier 3 (BS-intron)", color="#edf8b1", edgecolor="#bbb")
    axA.axhline(0.5, color="#999", ls=":", lw=0.8)
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axA.set_ylabel("AUC"); axA.set_ylim(0.4, 0.95)
    axA.set_title("a  Standard training: 3-tier probe")
    axA.legend(fontsize=7.5, frameon=False, loc="upper right")

    # Panel B: hnaug effect on Tier3
    hlabels = list(hn.keys())
    xb = np.arange(len(hlabels))
    std_t3 = [0.535, 0.508, 0.496]  # base, circcnn, fm standard Tier3
    hn_t3  = [hn[k][2] for k in hlabels]
    axB.bar(xb-0.2, std_t3, 0.4, label="Standard", color="#bdbdbd")
    axB.bar(xb+0.2, hn_t3, 0.4, label="Hard-neg aug.", color=C_HN)
    axB.axhline(0.5, color="#999", ls=":", lw=0.8)
    for xi, s, h in zip(xb, std_t3, hn_t3):
        axB.annotate("", xy=(xi+0.2, h), xytext=(xi-0.2, s),
                     arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))
    axB.set_xticks(xb); axB.set_xticklabels([k.replace("-hnaug","") for k in hlabels], rotation=20, ha="right", fontsize=8)
    axB.set_ylabel("Tier 3 AUC"); axB.set_ylim(0.4, 0.92)
    axB.set_title("b  Hard-neg augmented training")
    axB.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG/"Fig3_hardneg.png"); fig.savefig(FIG/"Fig3_hardneg.pdf")
    plt.close(fig); print("Fig3 done")


# ─────────────────────────────────────────────────────────────────────────────
def fig4_alu():
    ms = load("research_results/alu_multiscale_summary.csv")
    alu = [r for r in ms if r["rep"] == "alu"]
    windows = [int(r["window"]) for r in alu]
    bs_inv = [fnum(r["bs_inv_pct"]) for r in alu]
    ls_inv = [fnum(r["ls_inv_pct"]) for r in alu]
    bs_has = [fnum(r["bs_has_pct"]) for r in alu]
    ls_has = [fnum(r["ls_has_pct"]) for r in alu]

    # ALU-matched tier2
    am = load("research_results/alu_matched_tier2.csv")
    from collections import defaultdict
    agg = defaultdict(lambda: {"std":[], "mat":[]})
    for r in am:
        agg[r["model"]]["std"].append(fnum(r["tier2_std"]))
        agg[r["model"]]["mat"].append(fnum(r["tier2_alu_matched"]))
    am_models = ["bscan","circcnn","bscan_unified_fm"]
    am_lab = {"bscan":"BSCAN-base","circcnn":"CircCNN","bscan_unified_fm":"BSCAN-FM"}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: ALU inverted-pair % by window
    x = np.arange(len(windows)); w=0.36
    axA.bar(x-w/2, bs_inv, w, label="BS (circRNA)", color=C_FM)
    axA.bar(x+w/2, ls_inv, w, label="LS (linear)", color=C_BL)
    axA.set_xticks(x); axA.set_xticklabels([f"{wd} nt" for wd in windows])
    axA.set_ylabel("Inverted ALU pair (%)"); axA.set_xlabel("Intronic flank window")
    axA.set_title("a  ALU enrichment scales with window")
    axA.legend(fontsize=8, frameon=False)
    for xi,b,l in zip(x,bs_inv,ls_inv):
        axA.text(xi-w/2, b+0.15, f"{b:.1f}", ha="center", fontsize=7)
        axA.text(xi+w/2, l+0.15, f"{l:.1f}", ha="center", fontsize=7)

    # Panel B: ALU-matched Tier2
    xb = np.arange(len(am_models)); w=0.36
    std = [statistics.mean(agg[m]["std"]) for m in am_models]
    mat = [statistics.mean(agg[m]["mat"]) for m in am_models]
    axB.bar(xb-w/2, std, w, label="Standard Tier 2", color="#7fcdbb")
    axB.bar(xb+w/2, mat, w, label="ALU-matched Tier 2", color="#fdae6b")
    axB.axhline(0.5, color="#999", ls=":", lw=0.8)
    axB.set_xticks(xb); axB.set_xticklabels([am_lab[m] for m in am_models], fontsize=9)
    axB.set_ylabel("Tier 2 AUC"); axB.set_ylim(0.4, 0.82)
    axB.set_title("b  ALU-matched Tier 2: no change")
    axB.legend(fontsize=8, frameon=False, loc="upper right")
    for xi,s,mt in zip(xb,std,mat):
        axB.text(xi-w/2, s+0.01, f"{s:.3f}", ha="center", fontsize=7)
        axB.text(xi+w/2, mt+0.01, f"{mt:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG/"Fig4_alu.png"); fig.savefig(FIG/"Fig4_alu.pdf")
    plt.close(fig); print("Fig4 done")


# ─────────────────────────────────────────────────────────────────────────────
def fig5_duplex():
    rows = load("research_results/duplex_alpha_sensitivity.csv")
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["model"]][fnum(r["alpha"])].append(fnum(r["auc"]))
    models = ["bscan_unified_fm","bscan_unified_ernie","bscan_unified_msm","bscan_unified_bert"]
    mlab = {"bscan_unified_fm":"RNA-FM","bscan_unified_ernie":"RNAErnie","bscan_unified_msm":"RNAMSM","bscan_unified_bert":"RNABERT"}
    cols = ["#2c7fb8","#41b6c4","#7fcdbb","#253494"]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for m, c in zip(models, cols):
        alphas = sorted(agg[m].keys())
        means = [statistics.mean(agg[m][a]) for a in alphas]
        ax.plot(alphas, means, "-o", color=c, label=mlab[m], markersize=4, lw=1.6)
    ax.axvline(0.2, color="#e34a33", ls="--", lw=1, label="α = 0.2 (chosen)")
    ax.set_xlabel("Duplex weight α"); ax.set_ylabel("External AUC")
    ax.set_title("Thermodynamic duplex combination: α sensitivity")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"Fig5_duplex_alpha.png"); fig.savefig(FIG/"Fig5_duplex_alpha.pdf")
    plt.close(fig); print("Fig5 done")


if __name__ == "__main__":
    fig1_generalization()
    fig2_ablation()
    fig3_hardneg()
    fig4_alu()
    fig5_duplex()
    print("\nAll figures saved to figures/")
