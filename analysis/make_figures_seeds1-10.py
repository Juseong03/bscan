#!/usr/bin/env python3
"""Publication figures from the 10-seed (1-10) re-run.

Reads the server result set (default: /workspace/volume/bscan_server_results) and
writes PNG+PDF to figures/.
  Fig1  internal vs external AUC (generalization) — the headline
  Fig2  branch ablation (external AUC / drop by config)
  Fig3  AUG-RCM (FM vs FM+RCM, no gain)
"""
import os, csv, glob, sys, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "/workspace/volume/bscan_server_results"
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": .3, "figure.dpi": 150, "savefig.bbox": "tight"})
C_FM="#1b6cb3"; C_OH="#e8743b"; C_BL="#8a8a8a"

DISP={"bscan_unified_fm":"BSCAN-RNA-FM","bscan_unified_msm":"BSCAN-RNA-MSM",
"bscan_unified_ernie":"BSCAN-RNAErnie","bscan_unified_bert":"BSCAN-RNABERT",
"bscan_unified_onehot":"BSCAN-onehot","bscan":"BSCAN-base","bscan_seq_lite":"BSCAN-lite",
"circcnn":"CircCNN","circdc":"CircDC","circnet":"CircNet","circdeep":"CircDeep","jedi":"JEDI",
"deepcirccode":"DeepCircCode","circcnnsingle":"CircCNN-single","circcnndouble":"CircCNN-double",
"circcnndoubleshare":"CircCNN-dshare","circcnntri":"CircCNN-tri"}
FM={"bscan_unified_fm","bscan_unified_msm","bscan_unified_ernie","bscan_unified_bert"}
OH={"bscan_unified_onehot","bscan","bscan_seq_lite"}
def grp(m): return "FM" if m in FM else ("OH" if m in OH else "BL")
def col(m): return {"FM":C_FM,"OH":C_OH,"BL":C_BL}[grp(m)]

def load_internal():
    d=defaultdict(list)
    for f in glob.glob(f"{BASE}/research_results/model_comparison_valint_seed_*.csv"):
        for r in csv.DictReader(open(f)):
            try: d[r["model"]].append(float(r["test_auc"]))
            except: pass
    return {m:(st.mean(v),st.pstdev(v)) for m,v in d.items() if v}
def load_external():
    e={}
    for sf in ["all_model_external_control_summary.csv","all_fm_external_control_summary.csv"]:
        p=f"{BASE}/external_data/circatlas/exon_controls/{sf}"
        if os.path.exists(p):
            for r in csv.DictReader(open(p)): e[r["model"]]=(float(r["auc_mean"]),float(r["auc_std"]))
    return e

def fig1():
    I,E=load_internal(),load_external()
    models=[m for m in I if m in E and m in DISP]
    models.sort(key=lambda m: E[m][0])              # external 오름차순 (아래→위)
    y=range(len(models))
    fig,ax=plt.subplots(figsize=(8,6.5))
    for i,m in enumerate(models):
        ia=I[m][0]; ea,es=E[m]
        ax.plot([ea,ia],[i,i],color="#ccc",lw=2,zorder=1)          # drop 선
        ax.scatter(ia,i,color="#444",s=30,zorder=2)                # internal (회색)
        ax.scatter(ea,i,color=col(m),s=70,zorder=3)                # external (그룹색)
    ax.set_yticks(list(y)); ax.set_yticklabels([DISP[m] for m in models])
    ax.axvline(0.5,ls=":",color="r",alpha=.5)
    ax.set_xlabel("AUC"); ax.set_title("Internal (gray) → External (colored) generalization\nFM models stay high; one-hot/baselines collapse")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#444',label='Internal'),
        Line2D([],[],marker='o',ls='',color=C_FM,label='External · FM'),
        Line2D([],[],marker='o',ls='',color=C_OH,label='External · BSCAN one-hot'),
        Line2D([],[],marker='o',ls='',color=C_BL,label='External · baseline')],loc="center left",fontsize=9)
    fig.savefig(f"{FIG}/Fig1_generalization_s10.png"); fig.savefig(f"{FIG}/Fig1_generalization_s10.pdf"); plt.close(fig); print("Fig1 done")

def fig2():
    rows=list(csv.DictReader(open(f"{BASE}/research_results/ablation_results.csv")))
    rows.sort(key=lambda r:float(r["ext_auc"]))
    labels=[r["label"] for r in rows]; ext=[float(r["ext_auc"]) for r in rows]; drop=[float(r["drop_pct"]) for r in rows]
    fig,ax=plt.subplots(figsize=(8,5))
    cols=["#c0392b" if d>15 else ("#2980b9" if d<6 else "#f39c12") for d in drop]
    ax.barh(range(len(rows)),ext,color=cols)
    for i,(e,d) in enumerate(zip(ext,drop)): ax.text(e+.005,i,f"{e:.3f} ({d:.0f}%)",va="center",fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels)
    ax.set_xlabel("External AUC (drop% vs internal)"); ax.set_xlim(0.6,0.9)
    ax.axvline(0.5,ls=":",color="r",alpha=.4)
    ax.set_title("Branch ablation — CNN branch drives external generalization")
    fig.savefig(f"{FIG}/Fig2_ablation_s10.png"); fig.savefig(f"{FIG}/Fig2_ablation_s10.pdf"); plt.close(fig); print("Fig2 done")

def fig3():
    rows=list(csv.DictReader(open(f"{BASE}/results/rcm_aux_summary.csv")))
    fl=sorted(set(r["flanking_bps"] for r in rows))
    fig,ax=plt.subplots(figsize=(6,4.6)); w=.35; x=range(len(fl))
    base=[next(float(r["test_auc_mean"]) for r in rows if r["flanking_bps"]==f and "baseline" in r["label"]) for f in fl]
    rcm =[next(float(r["test_auc_mean"]) for r in rows if r["flanking_bps"]==f and "RCM" in r["label"]) for f in fl]
    be=[next(float(r["test_auc_std"]) for r in rows if r["flanking_bps"]==f and "baseline" in r["label"]) for f in fl]
    re=[next(float(r["test_auc_std"]) for r in rows if r["flanking_bps"]==f and "RCM" in r["label"]) for f in fl]
    ax.bar([i-w/2 for i in x],base,w,yerr=be,label="FM baseline",color=C_FM,capsize=4)
    ax.bar([i+w/2 for i in x],rcm,w,yerr=re,label="FM + RCM aux",color=C_OH,capsize=4)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"flank {f}" for f in fl]); ax.set_ylim(0.8,0.92)
    ax.set_ylabel("Internal AUC"); ax.legend(); ax.set_title("AUG-RCM: auxiliary RCM branch gives no gain")
    fig.savefig(f"{FIG}/Fig3_augrcm_s10.png"); fig.savefig(f"{FIG}/Fig3_augrcm_s10.pdf"); plt.close(fig); print("Fig3 done")

fig1(); fig2(); fig3()
print("All figures -> figures/*_s10.{png,pdf}")
