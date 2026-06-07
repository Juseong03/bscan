# BSCAN: Back-Splice CircRNA Attention Network

A deep learning framework for circRNA back-splicing site (BS) detection and expression prediction. BSCAN integrates frozen RNA foundation model embeddings with cross-attention over paired junction sequences to achieve state-of-the-art performance and generalization.

---

## Overview

circRNA biogenesis depends on back-splice junctions (BSJ), where a downstream 5' splice site joins to an upstream 3' splice site. BSCAN jointly encodes the upper and lower intronic flanks around each junction through a three-branch architecture:

1. **FM branch**: frozen RNA foundation model embeddings (RNA-FM, RNAErnie, RNABERT, RNA-MSM)
2. **Stem branch**: 2D convolutional map over WC/wobble base-pair probability matrix
3. **Cross-attention**: upper × lower junction interaction

### Key findings

**Cross-dataset generalization** (internal vs. circAtlas external):

| Model | Internal AUC | External AUC | Drop |
|-------|:---:|:---:|:---:|
| **BSCAN-RNA-FM / RNABERT / RNAMSM / RNAErnie** | 0.916–0.917 | 0.838–0.846 | **~8%** |
| BSCAN-base (3-branch, no FM) | 0.901 | 0.720 | 20% |
| CircCNN | 0.898 | 0.704 | 22% |
| CircCNN-single/tri, CircDC, JEDI | 0.85–0.89 | 0.47–0.54 | 39–45% |
| **BSCAN-onehot** (same architecture, no FM) | 0.916 | 0.572 | 38% |

The ~8% external drop for BSCAN-FM vs. 39–45% for CNN-only baselines shows that frozen RNA FM embeddings generalize across genomic contexts. The architecture-matched **BSCAN-onehot control drops 38%**, isolating the FM representation — not the architecture — as the source of generalization. Leakage is ruled out at two levels: sequence-disjoint (Δ ≤ 0.001) and **host-locus-disjoint** via hg19→hg38 liftOver (FM retains ~0.82 on loci absent from training).

**Branch ablation**: the local **CNN branch drives generalization** (CNN-only external AUC 0.845 ≈ full 0.850; removing CNN → 0.714). Stem and cross-attention branches are largely redundant once CNN is present.

**Hard-negative 3-tier probe**:
- Standard training: all models near-chance at Tier 3 (individual-intron discrimination); FM models are exon-dominant (Tier 2 < 0.5).
- Hard-negative augmented (one-hot): Tier 3 AUC 0.84 — **intron-pairing signal is learnable**.
- Hard-negative augmented (FM): Tier 3 AUC 0.51 — not recoverable from frozen FM embeddings.
- ALU enrichment is real (BS > LS, p < 10⁻⁷) but **ALU-density-matched Tier 2 is unchanged** — ALU is not the primary Tier-2 driver.

---

## Repository structure

```
bscan/
├── core/                 # importable library (run scripts import from here)
│   ├── dataloader.py             # DataSetPrep, circData_* Dataset classes
│   ├── trainer.py                # Trainer: model registry, early stopping, metrics
│   ├── utils.py                  # seeding, device, optimizers
│   ├── RCSFinder.py, CalPPM.py   # reverse-complement match helpers (RCM models)
│   └── *_regression.py           # regression variants
├── models/               # model definitions
│   ├── bscan_unified.py          # BSCANUnified (main model, all FM types + branch flags)
│   ├── circCNN*.py, deepCircCode.py, circDeep.py, jedi.py, circDC.py, ...  # baselines
│   ├── classifier.py, transformer.py, mamba.py / mamba2.py
│   └── *_regression.py
├── pipeline/             # runnable entry-point scripts (invoke from repo root)
│   ├── experiment.py             # main classification training entry point
│   ├── run_model_comparison.py   # multi-model sweep wrapper around experiment.py
│   ├── train_hard_negative_augmented[_fm].py   # hnaug training
│   ├── evaluate_hard_negative_pairing.py       # 3-tier hard-negative probe
│   ├── evaluate_circatlas_all_baselines.py     # external (circAtlas) evaluation
│   ├── extract_fm_embeddings.py / extract_external_fm_embeddings.py
│   ├── make_circatlas_exon_controls.py         # build external validation set
│   ├── smoke_models.py           # quick forward-pass sanity check
│   └── *_regression.py, summarize_*.py, generate_rcm_scores_subset.py
├── analysis/             # paper-supplement analysis & figures
│   ├── evaluate_ablation.py      # branch ablation (internal + external)
│   ├── analyze_statistics.py     # bootstrap CIs
│   ├── analyze_masking.py        # exon/intron masking
│   ├── analyze_alu_repeats.py / analyze_alu_multiscale.py / analyze_alu_matched_tier2.py
│   ├── analyze_duplex_alpha.py   # duplex α sensitivity
│   ├── analyze_external_b_disjoint.py / make_external_b_hostgene.py  # leakage controls
│   └── make_figures.py           # generate Fig 1–5
├── scripts/              # shell sweep runners (call pipeline/*.py)
├── data/                 # BS_LS_coordinates_final.csv, hg38_exon.bed, human_bed_v3.0/
│                         #   (large genome/seq files are gitignored — see Data section)
├── docs/                 # paper drafts, figure captions, setup & status notes
├── figures/              # Fig 1–5 (PNG + PDF)
├── results/ research_results/   # paper tables + analysis result CSVs
└── (gitignored, transfer separately) fm_embeddings/, saved_models/, external_data/
```

> **Path handling.** Scripts in `pipeline/` and `analysis/` auto-detect the repo root from
> `__file__`, so the repo works at any clone location. **Always run from the repo root**
> (e.g. `python pipeline/experiment.py …`) since data paths are relative.
> For deploying on a new machine see **[`docs/SETUP_NEW_SERVER.md`](docs/SETUP_NEW_SERVER.md)**.

---

## Installation

```bash
conda create -n bscan python=3.10
conda activate bscan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For Mamba support (optional):
```bash
pip install mamba-ssm causal-conv1d
```

---

## Data

The repository ships only small input files; large data is **gitignored** and must be
transferred or regenerated per machine (see [`docs/SETUP_NEW_SERVER.md`](docs/SETUP_NEW_SERVER.md)).

### In the repository

| File | Size | Description |
|------|------|-------------|
| `data/BS_LS_coordinates_final.csv` | 1.2 MB | circRNA/LS junction coordinates (hg19) |
| `data/hg38_exon.bed` | 15 MB | exon annotation for circAtlas external controls |
| `data/human_bed_v3.0/` | 30 MB | circAtlas v3 circRNA database |

### Transfer / regenerate separately (gitignored)

| Path | Size | How to obtain |
|------|-----:|---------------|
| `data/hg19_seq_dict.json` | 2.9 GB | hg19 genome dict — **required for all experiments**; transfer from source machine or Zenodo |
| `fm_embeddings/` | 64 GB | pre-extracted FM hidden states; or regenerate with `pipeline/extract_fm_embeddings.py` |
| `external_data/circatlas/exon_controls/` | 22 GB | external validation set; or rebuild with `pipeline/make_circatlas_exon_controls.py` + `extract_external_fm_embeddings.py` |
| `saved_models/` | 4.6 GB | trained checkpoints; auto-created when training |
| `data/rmsk_hg19.txt.gz` | 141 MB | UCSC RepeatMasker (ALU analysis only): `wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/rmsk.txt.gz -O data/rmsk_hg19.txt.gz` |

> Large files / checkpoints to be released at: **[Zenodo DOI — to be added]**

---

## Quick start

```bash
# Sanity check — forward pass for all models
python pipeline/smoke_models.py

# Train BSCAN-RNA-FM (10 seeds, transcript-grouped split)
python pipeline/experiment.py --model_name bscan_unified_fm --encoder_type rnafm \
    --epochs 100 --seed 42 --device 0 --split_strategy transcript

# Train all FM variants (multi-GPU)
bash scripts/run_gpu0.sh &  # RNA-FM, RNAErnie on GPU 0
bash scripts/run_gpu1.sh &  # RNABERT, RNAMSM on GPU 1

# Pre-extract FM embeddings (required for cached FM mode)
python pipeline/extract_fm_embeddings.py --enc_type rnafm --device 0

# Hard negative augmented training
python pipeline/train_hard_negative_augmented.py \
    --models bscan circcnn --seeds 42 123 315 \
    --eval-negative-modes lower_intron ls_lower_intron

python pipeline/train_hard_negative_augmented_fm.py \
    --enc-type rnafm --seeds 42 123 315 --device 0
```

---

## Reproducing paper results

### Table 1 — Internal validation

```bash
bash scripts/run_publication_classification_comparison.sh
```

### Table 2 — External validation (circAtlas controls)

```bash
# Build exon-length-matched controls from circAtlas v3
python pipeline/make_circatlas_exon_controls.py

# Evaluate all models
python pipeline/evaluate_circatlas_all_baselines.py --device 0
```

### Table 3 — Hard negative 3-tier analysis

```bash
bash scripts/run_hard_negative_pairing.sh
python pipeline/train_hard_negative_augmented.py --models bscan circcnn --seeds 42 123 315
python pipeline/train_hard_negative_augmented_fm.py --enc-type rnafm --seeds 42 123 315
```

### Branch ablation & supplementary analyses (Figures 2–5)

```bash
# Branch ablation (train 6 variants + full reference, then evaluate)
python pipeline/experiment.py --model_name bscan_unified_fm_cnnonly  --split_strategy transcript --device 0 --seed 42
#   ... (also: stemonly, attnonly, nocnn, nostem, noattn, fulltr) ×3 seeds
python analysis/evaluate_ablation.py            # → results/ablation_summary.csv

# Statistical tests, masking, ALU, duplex, leakage controls
python analysis/analyze_statistics.py           # bootstrap CIs
python analysis/analyze_masking.py              # exon/intron masking
python analysis/analyze_alu_repeats.py          # ALU/SINE (needs data/rmsk_hg19.txt.gz)
python analysis/analyze_alu_multiscale.py       # 100/250/500-nt windows
python analysis/analyze_alu_matched_tier2.py    # ALU-density-matched Tier 2
python analysis/analyze_duplex_alpha.py         # duplex α sensitivity
python analysis/analyze_external_b_disjoint.py  # sequence-disjoint leakage control
python analysis/make_external_b_hostgene.py     # host-locus-disjoint (needs pyliftover)

# Regenerate all paper figures
python analysis/make_figures.py                 # → figures/Fig1-5.{png,pdf}
```

Pre-computed result CSVs are in `results/` and `research_results/`; figures in `figures/`.

---

## Trained models

Pre-trained checkpoints available at: **[Zenodo DOI — to be added]**

| Model | File | Internal AUC | External AUC |
|-------|------|:---:|:---:|
| BSCAN-RNA-FM | `bscan_unified_fm_rnafm/` | 0.917 | 0.842 |
| BSCAN-RNAErnie | `bscan_unified_fm_rnaernie/` | 0.917 | 0.838 |
| BSCAN-RNAMSM | `bscan_unified_fm_rnamsm/` | 0.917 | 0.844 |
| BSCAN-RNABERT | `bscan_unified_fm_rnabert/` | 0.916 | 0.846 |

---

## Citation

```bibtex
@article{bscan2026,
  title   = {BSCAN: Back-Splice CircRNA Attention Network for circRNA Junction Detection},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI — to be added]}
}
```
