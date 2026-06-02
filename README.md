# BSCAN: Back-Splice Circuit Attention Network

A deep learning framework for circRNA back-splicing site (BS) detection and expression prediction. BSCAN integrates frozen RNA foundation model embeddings with cross-attention over paired junction sequences to achieve state-of-the-art performance and generalization.

---

## Overview

circRNA biogenesis depends on back-splice junctions (BSJ), where a downstream 5' splice site joins to an upstream 3' splice site. BSCAN jointly encodes the upper and lower intronic flanks around each junction through a three-branch architecture:

1. **FM branch**: frozen RNA foundation model embeddings (RNA-FM, RNAErnie, RNABERT, RNA-MSM)
2. **Stem branch**: 2D convolutional map over WC/wobble base-pair probability matrix
3. **Cross-attention**: upper × lower junction interaction

### Key findings

| Model | Internal AUC | External AUC | Drop |
|-------|:---:|:---:|:---:|
| **BSCAN-RNA-FM** | 0.917 | 0.842 | **8%** |
| BSCAN-base (CNN only) | 0.901 | 0.720 | 20% |
| CircCNN | 0.898 | 0.704 | 22% |
| CircCNN-single/tri | 0.889–0.894 | 0.538–0.539 | ~39% |

The ~8% external drop for BSCAN-FM vs. 39–45% for CNN-only baselines demonstrates that frozen RNA foundation model embeddings provide representations that generalize across genomic contexts.

**Hard negative analysis** (3-tier probe) reveals:
- Standard training: all models near-chance (AUC ≈ 0.50) at intron-specific discrimination
- Hard negative augmented (one-hot): Tier3 AUC 0.84 — intron pairing signal IS learnable
- Hard negative augmented (FM): Tier3 AUC 0.51 — FM embeddings suppress intron signals structurally

---

## Repository structure

```
bscan/
├── data/
│   ├── BS_LS_coordinates_final.csv   # circRNA/LS junction coordinates (hg19)
│   ├── hg19_seq_dict.json            # pre-extracted sequences (see Data section)
│   ├── hg38_exon.bed                 # exon annotation for external controls
│   └── human_bed_v3.0/              # circAtlas v3 circRNA database
├── models/
│   ├── bscan_unified.py             # BSCANUnified (main model, supports all FM types)
│   ├── bscan_seq.py / _lite.py      # shared token encoder variants
│   ├── bscan_v2.py                  # CircCNNATT + stem branch
│   ├── bscan_mamba_xattn.py         # Mamba SSM + cross-attention
│   ├── bscan_region_interact.py     # region-token compression
│   ├── classifier.py                # shared MLP head
│   ├── transformer.py               # multi-head attention
│   ├── mamba.py / mamba2.py         # Mamba state-space model
│   ├── circCNN.py / circCNNSingle.py / circCNNtri.py / circCNNDouble.py
│   ├── deepCircCode.py, circDeep.py, circNet.py, jedi.py, circDC.py
│   └── *_regression.py              # regression variants
├── dataloader.py                    # DataSetPrep, circData_* Dataset classes
├── dataloader_regression.py
├── trainer.py                       # Trainer with early stopping, model registry
├── trainer_regression.py
├── utils.py
├── experiment.py                    # main classification training entry point
├── experiment_regression.py
├── evaluate_hard_negative_pairing.py
├── train_hard_negative_augmented.py # hnaug training for one-hot models
├── train_hard_negative_augmented_fm.py  # hnaug training for FM models
├── extract_fm_embeddings.py         # pre-extract FM embeddings to disk
├── make_circatlas_exon_controls.py  # build external validation set
├── evaluate_circatlas_all_baselines.py
├── smoke_models.py                  # quick forward-pass sanity check
├── scripts/                         # shell scripts for full experiment sweeps
└── results/                         # pre-computed paper result CSVs
```

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

### Required files

| File | Size | Description |
|------|------|-------------|
| `data/BS_LS_coordinates_final.csv` | 1.2 MB | Included in repo |
| `data/hg19_seq_dict.json` | 2.9 GB | hg19 pre-extracted sequences |
| `data/hg38.json` | 3.1 GB | hg38 pre-extracted sequences |
| `data/seq_dict/` | 5.0 GB | Per-chromosome sequence dictionaries |

> Large files available at: **[Zenodo DOI — to be added]**

### Genome reference

hg19 FASTA is required to build `hg19_seq_dict.json` from scratch:

```bash
# Download hg19 and extract junction sequences
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
python extract_sequences.py --genome hg19.fa --coords data/BS_LS_coordinates_final.csv
```

---

## Quick start

```bash
# Sanity check — forward pass for all models
python smoke_models.py

# Train BSCAN-RNA-FM (10 seeds, transcript-grouped split)
python experiment.py --model_name bscan_unified_fm --encoder_type rnafm \
    --epochs 100 --seed 42 --device 0 --split_strategy transcript

# Train all FM variants (multi-GPU)
bash scripts/run_gpu0.sh &  # RNA-FM, RNAErnie on GPU 0
bash scripts/run_gpu1.sh &  # RNABERT, RNAMSM on GPU 1

# Pre-extract FM embeddings (required for cached FM mode)
python extract_fm_embeddings.py --enc_type rnafm --device 0

# Hard negative augmented training
python train_hard_negative_augmented.py \
    --models bscan circcnn --seeds 42 123 315 \
    --eval-negative-modes lower_intron ls_lower_intron

python train_hard_negative_augmented_fm.py \
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
python make_circatlas_exon_controls.py

# Evaluate all models
python evaluate_circatlas_all_baselines.py --device 0
```

### Table 3 — Hard negative 3-tier analysis

```bash
bash scripts/run_hard_negative_pairing.sh
python train_hard_negative_augmented.py --models bscan circcnn --seeds 42 123 315
python train_hard_negative_augmented_fm.py --enc-type rnafm --seeds 42 123 315
```

Pre-computed results are available in `results/`.

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
  title   = {BSCAN: Back-Splice Circuit Attention Network for circRNA Junction Detection},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI — to be added]}
}
```
