# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BSCAN (Back-Splice CircRNA Attention Network) is a deep learning framework for circRNA back-splicing site (BS) detection. It classifies paired upper/lower intronic junction sequences as BS (circRNA-producing) or linear-splice (LS) controls using a three-branch architecture: RNA foundation model (FM) embeddings + CNN + stem-map cross-attention.

## Environment setup

```bash
conda create -n bscan python=3.10
conda activate bscan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# Optional Mamba support:
pip install mamba-ssm causal-conv1d
```

Required large data files (not in repo — from Zenodo):
- `data/hg19_seq_dict.json` (2.9 GB) — pre-extracted hg19 junction sequences
- `data/hg38.json` (3.1 GB)
- `data/seq_dict/` (5.0 GB) — per-chromosome sequence dicts

## Common commands

```bash
# Forward-pass sanity check for all models (no data required beyond small batches)
python pipeline/smoke_models.py

# Train BSCAN-RNA-FM (recommended model) with transcript-grouped split
python pipeline/experiment.py --model_name bscan_unified_fm --encoder_type rnafm \
    --epochs 100 --seed 42 --device 0 --split_strategy transcript

# Quick test with subsampled data
python pipeline/experiment.py --model_name bscan_unified_fm --max_samples 200 --epochs 5 --device 0

# Pre-extract FM embeddings to disk (required before using cached FM mode)
python pipeline/extract_fm_embeddings.py --enc_type rnafm --device 0

# Hard negative augmented training
python pipeline/train_hard_negative_augmented.py --models bscan circcnn --seeds 42 123 315
python pipeline/train_hard_negative_augmented_fm.py --enc-type rnafm --seeds 42 123 315 --device 0

# External validation
python pipeline/make_circatlas_exon_controls.py
python pipeline/evaluate_circatlas_all_baselines.py --device 0

# Full paper experiment sweeps
bash scripts/run_bscan_comparison.sh
```

## Architecture

### Data flow

`DataSetPrep` (`dataloader.py`) loads `BS_LS_coordinates_final.csv` (hg19 coordinates) + `hg19_seq_dict.json` (sequences), extracts junction + intronic flanks, and returns upper/lower one-hot or tokenized tensors. Three split strategies: `sample` (stratified), `transcript` (group by transcript ID), `chromosome` (leave-one-chromosome-out).

`experiment.py` orchestrates: DataSetPrep → split → Dataset → Trainer → train/evaluate.

`Trainer` (`trainer.py`) owns the model registry (maps string names to classes), optimizer, early stopping, and metric logging. It writes checkpoints to `./saved_models/` and logs to `./logs/`.

### Model: BSCANUnified (`models/bscan_unified.py`)

The main model supports four operating modes controlled by `encoder_type` and `use_cached`:

| Mode | `encoder_type` | `use_cached` | Input |
|------|----------------|--------------|-------|
| One-hot | `onehot` | — | token indices |
| Live FM | `rnafm`/`rnaernie`/etc. | `False` | token IDs → frozen FM → project |
| Cached FM | `rnafm`/`rnaernie`/etc. | `True` | pre-extracted hidden states from `fm_embeddings/` |
| Embed-only | — | — | `BSCANUnifiedEmbedOnly` subclass |

Three branches fused via concatenation → `Classifier` MLP:
- **Branch A (CNN)**: `Conv1d` on projected FM features; upper and lower processed independently
- **Branch B (Stem)**: Hard Watson-Crick base-pair map (upper intron × lower_RC intron) → `Conv2d`; requires `upper_oh` and `lower_rc_oh` one-hot tensors
- **Branch C (Cross-attention)**: upper queries against lower keys/values across `n_attn_layers`

Optional `adapter_type` (`cnn` or `mamba`) refines FM embeddings after projection before branching.

### Baseline models

All baselines are in `models/` and registered in `Trainer.define_model()`. Published baselines include DeepCircCode, CircDeep, CircNet, JEDI, CircCNN, CircDC, CircCNNSingle/Double/Tri. Internal ablation variants: `bscan_v2`, `bscan_seq*`, `bscan_mamba_xattn`, `bscan_region_interact`, `bscan_region_stem`, `bscan_plus`.

### Dataset classes (`dataloader.py`)

- `circData_single` — concatenated upper+lower (one tensor)
- `circData_double` — (upper, lower) pair
- `circData_triple` — (upper, lower, lower_rc) for models needing explicit RC
- `circData_cached_fm` — (keys, labels, fm_name, upper_oh, lower_rc_oh); loads FM embeddings lazily from `fm_embeddings/<fm_name>/<key>.pt`
- `circData_triple_oh` — (upper_tokens, lower_tokens, lower_rc_tokens, upper_oh, lower_rc_oh)

### Regression variants

Parallel set of files for expression-level prediction: `dataloader_regression.py`, `trainer_regression.py`, `experiment_regression.py`, and `models/*_regression.py`.

## Repository layout

- `core/` — importable library: `dataloader.py`, `trainer.py`, `utils.py`, regression variants, `RCSFinder.py`, `CalPPM.py`
- `pipeline/` — runnable entry-point scripts: `experiment.py`, `train_*`, `evaluate_*`, `extract_*`, `make_circatlas_exon_controls.py`, `smoke_models.py`
- `models/` — model definitions (stays at repo root)
- `analysis/` — paper-supplement analysis scripts (`analyze_*`, `evaluate_ablation.py`, `make_figures.py`)
- `scripts/` — shell sweep runners

Scripts in `pipeline/` and `analysis/` carry a small `sys.path` shim that adds the repo root + `core/` + `pipeline/`, so bare imports (`from dataloader import …`) resolve regardless. **Always invoke from the repo root** (e.g. `python pipeline/experiment.py …`) because data paths are relative.

## Key conventions

- All experiments run from the repo root; data paths are relative (`./data/`, `./fm_embeddings/`, `./saved_models/`).
- `junction_bps=100` is the default window on each side of the splice site; `flanking_bps` controls wider intronic context.
- FM embeddings are stored per-sample as `.pt` files under `fm_embeddings/<enc_type>/<key>.pt` and loaded by `circData_cached_fm`.
- The `Trainer.define_model()` method is the single place that maps `model_name` strings to constructors — add new models there and in `experiment.py`'s `model_name` choices.
