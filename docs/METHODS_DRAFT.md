# Methods

## 2.1 Dataset and sequence extraction

The dataset consists of 24,216 human circRNA back-splice junction (BS) and matched linear-splice junction (LS) pairs, with near-equal class balance (12,105 BS, 12,111 LS) derived from hg19 genome coordinates. For each junction, we extract two 200-nucleotide composite sequences centered at the splice site:

- **Upper sequence**: [100 nt upper intronic flank] + [100 nt downstream exon]
- **Lower sequence**: [100 nt upstream exon] + [100 nt lower intronic flank]

This window (junction_bps = 100) captures canonical splice site signals (GT-AG dinucleotides at intron boundaries) plus local exonic context. The separation into upper and lower sequences preserves the strand orientation and positional relationship of the two junction arms, which is critical for the base-pairing stem analysis described below.

## 2.2 Data splitting

All BSCAN-FM models and the majority of baselines are evaluated using a **transcript-grouped split**, in which all junctions sharing a transcript identifier are assigned exclusively to one partition. This prevents information leakage between training and test sets that would arise from correlated splicing patterns within the same transcript. Splits are constructed with `StratifiedGroupKFold` (n_splits=5, shuffle=True) to produce a train-valid/test partition, followed by a second `StratifiedGroupKFold` (n_splits=4) to divide the train-valid portion into train/valid. The intended nominal ratio is approximately 60/20/20, but the realized proportions vary across seeds due to transcript group size constraints; for the primary seed (42), the resulting partition sizes are 15,122 / 4,161 / 4,933 (train/valid/test), corresponding to approximately 62% / 17% / 20% of the full dataset. Stratification ensures equal BS:LS ratios within each partition.

All results are averaged over 10 independent random seeds (seeds 42, 123, 315, 777, 1004, 2024, 2025, 2026, 3407, 9001), each producing a distinct transcript-grouped split and random model initialization.

## 2.3 RNA foundation model embeddings

Four pre-trained RNA language models were used as frozen sequence encoders, accessed via the multimolecule library (Taylor et al., 2024):

| Model | Architecture | Hidden dim | Layers | Total params |
|-------|---|:---:|:---:|:---:|
| RNA-FM | BERT-style | 640 | 12 | 99.5M |
| RNAErnie | BERT-style | 768 | 12 | 86.1M |
| RNABERT | BERT-style | 120 | 6 | 0.5M |
| RNA-MSM | MSA-style | 768 | 10 | 95.9M |

All FM parameters are frozen throughout training. FM embeddings were pre-extracted for the entire dataset and stored as per-sample `.pt` files prior to training, to avoid repeated FM inference costs. For each junction, the FM produces a per-nucleotide hidden state matrix of shape [L, d_FM] (L = 200, d_FM as above). CLS and SEP tokens are included in the raw FM output and cropped prior to downstream processing.

## 2.4 BSCAN architecture

### 2.4.1 Input representation

Each input to BSCAN-FM consists of:
- **Upper FM embedding**: [200, d_FM] hidden state for the upper sequence
- **Lower FM embedding**: [200, d_FM] hidden state for the lower sequence
- **Lower-RC FM embedding**: [200, d_FM] hidden state for the reverse complement of the lower sequence
- **Upper intron one-hot**: [4, 100] binary matrix for the upper intronic 100 nt
- **Lower intron RC one-hot**: [4, 100] binary matrix for the reverse complement of the lower intronic 100 nt (used only by the stem branch)

FM embeddings are projected to a common d_model = 128-dimensional space via a single linear layer followed by GELU activation (projection layers: 82,048 trainable parameters for RNA-FM).

### 2.4.2 Branch A: 1D convolutional branch

The projected upper and lower embeddings (each [B, 200, 128]) are processed independently through a shared-weight 1D-CNN:

```
Conv1d(128 → 256, kernel=11, padding=5) → BatchNorm1d → ReLU → MaxPool1d(4)
Conv1d(256 → 128, kernel=11, padding=5) → BatchNorm1d → ReLU → AdaptiveAvgPool1d(8)
→ Flatten → [B, 1024]
```

Upper and lower branch outputs are concatenated to yield [B, 2048].

### 2.4.3 Branch B: Watson-Crick stem map

To capture potential intron-intron base-pairing complementarity, we compute a hard Watson-Crick base-pair map between the upper intronic one-hot matrix and the reverse complement of the lower intronic one-hot matrix:

$$\mathbf{S} = \mathbf{U}_{\text{intron}}^\top \cdot \mathbf{L}_{\text{intron-RC}} \in \{0, 1\}^{100 \times 100}$$

where each entry S[i,j] indicates whether position i of the upper intron can form a Watson-Crick (A-U or G-C) base pair with position j of the lower intron's reverse complement. This binary matrix is processed by a 2D-CNN:

```
Conv2d(1 → 16, 3×3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(16 → 32, 3×3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, 3×3, padding=1) → ReLU → AdaptiveAvgPool2d(4)
→ Flatten → [B, 1024]
```

Additionally, the per-row and per-column maxima of **S** are concatenated ([B, 100] + [B, 100]) to capture position-level pairing strength, yielding a total stem branch output of [B, 1224].

### 2.4.4 Branch C: Cross-attention

Upper projected features serve as queries and lower projected features serve as keys and values in a two-layer multi-head attention module (4 heads, d_model=128, dropout=0.3). Each layer applies:

$$\text{query}^{(t+1)} = \text{LayerNorm}(\text{query}^{(t)} + \text{MHA}(\text{query}^{(t)}, \text{lower}, \text{lower}))$$

followed by a position-wise FFN (d_model → 2×d_model → d_model). The output is globally average-pooled to yield [B, 128].

### 2.4.5 Fusion and classification

Outputs of all three branches are concatenated into a [B, 3400]-dimensional vector (2048 + 1224 + 128), then passed through a LayerNorm and a two-layer MLP classifier:

```
LayerNorm(3400) → Linear(3400 → 256) → GELU → Dropout(0.3) → Linear(256 → 2)
```

Total trainable parameters: ~1.97M (RNA-FM), ~1.90M (RNABERT), ~1.99M (RNAErnie, RNA-MSM). FM backbone parameters are excluded from this count.

## 2.5 Baseline models

Eight published models were re-implemented and trained under identical data split and evaluation conditions:

| Model | Architecture | Reference |
|---|---|---|
| DeepCircCode | 1D-CNN on concatenated junction | Pan et al. (2019) |
| circDeep | CNN + conservation + structure | Dang et al. (2020) |
| CircNet | — | — |
| JEDI | k-mer encoding + attention | Park et al. (2021) |
| CircCNN | Dual-branch 1D-CNN | Liu et al. (2022) |
| CircCNN-single | Single-input CNN | Wang et al. (2024) |
| CircCNN-double | Concatenated-input CNN | Wang et al. (2024) |
| CircCNN-tri | Dual-CNN + RCM features | Wang et al. (2024) |
| CircDC | Dual-channel CNN | Chen et al. (2024) |

**BSCAN-base** refers to the full three-branch BSCAN architecture (CNN + stem + cross-attention) trained with learnable one-hot encoding rather than FM embeddings, serving as an architectural ablation control.

**BSCAN-onehot** uses tokenized integer indices with a learnable embedding table (vocabulary size 26, d_model=128) as a direct architectural match to BSCAN-FM, replacing frozen FM representations with a fully trainable token embedding. This control isolates the contribution of FM embeddings from architectural choices.

## 2.6 Training procedure

All models were trained with the AdamW optimizer (learning rate 1×10⁻⁴, weight_decay=0), batch size 128, and cross-entropy loss. Training ran for a maximum of 100 epochs for FM-based models and 300 epochs for non-FM baselines, with early stopping: training was halted if validation AUC did not improve for 30 consecutive epochs. The checkpoint with the highest validation AUC was retained and used for all subsequent evaluations. All experiments were run on NVIDIA RTX A6000 GPUs (48 GB VRAM).

## 2.7 External validation set

To evaluate cross-dataset generalization, we constructed an exon-length-matched external control set using circRNA coordinates from the circAtlas v3 database (Liu et al., 2023). Starting from 5,000 candidate circRNAs (sampled from the full human catalogue with `random_state=42`), we extracted 4,119 positives that yielded valid 200-nt BSCAN window sequences after mapping to the hg38 genome.

For each positive, a matched negative was drawn from exon-pair intervals of same-strand overlapping transcripts annotated in Ensembl/hg38 (hg38_exon.bed), selecting the candidate with the smallest relative length difference (within a 50% tolerance). This procedure prevents the length-composition bias that would arise from random exon sampling. A total of 4,098 length-matched negatives were successfully extracted (21 positives could not be matched), yielding 8,217 total evaluation samples.

Critically, the external validation set differs from the internal benchmark in two ways: (1) it is derived from an independent database (circAtlas, not the BS/LS coordinate collection used for training), and (2) the negatives are drawn from different genomic loci, preventing models from exploiting exon-sequence identity as a classification cue.

## 2.8 Hard negative probing framework

To assess whether trained models have internalized intronic pairing specificity rather than exon-composition features, we designed a three-tier hard negative evaluation:

- **Tier 1 (Standard)**: Original BS vs. LS junction sequences (equivalent to the internal test set).
- **Tier 2 (LS-intron swap)**: A real BS junction's exon sequences are retained, but its lower intronic 100 nt are replaced by the lower intronic 100 nt of a randomly selected LS junction. This tests whether a model can discriminate BS-type (ALU-enriched) introns from LS-type introns while holding exon content constant.
- **Tier 3 (BS-intron swap)**: As in Tier 2, but the replacement intron is drawn from a different BS junction. If a model scores such a hybrid lower than a real BS junction, it has learned that individual BS intron sequences are distinguishable — a strictly harder task, since both sources are circRNA-forming introns.

In all tiers, evaluation pairs consist of an equal number of real BS positive examples (real upper + real lower sequences) and synthetic hard negatives (real upper + swapped lower). AUC > 0.50 indicates that the model assigns higher probability to real junctions; AUC ≈ 0.50 indicates no discrimination; AUC < 0.50 indicates the model scores the synthetic negatives higher than real junctions (exon-dominant behavior when the exon is preserved from a real BS junction). Evaluations used three random seeds (42, 123, 315) per tier.

The T2−T3 gap (Tier2 AUC − Tier3 AUC) quantifies a model's ability to use intron-type identity (BS-type vs. LS-type, driven by ALU density differences) as opposed to individual intron specificity.

## 2.9 Hard negative augmented training

To determine whether intron-pairing specificity is learnable from sequence data, we trained models with explicit hard-negative supervision. In each training epoch, a batch of Tier-3-style synthetic negatives (BS exon + randomly swapped BS intron, label=0) is appended to the standard training data. Early stopping is applied to the joint objective:

$$\text{score} = 0.5 \times \text{AUC}_{\text{standard}} + 0.5 \times \text{AUC}_{\text{hard-negative}}$$

with patience=3 epochs. This formulation prevents collapse to either task (pure standard accuracy at the expense of hard-negative discrimination, or vice versa). Models trained in this manner are denoted with the suffix **-hnaug**. We trained BSCAN-hnaug (BSCAN-base) and CircCNN-hnaug across 3 seeds (42, 123, 315), and BSCAN-FM-hnaug (BSCAN-RNA-FM) under the same protocol.

## 2.10 Thermodynamic duplex combination

Thermodynamic intron complementarity was quantified using ViennaRNA (Lorenz et al., 2011) duplexfold, which computes the minimum free energy of hybridization between the upper intronic sequence and the reverse complement of the lower intronic sequence. The resulting duplex energy E_duplex (kcal/mol) was combined with the model's classification score in logit space:

$$\text{logit}(p_{\text{combined}}) = \text{logit}(p_{\text{model}}) + \alpha \cdot z_{\text{duplex}}$$

where $z_{\text{duplex}} = (- E_{\text{duplex}} - \mu) / \sigma$ is the z-score of the negated duplex energy (more negative energy = stronger pairing → higher score), $\mu$ and $\sigma$ are the mean and standard deviation over all test samples, and $\alpha = 0.2$ was set without validation-set tuning. This procedure was applied at inference time only, using pre-trained model checkpoints without any retraining.

## 2.11 Evaluation metrics

All models were evaluated on AUC (area under the ROC curve), AUPRC (area under the precision-recall curve), Matthews correlation coefficient (MCC), accuracy, and macro-averaged F1. The primary metric for model selection and comparison is AUC. All reported values are means ± standard deviations across independent seeds. For external validation and hard-negative probing, the same set of trained checkpoints was used for inference without any additional training.
