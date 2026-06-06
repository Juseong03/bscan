# Results

## 3.1 BSCAN-FM achieves state-of-the-art accuracy on internal validation

We evaluated all models on the internal transcript-grouped test split across 10 independent seeds. All four BSCAN-FM variants achieved AUC 0.916–0.917, outperforming every published baseline (**Table 1**). BSCAN-RNAErnie (0.9168 ± 0.0021) and BSCAN-RNAMSM (0.9167 ± 0.0023) ranked highest, with BSCAN-RNA-FM (0.9167 ± 0.0023) and BSCAN-RNABERT (0.9164 ± 0.0020) marginally behind; all four are statistically indistinguishable. The best-performing published baseline, CircCNN, reached 0.8983 ± 0.0039, followed by BSCAN-base (0.9009 ± 0.0019), CircCNN-single (0.8942 ± 0.0021), and CircCNN-tri (0.8892 ± 0.0037). Older architectures performed substantially lower: DeepCircCode (0.8765 ± 0.0035), CircDC (0.8619 ± 0.0058), JEDI (0.8538 ± 0.0028), and CircDeep (0.7899 ± 0.0074).

**Table 1. Internal validation — transcript-grouped split, 10 seeds.**

| Model | Params | AUC | AUPRC | MCC |
|-------|-------:|:---:|:-----:|:---:|
| **BSCAN-RNAErnie** | 2.0M | **0.9168 ± 0.0021** | 0.9394 | 0.7322 |
| **BSCAN-RNAMSM** | 2.0M | 0.9167 ± 0.0023 | 0.9388 | 0.7380 |
| **BSCAN-RNA-FM** | 2.0M | 0.9167 ± 0.0023 | 0.9392 | 0.7330 |
| **BSCAN-RNABERT** | 1.9M | 0.9164 ± 0.0020 | 0.9390 | 0.7359 |
| BSCAN-onehot | 2.0M | 0.9164 ± 0.0024 | 0.9366 | 0.7215 |
| BSCAN-base | 3.9M | 0.9009 ± 0.0019 | 0.9247 | 0.6853 |
| CircCNN | 2.6M | 0.8983 ± 0.0039 | 0.9223 | 0.6792 |
| CircCNN-single | 6.6M | 0.8942 ± 0.0021 | 0.9173 | 0.6713 |
| CircCNN-tri | 1.7M | 0.8892 ± 0.0037 | 0.9133 | 0.6625 |
| CircCNN-double | 8.4M | 0.8878 ± 0.0050 | 0.9110 | 0.6572 |
| DeepCircCode | 0.9M | 0.8765 ± 0.0035 | 0.9031 | 0.6112 |
| CircDC | 0.2M | 0.8619 ± 0.0058 | 0.8961 | 0.6455 |
| JEDI | 2.6M | 0.8538 ± 0.0028 | 0.8844 | 0.5916 |
| CircDeep | 1.2M | 0.7899 ± 0.0074 | 0.7990 | 0.4537 |

**Bold**: BSCAN-FM variants. Params = trainable parameters only; FM backbone excluded.

Notably, all four BSCAN-FM variants achieve this performance with ~2.0M trainable parameters, compared to 2.6M (CircCNN), 6.6M (CircCNN-single), and 8.4M (CircCNN-double), demonstrating that the frozen FM backbone enables highly parameter-efficient task-specific learning.

---

## 3.2 Frozen FM embeddings dramatically improve cross-dataset generalization

To assess generalization beyond the training benchmark, all models were evaluated on the circAtlas exon-length-matched external control set (8,217 samples; **Table 2**). The results reveal a striking divergence between internal accuracy and cross-dataset performance (**Figure 1**).

BSCAN-FM variants maintained external AUC 0.838–0.846, corresponding to an internal-to-external drop of only 7.7–8.6%. By contrast, CNN-only baselines collapsed: CircCNN dropped 21.6% to external AUC 0.704; CircCNN-single and CircCNN-tri dropped ~39% to AUC 0.539; CircDC dropped 41.9% to AUC 0.501; and JEDI dropped 45.3% to AUC 0.467 — effectively performing worse than chance on the external set.

**Table 2. External validation — circAtlas exon-length-matched controls.**

| Model | External AUC | External AUPRC | External MCC | Drop% |
|-------|:------------:|:--------------:|:------------:|:-----:|
| **BSCAN-RNABERT** | **0.8458 ± 0.0136** | 0.7222 | 0.6876 | **7.7%** |
| **BSCAN-RNAMSM** | 0.8443 ± 0.0144 | 0.7266 | 0.7125 | 7.9% |
| **BSCAN-RNA-FM** | 0.8418 ± 0.0143 | 0.7210 | 0.6991 | 8.2% |
| **BSCAN-RNAErnie** | 0.8378 ± 0.0162 | 0.7207 | 0.6836 | 8.6% |
| BSCAN-base | 0.7204 ± 0.0210 | 0.6255 | 0.2940 | 20.0% |
| CircCNN | 0.7041 ± 0.0127 | 0.6151 | 0.2530 | 21.6% |
| BSCAN-onehot | 0.5723 ± 0.0220 | 0.5202 | 0.0382 | 37.6% |
| DeepCircCode | 0.5556 ± 0.0101 | 0.5082 | 0.0343 | 36.6% |
| CircCNN-single | 0.5392 ± 0.0096 | 0.4992 | −0.012 | 39.7% |
| CircCNN-tri | 0.5380 ± 0.0091 | 0.4997 | −0.016 | 39.5% |
| CircCNN-double | 0.5092 ± 0.0081 | 0.4802 | −0.047 | 42.6% |
| CircDC | 0.5010 ± 0.0072 | 0.4759 | −0.075 | 41.9% |
| CircDeep | 0.4940 ± 0.0068 | 0.4927 | −0.010 | 37.5% |
| JEDI | 0.4672 ± 0.0049 | 0.4563 | −0.097 | 45.3% |

External MCC < 0 indicates classification worse than majority-class baseline.

A critical architectural control isolates the contribution of the FM representation from BSCAN's three-branch design: BSCAN-onehot, which uses the identical architecture as BSCAN-FM but replaces the frozen FM with a learnable token embedding, drops 37.6% to external AUC 0.572 — on par with the worst-performing CNN baselines and identical to CircCNN-single and CircCNN-tri in magnitude. This dissociation demonstrates that the generalization advantage of BSCAN-FM arises specifically from the frozen FM representation, not from architectural choices. Bootstrap confidence intervals confirm that all reported external AUC differences are highly significant (10,000 bootstrap resamples, all p < 0.001): BSCAN-RNA-FM vs. BSCAN-onehot: Δ = +0.270, 95% CI [+0.255, +0.285]; BSCAN-RNA-FM vs. CircCNN: Δ = +0.138, 95% CI [+0.126, +0.148].

**Leakage control.** To rule out the possibility that external performance is inflated by overlap between the internal training set and the external validation set, we applied two complementary controls. *(i) Sequence-level.* Only 45/8,217 (0.5%) of external samples shared an exact junction sequence with any training example, and re-evaluating all models on the sequence-disjoint subset (8,172 samples, 99.5%) left every AUC essentially unchanged (Δ ≤ 0.001; e.g., BSCAN-RNA-FM 0.8418 → 0.8408). *(ii) Host-locus-level.* We lifted all internal hg19 junction coordinates to hg38 (pyliftover; 24,220/24,226 lifted) and removed any circAtlas sample whose genomic span overlapped an internal locus (same chromosome and strand, ±100-nt padding). On the resulting host-locus-disjoint subset (5,305 samples, 64.6%; drawn entirely from loci absent from training), all FM models retained strong performance with only a small, uniform decrease (BSCAN-RNA-FM 0.8418 → 0.8182, Δ = −0.024; RNABERT 0.8458 → 0.8227; RNAMSM 0.8443 → 0.8209; RNAErnie 0.8378 → 0.8150), while baselines dropped comparably (CircCNN-tri −0.014, JEDI −0.012). The FM models thus maintain an AUC of ~0.82 — and their large margin over baselines (0.52–0.70) — even on genomic loci entirely disjoint from the training set, confirming that the external generalization is a genuine property of the learned representation rather than an artifact of sequence or host-locus leakage.

---

## 3.3 Branch ablation: the convolutional branch drives generalization

To quantify the contribution of each of BSCAN's three branches (CNN, WC stem map, cross-attention), we trained seven architectural variants on the RNA-FM backbone under identical conditions — transcript-grouped split, 3 seeds (42, 123, 315) — and evaluated each on both the internal test set and the circAtlas external set (**Table 3**, **Figure 2**). All variants share the same frozen FM encoder and projection; only the downstream branch composition differs.

**Table 3. Branch ablation (RNA-FM backbone, transcript-grouped split, 3 seeds).**

| Configuration | Branches | Internal AUC | External AUC | Drop% |
|---------------|----------|:---:|:---:|:---:|
| **Full** | CNN + Stem + Attn | 0.8907 ± 0.005 | 0.8496 ± 0.006 | 4.6% |
| Full − Attn | CNN + Stem | 0.8838 ± 0.002 | **0.8498 ± 0.007** | 3.8% |
| Full − Stem | CNN + Attn | 0.8838 ± 0.003 | 0.8487 ± 0.001 | 4.0% |
| FM + CNN | CNN only | 0.8911 ± 0.007 | 0.8450 ± 0.007 | 5.2% |
| Full − CNN | Stem + Attn | 0.8695 ± 0.012 | 0.7137 ± 0.030 | 17.9% |
| FM + Stem | Stem only | 0.8402 ± 0.005 | 0.7248 ± 0.017 | 13.7% |
| FM + Attn | Attn only | 0.8674 ± 0.010 | 0.6852 ± 0.077 | 21.0% |

Three findings emerge:

1. **The CNN branch is the dominant contributor to cross-dataset generalization.** Removing it (Full − CNN) is the single most damaging ablation, collapsing external AUC from 0.850 to 0.714 (drop 4.6% → 17.9%). Conversely, the CNN branch alone (FM + CNN) nearly reproduces the full model's external performance (0.845 vs. 0.850).

2. **The stem and cross-attention branches are largely redundant given the CNN branch.** Removing either from the full model leaves external AUC essentially unchanged (Full − Stem: 0.849; Full − Attn: 0.850), and individually each generalizes poorly (FM + Stem: 0.725; FM + Attn: 0.685). They contribute marginally to internal accuracy but little additional cross-dataset robustness once the local convolutional branch is present.

3. **Cross-attention alone generalizes worst** (external AUC 0.685, drop 21.0%), indicating that the global pairwise interaction it models is the most susceptible to the exon-composition shortcut, whereas the local motif features captured by the CNN branch are the most transferable.

Together these results identify the local convolutional branch operating on FM-projected features as the principal driver of BSCAN's generalization, with the stem and attention branches providing interpretability (explicit base-pairing maps and junction interaction weights) and small accuracy gains rather than independent generalization benefit. Note that under the transcript-grouped protocol the full model reaches 0.891 internal AUC (lower than the 0.917 reported under the stratified-sample protocol in Table 1), reflecting the harder cross-transcript generalization demanded by grouped splitting.

---

## 3.4 Hard negative probing reveals mechanistic differences across model families

We applied the three-tier hard negative probing framework to all models trained under the standard protocol (**Table 4**, **Figure 3a**). Tier 1 performance replicates the internal validation results. Tier 2 and Tier 3 AUCs reveal substantial heterogeneity across model families.

**Table 4. Hard negative probing results (3 seeds: 42, 123, 315). Standard training, inference only.**

| Model | Tier 1 (Standard) | Tier 2 (LS-intron) | Tier 3 (BS-intron) | T2−T3 |
|-------|:-----------------:|:-----------------:|:------------------:|:------:|
| BSCAN-base | 0.901 | **0.727 ± 0.070** | 0.535 ± 0.022 | +0.192 |
| CircCNN | 0.898 | 0.719 ± 0.051 | 0.508 ± 0.005 | +0.211 |
| CircCNN-tri | 0.889 | 0.715 ± 0.044 | **0.558 ± 0.020** | +0.157 |
| CircCNN-single | 0.894 | 0.671 ± 0.052 | 0.536 ± 0.015 | +0.135 |
| DeepCircCode | 0.877 | 0.663 ± 0.005 | 0.518 ± 0.003 | +0.145 |
| CircCNN-double | 0.888 | 0.657 ± 0.060 | 0.513 ± 0.003 | +0.144 |
| CircDC | 0.862 | 0.638 ± 0.020 | 0.514 ± 0.002 | +0.124 |
| JEDI | 0.854 | 0.549 ± 0.009 | 0.503 ± 0.003 | +0.046 |
| BSCAN-RNABERT | 0.916 | 0.569 ± 0.008 | 0.496 ± 0.002 | +0.073 |
| BSCAN-RNA-FM | 0.917 | 0.475 ± 0.019* | 0.496 ± 0.002 | −0.021 |
| BSCAN-RNAErnie | 0.917 | 0.477 ± 0.029* | 0.492 ± 0.006 | −0.015 |
| BSCAN-RNAMSM | 0.917 | 0.450 ± 0.021* | 0.494 ± 0.005 | −0.044 |

*AUC < 0.50: model assigns higher probability to synthetic hard negatives than to real BS junctions.

**Tier 3 results.** All models trained under the standard protocol show near-chance Tier 3 discrimination (AUC 0.492–0.558), confirming that no model can reliably distinguish the actual intronic sequences of individual BS junctions from those of other BS junctions. The modest above-chance performance of some CNN models (CircCNN-tri: 0.558 ± 0.020; CircCNN-single: 0.536 ± 0.015; BSCAN-base: 0.535 ± 0.022) is consistent with subtle differences in ALU density or orientation between BS junctions, but the signal is too weak for practical discrimination. FM-based models cluster near or below chance (0.492–0.496).

**Tier 2 results.** Performance diverges markedly by model family. All CNN-based models achieve substantial Tier 2 AUC, indicating sensitivity to the distinction between BS-type (ALU-enriched) and LS-type intronic sequences: BSCAN-base attains the highest Tier 2 AUC (0.727 ± 0.070), surpassing CircCNN (0.719 ± 0.051) and CircCNN-tri (0.715 ± 0.044). The superiority of BSCAN-base over single-CNN architectures is consistent with its richer intron feature extraction via the WC stem map and cross-attention branches. By contrast, raw BSCAN-FM models score near or below chance in Tier 2 (BSCAN-RNAMSM: 0.450 ± 0.021; BSCAN-RNA-FM: 0.475 ± 0.019; BSCAN-RNAErnie: 0.477 ± 0.029), indicating that the FM classifier assigns higher scores to BS-exon + LS-intron hybrids than to some real BS junctions — the hallmark of a complete exon classifier that ignores intronic content.

**T2−T3 gap.** The T2−T3 gap quantifies the ability to distinguish intron origin (BS-type vs. LS-type) independently of individual intron specificity. CNN models exhibit consistently positive gaps (+0.046 to +0.211), confirming that they use intron-type information as a classification cue. FM models show near-zero or negative gaps (−0.044 to +0.073), with BSCAN-RNAMSM showing the most negative gap (−0.044), indicating the strongest exon dominance.

**Extended probing: upper intron and both introns.** To test whether the Tier 2/3 patterns generalize beyond lower-intron replacement, we conducted two additional probing modes: upper intron swap (replacing only the upper intronic flank) and both introns swap (replacing both flanks simultaneously). Across both modes, the model-family pattern was preserved: CNN-based models showed small but consistent above-chance performance (upper: 0.507–0.516; both: 0.522–0.553), while FM models remained near or below chance (upper: 0.497–0.501; both: 0.494–0.499). The consistency of this pattern across three intron replacement strategies — lower only, upper only, and both — strengthens the conclusion that the Tier 2 discrimination signal in CNN models reflects a general property of BS-type intronic sequences rather than a sequence specific to the lower flank.

---

## 3.5 Intron-pairing specificity is learnable but FM embeddings suppress it

To determine whether the near-chance Tier 3 performance reflects a fundamental limitation of sequence data or merely a consequence of task design, we trained models with hard-negative augmented supervision (hnaug; **Table 5**, **Figure 3b**).

**Table 5. Hard negative augmented training results (3 seeds, 42/123/315). Early stopping on 0.5×AUC + 0.5×HN-AUC.**

| Model | Tier 1 AUC | Tier 2 AUC | Tier 3 AUC | ΔTier 1 | ΔTier 3 |
|-------|:----------:|:----------:|:----------:|:-------:|:-------:|
| BSCAN-base (standard) | 0.901 | 0.727 | 0.535 | — | — |
| **BSCAN-hnaug** | **0.872 ± 0.004** | **0.901 ± 0.005** | **0.843 ± 0.007** | −0.029 | **+0.308** |
| CircCNN (standard) | 0.898 | 0.719 | 0.508 | — | — |
| **CircCNN-hnaug** | **0.872 ± 0.010** | **0.897 ± 0.005** | **0.836 ± 0.009** | −0.026 | **+0.328** |
| BSCAN-FM (standard) | 0.917 | 0.475 | 0.496 | — | — |
| **BSCAN-FM-hnaug** | 0.909 ± 0.002 | 0.533 ± 0.033 | 0.509 ± 0.008 | −0.008 | +0.013 |

Hard-negative augmented training produced dramatically different outcomes depending on encoder type. For one-hot models, hnaug training increased Tier 3 AUC from near-chance to 0.843 (BSCAN-hnaug, +0.308) and 0.836 (CircCNN-hnaug, +0.328), demonstrating conclusively that intron-pairing specificity is learnable from sequence data alone when models are explicitly incentivized to acquire it. The Tier 2 AUC simultaneously reached 0.897–0.901, confirming that both intron-type and intron-specificity discrimination are accessible to sequence-based models under appropriate supervision. The cost was a modest reduction in standard classification accuracy (−0.026 to −0.029 in Tier 1 AUC), a favorable trade-off if intron discrimination is the primary objective.

For FM-based models, hnaug training produced negligible improvement in Tier 3 AUC (0.509 vs. 0.496 standard; +0.013) despite identical training protocol. This near-null result — in contrast to the +0.30 improvement in one-hot models — indicates that frozen FM embeddings suppress intron-discriminative signals structurally: the downstream projection and attention layers cannot recover intron-specific features from FM representations regardless of the training objective. The FM backbone's exon-dominant encoding constitutes an irreversible information bottleneck for intron specificity.

---

## 3.6 Sequence masking reveals intronic contribution in all model families

To directly assess whether trained models depend on exonic or intronic sequence content, we evaluated each model on the internal test set under five masking conditions: full input (baseline), exon-masked (exon positions zeroed), intron-masked (intron positions zeroed), upper-intron-masked, and lower-intron-masked (**Table S3**).

Across all models, intron masking caused a substantially larger AUC drop than exon masking:

| Model | Full AUC | Exon masked (Δ) | Intron masked (Δ) |
|-------|:---:|:---:|:---:|
| BSCAN-FM | 0.977 | −0.021 | **−0.080** |
| BSCAN-base | 0.942 | −0.034 | **−0.074** |
| CircCNN | 0.926 | −0.038 | **−0.084** |

Values are means over seeds 42, 123, 315 on the transcript-grouped internal test split.

These results indicate that sequence information at the intronic positions — which include the branch point, polypyrimidine tract, and GT-AG splice site dinucleotides — is essential for all models. For BSCAN-FM, lower-intron masking caused a larger drop (−0.070) than upper-intron masking (−0.009), consistent with lower-intron positions (including the canonical 3′ splice site AG) being the primary classification-relevant region.

An important caveat applies to the FM model: because FM embeddings are computed in a fully context-aware manner (each position's representation depends on the whole sequence), zeroing specific positions' embeddings does not cleanly isolate sequence-level exon or intron information — contextual dependencies between positions are already encoded in every token's representation. The masking analysis is therefore more interpretable for the CNN and BSCAN-base models (which use position-independent one-hot encoding) than for FM models. Future work with FM layer-wise probing would better isolate exon-specific versus intron-specific representational content.

---

## 3.7 Thermodynamic duplex energy provides an orthogonal signal

Combining the BSCAN-FM classification score with ViennaRNA duplex folding energy in logit space (α = 0.2) consistently improved both external AUC and AUPRC across all four FM variants (**Table 6**); an α sweep (**Figure 5**) confirms α = 0.2 is near-optimal for all four.

**Table 6. Effect of thermodynamic duplex combination on external validation (circAtlas, 10 seeds).**

| Model | AUC (raw) | AUC (+duplex) | ΔAUC | AUPRC (raw) | AUPRC (+duplex) | ΔAUPRC |
|-------|:---------:|:-------------:|:----:|:-----------:|:---------------:|:------:|
| BSCAN-RNA-FM | 0.8418 | 0.8449 | +0.0031 | 0.7210 | 0.7375 | +0.0165 |
| BSCAN-RNAErnie | 0.8378 | 0.8412 | +0.0034 | 0.7207 | 0.7349 | +0.0142 |
| BSCAN-RNAMSM | 0.8443 | 0.8475 | +0.0032 | 0.7266 | 0.7438 | +0.0172 |
| BSCAN-RNABERT | 0.8458 | 0.8478 | +0.0020 | 0.7222 | 0.7333 | +0.0111 |

AUC improvements ranged from +0.002 to +0.003, and AUPRC improvements from +0.011 to +0.017, consistently across all four FM variants. Bootstrap confidence intervals confirm that all improvements are statistically significant (paired bootstrap CI, 10,000 iterations; all p < 0.001; e.g., BSCAN-RNA-FM: 95% CI [+0.0026, +0.0037]). The duplex combination was also beneficial in the hard negative Tier 3 evaluation, increasing AUC from 0.492–0.496 (raw) to 0.502–0.533 (+0.005 to +0.037), confirming that thermodynamic intron complementarity provides signal orthogonal to the learned sequence classifier. Threshold-based metrics (MCC, accuracy) were unaffected, consistent with logit rescaling being a ranking operation. An α sensitivity analysis (α = 0–1.0) confirmed that α = 0.2 is near-optimal for all four FM variants, with consistent improvement in the range α = 0.1–0.5 and performance decline at α = 1.0 (Supplementary Figure S2).

---

## 3.8 RepeatMasker analysis: ALU enrichment is significant but limited within 100-nt flanks

To directly quantify ALU/SINE repeat enrichment in BS vs. LS intronic flanks, we annotated all 24,216 junctions using hg19 RepeatMasker coordinates (1.12M ALU, 1.77M SINE intervals) with bisect-based interval queries (**Figure 4**).

| | BS (n=12,105) | LS (n=12,111) | p-value |
|---|:---:|:---:|:---:|
| Has any ALU (either 100-nt flank) | **8.7%** | 7.1% | 9.1×10⁻⁷ |
| Has inverted ALU pair (both flanks) | **0.55%** | 0.27% | — |
| Mean ALU coverage | **1.94%** | 1.62% | 9.1×10⁻⁷ |
| Mean SINE coverage | **3.02%** | 2.85% | 5.2×10⁻³ |

*Mann-Whitney U (one-sided: BS > LS).*

BS junctions showed significantly higher ALU and SINE content at all three window sizes (**Table S4**):

| Window | BS has ALU | LS has ALU | BS inv. pair | LS inv. pair | p (MW) |
|--------|:---:|:---:|:---:|:---:|:---:|
| 100 nt | 8.7% | 7.1% | 0.55% | 0.27% | 9×10⁻⁷ |
| 250 nt | 23.1% | 19.3% | 3.02% | 1.36% | 3×10⁻¹⁴ |
| 500 nt | **40.8%** | 36.0% | **9.25%** | 5.17% | 7×10⁻¹⁸ |

The ALU enrichment strengthens with larger windows, confirming that ALU elements are distributed across the full intronic flank rather than concentrated near splice sites. At 500 nt, 40.8% of BS junctions have at least one ALU flank (vs. 36.0% in LS), and 9.25% have an inverted ALU pair in both flanks (vs. 5.17% in LS) — a 1.8× enrichment consistent with the established ALU-driven back-splicing model (Jeck et al., 2013). When ALU is present, it occupies ~22% of the relevant window on average.

To directly test whether ALU density drives CNN Tier 2 discrimination, we conducted an **ALU-density-matched Tier 2** analysis: for each BS junction, we selected LS donors with similar ALU coverage in the lower intron (within ±5%). If ALU density were the primary discriminative signal, ALU-matched Tier 2 should show substantially lower AUC than standard Tier 2. The results were:

| Model | Standard Tier 2 | ALU-matched Tier 2 | Δ |
|-------|:---:|:---:|:---:|
| BSCAN-base | 0.735 ± 0.065 | 0.735 ± 0.061 | +0.001 |
| CircCNN | 0.721 ± 0.041 | 0.724 ± 0.043 | +0.002 |
| BSCAN-FM | 0.478 ± 0.028 | 0.479 ± 0.031 | +0.000 |

ALU-matching produced essentially no change in Tier 2 AUC for any model, demonstrating that **ALU density is not the primary driver of CNN Tier 2 discrimination**. Despite statistically significant ALU enrichment in BS vs. LS introns (8.7% vs. 7.1%, p=9×10⁻⁷), the CNN Tier 2 signal arises from other features that co-vary with BS-type intron identity — likely a combination of splice site strength, polypyrimidine tract composition, or broader repeat element patterns beyond the 100-nt window. Multi-scale analyses (250–500 nt) would help characterize these features further.

---

## 3.9 A-rich motif over-reliance identifies biologically invalid classifiers

De novo motif analysis assessed the degree to which each model's predictions co-vary with polyadenylation-associated A-rich sequence motifs (ATAAA, TAAAA, AATAAA, etc.) that are biologically unrelated to back-splice site selection. A-rich overreliance risk scores were near zero for all BSCAN-FM variants (0.001–0.007) and for BSCAN-base (0.012) and CircCNN (0.006), indicating that these models do not exploit polyadenylation signals. By contrast, several CNN baselines showed elevated risk: CircCNN-single (0.026), CircCNN-tri (0.046), CircDC (0.082), and JEDI (0.251). JEDI's high risk score is particularly notable: the k-mer encoding used by JEDI appears to capture polyadenylation signal hexamers as high-weight features, a spurious correlation that partly explains its poor external generalization (external AUC 0.467, drop 45.3%). The A-rich risk scores and the external generalization drops are positively correlated across models, providing orthogonal biological evidence that the two measures capture the same underlying exon-composition bias.

---

## 3.10 Integrated biological profile

Combining external generalization (weight 0.35), A-rich motif safety (0.25), thermodynamic duplex gain (0.25), and hard-negative robustness (0.15) into a composite biological profile score, the BSCAN-FM+duplex variants occupy the top four positions (**Table 7**). BSCAN-RNA-FM+duplex ranks first (profile score 0.942), followed by BSCAN-RNAErnie+duplex (0.940), BSCAN-RNAMSM+duplex (0.912), and BSCAN-RNABERT+duplex (0.892). BSCAN-base and CircCNN follow at ranks 9 and 10 (profile scores 0.507 and 0.504), and no purely CNN-based baseline exceeds 0.51. CircCNN-tri, despite its competitive Tier 2 and Tier 3 performance, ranks 11th (0.391) owing to its elevated A-rich risk (0.046) and near-random external AUC (0.538).

**Table 7. Integrated biological profile (top models).**

| Rank | Model | Profile Score | Ext AUC | A-rich Risk | Duplex ΔAUPRC | Tier 3 AUC |
|:----:|-------|:-------------:|:-------:|:-----------:|:-------------:|:----------:|
| 1 | **BSCAN-RNA-FM + duplex** | **0.942** | 0.845 | 0.001 | +0.017 | 0.508 |
| 2 | **BSCAN-RNAErnie + duplex** | 0.940 | 0.841 | 0.005 | +0.014 | 0.533 |
| 3 | **BSCAN-RNAMSM + duplex** | 0.912 | 0.847 | 0.007 | +0.017 | 0.509 |
| 4 | **BSCAN-RNABERT + duplex** | 0.892 | 0.848 | 0.007 | +0.011 | 0.502 |
| 9 | BSCAN-base | 0.507 | 0.720 | 0.012 | — | 0.535 |
| 10 | CircCNN | 0.504 | 0.704 | 0.006 | — | 0.508 |
| 11 | CircCNN-tri | 0.391 | 0.538 | 0.046 | — | 0.558 |
| 14 | JEDI | 0.071 | 0.467 | 0.251 | — | 0.503 |

Weights: external generalization 0.35, A-rich motif safety 0.25, duplex channel 0.25, hard-negative robustness 0.15.

On the basis of this analysis, we recommend **BSCAN-RNA-FM + duplex** as the default model for cross-context circRNA back-splice site prediction, offering the best combination of cross-dataset generalization (external AUC 0.845), thermodynamic grounding, biological feature safety, and parameter efficiency (~2.0M trainable parameters).
