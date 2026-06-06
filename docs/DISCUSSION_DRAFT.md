# Discussion

## 1. BSCAN-FM achieves state-of-the-art accuracy with parameter efficiency

All four BSCAN-FM variants (RNA-FM, RNAErnie, RNAMSM, RNABERT) achieved AUC 0.916–0.917 on the internal transcript-grouped test split across 10 independent seeds, outperforming every published baseline evaluated here: CircCNN (0.898), BSCAN-base (0.901), CircDC (0.862), and JEDI (0.854). Crucially, this performance is achieved with fewer trainable parameters (~2.0M) than most baselines (CircCNN 2.6M, CircCNN-single 6.6M, CircCNN-double 8.4M), demonstrating that the frozen FM backbone provides a high-capacity representation that allows the downstream architecture to remain compact. The consistency across all four RNA foundation models suggests that performance is robust to the choice of FM.

A branch ablation (Results §3.3) clarifies which architectural components drive this performance. Among the three branches, the local convolutional branch is the principal contributor to cross-dataset generalization: the CNN branch alone reproduces nearly the full external AUC (0.845 vs. 0.850), whereas removing it collapses external AUC to 0.714. The WC stem-map and cross-attention branches generalize poorly in isolation (0.725 and 0.685) and are largely redundant once the CNN branch is present — removing either leaves external AUC essentially unchanged. We therefore interpret the stem and attention branches as contributing primarily interpretability (explicit intronic base-pairing maps and junction interaction weights) rather than independent generalization benefit, while the convolutional branch operating on FM-projected features captures the transferable local sequence motifs that underlie BSCAN's robustness.

---

## 2. Exon composition bias drives the internal–external performance gap

The most important finding of this study emerges when internal accuracy is compared against external generalization on the circAtlas exon-length-matched control set:

| Model family | Internal AUC | External AUC | Drop |
|---|:---:|:---:|:---:|
| BSCAN-FM (4 variants) | 0.916–0.917 | 0.838–0.846 | ~8% |
| BSCAN-base | 0.901 | 0.720 | 20% |
| CircCNN | 0.898 | 0.704 | 22% |
| CircCNN-single/tri | 0.889–0.894 | 0.538–0.539 | ~39% |
| CircDC / JEDI | 0.854–0.862 | 0.467–0.501 | 42–45% |

The external validation set removes the exon composition shortcut available in the internal benchmark: internal negatives are linear splice sites drawn from the same transcripts as the circRNA positives, sharing similar exonic sequence context, whereas external negatives are exon-length-matched intervals from unrelated genomic loci. Models that have learned to classify by exon identity rather than by junction-intrinsic signals therefore fail catastrophically on the external set.

The most direct evidence that FM embeddings — not architecture — are the key factor comes from an internal control: BSCAN-onehot, which shares the identical three-branch architecture as BSCAN-FM but uses a learnable token embedding instead of a frozen FM, drops 37.5% on external validation, on par with the worst-performing baselines. This dissociation confirms that frozen RNA FM representations encode a sequence-context-general signal that resists the exon-composition shortcut, while any learnable encoder — regardless of architectural sophistication — can exploit it.

---

## 3. Hard negative probing reveals a fundamental divide between CNN and FM-based learning

To test whether models have learned intronic pairing complementarity — the proposed biological driver of back-splicing — we constructed a three-tier hard negative probe based on systematic intron replacement:

- **Tier 2** (LS-intron swap): real BS exon + linear-splice intron → tests discrimination of BS-type vs. LS-type intronic sequences
- **Tier 3** (BS-intron swap): real BS exon + a different BS junction's intron → tests individual intron specificity

**Tier 2 results reveal a clear model-family ranking** (sorted by Tier2 AUC):

| Model | Tier2 AUC | Tier3 AUC | T2−T3 |
|---|:---:|:---:|:---:|
| BSCAN-base | **0.727** | 0.535 | +0.192 |
| CircCNN | 0.719 | 0.508 | +0.211 |
| CircCNN-tri | 0.715 | 0.558 | +0.157 |
| CircCNN-single | 0.671 | 0.536 | +0.135 |
| DeepCircCode | 0.663 | 0.518 | +0.145 |
| CircCNN-double | 0.657 | 0.513 | +0.144 |
| CircDC | 0.638 | 0.514 | +0.124 |
| JEDI | 0.549 | 0.503 | +0.046 |
| BSCAN-FM (raw) | 0.45–0.57† | 0.492–0.496† | ≤0 |

†Below or barely above chance; raw FM models including BSCAN-RNAErnie/MSM/FM show Tier2 AUC < 0.50.

Notably, BSCAN-base achieves the highest Tier2 AUC (0.727) among all models, surpassing even CircCNN-tri (0.715). BSCAN-base's three-branch architecture (CNN backbone + WC stem map + cross-attention) provides richer intronic feature extraction than a single CNN, even without FM embeddings. By contrast, JEDI's low Tier2 (0.549) suggests that k-mer frequency encoding provides limited access to intron-type discriminative patterns.

**Tier 3 results are near-chance for all standard-trained models** (0.49–0.56). This is biologically expected: circRNA-forming introns are broadly enriched for ALU/SINE repeat elements, making their reverse-complementary sequences compositionally similar across different BS junctions (Jeck et al., 2013; Ivanov et al., 2015; Kramer et al., 2015). The small positive AUC seen in CNN models (0.51–0.56) may reflect subtle differences in ALU density or orientation, but the signal is too weak to be practically meaningful.

**The positive T2−T3 gap** common to all CNN-based models (range +0.12 to +0.21) quantifies the ability to distinguish BS-type from LS-type intronic sequences when exon content is held constant. We investigated this signal using hg19 RepeatMasker annotations: BS junctions showed significantly higher ALU coverage within the 100-nt intronic flanks (8.7% vs. 7.1% with any ALU; Mann-Whitney p = 9×10⁻⁷), consistent with ALU elements contributing to BS-type intron identity. However, 91.3% of BS junctions have no detectable ALU in the 100-nt window, indicating that ALU enrichment alone is insufficient to fully explain the Tier 2 discrimination signal. Other sequence properties — polypyrimidine tract composition, splice site motif strength, or broader ALU elements beyond the 100-nt window — likely contribute alongside ALU content. Multi-scale window analyses (500–3,000 nt) would be required to characterize the full role of ALU elements in this discrimination. FM models exhibit a negative or near-zero T2−T3 gap (≤0.07), indicating that under the standard frozen-embedding setup, these models did not utilize intron-origin information; in three of four FM variants, the gap is negative, confirming exon-dominant scoring.

---

## 4. Hard negative augmented training proves intron specificity is learnable, but FM suppresses it

To determine whether intron-pairing specificity is a fundamental limitation of sequence data or merely a consequence of task design, we trained models with explicit hard negative supervision (hnaug), jointly optimizing standard classification loss and hard-negative discrimination:

| Model | Tier1 AUC | Tier2 AUC | Tier3 AUC |
|---|:---:|:---:|:---:|
| BSCAN-base (standard) | 0.901 | 0.727 | 0.535 |
| **BSCAN-hnaug** | 0.872 | **0.901** | **0.843** |
| CircCNN (standard) | 0.898 | 0.719 | 0.508 |
| **CircCNN-hnaug** | 0.872 | **0.897** | **0.836** |
| BSCAN-FM (standard) | 0.917 | 0.475 | 0.496 |
| **BSCAN-FM-hnaug** | 0.909 | 0.533 | 0.509 |

One-hot models achieve Tier3 AUC 0.84 under hnaug training — a +0.31 improvement over standard training. This conclusively demonstrates that **intron-pairing specificity is learnable from sequence alone**; the signal is present in the data, but standard BS-vs-LS classification does not incentivize its extraction because exon features are sufficient to reach high accuracy.

The trade-off is modest: standard AUC decreases by only 0.026–0.029 while Tier3 AUC increases by 0.31 — a strongly favorable exchange if intron-discriminative features are the goal.

FM models, however, remained near chance even with hnaug (Tier3 0.509 vs. 0.496 standard). This contrast with the +0.31 improvement in one-hot models — under identical training protocol — indicates that, in the tested frozen-embedding setting, intron-specific signals were not readily recoverable by the downstream classifier. Whether this reflects a fundamental property of FM representations, a limitation of the frozen-embedding setup, or an insufficiency of the downstream architecture cannot be determined from the current experiments alone; distinguishing these possibilities would require layer-wise probing of FM activations, partial fine-tuning experiments, or intron-only representation analyses. Provisionally, this pattern is consistent with FM embeddings encoding exon identity as a dominant representational axis that the projection and attention layers we tested could not overcome — an interpretation supported by the parallel finding that FM models show below-chance Tier 2 AUC (negative T2−T3 gap), behavior that cannot arise from random noise alone and suggests active exon-biased scoring. This observation provides a working hypothesis — rather than a confirmed mechanism — for both the generalization advantage of FM and its intron-specificity limitation.

---

## 5. Thermodynamic duplex energy provides an orthogonal signal to learned classifiers

Combining BSCAN-FM classification scores with ViennaRNA duplex folding energy in logit space (`logit(p_combined) = logit(p_model) + 0.2 × zscore(−E_duplex)`) consistently improved external AUC by +0.002–0.003 and AUPRC by +0.011–0.017 across all four FM variants. The improvement was also observed in Tier3 (+0.005 to +0.037), consistent with thermodynamic intron complementarity providing the only intron-specific signal available to FM-based models in that probe.

The modest but robust nature of this improvement suggests that duplex folding energy captures information orthogonal to the sequence-level classifier: it reflects a physical property of RNA secondary structure that is independent of primary sequence features learned by the FM. We recommend reporting AUC and AUPRC when evaluating this combination, as threshold-based metrics (MCC, accuracy) are unaffected by logit rescaling.

---

## 6. A-rich motif over-reliance as a biological validity check

De novo motif analysis identified statistically elevated response to polyadenylation-associated A-rich pentamers (ATAAA, TAAAA, etc.) in several baseline models: JEDI (A-rich risk score 0.251) and CircDC (0.082) show the highest over-reliance, while BSCAN-FM variants are negligible (0.001–0.007) and BSCAN-base is safe (0.012). Polyadenylation signals are biologically unrelated to back-splice site selection, and their presence as classification cues indicates that these models are exploiting non-functional exonic sequence correlates of the training data.

This analysis provides biological sanity evidence that is orthogonal to the external generalization results: both the cross-dataset AUC drop and the A-rich risk score converge on the same models (CircCNN-single, CircCNN-tri, CircDC, JEDI) as the most exon-biased, while BSCAN-FM variants pass both criteria.

---

## 7. Integrated biological profile

We constructed a composite biological profile score by weighting four evaluation axes: external generalization (0.35), A-rich motif safety (0.25), thermodynamic duplex gain (0.25), and hard-negative robustness (0.15). Under this scoring, BSCAN-FM+duplex variants occupy ranks 1–4 and no CNN-only baseline exceeds 0.51. The full scoring table is provided in Supplementary Table S1; the weights were defined prior to model evaluation and reflect our assessment of biological relevance, but the specific values are inherently subjective. We do not recommend using the composite score as the sole basis for model selection; instead, we present it as a multi-dimensional summary, and encourage readers to consult the individual metric tables (Tables 2–5) for their specific use case.

On the basis of the individual metrics — external AUC, AUPRC, motif safety, and duplex gain — **BSCAN-RNA-FM + duplex** (external AUC 0.845) is our recommended default model for cross-context back-splice site prediction.

---

## 8. Limitations and future directions

**Preliminary adapter evidence.** Exploratory experiments with lightweight local (CNN) and sequential (Mamba) adapters on top of frozen FM embeddings suggested reductions in the internal-to-external drop from ~8% to ~5%, but these were conducted under different experimental conditions (sample split, 5 seeds) than the core FM models and are therefore excluded from the main comparison. Confirming this signal under transcript-grouped conditions is a natural next step and could motivate a more systematic study of FM adaptation strategies for splice site prediction.

**CircAtlas as OOD benchmark and leakage control.** All core models have been compared on the circAtlas external set, which functions as an out-of-distribution benchmark. We verified that the external results are not driven by train–test leakage at two levels (Results §3.2): at the sequence level (only 0.5% exact-sequence overlap; sequence-disjoint AUCs unchanged, Δ ≤ 0.001), and at the host-locus level (hg19→hg38 liftOver-based removal of all loci overlapping training; on the 64.6% host-locus-disjoint subset FM models retained AUC ~0.82 with only Δ ≈ −0.024). The consistent ~0.02 decrease across all models on the host-disjoint set — far smaller than the FM-vs-baseline margin — indicates genuine generalization to unseen genomic loci. A separate attempt to build a fully cross-database external set from circBase (positives) and RJunBase (negatives) was abandoned because those databases encode junction coordinates under a convention incompatible with the training data (circBase 81% vs. RJunBase 4% canonical GT-AG, the inverse of internal BS 43%/LS 99.7%), causing systematic prediction inversion; a valid cross-database set would require per-database coordinate re-anchoring to canonical splice sites and is left to future work.

**ALU and Tier 2 interpretation.** RepeatMasker analysis confirmed that ALU enrichment in BS vs. LS intronic flanks is statistically significant (8.7% vs. 7.1% at 100 nt, p = 9×10⁻⁷; 40.8% vs. 36.0% at 500 nt, p = 7×10⁻¹⁸). However, an ALU-density-matched Tier 2 experiment — replacing LS intron donors with donors matched on ALU coverage within ±5% — produced essentially no change in CNN Tier 2 AUC (BSCAN-base: standard 0.735, matched 0.735, Δ = +0.001; CircCNN: 0.721 vs 0.724, Δ = +0.002). This directly demonstrates that **CNN Tier 2 discrimination does not depend primarily on ALU density differences**, despite ALU enrichment being a real genomic feature of BS-producing introns. The CNN models likely exploit other sequence properties that co-distribute with BS-type introns — such as polypyrimidine tract strength, branch point signals, or canonical splice site quality — to achieve Tier 2 discrimination. Characterizing these features through targeted sequence ablations and positional enrichment analysis is a priority for future work.

**FM intron suppression.** The failure of BSCAN-FM-hnaug to recover Tier 3 discrimination is consistent with frozen FM representations being exon-dominant, but does not conclusively establish the mechanism. Layer-wise probing of FM activations, partial fine-tuning of the FM upper layers, and intron-only FM embedding analyses would help determine whether the limitation resides in the FM representation itself or in the frozen-embedding setup.

**Future directions.** The hard negative augmented results demonstrate that intron-pairing specificity is learnable from one-hot sequence models; combining FM representations with hnaug training objectives, partial FM fine-tuning, or explicit intron-level encoders represents a promising direction. Branch ablation experiments (FM+MLP-only, no-CNN branch, no-stem branch, no-attention) would clarify the contribution of each BSCAN component and directly address reviewer questions about architectural necessity. Statistical confirmation of duplex combination improvements (DeLong test, bootstrap CI) would strengthen AUPRC-level claims. Extension to multi-scale intronic windows and RBP motif analysis would further characterize the sequence determinants of back-splice site specificity.

---

## Summary of key claims

| Claim | Evidence |
|---|---|
| BSCAN-FM achieves SoTA internal AUC with fewer parameters | AUC 0.916–0.917 vs. CircCNN 0.898; ~2.0M params vs. up to 8.4M |
| FM embeddings drive generalization; architecture does not | BSCAN-onehot (same architecture, no FM): 37.5% drop vs. FM 8% |
| Intron-type discrimination is architecture-dependent | BSCAN-base Tier2 0.727 (highest); JEDI Tier2 0.549 (lowest) |
| Intron-pairing specificity is learnable but not incentivized | hnaug one-hot: Tier3 0.84; standard: 0.50–0.56 |
| Intron signals not recoverable in tested frozen-FM setting | FM-hnaug Tier3 0.509; mechanism (representation vs. architecture) requires further probing |
| Thermodynamic duplex provides orthogonal signal | External AUC +0.002–0.003; Tier3 +0.005–0.037 |
| A-rich motif over-reliance identifies biologically invalid models | JEDI 0.251, CircDC 0.082 vs. BSCAN-FM 0.001–0.007 |
| BSCAN-FM+duplex achieves best integrated biological profile | Profile score 0.942 (rank 1) |
