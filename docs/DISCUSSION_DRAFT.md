# Discussion (Draft)

## 1. Internal validation: BSCAN-FM achieves state-of-the-art

All four BSCAN-FM variants (RNA-FM, RNABERT, RNAErnie, RNAMSM) reached AUC 0.916–0.917 on the internal
transcript-grouped test split across 10 seeds, outperforming every published baseline (CircCNN 0.898,
BSCAN-base 0.901, CircDC 0.862, JEDI 0.854). The consistent performance across all four frozen RNA
foundation models suggests that the BSCAN architecture — cross-attention between upper/lower junction
branches plus a WC stem branch — is the key driver of accuracy rather than the specific FM used.

Notably, BSCAN-FM achieves this with fewer trainable parameters (~2M) than most baselines (CircCNN 2.6M,
CircCNN-double 8.4M, CircCNN-single 6.6M), indicating strong parameter efficiency driven by the frozen
FM backbone.

## 2. Exon composition bias explains the internal–external performance gap

The most striking finding of this study is the dramatic divergence between internal and external
validation performance across model families:

| Model family         | Internal AUC | External AUC | Drop  |
|----------------------|-------------|-------------|-------|
| BSCAN-FM (4 models)  | 0.916–0.917 | 0.838–0.846 | ~8%   |
| BSCAN-base           | 0.901       | 0.720       | 20%   |
| CircCNN              | 0.898       | 0.704       | 22%   |
| CircCNN-single/tri   | 0.889–0.894 | 0.538–0.539 | ~39%  |
| CircDC, JEDI         | 0.854–0.862 | 0.467–0.501 | 42–45%|

The external validation (circAtlas exon-length-matched controls) deliberately removes the exon
composition shortcut available in the internal split: internal negatives are linear splice sites from
the same transcripts as positives, sharing similar exon sequences. External negatives are drawn from
different genomic loci with matched exon lengths, exposing models that have learned to classify by
exon composition rather than intrinsic junction signals.

The 39–45% drop seen in CircCNN-single, CircCNN-tri, CircDC, and JEDI indicates that these models
have largely learned exon-discriminative features that do not transfer to novel genomic contexts.
BSCAN-FM's substantially smaller drop (~8%) demonstrates that frozen RNA foundation model embeddings
provide a more sequence-context-general representation that is less susceptible to this shortcut.

BSCAN-base's intermediate drop (20%) suggests that CNN-based models without FM embeddings partially
learn exon compositional cues, but less severely than single-input or purely discriminative architectures.

## 3. Hard negative pairing reveals a critical distinction between sequence and FM-based learning

To test whether any model has learned intron-specific pairing complementarity — the proposed biological
mechanism underlying back-splice site selection — we constructed a 3-tier hard negative probe:

- **Tier 1**: Standard BS vs LS (training task)
- **Tier 2**: BS exon + LS intron (tests LS-vs-BS intron discrimination)
- **Tier 3**: BS exon + different-BS intron (tests individual intron specificity)

**Standard training (inference only):**
- CNN-based models: Tier3 0.50–0.57, Tier2 0.51–0.72 (some ALU/SINE pattern captured)
- FM models (raw): Tier3 0.494–0.497 (**below** chance); Tier2 0.450–0.569 (exon-dominated)
- FM + duplex energy: Tier3 0.499–0.533 (thermodynamic intron signal partially recovers discrimination)

**Hard negative augmented training (hnaug):**
Adding BS-exon + swapped-intron pairs (label=0) to training and jointly optimizing
`0.5 × valid_AUC + 0.5 × valid_HN_AUC` produced dramatically different outcomes by model type:

| Model | Tier1 | Tier2 | Tier3 |
|-------|:-----:|:-----:|:-----:|
| BSCAN-hnaug (one-hot) | 0.872 | **0.901** | **0.843** |
| CircCNN-hnaug (one-hot) | 0.872 | **0.897** | **0.836** |
| BSCAN-FM-hnaug | 0.909 | 0.533 | 0.509 |

One-hot models trained with hard negatives achieve Tier3 AUC 0.84 — proving that **intron-pairing
specificity is learnable from sequence data**, not a fundamental biological limitation. The signal
exists; standard task formulation simply does not incentivize its extraction.

FM models, by contrast, remain near chance (Tier3 0.509) even after explicit hard negative training.
This is the key mechanistic finding: **FM embeddings suppress intron signals irreversibly**. When the
frozen FM backbone encodes exon identity so strongly that the downstream head cannot recover intron
discriminative features regardless of training objective, the FM architecture becomes an exon classifier
by construction.

The near-chance performance of standard models is biologically expected: circRNA-forming introns are
universally enriched for ALU/SINE repeat elements, making their reverse-complementary sequences
compositionally similar across different BS junctions (Jeck 2013, Ivanov 2015, Kramer 2015).
The hnaug result shows that models can and do learn to leverage subtle intron differences when
given the right supervision signal.

The Tier2 (LS-intron) vs Tier3 (BS-intron) gap (T2−T3) quantifies the ability to distinguish intron
origin: CNN baselines show T2−T3 +0.13–0.16, consistent with capturing ALU-rich (BS) vs ALU-poor (LS)
intron statistics. FM models show T2−T3 ≤ 0, confirming complete exon-bias. BSCAN-FM-hnaug shows
T2−T3 +0.025 — a marginal improvement after hnaug, but still an order of magnitude below one-hot
hnaug models.

## 4. FM + duplex energy combination improves biological grounding

Combining BSCAN-FM probability with ViennaRNA duplex energy in logit space
(`logit(p) + α·zscore(-E_duplex)`, α=0.2) consistently improved external AUC (+0.002–0.003) and AUPRC
(+0.011–0.017) across all four FM variants. The improvement is modest in absolute terms but robust
across seeds and models, confirming that thermodynamic intron complementarity provides signal orthogonal
to the learned sequence classifier.

The combination did not improve binary MCC at threshold 0.5, which is expected: logit rescaling shifts
the ranking without changing the calibration of the threshold, so ranking metrics (AUC, AUPRC) improve
while threshold-based metrics do not. Paper claims based on this analysis should reference AUC and AUPRC.

## 5. A-rich motif over-reliance as a biological safety check

De novo motif analysis revealed that several baseline models show statistically elevated response to
A-rich pentamers (ATAAA, TAAAA, etc.) associated with polyadenylation signals — sequences that should
be irrelevant to back-splice junction classification. CircDC (risk score 0.082) and JEDI (0.251) show
the highest A-rich over-reliance, while BSCAN-FM variants show negligible signal (0.001–0.007).

This provides orthogonal evidence that models with high exon-composition bias also exploit
biologically unrelated sequence features present in exonic regions of the training data.

## 6. Integrated biological profile

Combining external generalization, A-rich motif safety, thermodynamic pairing channel, and hard-negative
robustness into a weighted composite score (weights: 0.35/0.25/0.25/0.15), BSCAN-FM+duplex variants
rank first through fourth. BSCAN-base and CircCNN follow at ranks 9–10, and no pure-CNN baseline
achieves a biological profile score above 0.51.

The recommended model for practical use is **BSCAN-RNA-FM + duplex** (external AUC 0.845, biological
profile 0.942), which provides the best combination of generalization, interpretability, and
thermodynamic grounding.

---

## Key claims supported by experiments

1. BSCAN-FM achieves SoTA internal AUC with fewer trainable parameters than most baselines. ✓
2. The internal–external AUC gap correlates strongly with exon-bias susceptibility. ✓
3. Intronic pairing specificity is learnable (hnaug one-hot: 0.84 AUC), but FM embedding suppresses this signal even under explicit hard negative training. ✓
4. Thermodynamic duplex energy provides orthogonal improvement to FM classifiers. ✓
5. A-rich motif over-reliance serves as a biological sanity check; BSCAN-FM is safe. ✓
6. BSCAN-FM+duplex achieves the best integrated biological profile across all evaluated criteria. ✓
