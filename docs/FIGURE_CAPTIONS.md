# Figure Captions — BSCAN

> 그림 파일: `figures/Fig{1-5}_*.png` (300 dpi) / `.pdf` (vector). 생성: `python analysis/make_figures.py`.
> 본문 참조 위치는 `docs/RESULTS_DRAFT.md` 참조.

---

**Figure 1. Frozen RNA foundation-model embeddings drive cross-dataset generalization.**
(a) Internal (grey) versus external (colored) AUC for every model, linked by a connector; external points colored by family (blue, BSCAN-FM; teal, one-hot architecture controls; grey, baselines). Dotted line marks chance (AUC = 0.5). All four BSCAN-FM variants retain external AUC ≈ 0.84, whereas CNN-only baselines collapse toward chance. (b) Internal→external AUC drop (%). BSCAN-FM models drop ~8% versus 20–45% for one-hot/CNN models; the architecture-matched BSCAN-onehot control drops 38%, isolating the FM representation as the source of generalization. *(Results §3.2; Tables 1–2.)*

**Figure 2. The convolutional branch is the principal driver of generalization.**
External AUC for seven RNA-FM ablation variants (transcript-grouped split, 3 seeds; mean ± s.d.). Bars colored by whether the CNN branch is present (blue) or absent (red). Configurations retaining the CNN branch (Full, Full−Attn, Full−Stem, FM+CNN) all reach external AUC ≈ 0.85, while removing the CNN branch (Full−CNN, FM+Stem, FM+Attn) drops external AUC to 0.69–0.72. The stem and cross-attention branches are largely redundant once CNN is present. *(Results §3.3; Table 3.)*

**Figure 3. Hard-negative probing dissociates exon-bias from learnable intron specificity.**
(a) Three-tier probe under standard training: Tier 1 (BS vs. LS), Tier 2 (BS exon + LS intron), Tier 3 (BS exon + different-BS intron). CNN-based models (BSCAN-base, CircCNN, CircCNN-tri) discriminate intron type (Tier 2 > 0.7) but not individual introns (Tier 3 ≈ 0.5); FM models are exon-dominant (Tier 2 < 0.5). (b) Hard-negative augmented training raises Tier 3 AUC from ≈ 0.50 to 0.84 for one-hot models (BSCAN, CircCNN) but leaves the FM model near chance (0.51), indicating that intron specificity is learnable from sequence yet not recoverable from frozen FM embeddings in this setup. *(Results §3.4–3.5; Tables 4–5.)*

**Figure 4. ALU enrichment is real but does not explain CNN Tier-2 discrimination.**
(a) Fraction of junctions with an inverted ALU pair (ALU in both intronic flanks) at 100/250/500-nt window sizes, for BS (circRNA, blue) and LS (linear, grey). ALU enrichment in BS introns grows with window size (500 nt: 9.2% vs. 5.2%; Mann–Whitney p = 7×10⁻¹⁸). (b) Standard versus ALU-density-matched Tier 2 AUC. Matching the lower-intron ALU coverage of LS donors to each BS junction leaves Tier 2 AUC essentially unchanged (Δ ≈ 0), demonstrating that ALU density is not the primary driver of CNN Tier-2 discrimination. *(Results §3.8.)*

**Figure 5. Thermodynamic duplex combination is robust to the weighting coefficient.**
External AUC versus duplex weight α (logit-space combination of model score and ViennaRNA duplex energy) for the four BSCAN-FM variants. All four peak near α = 0.2 (dashed line, the fixed value used throughout) and decline only at α = 1.0, confirming the combination is stable and not finely tuned. *(Results §3.7; Table 6.)*

---

## Supplementary figures (optional)
- Masking analysis (Results §3.6, Table S3) — intron vs. exon masking AUC drop.
- Host-locus-disjoint External-B (Results §3.2) — `research_results/external_b_hostgene_disjoint.csv`.
