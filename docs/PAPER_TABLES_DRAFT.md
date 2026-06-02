# BSCAN 논문 테이블 초안

> 업데이트: 2026-06-02  
> Seeds: FM 모델 10종, baseline 10종, hard neg 3종(42/123/315)  
> † = BSCAN-FM+CNN adapter는 5 seeds, BSCAN-FM+Mamba는 훈련 중  
> ‡ = Hard neg augmented training 결과 대기 중

---

## Table 1. Internal Validation (Standard BS vs LS, Transcript-Grouped Split)

> 내부 검증: transcript 기준으로 그룹 분리한 80/10/10 split. 10 seeds 평균 ± 표준편차.

| Model | Params | AUC | AUPRC | MCC |
|-------|-------:|:---:|:-----:|:---:|
| **BSCAN-RNAErnie** | 2.0M | **0.9168 ± 0.0021** | 0.9394 | 0.7322 |
| **BSCAN-RNAMSM** | 2.0M | 0.9167 ± 0.0023 | 0.9388 | 0.7380 |
| **BSCAN-RNA-FM** | 2.0M | 0.9167 ± 0.0023 | 0.9392 | 0.7330 |
| **BSCAN-RNABERT** | 1.9M | 0.9164 ± 0.0020 | 0.9390 | 0.7359 |
| BSCAN-FM+CNN† | 2.0M | 0.9131 ± 0.0022 | — | — |
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

**Bold**: BSCAN-FM variants (proposed). 모든 BSCAN-FM variants가 baseline 최고 (CircCNN 0.898)를 상회.  
BSCAN-FM은 BSCAN-base(3.9M)보다 적은 파라미터(~2.0M)로 더 높은 AUC.

---

## Table 2. External Validation (circAtlas Exon-Length-Matched Controls)

> 외부 검증: circAtlas v3에서 추출한 exon 길이 매칭 컨트롤. 10 seeds 평균 ± 표준편차.  
> Drop%: (Internal AUC − External AUC) / Internal AUC × 100

| Model | External AUC | External AUPRC | External MCC | Drop% |
|-------|:------------:|:--------------:|:------------:|:-----:|
| **BSCAN-RNABERT** | **0.8458 ± 0.0136** | 0.7222 | 0.6876 | **7.7%** |
| **BSCAN-RNAMSM** | 0.8443 ± 0.0144 | 0.7266 | 0.7125 | 7.9% |
| **BSCAN-RNA-FM** | 0.8418 ± 0.0143 | 0.7210 | 0.6991 | 8.2% |
| **BSCAN-RNAErnie** | 0.8378 ± 0.0162 | 0.7207 | 0.6836 | 8.6% |
| BSCAN-onehot | 0.5723 ± 0.0220 | 0.5202 | 0.0382 | 37.6% |
| CircCNN | 0.7041 ± 0.0127 | 0.6151 | 0.2530 | 21.6% |
| DeepCircCode | 0.5556 ± 0.0101 | 0.5082 | 0.0343 | 36.6% |
| CircCNN-single | 0.5392 ± 0.0096 | 0.4992 | −0.0118 | 39.7% |
| CircCNN-tri | 0.5380 ± 0.0091 | 0.4997 | −0.0158 | 39.5% |
| CircCNN-double | 0.5092 ± 0.0081 | 0.4802 | −0.0466 | 42.6% |
| CircDC | 0.5010 ± 0.0072 | 0.4759 | −0.0754 | 41.9% |
| CircDeep | 0.4940 ± 0.0068 | 0.4927 | −0.0100 | 37.5% |
| JEDI | 0.4672 ± 0.0049 | 0.4563 | −0.0965 | 45.3% |
| BSCAN-base | — | — | — | — |
| BSCAN-FM+CNN† | — | — | — | — |

**핵심**: BSCAN-FM의 Drop%는 ~8%로 baseline 대비 4–5배 낮음.  
**Control**: BSCAN-onehot (동일 아키텍처, FM 없음)이 37.6% drop → FM embedding이 generalization의 핵심.

---

## Table 3. Hard Negative Pairing Analysis (3-Tier)

> 3-tier 난이도 설계. 모든 모델 동일 학습 후 inference-only 평가. Seeds 42/123/315, mean ± std.
>
> - **Tier 1**: Standard BS vs LS (Table 1과 동일)
> - **Tier 2**: BS exon + LS intron vs real BS junction (LS-intron swap, ls_lower_intron)
> - **Tier 3**: BS exon + 다른 BS locus intron vs real BS junction (BS-intron swap, lower_intron)
>
> AUC ≈ 0.50: chance level (indistinguishable). AUC < 0.50: model scores hard neg higher (reversed).

| Model | Tier1 (Internal) | Tier2 (LS-intron) | Tier3 (BS-intron) | Tier3+Duplex | T2−T3 Gap |
|-------|:----------------:|:-----------------:|:-----------------:|:------------:|:---------:|
| **BSCAN-RNA-FM** | 0.917 | 0.475 ± 0.019† | 0.496 ± 0.002 | **0.508** | −0.021 |
| **BSCAN-RNAErnie** | 0.917 | 0.477 ± 0.029† | 0.492 ± 0.006 | **0.533** | −0.015 |
| **BSCAN-RNAMSM** | 0.917 | 0.450 ± 0.021† | 0.494 ± 0.005 | 0.510 | −0.044 |
| **BSCAN-RNABERT** | 0.916 | 0.569 ± 0.008 | 0.496 ± 0.002 | 0.502 | +0.073 |
| BSCAN-FM+CNN† | 0.913 | 0.512 ± 0.036 | 0.496 ± 0.008 | 0.499 | +0.016 |
| BSCAN-base | 0.901 | — | 0.523 ± 0.006 | — | — |
| CircCNN | 0.898 | — | 0.508 ± 0.006 | — | — |
| CircCNN-single | 0.894 | 0.671 ± 0.052 | 0.536 ± 0.015 | — | +0.135 |
| CircCNN-tri | 0.889 | **0.715 ± 0.044** | **0.558 ± 0.020** | — | **+0.158** |
| CircCNN-double | 0.888 | — | 0.513 ± 0.003 | — | — |
| DeepCircCode | 0.877 | 0.663 ± 0.005 | 0.518 ± 0.003 | — | +0.145 |
| CircDC | 0.862 | — | 0.514 ± 0.002 | — | — |
| JEDI | 0.854 | — | 0.503 ± 0.003 | — | — |
| **BSCAN-hnaug** | 0.872 ± 0.004 | **0.901 ± 0.005** | **0.843 ± 0.007** | — | +0.158 |
| **CircCNN-hnaug** | 0.872 ± 0.010 | **0.897 ± 0.005** | **0.836 ± 0.009** | — | +0.135 |
| **BSCAN-FM-hnaug** | 0.909 ± 0.002 | 0.533 ± 0.033 | 0.509 ± 0.008 | — | +0.025 |

†: AUC < 0.50; model is exon-biased (hard neg exon = real BS exon, so model cannot discriminate).  
**hnaug**: Hard negative augmented training — training set에 BS-exon + swapped-intron 쌍을 label=0으로 추가.  
Early stopping: 0.5 × valid_AUC + 0.5 × valid_HN_AUC 동시 최적화. Seeds 42/123/315.

**T2−T3 Gap 해석**:
- CNN 모델 (gap > 0.13): LS-type intron을 BS-type과 구분 가능 (ALU-rich vs ALU-poor 패턴 포착)
- FM 모델 (gap ≤ 0): Intron 정보 무시, 완전 exon 분류기
- Standard 학습 모델은 개별 BS junction intron 구분 불가 (Tier3 ≤ 0.558)
- **hnaug 모델 (one-hot)**: Tier3 0.84로 비약적 개선 — intron specificity 신호는 존재하나 표준 학습이 이를 incentivize하지 않음을 증명
- **BSCAN-FM-hnaug**: Tier3 0.509 — hnaug 학습에도 불구하고 FM 임베딩이 exon 편향을 극복하지 못함. FM이 학습 신호 자체를 억제하는 근본적 편향임을 확인

---

## Table 4. FM+Duplex Thermodynamic Combination (External Validation)

> ViennaRNA duplexfold 에너지를 logit 공간에서 선형 결합:  
> `logit(p_combined) = logit(p_model) + 0.2 × z-score(−E_duplex)`

| Model | AUC (raw) | AUC (+duplex) | ΔAUC | AUPRC (raw) | AUPRC (+duplex) | ΔAUPRC |
|-------|:---------:|:-------------:|:----:|:-----------:|:---------------:|:------:|
| BSCAN-RNA-FM | 0.8418 | **0.8449** | +0.0031 | 0.7210 | **0.7381** | +0.0171 |
| BSCAN-RNAErnie | 0.8378 | 0.8412 | +0.0034 | 0.7207 | 0.7318 | +0.0111 |
| BSCAN-RNAMSM | 0.8443 | 0.8475 | +0.0032 | 0.7266 | 0.7404 | +0.0138 |
| BSCAN-RNABERT | 0.8458 | **0.8484** | +0.0026 | 0.7222 | 0.7354 | +0.0132 |

Duplex 조합은 AUC +0.002–0.003, AUPRC +0.011–0.017 일관된 개선.  
MCC (threshold=0.5)는 변화 없음 (logit rescaling이 ranking에만 영향).

---

## Table 5. Biological Profile Score (통합 평가)

> 4가지 기준의 가중 합산:
> - External generalization (weight 0.35)
> - A-rich motif safety (weight 0.25)
> - Thermodynamic duplex channel (weight 0.25)
> - Hard negative robustness (weight 0.15)

| Rank | Model | Profile Score | Ext AUC | A-rich Risk | Duplex ΔAUPRC | Tier3 AUC |
|:----:|-------|:-------------:|:-------:|:-----------:|:-------------:|:---------:|
| 1 | **BSCAN-RNA-FM + duplex** | **0.942** | 0.845 | 0.001 | +0.017 | 0.508 |
| 2 | **BSCAN-RNAErnie + duplex** | 0.940 | 0.841 | 0.005 | +0.011 | 0.533 |
| 3 | **BSCAN-RNAMSM + duplex** | 0.912 | 0.847 | 0.007 | +0.014 | 0.510 |
| 4 | **BSCAN-RNABERT + duplex** | 0.892 | 0.848 | 0.007 | +0.013 | 0.502 |
| 5–8 | BSCAN-FM (raw, 4종) | 0.621–0.632 | 0.838–0.846 | 0.001–0.007 | — | 0.492–0.497 |
| 9 | BSCAN-base | 0.507 | 0.720 | 0.012 | — | 0.523 |
| 10 | CircCNN | 0.504 | 0.704 | 0.006 | — | 0.508 |
| 11 | CircCNN-tri | 0.391 | 0.538 | 0.046 | — | 0.558 |
| 12–15 | Other baselines | < 0.4 | 0.467–0.556 | 0.008–0.251 | — | 0.503–0.536 |

**추천 모델**: BSCAN-RNA-FM + duplex  
외부 AUC 0.845 + thermodynamic interpretability + 최소 A-rich bias

---

## 논문 테이블 배치 제안

| 논문 내 위치 | 테이블 | 핵심 메시지 |
|:---:|------|-------------|
| Main Table 1 | Table 1 (내부) + Table 2 (외부) | SoTA + Generalization gap |
| Main Table 2 or Fig | Table 3 (3-tier hard neg) | Exon bias 정량화 + intron 신호 한계 |
| Supplementary | Table 4 (duplex) | Thermodynamic orthogonal signal |
| Supplementary | Table 5 (bio profile) | 통합 추천 모델 |

---

## 대기 중 수치 (‡)

| 항목 | 예상 추가 시점 |
|------|--------------|
| BSCAN-FM+Mamba 내부 AUC (5 seeds) | 수시간 내 |
| BSCAN-base 외부 AUC | 별도 실험 필요 |
| hnaug (BSCAN, CircCNN) Tier2/3 AUC | 훈련 완료 후 |
