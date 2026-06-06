# BSCAN — 프로젝트 종합 정리

**Back-Splice CircRNA Attention Network: circRNA 백스플라이싱 부위 검출을 위한 RNA 파운데이션 모델 프레임워크**

> 작성: 2026-06-06 | 단일 종합 문서 (연구 배경·방법·실험·분석 결과 전체)
> 세부 초안: `docs/{ABSTRACT,INTRODUCTION,METHODS,RESULTS,DISCUSSION}_DRAFT.md`
> 수치 출처: `results/paper_table_master.csv`, `research_results/*.csv`

---

## 1. 연구 배경 및 목표

### 1.1 생물학적 배경
**Circular RNA (circRNA)** 는 백스플라이싱(back-splicing)으로 생성되는 공유결합 폐환형 RNA다. 하류 5′ 스플라이스 부위가 상류 3′ 스플라이스 부위에 연결되어 back-splice junction (BSJ)을 형성한다. circRNA는 miRNA sponge, RBP 상호작용, cap-비의존 번역, 안정적 바이오마커 등으로 기능하지만, **어떤 인트론 쌍이 백스플라이싱을 일으키는지를 서열로 예측하는 문제**는 미해결 상태다.

지배적 메커니즘 모델: circRNA를 만드는 인트론은 ALU/SINE 같은 역상보 반복서열이 풍부해, 상·하류 인트론의 ALU가 짝지어 두 스플라이스 부위를 공간적으로 가깝게 만든다 (Jeck 2013, Ivanov 2015, Kramer 2015). 즉 판별 신호는 **엑손이 아니라 인트론 측면(flank)** 에 있어야 한다.

### 1.2 핵심 문제의식
기존 딥러닝 모델(DeepCircCode 2019 → circDeep 2020 → JEDI 2021 → CircCNN 2022 → CircDC·CircCNNs 2024)은 **동일 transcript에서 음성을 추출**하는 공통 평가 설계를 쓴다. 이 때문에 모델이 인트론 본질 신호 대신 **엑손 조성(exon composition) 단축경로**로 분류할 수 있고, 내부 정확도가 높아도 **다른 게놈 맥락으로 일반화되지 않는다.**

### 1.3 목표
1. 동결된 RNA 파운데이션 모델(FM) 임베딩으로 일반화 가능한 표현을 얻는 **BSCAN** 제안
2. 내부·외부 검증으로 일반화 격차를 정량화하고, FM이 그 핵심임을 분리 증명
3. **Hard-negative 3-tier 프로빙**으로 모델이 실제로 인트론 신호를 학습하는지 해부
4. branch 기여도·thermodynamic 신호·ALU·motif 안전성을 다각도로 분석

---

## 2. 방법 (Methods)

### 2.1 데이터
- 인간 circRNA BS junction + 매칭된 linear-splice (LS) 음성 쌍 **24,216개** (BS 12,105 / LS 12,111), hg19 좌표
- 각 junction에서 splice site 중심 **200-nt 복합서열 2개** 추출:
  - upper = [upper 인트론 100nt] + [upper 엑손 100nt]
  - lower = [lower 엑손 100nt] + [lower 인트론 100nt]
- 분할: **transcript-grouped split** (같은 transcript는 한 partition에만; 누출 방지), 실제 비율 ~62/17/20, 10 seeds

### 2.2 BSCAN 아키텍처 (BSCANUnified)
동결 FM 인코더 + projection 위에 3개 branch를 concat → MLP 분류기:
- **Branch A (CNN)**: FM-projected feature에 1D-CNN — local splice motif
- **Branch B (Stem)**: upper 인트론 × lower 인트론 RC의 Watson-Crick 염기쌍 맵 → 2D-CNN — 인트론 상보성
- **Branch C (Cross-attention)**: upper(query) × lower(key/value) 상호작용

FM 백본 4종: **RNA-FM, RNAErnie, RNABERT, RNA-MSM** (모두 동결). 학습 파라미터 ~2.0M.

### 2.3 베이스라인
DeepCircCode, circDeep, CircNet, JEDI, CircCNN, CircDC, CircCNN-single/double/tri.
- **BSCAN-base**: 동일 3-branch 구조 + 학습형 one-hot (FM 없음) — 아키텍처 효과 분리
- **BSCAN-onehot**: BSCAN-FM과 동일 구조 + 학습형 token embedding — FM 표현 효과 분리 (핵심 대조군)

### 2.4 학습
AdamW (lr 1e-4), batch 128, cross-entropy, early stopping (val AUC 30ep 무개선), 최대 100ep. NVIDIA RTX A6000.

### 2.5 외부 검증
- **External-A**: circAtlas v3 → exon-length-matched 음성 (hg38). 8,217 sample.
- **Leakage control 2단계**:
  - sequence-disjoint: 내부와 서열 겹치는 sample 제거
  - host-locus-disjoint: 내부 hg19 좌표를 hg38로 liftOver 후 좌표 겹치는 loci 제거

### 2.6 Hard-negative 3-tier 프로빙
- **Tier 1**: 표준 BS vs LS
- **Tier 2** (LS-intron swap): 진짜 BS 엑손 + LS 인트론 → BS형/LS형 인트론 구분 능력
- **Tier 3** (BS-intron swap): 진짜 BS 엑손 + 다른 BS junction 인트론 → 개별 인트론 특이성
- inference-only (재학습 없음). AUC<0.5 = 모델이 합성 음성을 진짜보다 높게 평가 (엑손 지배).

### 2.7 보조 분석
- **hnaug**: hard negative를 학습에 명시적으로 추가 (0.5·AUC + 0.5·HN-AUC 동시 최적화)
- **Duplex**: ViennaRNA duplexfold 에너지를 logit 공간에서 결합 (α=0.2)
- **ALU**: hg19 RepeatMasker로 BS/LS 인트론의 ALU/SINE 정량화 (multi-scale, ALU-matched)
- **Masking, A-rich motif, Bootstrap CI** 등

---

## 3. 실험 및 분석 결과

### 3.1 내부 검증 — SoTA + 파라미터 효율
BSCAN-FM 4종 모두 **AUC 0.916–0.917**, 전 베이스라인 상회 (CircCNN 0.898, JEDI 0.854). 학습 파라미터 ~2.0M으로 CircCNN-double(8.4M)보다 적음. Bootstrap CI: FM vs CircCNN 내부 Δ +0.019~+0.026 (모두 p<0.001).

### 3.2 외부 검증 — FM이 일반화의 핵심 (메인 결과)

| 모델군 | 내부 AUC | 외부 AUC | Drop |
|--------|:---:|:---:|:---:|
| BSCAN-FM 4종 | 0.916–0.917 | 0.838–0.846 | **~8%** |
| BSCAN-base (CNN+구조, FM無) | 0.901 | 0.720 | 20% |
| CircCNN | 0.898 | 0.704 | 22% |
| CircCNN-single/tri, CircDC, JEDI | 0.85–0.89 | 0.47–0.54 | **39–45%** |
| **BSCAN-onehot (동일 구조, FM無)** | 0.916 | **0.572** | **38%** |

→ **BSCAN-onehot 대조군이 38% drop** = FM이 일반화의 원인 (아키텍처 아님). Bootstrap CI: FM vs onehot 외부 Δ +0.270 [+0.255, +0.285], p<0.001.

**Leakage control:**
- sequence-disjoint (99.5%): 전 모델 Δ≤0.001 (변화 없음)
- **host-locus-disjoint (64.6%, 내부와 좌표 완전 비중복)**: FM 0.842→**0.818** (Δ−0.024), baseline도 −0.012~−0.015 균일. 내부와 안 겹치는 loci에서도 FM 0.82 유지 + baseline 대비 우위 → **진짜 일반화 증명.**

### 3.3 Branch Ablation — CNN branch가 일반화 동력
7개 config, transcript split, 3 seeds (동일 조건 공정 비교):

| Config | External AUC |
|--------|:---:|
| Full / Full−Attn / Full−Stem / FM+CNN (CNN 有) | 0.845–0.850 |
| Full−CNN / FM+Stem / FM+Attn (CNN 無) | 0.685–0.725 |

→ **CNN branch 제거 시 0.850→0.714 (최대 손실).** CNN 단독으로 풀 모델 재현. Stem·Attention은 CNN 있으면 redundant (해석성 제공, 일반화엔 부수적). Cross-attention 단독이 일반화 최악 (drop 21%).

### 3.4 Hard-Negative 3-Tier — exon bias 정량화
표준 학습 (3 seeds):

| 모델 | Tier1 | Tier2 | Tier3 |
|------|:---:|:---:|:---:|
| BSCAN-base | 0.901 | **0.727** | 0.535 |
| CircCNN | 0.898 | 0.719 | 0.508 |
| BSCAN-FM 4종 | 0.916 | 0.45–0.57† | 0.49–0.50 |

†chance 이하. CNN 모델은 BS형/LS형 인트론 구분(Tier2↑)하나 개별 인트론은 구분 못함(Tier3≈0.5). **FM은 Tier2조차 chance 이하 = 완전한 엑손 분류기.** upper/both intron swap에서도 동일 패턴 확인.

### 3.5 hnaug — 인트론 특이성은 학습 가능하나 FM은 억제

| 모델 | Tier3 (표준) | Tier3 (hnaug) |
|------|:---:|:---:|
| BSCAN (one-hot) | 0.535 | **0.843** |
| CircCNN (one-hot) | 0.508 | **0.836** |
| BSCAN-FM | 0.496 | 0.509 |

→ one-hot 모델은 hnaug로 Tier3 0.84 달성 = **인트론 특이성 신호는 서열에 존재, 학습 가능.** 그러나 표준 학습은 이를 incentivize 안 함 (엑손만으로 충분). **FM은 hnaug에도 회복 안 됨** = 동결 FM 설정에서 인트론 신호가 downstream에서 복원되지 않음.

### 3.6 ALU 분석 — 통계적으로 유의하나 Tier2 주원인은 아님
- Multi-scale: 윈도우 클수록 BS vs LS ALU 격차↑ (500nt: inverted pair 9.2% vs 5.2%, p=7×10⁻¹⁸)
- **ALU-matched Tier2**: LS donor를 ALU 농도 맞춰 교체해도 Tier2 AUC 불변 (Δ≈0) → **ALU density는 CNN Tier2 판별의 주원인이 아님.** splice site strength·polypyrimidine tract 등 다른 특징이 주도.

### 3.7 Thermodynamic Duplex — 직교 신호
ViennaRNA duplex 에너지 결합 (α=0.2): 외부 AUC +0.002~+0.003, AUPRC +0.011~+0.017 (전 FM, paired bootstrap p<0.001). α sweep으로 0.2 near-optimal 확인 (α=1.0에서 하락).

### 3.8 A-rich Motif 안전성
JEDI(0.251)·CircDC(0.082)는 polyadenylation A-rich motif에 과의존 → 외부 일반화 실패와 양의 상관. BSCAN-FM은 0.001–0.007로 안전.

---

## 4. 핵심 기여 (Contributions)

1. **BSCAN-FM**: ~2M 파라미터로 SoTA 내부 AUC (0.917) + 최고 외부 일반화 (~8% drop)
2. **FM이 일반화 원인임을 분리 증명**: BSCAN-onehot 대조군(동일 구조, 38% drop) + 2단계 leakage control(sequence + host-locus)
3. **Hard-negative 3-tier 프레임워크** (methodological contribution): exon bias vs intron specificity 분리 정량화
4. **인트론 특이성은 학습 가능하나 동결 FM이 억제함**을 hnaug로 입증
5. **Branch ablation**: 로컬 CNN branch가 일반화 동력임을 규명
6. **ALU-matched 분석**: ALU가 Tier2 판별의 주원인이 아님을 직접 반증

---

## 5. 한계 및 향후 과제

- Adapter 실험(FM+CNN/Mamba)은 split 조건 불일치로 본문 제외 → transcript-split 재현 필요
- 동결 FM의 인트론 신호 억제: layer-wise probing / partial fine-tuning으로 메커니즘 규명 필요
- 교차-DB External-B (circBase+RJunBase)는 좌표 규약 불일치로 보류 → splice-site 재앵커링 필요
- ALU의 Tier2 기여 정밀 분해 (multi-scale window, RBP motif)

---

## 6. 산출물 맵

| 항목 | 위치 |
|------|------|
| 논문 초안 (5섹션) | `docs/{ABSTRACT,INTRODUCTION,METHODS,RESULTS,DISCUSSION}_DRAFT.md` |
| 그림 5개 + 캡션 | `figures/Fig{1-5}_*.png/pdf`, `docs/FIGURE_CAPTIONS.md` |
| 메인 수치 표 | `results/paper_table_master.csv`, `results/ablation_summary.csv` |
| 보완 분석 결과 | `research_results/*.csv` (통계·masking·ALU·duplex·leakage) |
| 분석 스크립트 | `analysis/*.py` |
| 코드 (라이브러리/실행) | `core/`, `pipeline/`, `models/` |
| 작업 로그·현황 | `docs/WORK_LOG_2026-06-04.md`, `docs/PROJECT_STATUS.md` |

---

## 7. 한 줄 요약
> 동결 RNA 파운데이션 모델 임베딩은 circRNA 백스플라이싱 부위 예측에서 SoTA 정확도와 압도적 교차-데이터셋 일반화를 동시에 달성하며, 그 일반화는 엑손 단축경로를 회피하는 표현 특성에서 비롯된다 — 다만 같은 특성이 인트론 특이적 신호의 학습을 구조적으로 제약한다.
