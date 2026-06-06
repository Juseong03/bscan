# BSCAN 프로젝트 작업 현황 요약

> 최종 업데이트: 2026-06-05  
> 이 문서는 지금까지 수행한 모든 작업의 단일 참조점입니다.

---

## 1. 프로젝트 개요

**BSCAN** (Back-Splice CircRNA Attention Network) — circRNA 백스플라이싱 접합부(BSJ) 검출 딥러닝 프레임워크.

**핵심 아키텍처 (BSCANUnified):**
- Branch A: 1D-CNN (FM projected features, local motif)
- Branch B: Watson-Crick stem map (intron 상보성 2D-CNN)
- Branch C: Cross-attention (upper × lower junction)
- FM 백본 4종 (RNA-FM, RNAErnie, RNAMSM, RNABERT) — 동결(frozen)

---

## 2. 최종 수치 (results/paper_table_master.csv 기준)

### 내부 검증 (transcript-grouped split, 10 seeds, sample-split 헤드라인)

| 모델 | 파라미터 | 내부 AUC | 외부 AUC | Drop% |
|------|-------:|:---:|:---:|:---:|
| BSCAN-RNAErnie | 2.0M | 0.9168 | 0.8378 | 8.6% |
| BSCAN-RNAMSM | 2.0M | 0.9167 | 0.8443 | 7.9% |
| BSCAN-RNA-FM | 2.0M | 0.9167 | 0.8418 | 8.2% |
| BSCAN-RNABERT | 1.9M | 0.9164 | 0.8458 | 7.7% |
| BSCAN-onehot | 2.0M | 0.9164 | 0.5723 | 37.6% |
| BSCAN-base | 3.9M | 0.9009 | 0.7204 | 20.0% |
| CircCNN | 2.6M | 0.8983 | 0.7041 | 21.6% |
| CircCNN-single/tri/double | 6.6/1.7/8.4M | 0.888–0.894 | 0.51–0.54 | 39–43% |
| DeepCircCode / CircDC / JEDI / CircDeep | — | 0.79–0.88 | 0.47–0.56 | 37–45% |

### Hard Negative 3-Tier (standard training, 3 seeds)

| 모델 | Tier1 | Tier2 (LS-intron) | Tier3 (BS-intron) |
|------|:---:|:---:|:---:|
| BSCAN-base | 0.901 | **0.727** | 0.535 |
| CircCNN | 0.898 | 0.719 | 0.508 |
| BSCAN-FM 4종 | 0.916–0.917 | 0.45–0.57† | 0.49–0.50 |
| **BSCAN-hnaug** | 0.872 | **0.901** | **0.843** |
| **CircCNN-hnaug** | 0.872 | **0.897** | **0.836** |
| **BSCAN-FM-hnaug** | 0.909 | 0.533 | 0.509 |

†FM은 chance 이하 (exon-dominant)

---

## 3. 보완 실험 결과 (research_results/) — 2026-06-04~05 완료

### ✅ P1: Branch Ablation (results/ablation_summary.csv)
7개 config, transcript split, 3 seeds, 동일 평가 — 완전 공정 비교.

| Config | Branches | Int AUC | Ext AUC | Drop% |
|--------|----------|:---:|:---:|:---:|
| **Full** | CNN+Stem+Attn | 0.8907 | 0.8496 | 4.6% |
| Full−Attn | CNN+Stem | 0.8838 | 0.8498 | 3.8% |
| Full−Stem | CNN+Attn | 0.8838 | 0.8487 | 4.0% |
| FM+CNN | CNN only | 0.8911 | 0.8450 | 5.2% |
| Full−CNN | Stem+Attn | 0.8695 | 0.7137 | 17.9% |
| FM+Stem | Stem only | 0.8402 | 0.7248 | 13.7% |
| FM+Attn | Attn only | 0.8674 | 0.6852 | 21.0% |

**결론: CNN branch가 일반화의 핵심**(제거 시 ext 0.850→0.714). Stem·Attn은 CNN 있으면 redundant. Cross-attention 단독이 일반화 최악(21% drop).
> 주의: transcript split Full 내부 AUC=0.891 (sample split 0.917보다 낮음). Ablation은 헤드라인 Table 1과 별도 프로토콜.

### ✅ P3: Bootstrap CI (statistical_tests.csv)
| 비교 | Δ AUC | p |
|------|-------|---|
| FM vs onehot (외부) | +0.270 [+0.255,+0.285] | *** |
| FM vs CircCNN (외부) | +0.138~+0.142 | *** |
| FM+duplex vs raw (paired) | +0.002~+0.003 | *** |

### ✅ P4: Masking (masking_analysis.csv)
intron masking 타격 > exon masking (전 모델, 내부 test). FM은 context-aware라 해석 주의.

### ✅ P5: ALU/SINE RepeatMasker
- 100/250/500nt multi-scale: 500nt에서 BS 40.8% vs LS 36.0% ALU, inverted pair 1.8× (p=7e-18)
- **ALU-density-matched Tier2** (가장 중요): ALU 맞춰도 Tier2 불변(Δ≈0) → **ALU는 Tier2 주원인 아님**
- 파일: alu_summary.csv, alu_multiscale_summary.csv, alu_matched_tier2.csv

### ✅ P6: Duplex α sweep (duplex_alpha_sensitivity.csv)
α=0.2 near-optimal, α=0.1–0.5 안정, α=1.0 하락.

### ✅ HN-extra: upper/both intron swap
세 방향(lower/upper/both) 모두 동일 패턴 — CNN 소폭 >chance, FM ~chance.

### ✅ External leakage 감사 (external_b_sequence_disjoint.csv)
- External-A exact 서열 중복 0.5% (45/8217)
- Sequence-disjoint subset(99.5%)에서 전 모델 Δ≤0.001 → **외부 결과는 leakage 아님**

---

## 4. 논문 초안 현황 (docs/)

| 파일 | 내용 |
|------|------|
| `ABSTRACT_DRAFT.md` | Abstract (~370단어) |
| `INTRODUCTION_DRAFT.md` | Introduction (7단락) |
| `METHODS_DRAFT.md` | Methods §2.1–2.11 |
| `RESULTS_DRAFT.md` | Results §3.1–3.10 |
| `DISCUSSION_DRAFT.md` | Discussion §1–8 + Summary |

### Results 섹션 구성 (최신)
| § | 내용 |
|---|------|
| 3.1 | 내부 검증 SoTA + 파라미터 효율 |
| 3.2 | FM 외부 일반화 + Bootstrap CI + leakage 감사 |
| **3.3** | **Branch ablation (신규)** |
| 3.4 | Hard negative 3-tier + upper/both swap |
| 3.5 | hnaug: intron specificity 학습 가능, FM은 불가 |
| 3.6 | Masking 분석 |
| 3.7 | Duplex thermodynamic + α sensitivity |
| 3.8 | RepeatMasker ALU/SINE + ALU-matched Tier2 |
| 3.9 | A-rich motif over-reliance |
| 3.10 | Integrated biological profile (Supplementary 권장) |

---

## 5. 중요 수정/결정 사항

| 항목 | 내용 |
|------|------|
| Adapter 실험 제외 | split 불일치(sample vs transcript)로 본문 제외, Limitations에 future work 언급 |
| Split 비율 수정 | "80/10/10" → 실제 **~62/17/20** |
| FM "구조적 억제" 표현 완화 | "frozen-embedding setting에서 recoverable하지 않음"으로 조정 |
| Composite score | Supplementary로 이동, 개별 지표 우선 |
| BSCAN-base Tier3 수정 | 0.523 → 0.535 (bscan_seq_lite 오기재) |
| ALU 해석 수정 | ALU-matched Tier2로 "ALU가 주원인 아님" 확인, 해석 보강 |
| External-B (circBase) 폐기 | DB간 좌표 규약 불일치(canonical 81% vs 4%) → sequence-disjoint 감사로 대체 |
| mlponly ablation 제외 | 전 branch off는 degenerate (torch.cat 빈 리스트) |

---

## 6. 남은 작업 (투고 전)

### 완료됨 ✅
- P1 Branch ablation, P3 통계, P4 Masking, P5 ALU(+multiscale+matched), P6 Duplex α
- External leakage 감사 (sequence-disjoint)

### 남음
| 작업 | 비고 |
|------|------|
| host-gene-disjoint External-B | hg19↔hg38 liftOver 필요 (현재 도구 부재). sequence-disjoint가 leakage 핵심은 커버 |
| circ2LO (2025) baseline | Related Work / 비교 |
| 그림(Figure) 생성 | Table → Figure 변환 |
| Exact sequence dup 제거 (P2 잔여) | 외부셋 정제 |

---

## 7. 파일 구조

```
bscan/
├── results/
│   ├── paper_table_master.csv          ← 메인 모델 최종 수치
│   └── ablation_summary.csv            ← branch ablation 7행
├── research_results/
│   ├── statistical_tests.csv           P3 Bootstrap CI
│   ├── masking_analysis.csv            P4
│   ├── alu_summary.csv / alu_coverage.csv          P5 (100nt)
│   ├── alu_multiscale_summary.csv      P5 multi-scale
│   ├── alu_matched_tier2.csv           P5 ALU-matched
│   ├── duplex_alpha_sensitivity.csv    P6
│   ├── external_b_sequence_disjoint.csv  leakage 감사
│   ├── ablation_results.csv            ablation 원본
│   └── hard_negative_pairing_*_summary.csv  (lower/ls_lower/upper/both)
├── docs/
│   ├── ABSTRACT/INTRODUCTION/METHODS/RESULTS/DISCUSSION_DRAFT.md
│   ├── PAPER_TABLES_DRAFT.md
│   ├── ANALYSIS_SUMMARY.md             한국어 분석 요약
│   ├── REVISION_PLAN.md                보완 작업 계획
│   ├── WORK_LOG_2026-06-04.md          이번 세션 상세 로그
│   └── PROJECT_STATUS.md               이 파일
├── saved_models/                       → symlink (circRNA/BSCAN)
│   └── bscan_unified_fm_{cnnonly,stemonly,attnonly,nocnn,nostem,noattn,fulltr}/  ablation
├── external_data/                      → symlink
│   ├── circatlas/exon_controls/        External-A (circAtlas)
│   └── circbase_b/                     ⚠️ 폐기 (좌표 규약 불일치)
├── fm_embeddings/                      사전 추출 FM hidden states
├── figures/                            논문 그림 5개 (PNG+PDF) + FIGURES.md
├── analysis/                           ← 이번 세션 분석 스크립트 (정리됨)
└── (루트 .py)                          원본 프레임워크 + CalPPM/RCSFinder 의존성
```

### analysis/ — 이번 세션 분석 스크립트 (루트에서 `python analysis/xxx.py`로 실행)
```
analyze_statistics.py          P3 Bootstrap CI
analyze_masking.py             P4 Masking
analyze_alu_repeats.py         P5 ALU (100nt)
analyze_alu_multiscale.py      P5 Multi-scale ALU
analyze_alu_matched_tier2.py   P5 ALU-matched Tier2
analyze_duplex_alpha.py        P6 Duplex α
analyze_external_b_disjoint.py leakage 감사
evaluate_ablation.py           Branch ablation 평가 (내부+External-A)
evaluate_circatlas_fm_unified.py  circAtlas FM 외부 평가
make_figures.py                논문 그림 5개 생성
update_mamba_results.py        (adapter, 폐기됨)
make_circbase_external.py      External-B (⚠️ 폐기, convention 문제)
rebuild_external_b.py          External-B 재구축 시도 (⚠️ 폐기)
```
> 주의: 상대 데이터 경로(`research_results/`, `data/` 등)를 쓰므로 **repo 루트에서** 실행해야 함.
> 루트 유지: 원본 프레임워크(dataloader/trainer/experiment/...) + CalPPM.py·RCSFinder.py(evaluate_hard_negative_pairing 의존성).

---

## 8. 핵심 수치 Quick Reference

```
내부 AUC (best, sample split):  BSCAN-FM 0.916–0.917
외부 AUC (best raw):            BSCAN-RNABERT 0.846
외부 AUC drop (FM):             ~8%  (vs CNN 39–45%)
BSCAN-onehot drop:              37.6% → FM이 generalization 원인

Branch ablation (transcript, 3-seed):
  Full ext=0.850 | FM+CNN ext=0.845 | Full−CNN ext=0.714
  → CNN branch가 핵심, stem/attn은 redundant

Tier2 (CNN best):    BSCAN-base 0.727 (ALU-matched 후에도 0.735 유지)
Tier3 (std):         0.49–0.56 / (hnaug one-hot) 0.84 / (hnaug FM) 0.509
Duplex α=0.2:        AUC +0.002–0.003, AUPRC +0.011–0.017 (p<0.001)
ALU 500nt:           BS 40.8% vs LS 36.0%, inv pair 1.8× (p=7e-18)
Leakage:             외부 exact dup 0.5%, disjoint subset Δ≤0.001
Split 실제 비율:      ~62/17/20 (transcript-grouped)
```
