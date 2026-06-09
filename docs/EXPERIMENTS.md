# BSCAN 실험 레지스트리 (Experiment Registry)

> 모든 실험의 단일 명칭·설정·핵심결과·재현명령·산출물 추적표.
> 새 실험은 여기 ID를 부여하고 시작. 명칭 규칙: `<GROUP>-<SHORT>`.
> 모든 명령은 **repo 루트에서** 실행 (`python pipeline/...`).

---

## 명칭 체계 (4 그룹)

| 그룹 | 의미 | prefix |
|------|------|--------|
| **A. Performance** | 성능·일반화 (얼마나 잘하나) | `VAL` |
| **B. Architecture** | 구조 기여 (무엇이 성능을 만드나) | `ABL` |
| **C. Mechanism** | 학습 내용 해부 (무엇을 보고 판단하나) | `MECH` |
| **D. Enhancement** | 보강 신호 (더 좋게 만들 수 있나) | `AUG` |

공통 설정: junction_bps=100, transcript-grouped split, AdamW lr 1e-4. 통계검증(Bootstrap CI)은 각 실험 부속.
메인 모델 = **`bscan_unified_fm`** (BSCAN-RNA-FM). 핵심 대조군 = `bscan_unified_onehot`(동일구조·FM무), `bscan`(BSCAN-base).

---

## A. Performance — `VAL`

### VAL-INT · Internal validation ✅
- **질문:** 내부 transcript-split test에서 SoTA인가
- **모델:** BSCAN-FM 4종 + baseline 10종, 10 seeds
- **핵심결과:** BSCAN-FM **0.916–0.917** > CircCNN 0.898 > JEDI 0.854. ~2M 파라미터로 최고.
- **산출물:** `results/paper_table_master.csv` (int_auc)
- **재현:** `bash scripts/run_publication_classification_comparison.sh`

### VAL-EXT · External validation (circAtlas) ✅
- **질문:** 다른 데이터셋(circAtlas, hg38)으로 일반화되나
- **핵심결과:** FM drop **~8%** (0.84) vs CNN baseline 39–45% (0.47–0.54). **BSCAN-onehot(동일구조,FM무) 38% drop** → FM이 원인.
- **산출물:** `results/paper_table_master.csv` (ext_auc, drop_pct)
- **재현:** `python pipeline/make_circatlas_exon_controls.py` → `python pipeline/evaluate_circatlas_all_baselines.py --device 0`

### VAL-LEAK · Leakage control ✅
- **질문:** 외부 일반화가 train-test 누출 때문은 아닌가
- **핵심결과:** sequence-disjoint(99.5%) Δ≤0.001; **host-locus-disjoint(64.6%, liftOver) FM 0.842→0.818** (baseline도 −0.01~0.02 균일) → 누출 아님.
- **산출물:** `research_results/external_b_sequence_disjoint.csv`, `external_b_hostgene_disjoint.csv`
- **재현:** `python analysis/analyze_external_b_disjoint.py` / `python analysis/make_external_b_hostgene.py`

---

## B. Architecture — `ABL`

### ABL-BRANCH · Branch ablation ✅
- **질문:** CNN/Stem/Attn 중 무엇이 일반화를 만드나
- **모델:** 7 config (full, −CNN/−Stem/−Attn, CNN/Stem/Attn-only), rnafm, transcript, 3 seeds
- **핵심결과:** **CNN branch가 핵심** — 제거 시 ext 0.850→0.714. CNN 단독 0.845≈full. Stem·Attn은 CNN 있으면 redundant.

  | config | int | ext |
  |---|---|---|
  | Full | 0.891 | 0.850 |
  | FM+CNN | 0.891 | 0.845 |
  | Full−CNN | 0.870 | 0.714 |
  | FM+Stem | 0.840 | 0.725 |
  | FM+Attn | 0.867 | 0.685 |
- **산출물:** `results/ablation_summary.csv`
- **재현:** ablation 7종 학습(`experiment.py --model_name bscan_unified_fm_{cnnonly,...}`) → `python analysis/evaluate_ablation.py`

### ABL-CTX · Context-window study 🔬 (NEW)
- **질문:** BSCAN이 **더 넓은 flanking**을 보면 외부 일반화가 개선되나, 아니면 exon-bias가 구조적인가
- **하위:**
  - **ABL-CTX-WIN**: junction_bps 100→250→500 확장 (전 모델 동일, FM 임베딩 재추출 필요)
  - **ABL-CTX-BASE**: circcnntri flanking 100→500 민감도 (RCM 베이스라인 단독, supplementary)
- **선결 과제:** `fm_embeddings/{enc}/` 캐시가 junction_bps로 키 안 됨 → 윈도우별 분리 수정 필요
- **예상:** duplex/hnaug가 효과 없었던 점으로 보아 외부 AUC 큰 변화 없을 가능성 (그 자체가 "exon-bias는 구조적" 강화)

---

## C. Mechanism — `MECH`

### MECH-HN3 · Hard-negative 3-tier probe ✅
- **질문:** intron-type(BS형/LS형) vs intron-specificity(개별 intron) 구분하나
- **설계:** Tier1(BS vs LS) / Tier2(LS-intron swap) / Tier3(BS-intron swap), inference-only, 3 seeds. lower/upper/both swap 3방향.
- **핵심결과:** CNN계열 Tier2 0.65–0.73(intron-type 포착), Tier3 ~0.5. **FM은 Tier2도 chance 이하(exon-dominant)**.
- **산출물:** `research_results/hard_negative_pairing_{lower,ls_lower,upper,both}_intron_summary.csv`
- **재현:** `bash scripts/run_hard_negative_pairing.sh` + `python pipeline/evaluate_hard_negative_pairing.py --negative-mode <mode>`

### MECH-HNAUG · Hard-negative augmented training ✅
- **질문:** intron-pairing 신호가 학습 가능한가, FM은 회복되나
- **핵심결과:** one-hot 모델 Tier3 0.50→**0.84**(학습 가능!), **FM-hnaug 0.51(회복 안 됨)** → frozen FM이 intron 신호 억제.
- **산출물:** `results/hard_neg_augmented_summary.csv`
- **재현:** `python pipeline/train_hard_negative_augmented.py --models bscan circcnn --seeds 42 123 315` / `..._fm.py`

### MECH-MASK · Exon/Intron masking ✅
- **질문:** exon vs intron 중 무엇에 의존하나
- **핵심결과:** intron masking이 exon masking보다 큰 하락(전 모델). FM은 context-aware라 해석 주의.
- **산출물:** `research_results/masking_analysis.csv`
- **재현:** `python analysis/analyze_masking.py`

### MECH-ALU · RepeatMasker ALU/SINE ✅
- **질문:** ALU가 CNN Tier2 판별의 원인인가
- **핵심결과:** ALU 유의(BS>LS, 500nt p=7e-18) **그러나 ALU-matched Tier2 불변(Δ≈0)** → ALU는 주원인 아님.
- **산출물:** `research_results/alu_summary.csv`, `alu_multiscale_summary.csv`, `alu_matched_tier2.csv`
- **선결:** `data/rmsk_hg19.txt.gz` (UCSC)
- **재현:** `python analysis/analyze_alu_repeats.py` / `analyze_alu_multiscale.py` / `analyze_alu_matched_tier2.py`

### MECH-MOTIF · A-rich motif over-reliance ✅
- **질문:** polyadenylation 등 spurious motif에 과의존하나
- **핵심결과:** JEDI 0.251·CircDC 0.082(위험) vs BSCAN-FM 0.001–0.007(안전). A-rich risk ∝ 외부 drop.
- **산출물:** `results/biological_profile_summary.csv` (arich 열)

---

## D. Enhancement — `AUG`

### AUG-DUPLEX · Thermodynamic duplex combination ✅
- **질문:** ViennaRNA duplex 에너지가 직교 신호인가
- **핵심결과:** 외부 AUC +0.002~0.003, AUPRC +0.011~0.017 (paired bootstrap p<0.001). **α=0.2 near-optimal**.
- **산출물:** `results/fm_duplex_combination_summary.csv`, `research_results/duplex_alpha_sensitivity.csv`
- **재현:** `python analysis/analyze_duplex_alpha.py`

### AUG-RCM · RCM auxiliary branch for BSCAN 🔬 (NEW)
- **질문:** 200nt junction 유지 + flanking RCM 피처를 BSCAN 4번째 branch로 추가하면 도움 되나 (AUG-DUPLEX의 RCM판)
- **선결:** `pipeline/generate_rcm_scores_subset.py`로 RCM 추출, BSCAN에 RCM branch 추가 코드
- **예상:** duplex가 미미했던 점으로 보아 비슷할 가능성

---

## 종합 — `PROFILE`

### PROFILE · Integrated biological profile ✅ (Supplementary 권장)
- 4기준 가중합(외부 0.35 / motif 0.25 / duplex 0.25 / hardneg 0.15). BSCAN-RNA-FM+duplex 1위(0.942).
- **산출물:** `results/biological_profile_summary.csv`

---

## 논문 Results 섹션 ↔ 실험 ID 매핑

| Results § | 실험 ID |
|-----------|---------|
| 3.1 | VAL-INT |
| 3.2 | VAL-EXT + VAL-LEAK |
| 3.3 | ABL-BRANCH |
| 3.4 | MECH-HN3 |
| 3.5 | MECH-HNAUG |
| 3.6 | MECH-MASK |
| 3.7 | AUG-DUPLEX |
| 3.8 | MECH-ALU |
| 3.9 | MECH-MOTIF |
| 3.10 | PROFILE |
| (예정) | ABL-CTX, AUG-RCM |

---

## 상태 범례
✅ 완료 | 🔬 설계 중 | 🔄 실행 중 | ⏸ 보류
