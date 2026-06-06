# BSCAN 논문 그림

> 생성: `python make_figures.py` → `figures/` (PNG 300dpi + PDF)
> 모든 수치는 results/ 및 research_results/ CSV에서 자동 생성.

---

## Figure 1 — Internal vs External generalization (`Fig1_generalization`)
**메인 결과.** 
- (a) 모델별 internal→external AUC dumbbell. FM 4종은 external 0.84+ 유지, baseline은 chance 수준으로 붕괴.
- (b) Internal→External drop%. FM ~8% vs CNN baseline 39–45%. BSCAN-onehot(같은 구조, FM 없음) 38% → FM이 일반화의 원인.

## Figure 2 — Branch ablation (`Fig2_ablation`)
**P1 신규.** External AUC by branch 구성. CNN 있는 4개(파랑) 0.845–0.850 vs CNN 없는 3개(빨강) 0.685–0.725. → **CNN branch가 일반화 핵심.**

## Figure 3 — Hard-negative 3-tier (`Fig3_hardneg`)
- (a) 표준 학습 Tier1/2/3. CNN 계열은 Tier2 높음(intron-type 구분), FM은 Tier2 chance 이하(exon-dominant).
- (b) Hard-neg augmented: one-hot 모델 Tier3 0.50→0.84 급등, FM은 0.51 (회복 안 됨).

## Figure 4 — ALU 분석 (`Fig4_alu`)
- (a) Multi-scale: 윈도우 클수록 BS vs LS inverted ALU pair 격차 증가 (500nt: 9.2% vs 5.2%).
- (b) ALU-matched Tier2: ALU 농도 맞춰도 Tier2 불변 → **ALU는 Tier2 주원인 아님.**

## Figure 5 — Duplex α sensitivity (`Fig5_duplex_alpha`)
4개 FM 모두 α=0.2 부근 최적, α=1.0에서 하락. validation 없이 α=0.2 고정 정당화.

---

## 논문 배치 제안
| Figure | 위치 | 핵심 메시지 |
|--------|------|-------------|
| Fig 1 | Main | FM → generalization (메인 클레임) |
| Fig 2 | Main | CNN branch가 핵심 (ablation) |
| Fig 3 | Main | Exon bias + intron signal 학습 가능성 |
| Fig 4 | Main/Supp | ALU 정량화 + ALU≠Tier2 원인 |
| Fig 5 | Supp | Duplex α robustness |
