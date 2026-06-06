# BSCAN 원고 보완 계획

> 업데이트: 2026-06-03  
> 상태: 즉시 수정 완료. 단기/장기 실험 계획 수립.

---

## ✅ 즉시 수정 완료 (문서)

| 항목 | 내용 |
|------|------|
| Mamba 5-seed 복원 | seed 1004 포함, paper_table_master.csv 및 모든 draft 수정 |
| FM "억제" 표현 완화 | "structurally suppress" → "not readily recoverable in the tested frozen-embedding setting" |
| Composite score 위상 조정 | Supplementary로 이동 명시, 개별 지표 우선 권장 |
| Adapter split 불일치 명시 | Methods + Results + Discussion + Abstract에 caveat 추가 |
| Split 비율 수정 | "80/10/10" → 실제 ~62/17/20 (transcript-grouped), 60/20/20 (sample) |
| Limitations 섹션 확장 | circAtlas OOD benchmark 지위, ALU 가설, FM probing 필요성 |

---

## 🔬 단기 실험 (재훈련 불필요, 기존 checkpoint 활용)

### P3. 통계 검증
- Bootstrap confidence intervals: BSCAN-FM vs BSCAN-onehot, raw FM vs +duplex
- DeLong test: duplex 결합 AUC 상승이 유의한지
- Paired t-test: seed-wise AUC 비교 (BSCAN-FM vs CircCNN)
- **스크립트**: `analyze_statistical_significance.py` (old project에 있음)

### P4. Exon/Intron Masking 분석
모든 기존 checkpoint에서 inference-only:
- full input (baseline)
- exon masked (N 또는 zero embedding)
- intron masked
- upper intron only masked
- lower intron only masked
- exon-only, intron-only
모델: BSCAN-FM, BSCAN-onehot, BSCAN-base, CircCNN, BSCAN-hnaug, BSCAN-FM-hnaug

### P5. RepeatMasker 기반 ALU 분석
- hg19 RepeatMasker track 다운로드 (UCSC)
- BS/LS intron의 ALU coverage 비교
- Tier 2 real vs. synthetic의 ALU density 비교
- ALU-density-matched Tier 2 생성 및 재평가
- 모델 score와 ALU coverage 상관 분석

### P6. Duplex α sensitivity
- α = 0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 sweep
- validation set으로 α 선택, external test에 고정
- sensitivity plot 생성

### HN 추가 probe
- Upper intron swap (현재는 lower만)
- Both introns swap
- ALU-density-matched Tier 2
- positive당 5 random swap → bootstrap CI

### P2. External set 감사 (부분)
- 내부 train과 circAtlas의 BSJ coordinate overlap 점검
- exact sequence duplicate (200nt) 제거
- GC%, repeat 비율 covariate distribution 비교

---

## 🔨 장기 실험 (재훈련 필요)

### P0 (필수). Adapter transcript-split 재훈련
```bash
# CNN adapter (5→10 seeds, transcript split)
for SEED in 42 123 315 777 1004 2024 2025 2026 3407 9001; do
  python experiment.py --model_name bscan_unified_fm_cnnadapter \
    --split_strategy transcript --epochs 100 --seed $SEED --device 0
done

# Mamba adapter (동일)
for SEED in ...; do
  python experiment.py --model_name bscan_unified_fm_mambaadapter \
    --split_strategy transcript --epochs 100 --seed $SEED --device 1
done
```
예상 시간: ~3–5일 (A6000 2장 병렬)

### P1 (필수). Architecture ablation
추가할 모델 조건:
| 모델명 | 설명 |
|--------|------|
| `bscan_unified_fm_mlponly` | FM + projection + MLP (no 3-branch) |
| `bscan_unified_fm_nocnn` | FM + stem + attn (CNN branch 제거) |
| `bscan_unified_fm_nostem` | FM + CNN + attn (stem branch 제거) |
| `bscan_unified_fm_noattn` | FM + CNN + stem (cross-attention 제거) |
| `bscan_unified_fm_stemonly` | FM + stem만 |

각 조건: 5 seeds (42, 123, 315, 777, 1004), transcript split  
평가: Internal AUC, External AUC, Tier2, Tier3

### P7. Multi-scale flank (선택)
- 250 / 500 / 1000 nt window
- BSCAN-RNA-FM, BSCAN-base, BSCAN-hnaug 3모델만

---

## 📊 데이터 파이프라인 정리 (P0 선행 작업)

```
1. master_manifest.csv 생성
   - columns: sample_id, chr, start, end, strand, label, transcript_id, gene_id,
              split_seed42, split_seed123, ..., sequence_200nt
   - 용도: 재현성 확보, coordinate overlap 점검

2. master_result.csv 생성
   - 모든 표와 그림이 하나의 CSV에서 자동 생성
   - 현재: paper_table_master.csv (부분 완성)

3. Genome build 확인
   - 내부 데이터: hg19 (BS_LS_coordinates_final.csv)
   - 외부 데이터: hg38 (circAtlas)
   - hg19→hg38 liftover 좌표 overlap 점검
```

---

## 🆕 추가할 baseline (Related Work / Comparison)

| 논문 | 년도 | 처리 방법 |
|------|------|----------|
| CircCNNs (single/double/tri) | 2024 Scientific Reports | ✅ 이미 포함 |
| circ2LO | 2025 | Task 동일성 확인 후 baseline 또는 Related Work |
| 2026 preprint (genomic LM) | 2026 | Discussion 최신 동향으로 언급 |

---

## 우선순위 요약

```
지금 당장:  ✅ 문서 수정 완료
1–2주 내:   P3 (통계), P4 (masking), P5 (ALU), P6 (duplex α), HN 추가
1달 내:     P0 (adapter transcript split 재훈련) — 논문 핵심 claim 확정
2달 내:     P1 (branch ablation), P2 (external audit 완성)
선택사항:   P7 (multi-scale), P8 (RBP motif)
```
