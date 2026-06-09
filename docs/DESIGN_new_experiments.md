# 신규 실험 상세 설계: ABL-CTX & AUG-RCM

> 두 실험 모두 핵심 질문은 같다: **"BSCAN-FM에 더 넓은 intronic 맥락(flanking)을 주면
> 외부 일반화가 개선되는가, 아니면 exon-bias가 구조적 한계인가?"**
> 접근이 다르다 — ABL-CTX는 *입력 윈도우 확장*, AUG-RCM은 *보조 피처 추가*.
> 참조: `docs/EXPERIMENTS.md` (레지스트리), `docs/RESULTS_DRAFT.md` (기존 결과)

---

## 공통 배경 (왜 지금 이 실험인가)

기존 결과가 만든 긴장:
- **VAL-EXT/VAL-LEAK**: BSCAN-FM은 외부 일반화 1위(~0.84), exon shortcut에 강함.
- **MECH-HN3/HNAUG**: 그러나 FM은 intron 신호를 못 본다 — Tier2 chance 이하, hnaug로도 Tier3 회복 안 됨(0.51).
- **MECH-ALU**: ALU(역상보 매치)는 BS 인트론에 실재하나 100nt 창엔 91%가 ALU 없음 — 신호가 **더 넓은 창**에 있음.
- **AUG-DUPLEX**: flanking 기반 thermodynamic 신호를 줘도 외부 +0.002 (거의 무효).

→ 자연스러운 다음 질문: **"100nt 창이 좁아서 FM이 intron 신호를 못 본 것인가? 창을 넓히거나 RCM을 직접 주면 달라지나?"**
이 실험은 우리 논문의 핵심 주장("FM exon-bias는 구조적")을 **반증 시도(falsification)** 한다. 어느 결과든 논문에 기여한다.

---

# 실험 1 — ABL-CTX (Context-Window Study)

## 1.1 정의
입력 junction 윈도우 `junction_bps`를 100 → 250 → 500으로 확장하여, **모든 모델이 동일하게 더 넓은 intronic 맥락**을 보도록 한 뒤 내부·외부 성능을 재측정한다.
- `upper_seq` = [intron junction_bps] + [exon junction_bps] → 길이 2·junction_bps
- FM 임베딩을 각 윈도우에서 재추출, BSCAN(및 비교군) 재학습

하위 실험:
- **ABL-CTX-WIN** (main): junction_bps ∈ {100, 250, 500}에서 BSCAN-FM + 핵심 대조군 재학습
- **ABL-CTX-BASE** (supplementary): circcnntri의 RCM flanking ∈ {100…500} 민감도 (공정비교 아님 — 별도 보고)

## 1.2 의미 (무엇을 검증하나)
| 관찰 | 해석 |
|------|------|
| 창 ↑ 해도 외부 AUC 정체 | **exon-bias는 구조적** — 맥락 부족이 아니라 FM 표현 자체의 한계 (핵심 주장 강화) |
| 창 ↑ 시 외부 AUC 상승 | intron 맥락이 실제로 도움 — "100nt가 좁았다"는 새 발견 (주장 수정) |
| 창 ↑ 시 Tier2/3도 상승 | FM이 더 넓은 창에선 intron-type 신호를 일부 포착 |

핵심: **공정성**. 모든 모델이 같은 junction_bps를 받으므로 head-to-head 비교가 유효 (circcnntri만 flank 키우는 것과 다름).

## 1.3 가설
duplex(+0.002)·hnaug(FM Tier3 0.51)가 무효였던 점으로 보아 → **외부 AUC 큰 변화 없을 것**으로 예상. 정체가 관찰되면 "맥락을 5배 늘려도 FM exon-bias는 견고하다"는 강한 진술 가능.

## 1.4 방법
**데이터:** 각 junction_bps마다 `data/seq_dict/{jbps}/junction.json` 게놈에서 생성 (hg19_seq_dict.json 필요).
**FM 임베딩:** 윈도우별 재추출.
- ⚠️ **선결 과제(캐시 키 함정):** 현재 `circData_cached_fm`는 `./fm_embeddings/{enc}/`로만 키됨 → junction_bps가 달라도 같은 폴더에 덮어씀. **`fm_embeddings/{enc}_jb{jbps}/`로 분리**하도록 `extract_fm_embeddings.py`와 `circData_cached_fm`(dataloader) 수정 필요.
- FM 길이 한계: junction_bps=500 → 1000토큰, RNA-FM max(~1024) 근처. 500이 상한.

**모델:** `bscan_unified_fm` + 대조군 `bscan`(onehot), `circcnn`. (전 모델 동일 윈도우)
**설정:** transcript split, seeds 42/123/315, 그 외 기본.
**평가:** 내부 AUC, 외부 AUC(circAtlas — 외부도 동일 jbps로 재구축·재추출 필요), Tier2/3.

## 1.5 산출물
`results/context_window_summary.csv` — 행: (model × junction_bps), 열: int_auc, ext_auc, tier2, tier3.
그림: junction_bps(x) vs 외부 AUC(y), 모델별 선.

## 1.6 구현 단계
1. 캐시 키 수정 (`extract_fm_embeddings.py` out_dir, `dataloader.circData_cached_fm` cache_dir → `{enc}_jb{jbps}`)
2. 외부셋도 jbps별 재구축 (`make_circatlas_exon_controls` + 외부 임베딩)
3. 스윕 스크립트 `scripts/run_context_window_sweep.sh` (jbps × model × seed)
4. 평가 집계 `analysis/evaluate_context_window.py`

## 1.7 리스크
- 비용 큼: jbps 3종 × (내부24K+외부8K) × 4 encoder 임베딩 재추출 + 재학습. **먼저 rnafm 1종 × jbps 250으로 파일럿** 권장.
- 외부셋 재구축까지 필요 → 누락 시 외부 AUC 비교 불가.

---

# 실험 2 — AUG-RCM (RCM Auxiliary Branch)

## 2.1 정의
200nt junction 윈도우는 **그대로 두고**, flanking에서 추출한 **RCM(reverse-complement match) 피처를 BSCAN의 4번째 branch**로 추가하여, intron 상보성 신호를 명시적으로 주입했을 때 성능 변화를 측정한다. (AUG-DUPLEX의 RCM 버전; duplex=열역학 스칼라, RCM=k-mer 매치 분포)

## 2.2 의미
- AUG-DUPLEX는 thermodynamic 스칼라 1개를 logit에 더하는 **post-hoc 결합**이었고 거의 무효(+0.002).
- AUG-RCM은 RCM 분포(5 kmer × 3 region)를 **학습 가능한 branch**로 융합 — 더 강한 주입.
- **차별점:** 모델이 학습 중 RCM을 쓸 수 있게 함 → "FM이 못 본 intron 신호를 외부에서 떠먹여주면 일반화가 오르나?"

| 관찰 | 해석 |
|------|------|
| 외부 AUC 상승 | intron 상보성 신호가 보조로 주입되면 도움 — 실용적 개선 |
| 변화 없음 | FM exon-bias가 보조 신호로도 보정 안 됨 (구조적 한계 재확인, duplex와 일관) |

## 2.3 가설
duplex가 무효였으므로 RCM도 외부 AUC는 미미할 가능성. 단 Tier2/3(intron 구분)에서는 RCM이 명시적 신호라 상승할 수 있음 → "RCM branch는 intron-type은 보지만 외부 일반화엔 무영향" 같은 분리된 결과 가능.

## 2.4 방법
**RCM 데이터:** `pipeline/generate_rcm_scores_subset.py --max_samples 100000 --flanking_bps {100|500}` → `rcm_scores/{flanking,upper,lower}_{F}_bps_{k}mer.json`. 학습 시 `seq_to_tensor_w_rcm`로 로드 (이미 `circData_rcm`, `circcnntri` 경로 존재).
**모델 변경 (`models/bscan_unified.py`):**
- `use_rcm: bool` 플래그 + `self.rcm_mlp` (입력: flanking/upper/lower × 5 kmer × 25 = 375차원 → MLP → d_rcm)
- forward에서 `parts.append(rcm_feat)`, `total_d += d_rcm`
- 입력 경로: `circData_*`에 rcm 텐서 추가 전달 (circcnntri의 `circData_rcm` 패턴 재사용)
**변형:** `bscan_unified_fm` vs `bscan_unified_fm_rcm` (flanking 100, 500 각각).
**설정:** transcript split, seeds 42/123/315.
**평가:** 내부/외부 AUC + Tier2/3 (RCM이 intron 신호를 주입했으니 Tier 변화가 핵심 관전 포인트).

## 2.5 산출물
`results/rcm_aux_summary.csv` — 행: (bscan_unified_fm, +rcm@100, +rcm@500), 열: int/ext/tier2/tier3.

## 2.6 구현 단계
1. RCM 추출 (`generate_rcm_scores_subset.py`, 전체 샘플)
2. `BSCANUnified`에 `use_rcm` branch 추가 + trainer/experiment 입력 배선
3. 학습 + `analysis/evaluate_rcm_aux.py` 집계

## 2.7 리스크
- 가벼움(임베딩 재추출 불필요)이 장점. 주 리스크는 BSCAN forward에 rcm 입력 배선(데이터로더 분기) 복잡도.
- RCM은 flanking 기반 → flanking_500 쓰려면 게놈 필요(ABL-CTX와 공유).

---

## 우선순위 권장
1. **AUG-RCM 먼저** (가벼움, 임베딩 재추출 불필요) — "보조 intron 신호" 질문에 빠르게 답
2. **ABL-CTX 파일럿** (rnafm × jbps 250) — 효과 있으면 500·4encoder로 확대
3. 두 결과를 묶어 논문에 **"intron 맥락을 윈도우(ABL-CTX)로도 보조피처(AUG-RCM)로도 줬으나 FM 외부 일반화는 exon-bias에 지배된다"** 단일 메시지로 통합 가능

## 예상 논문 기여
- 긍정/부정 어느 쪽이든 §3에 1개 섹션 추가 (예: §3.11 "Context and auxiliary intron signals do not overcome FM exon-bias")
- 핵심 주장(exon-bias 구조성)을 **2개의 독립적 개입(window, aux-feature)** 으로 추가 검증 → robustness 강화
