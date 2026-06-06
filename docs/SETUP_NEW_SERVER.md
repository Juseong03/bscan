# 새 GPU 서버 셋업 가이드

git에는 **코드와 소형 입력 데이터만** 들어있습니다. 대용량 데이터·임베딩·체크포인트는 직접 전송해야 합니다.

---

## 1. 코드 받기 (git이 자동 제공)

```bash
git clone https://github.com/Juseong03/bscan.git
cd bscan
```

git이 주는 것 (수정→push→pull로 자동 동기화):
- 전체 코드: `core/`, `pipeline/`, `analysis/`, `models/`, `scripts/`
- 소형 입력 데이터: `data/BS_LS_coordinates_final.csv` (좌표), `data/hg38_exon.bed`, `data/human_bed_v3.0/`
- 문서·그림·결과 요약: `docs/`, `figures/`, `results/`, `research_results/*_summary.csv`

> **경로 자동 탐지**: 모든 스크립트는 `__file__` 기준으로 repo 루트를 찾습니다. clone 경로가 어디든(`/home/you/bscan` 등) 그대로 작동합니다. **단, 항상 repo 루트에서 실행** (`python pipeline/xxx.py`).

---

## 2. 직접 전송해야 하는 파일 (git 제외, 용량순)

| 경로 | 크기 | 필수도 | 대안 |
|------|-----:|--------|------|
| `fm_embeddings/` | 64G | 학습/평가 필수 | `pipeline/extract_fm_embeddings.py`로 재생성 (GPU 시간 소요) |
| `external_data/circatlas/exon_controls/` | 22G | 외부검증 필수 | `make_circatlas_exon_controls.py` + `extract_external_fm_embeddings.py`로 재생성 |
| `saved_models/` | 4.6G | 기존 체크포인트 재사용 시 | 재학습하면 자동 생성됨 (전송 불필요) |
| `data/hg19_seq_dict.json` | 2.9G | **모든 실험 필수** (게놈) | 없음 — 반드시 전송 |
| `data/rmsk_hg19.txt.gz` | 141M | ALU 분석만 | UCSC에서 재다운로드 (아래) |

### 전송 예시 (rsync)
```bash
# 이 서버 → 새 서버 (최소 구성: 게놈 + 임베딩)
rsync -av data/hg19_seq_dict.json   NEWSERVER:~/bscan/data/
rsync -av fm_embeddings/            NEWSERVER:~/bscan/fm_embeddings/
# 외부검증/기존 체크포인트도 쓰려면
rsync -av external_data/            NEWSERVER:~/bscan/external_data/
rsync -av saved_models/             NEWSERVER:~/bscan/saved_models/
```

### rmsk 재다운로드 (ALU 분석 시에만)
```bash
wget -O data/rmsk_hg19.txt.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/rmsk.txt.gz
```

---

## 3. symlink 주의 (해결됨)

이 서버에서 `saved_models/`, `external_data/`는 옛 프로젝트를 가리키는 **symlink**였습니다.
이들은 `.gitignore`에 등록돼 **git에 포함되지 않습니다** → 새 서버 clone엔 애초에 따라오지 않습니다.

새 서버에서는:
- `saved_models/` — 학습 시 `trainer.py`가 **자동 생성** (일반 디렉토리). 전송 불필요.
- `external_data/` — 위 2절대로 전송하거나 재생성하면 **일반 디렉토리**로 생성됨.

→ 새 서버엔 깨진 symlink가 생기지 않습니다.

---

## 4. 환경

```bash
conda create -n bscan python=3.10 && conda activate bscan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# ALU 분석 시: pip install pyliftover  (hg19→hg38)
```

---

## 5. 실행 확인

```bash
# 1) 빠른 무결성 체크 (게놈만 있으면 OK)
python pipeline/smoke_models.py

# 2) 단일 학습
python pipeline/experiment.py --model_name bscan_unified_fm \
    --split_strategy transcript --device 0 --seed 42

# 3) 모델 비교 스윕
python pipeline/run_model_comparison.py --models bscan circcnn --epochs 100 --device 0
```

---

## 6. 양방향 동기화

```bash
# 새 서버에서 수정 후 올리기
git add -A && git commit -m "..." && git push

# 다른 곳에서 최신 받기
git pull
```

> 데이터·임베딩은 `.gitignore`로 제외돼 있어 push에 섞이지 않습니다. 코드만 깔끔하게 동기화됩니다.
