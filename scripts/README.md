# BSCAN scripts — quick reference

All scripts are run **from the repo root** (e.g. `bash scripts/<name>.sh ...`).
GPU ids assume a 3-GPU server (0,1,2); adjust args as needed.

---

## 🟢 Main reproduction workflow (use these)

| Script | What it does | Typical call |
|---|---|---|
| **run_multi_gpu.sh** | 3-GPU dispatcher. Splits everything across GPUs (seeds packed per GPU). | see below |
| **run_all_experiments.sh** | Single-GPU orchestrator (phased). `run_multi_gpu` calls into this. | `bash scripts/run_all_experiments.sh <phase> <gpu> "<seeds>"` |

### `run_multi_gpu.sh [STAGE] [SEEDS] [JOBS_PER_GPU]`
- **STAGE**: `emb` · `main` · `external` · `newexp` · `all` (default `all`)
- **SEEDS**: quoted list (default = paper's 10 seeds). Pilot: `"1 2 3"`
- **JOBS_PER_GPU**: concurrent jobs per GPU (default 2; use **3–4 on 40GB**)

```bash
# Recommended full run (10 seeds, pack 4/GPU), under tmux:
tmux new -s bscan
bash scripts/run_multi_gpu.sh main "" 4 && \
bash scripts/run_multi_gpu.sh external && \
bash scripts/run_multi_gpu.sh newexp

# Fast pilot first (3 seeds, 2/GPU) to gauge per-job memory:
bash scripts/run_multi_gpu.sh main "1 2 3" 2
```

Stage → GPU split: `main` packs seeds across GPUs · `external` runs once on GPU0
· `newexp` = AUG-RCM(GPU0) / ABL-CTX jb250(GPU1) / jb500(GPU2).

### Manual per-experiment GPU assignment (alternative to the dispatcher)
Every script below takes an explicit GPU id, so you can place experiments yourself:
```bash
bash scripts/run_all_experiments.sh train 0 "1 2 3 4" &   # GPU0
bash scripts/run_all_experiments.sh train 1 "5 6 7"     &   # GPU1
bash scripts/run_all_experiments.sh train 2 "8 9 10"     &   # GPU2
wait
```
`run_all_experiments.sh` phases: `emb · train · external · ablation · hardneg · analysis · all`.

---

## 🖥️ Server launchers — one experiment fills all GPUs (scripts/server/)

`scripts/server/<exp>.sh` runs a SINGLE experiment but spreads its seeds across
all GPUs (round-robin + packing). Run experiments one at a time on the server.

```bash
# 0) once
bash scripts/server/prep.sh 0

# 1) headline (10 seeds, packed 2/GPU over GPUs 0,1,2) — under tmux/nohup:
nohup bash scripts/server/val_int.sh 2      > logs/exp/val_int.run      2>&1 &
#   ...wait for val_int, then:
nohup bash scripts/server/abl_branch.sh 2   > logs/exp/abl_branch.run   2>&1 &
nohup bash scripts/server/mech_hardneg.sh 2 > logs/exp/mech_hardneg.run 2>&1 &

# 2) external + analysis (after val_int; inference-only, GPU0)
bash scripts/server/val_ext.sh 0
bash scripts/server/analysis.sh 0

# 3) auxiliary (5 seeds; each splits its configs across 2 GPUs)
bash scripts/server/aug_rcm.sh        # flanking 100→GPU0, 500→GPU1
bash scripts/server/abl_ctx.sh        # jb250→GPU1, jb500→GPU2
```

- arg 1 of the seed-parallel launchers = **JOBS_PER_GPU** (default 2; 3–4 on 40GB).
- 2-GPU box: `export BSCAN_GPUS="0 1"` first.
- progress: `bash scripts/exp_status.sh`. Don't start two heavy launchers at once
  (each already uses all GPUs).

---

## 🧩 Per-experiment scripts (one GPU each, manual)

`scripts/exp/<name>.sh [GPU] [SEEDS]` — run a single experiment on a GPU you
pick. Each appends an `EXP_DONE <name> seed=<S>` marker to `logs/exp/`, so
completion is counted straight from the logs (no bookkeeping).

```bash
bash scripts/exp/prep.sh 0            # ONCE first: rcm_scores + external seq_dict/emb

# then place experiments on GPUs yourself (40GB → run several at once):
bash scripts/exp/val_int.sh 0 "1 2 3 4" &   # GPU0
bash scripts/exp/val_int.sh 1 "5 6 7"    &   # GPU1
bash scripts/exp/val_int.sh 2 "8 9 10"    &   # GPU2
wait
bash scripts/exp/val_ext.sh 0                        # external (after val_int)
bash scripts/exp/analysis.sh 0
bash scripts/exp/aug_rcm.sh 1 "1 2 3 4 5"
bash scripts/exp/abl_ctx.sh 2 "1 2 3 4 5" "250"
```

Experiments: `prep · val_int · abl_branch · mech_hardneg · val_ext · analysis ·
aug_rcm · abl_ctx`. Check completion any time:

```bash
bash scripts/exp_status.sh            # done/fail counts per experiment + seeds done
watch -n 30 bash scripts/exp_status.sh
```

---

## 🧪 New experiments (standalone)

| Script | Experiment | Call |
|---|---|---|
| **run_rcm_aux.sh** | AUG-RCM — RCM auxiliary branch | `bash scripts/run_rcm_aux.sh "100 500" <gpu> "<seeds>"` |
| **run_context_window_sweep.sh** | ABL-CTX — context-window sweep | `bash scripts/run_context_window_sweep.sh "250" rnafm <gpu> "<seeds>"` |

Both auto-build their inputs (RCM scores / wider-window embeddings). Aggregate
AUG-RCM with `python analysis/evaluate_rcm_aux.py`.

---

## 🔧 Data prep

| Script | What it does | Call |
|---|---|---|
| **extract_all_fm_embeddings.sh** | Extract FM embeddings (internal/external). Resumable (skips existing). | `bash scripts/extract_all_fm_embeddings.sh <gpu> "rnafm rnabert rnaernie rnamsm" both 256` |

(One-off external seq_dict build is `python pipeline/build_circatlas_seq_dict.py`,
which `run_all_experiments.sh` invokes automatically.)

---

## 📊 Monitoring / checks

| Script | What it shows | Call |
|---|---|---|
| **check_embeddings.sh** | Per-encoder `.pt` counts vs expected (internal + external). | `bash scripts/check_embeddings.sh` |
| **check_progress.sh** | GPU activity, live log tails, per-seed completion, checkpoints, result CSVs. | `bash scripts/check_progress.sh` · `watch -n 30 bash scripts/check_progress.sh` |

Handy raw commands:
```bash
nvidia-smi                                   # GPU utilisation
tail -f logs/multigpu/main_w0_gpu0.log        # live worker log
find -L saved_models -name model.pth | wc -l     # checkpoints done
grep -riE "error|traceback|out of memory" logs/multigpu/ | tail
```

---

## 🗄️ Standalone / legacy (kept for reference)

| Script | Note |
|---|---|
| run_publication_classification_comparison.sh | Original onehot-baseline comparison (3 seeds). Superseded by `run_all_experiments train`. |
| run_publication_regression_comparison.sh | Expression-regression variant. |
| run_rcm_flanking_sweep.sh | RCM-score generation across flanking widths (used by AUG-RCM/ABL-CTX prep). |
| run_bscan_comparison.sh | Old 5-seed internal sweep of early BSCAN variants. |
| run_transcript_holdout_bscan_foundation.sh | Early transcript-holdout probe. |
| run_hard_negative_pairing.sh | Thin wrapper around `evaluate_hard_negative_pairing.py` (env-var driven). |
| run_gpu0.sh / run_gpu1.sh | **Deprecated** ad-hoc 2-model launchers — replaced by `run_multi_gpu.sh`. |
