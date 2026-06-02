#!/bin/bash
# BSCAN 5-seed final comparison
# BSCAN vs key baselines: circcnnatt, circcnn, circstemv2, circbialign, circmotif

SEEDS=(42 123 315 777 1004)
DEVICE=0
MODELS="bscan circcnnatt circcnn circstemv2 circbialign circmotif"

echo "BSCAN 5-seed comparison: $MODELS"
echo "=================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">> Seed $SEED"
    python run_model_comparison.py \
        --epochs 100 \
        --earlystop 10 \
        --batch_size 128 \
        --device $DEVICE \
        --seed $SEED \
        --models $MODELS \
        --tag "bscan_final_seed_${SEED}" \
        --out_dir "research_results"
    echo "[done] Seed $SEED"
done

echo ""
echo "All seeds done. Computing summary..."

python3 - << 'PYEOF'
import csv, glob, statistics
from collections import defaultdict

model_aucs = defaultdict(list)
model_accs = defaultdict(list)
model_f1s  = defaultdict(list)
model_mccs = defaultdict(list)

for path in sorted(glob.glob('research_results/model_comparison_bscan_final_seed_*.csv')):
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['success'] == 'True' and row['test_auc']:
                m = row['model']
                model_aucs[m].append(float(row['test_auc']))
                model_accs[m].append(float(row['test_acc']))
                model_f1s[m].append(float(row['test_macro_f1']))
                model_mccs[m].append(float(row['test_mcc']))

ORDER = ['bscan', 'circcnnatt', 'circstemv2', 'circbialign', 'circmotif', 'circcnn']

print('\n' + '='*85)
print(f"{'Model':<22} {'AUC':>14} {'Acc':>14} {'F1':>10} {'MCC':>10}  n")
print('='*85)

for m in ORDER:
    aucs = model_aucs.get(m, [])
    if not aucs:
        continue
    n = len(aucs)
    am = statistics.mean(aucs); as_ = statistics.stdev(aucs) if n > 1 else 0
    cm = statistics.mean(model_accs[m]); cs = statistics.stdev(model_accs[m]) if n > 1 else 0
    f1 = statistics.mean(model_f1s[m])
    mc = statistics.mean(model_mccs[m])
    marker = " ←" if m == 'bscan' else ""
    print(f"{m:<22} {am:.4f}±{as_:.4f}  {cm:.4f}±{cs:.4f}  {f1:.4f}  {mc:.4f}  {n}{marker}")

print('='*85)

# Save CSV
import os
os.makedirs('research_results', exist_ok=True)
with open('research_results/bscan_final_comparison.csv', 'w', newline='') as f:
    import csv as csv_mod
    w = csv_mod.DictWriter(f, fieldnames=['model','n_seeds','auc_mean','auc_std','acc_mean','acc_std','f1_mean','mcc_mean'])
    w.writeheader()
    for m in ORDER:
        aucs = model_aucs.get(m, [])
        if not aucs:
            continue
        n = len(aucs)
        w.writerow(dict(
            model=m, n_seeds=n,
            auc_mean=round(statistics.mean(aucs), 4),
            auc_std=round(statistics.stdev(aucs) if n > 1 else 0, 4),
            acc_mean=round(statistics.mean(model_accs[m]), 4),
            acc_std=round(statistics.stdev(model_accs[m]) if n > 1 else 0, 4),
            f1_mean=round(statistics.mean(model_f1s[m]), 4),
            mcc_mean=round(statistics.mean(model_mccs[m]), 4),
        ))
print('\nSaved: research_results/bscan_final_comparison.csv')
PYEOF
