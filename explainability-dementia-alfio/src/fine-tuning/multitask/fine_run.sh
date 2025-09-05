#!/usr/bin/env bash
set -euo pipefail

# ================= CONFIGURAZIONE PERCORSI ==========================
# 1) Dataset
DS_PARENT_DIR="/home/tom/dataset_eeg"
DS_NAME="miltiadous_deriv_uV_d1.0s_o0.0s"

# 2) CSV monitor-CNN
MONITOR_DIR="/home/alfio/improving_dementia_detection_model/results_cnn_weighted"

# 3) Checkpoint GNN pre-addestrato
CKPT_GNN="/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt"

# 4) Classi
CLASSES="hc-ftd-ad"

# 5) Cartella output finale
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_ROOT="/home/alfio/improving_dementia_detection_model/results_multitask_ft"
SAVE_DIR="${SAVE_ROOT}/${TIMESTAMP}"
mkdir -p "${SAVE_DIR}"

# ================= FINE-TUNING MULTI-TASK ===========================
python3 train_multitask.py \
  -n "${DS_NAME}" -c "${CLASSES}" -p "${DS_PARENT_DIR}" \
  -m "${MONITOR_DIR}" -k "${CKPT_GNN}" -l 0.3 \
  --device "cuda:0" --num_epochs 30 --batch_size 64 --seed 1234 \
  --patience 5 --freeze_epochs 3

BEST_CKPT=$(ls -td local/checkpoints/multitask_* | head -n1)/best.pt
echo -e "\n>>> Best checkpoint individuato in: ${BEST_CKPT}"

# Esportiamo per il blocco Python
export BEST_CKPT SAVE_DIR DS_PARENT_DIR DS_NAME MONITOR_DIR

# ================= GENERAZIONE REPORT FINAL =========================
python3 - << 'PYCODE'
import os, torch, pandas as pd, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from torch_geometric.loader import DataLoader

from multitask_dataset import MultiTaskCWTGraphDataset
from multitask_model   import MultiTaskGNNCWT2D_Mk11_1sec

BEST_CKPT = os.getenv('BEST_CKPT')
SAVE_DIR  = os.getenv('SAVE_DIR')
DS_PARENT_DIR = os.getenv('DS_PARENT_DIR')
DS_NAME   = os.getenv('DS_NAME')
MONITOR_DIR = os.getenv('MONITOR_DIR')

# ===== Dataset & split (hard-coded come nel training) ===============
annot_fp = os.path.join(DS_PARENT_DIR, DS_NAME, 'annot_all_hc-ftd-ad.csv')
crop_dir = os.path.join(DS_PARENT_DIR, DS_NAME, 'cwt')
annot = pd.read_csv(annot_fp)

def sub(lst): return [f"sub-{s:03d}" for s in lst]
split_sub = {
    "train": sub([37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                  66,67,68,69,70,71,72,73,74,75,76,77,78,
                  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]),
    "val":   sub([54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28]),
    "test":  sub([60,61,62,63,64,65,84,85,86,87,88,
                  29,30,31,32,33,34,35,36])
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiTaskGNNCWT2D_Mk11_1sec.from_pretrained(BEST_CKPT, device)
model.eval()

os.makedirs(SAVE_DIR, exist_ok=True)

for split, subjects in split_sub.items():
    df = annot[annot.original_rec.isin(subjects)].reset_index(drop=True)
    csv_path = os.path.join(MONITOR_DIR, f'{split}_predictions_detailed.csv')
    ds = MultiTaskCWTGraphDataset(df, crop_dir, csv_path)
    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    num_workers=4, pin_memory=False)

    rows, y_true, y_pred = [], [], []
    idx = 0
    for batch in tqdm(dl, desc=f'{split.upper()}'):
        batch = batch.to(device)
        with torch.no_grad():
            log_main, _ = model(batch.x, batch.edge_index, batch.batch)
            soft = F.softmax(log_main, dim=1).cpu().numpy()
            act  = log_main.cpu().numpy()
            pred = soft.argmax(1)
        bs = pred.shape[0]
        for i in range(bs):
            row = df.iloc[idx+i]
            rows.append([row.crop_file,
                         act[i].tolist(),
                         split,
                         soft[i].tolist(),
                         int(pred[i]),
                         int(row.label),
                         row.original_rec,
                         row.crop_start_sample,
                         row.crop_end_sample])
        y_true.extend(df.label.iloc[idx:idx+bs])
        y_pred.extend(pred)
        idx += bs

    pd.DataFrame(rows, columns=[
        'crop_file','activation_values','dataset','softmax_values',
        'pred_label','true_label','original_rec',
        'crop_start_sample','crop_end_sample'
    ]).to_csv(os.path.join(SAVE_DIR, f'{split}_results.csv'), index=False)

    with open(os.path.join(SAVE_DIR, f'{split}_classification_report.txt'),'w') as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HC','FTD','AD'],
                yticklabels=['HC','FTD','AD'])
    plt.title(f'Confusion Matrix – {split}')
    plt.xlabel('Pred'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{split}_confusion_matrix.png'))
    plt.close()

print(f'\n[INFO] Risultati salvati in: {SAVE_DIR}')
PYCODE

echo -e "\n>>> Completato!"
