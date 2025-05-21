#!/usr/bin/env bash
set -euo pipefail

# ===================== CONFIGURAZIONE ================================
# 1) Dataset preprocessato
DS_PARENT="/home/tom/dataset_eeg"
DS_NAME="miltiadous_deriv_uV_d1.0s_o0.0s"

# 2) Cartella con i CSV del monitor-CNN
MONITOR_DIR="/home/alfio/improving_dementia_detection_model/results_cnn_weighted"

# 3) Checkpoint GNN (pre-allenato su hc-ftd-ad)
CKPT_GNN="/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt"

# 4) Parametri Re-weighting
ALPHA=0.3          # (ignora se usi --invert nello script python)
INVERT=false       # true  => w_i = 1 - goodness
                   # false => w_i = α + (1-α)*goodness

# 5) Cartella output
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="/home/alfio/improving_dementia_detection_model/results_reweight"
OUT_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT_DIR}"

# ===================== TRAINING RE-WEIGHT ============================
echo ">>> Avvio fine-tuning con Re-weighting ..."
python3 train_reweight.py \
  -n "${DS_NAME}" \
  -p "${DS_PARENT}" \
  -m "${MONITOR_DIR}" \
  -k "${CKPT_GNN}" \
  --alpha ${ALPHA} \
  $( $INVERT && echo "--invert" ) \
  --device "cuda:0" \
  --num_epochs 15 \
  --batch_size 64 \
  --seed 1234

BEST_CKPT="best_reweight.pt"   # salvato dallo script python
echo ">>> Best checkpoint: ${BEST_CKPT}"

# Esporta variabili d’ambiente per il blocco Python
export BEST_CKPT OUT_DIR DS_PARENT DS_NAME MONITOR_DIR

# ===================== GENERAZIONE REPORT ============================
python3 - << 'PYCODE'
import os, torch, pandas as pd, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from torch_geometric.loader import DataLoader

from reweight_dataset import ReweightCWTGraphDataset
from models           import GNNCWT2D_Mk11_1sec

# -------- variabili dal bash ----------------------------------------
BEST_CKPT  = os.getenv('BEST_CKPT')
OUT_DIR    = os.getenv('OUT_DIR')
DS_PARENT  = os.getenv('DS_PARENT')
DS_NAME    = os.getenv('DS_NAME')
MONITOR_DIR= os.getenv('MONITOR_DIR')

# -------- dataset e split come in train_reweight.py ------------------
ANNOT_FP = os.path.join(DS_PARENT, DS_NAME, 'annot_all_hc-ftd-ad.csv')
CROP_DIR = os.path.join(DS_PARENT, DS_NAME, 'cwt')
annot    = pd.read_csv(ANNOT_FP)

def sub(lst): return [f'sub-{s:03d}' for s in lst]
split_subj = {
    "train": sub([37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                  66,67,68,69,70,71,72,73,74,75,76,77,78,
                  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]),
    "val":   sub([54,55,56,57,58,59,79,80,81,82,83,
                  22,23,24,25,26,27,28]),
    "test":  sub([60,61,62,63,64,65,84,85,86,87,88,
                  29,30,31,32,33,34,35,36])
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GNNCWT2D_Mk11_1sec(19,(40,500),3).to(device)
state = torch.load(BEST_CKPT, map_location=device)
model.load_state_dict(state)
model.eval()

os.makedirs(OUT_DIR, exist_ok=True)

for split, subj in split_subj.items():
    df_split = annot[annot.original_rec.isin(subj)].reset_index(drop=True)
    csv_path = os.path.join(MONITOR_DIR, f'{split}_predictions_detailed.csv')
    ds = ReweightCWTGraphDataset(df_split, CROP_DIR, csv_path)
    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    num_workers=4, pin_memory=False)

    rows, y_true, y_pred = [], [], []
    idx = 0
    for batch in tqdm(dl, desc=f'{split.upper()}'):
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.batch)
            soft   = F.softmax(logits, dim=1).cpu().numpy()
            act    = logits.cpu().numpy()
            pred   = soft.argmax(1)

        bs = pred.shape[0]
        for i in range(bs):
            row = df_split.iloc[idx+i]
            rows.append([row.crop_file,
                         act[i].tolist(),
                         split,
                         soft[i].tolist(),
                         int(pred[i]),
                         int(row.label),
                         row.original_rec,
                         row.crop_start_sample,
                         row.crop_end_sample])
        y_true.extend(df_split.label.iloc[idx:idx+bs])
        y_pred.extend(pred)
        idx += bs

    # CSV risultati
    pd.DataFrame(rows, columns=[
        'crop_file','activation_values','dataset','softmax_values',
        'pred_label','true_label','original_rec',
        'crop_start_sample','crop_end_sample'
    ]).to_csv(os.path.join(OUT_DIR, f'{split}_results.csv'), index=False)

    # classification report
    with open(os.path.join(OUT_DIR, f'{split}_classification_report.txt'),'w') as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HC','FTD','AD'],
                yticklabels=['HC','FTD','AD'])
    plt.title(f'Confusion Matrix – {split}')
    plt.xlabel('Pred'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{split}_confusion_matrix.png'))
    plt.close()

print(f"\n[INFO] Tutti i risultati sono in: {OUT_DIR}")
PYCODE

echo -e "\n>>> Pipeline completata!"
