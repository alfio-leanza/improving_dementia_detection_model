#!/usr/bin/env bash
set -euo pipefail

# ============== CONFIGURAZIONE PERCORSI ===============
# 1) Dataset CWT (cartella che contiene 'cwt/' e i file annot)
DS_PARENT_DIR="/home/tom/dataset_eeg"
DS_NAME="miltiadous_deriv_uV_d1.0s_o0.0s"

# 2) Cartella con i CSV del monitor-CNN (contiene i tre *_predictions_detailed.csv)
MONITOR_DIR="/home/alfio/improving_dementia_detection_model/results_cnn_weighted"

# 3) Checkpoint GNN pre-addestrato da cui partire
CKPT_GNN="/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt"

# 4) Classi usate
CLASSES="hc-ftd-ad"

# 5) Cartella di destinazione dei RISULTATI FINALI
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_ROOT="/home/alfio/improving_dementia_detection_model/results_multitask_ft"
SAVE_DIR="${SAVE_ROOT}/${TIMESTAMP}"
mkdir -p "${SAVE_DIR}"

# ============== FINE-TUNING MULTI-TASK ================
echo ">>> Avvio fine-tuning multi-task ..."
python train_multitask.py \
  -n "${DS_NAME}" \
  -c "${CLASSES}" \
  -p "${DS_PARENT_DIR}" \
  -m "${MONITOR_DIR}" \
  -k "${CKPT_GNN}" \
  -l 0.3 \
  --device "cuda:0" \
  --num_epochs 30 \
  --batch_size 64 \
  --seed 1234

# Recupera automaticamente l’ultimo run multitask_* e prende il best.pt
BEST_CKPT=$(ls -td local/checkpoints/multitask_* | head -n1)/best.pt
echo ">>> Best checkpoint individuato in: ${BEST_CKPT}"

# ============== GENERAZIONE REPORT ====================
echo ">>> Generazione report (CSV, TXT, PNG) ..."
python - << 'PYCODE'
import os, torch, pandas as pd, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from torch_geometric.loader import DataLoader
from multitask_dataset import MultiTaskCWTGraphDataset
from multitask_model   import MultiTaskGNNCWT2D_Mk11_1sec

# ---------- PATH passati dal wrapper bash ----------------
BEST_CKPT = os.environ['BEST_CKPT']
SAVE_DIR  = os.environ['SAVE_DIR']
DS_PARENT_DIR = os.environ['DS_PARENT_DIR']
DS_NAME   = os.environ['DS_NAME']
MONITOR_DIR = os.environ['MONITOR_DIR']

# ---------- Dataset & split identici a train_multitask ----
ANNOT_FP = os.path.join(DS_PARENT_DIR, DS_NAME, 'annot_all_hc-ftd-ad.csv')
CROP_PATH = os.path.join(DS_PARENT_DIR, DS_NAME, 'cwt')
annot = pd.read_csv(ANNOT_FP)

split_subj = {
    "train": [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    "val":   [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    "test":  [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}
for k in split_subj:
    split_subj[k] = [f"sub-{s:03d}" for s in split_subj[k]]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskGNNCWT2D_Mk11_1sec.from_pretrained(BEST_CKPT, device)
model.eval()

os.makedirs(SAVE_DIR, exist_ok=True)

for split, subjects in split_subj.items():
    df_split = annot[annot.original_rec.isin(subjects)].reset_index(drop=True)
    csv_path = os.path.join(MONITOR_DIR, f"{split}_predictions_detailed.csv")
    ds = MultiTaskCWTGraphDataset(df_split, CROP_PATH, csv_path)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    records, y_true, y_pred = [], [], []
    idx = 0
    for data in tqdm(dl, desc=f"{split.upper()}"):
        data = data.to(device)
        with torch.no_grad():
            logits_main, _ = model(data.x, data.edge_index, data.batch)
            soft = F.softmax(logits_main, dim=1).cpu().numpy()
            acts = logits_main.cpu().numpy()
            preds = soft.argmax(1)

        batch_size = preds.shape[0]
        # per ogni sample ricava meta-info dal DF originale (stessa ordine se shuffle=False)
        for i in range(batch_size):
            row = df_split.iloc[idx + i]
            records.append([
                row.crop_file,
                acts[i].tolist(),
                split,
                soft[i].tolist(),
                int(preds[i]),
                int(row.label),
                row.original_rec,
                row.crop_start_sample,
                row.crop_end_sample
            ])
        y_true.extend(df_split.label.iloc[idx:idx+batch_size])
        y_pred.extend(preds)
        idx += batch_size  # avanza l’indice globale

    # ---- salva CSV ----
    out_csv = os.path.join(SAVE_DIR, f"{split}_results.csv")
    pd.DataFrame(records, columns=[
        "crop_file","activation_values","dataset","softmax_values",
        "pred_label","true_label","original_rec",
        "crop_start_sample","crop_end_sample"
    ]).to_csv(out_csv, index=False)

    # ---- classification report + confusion matrix ----
    report_txt = os.path.join(SAVE_DIR, f"{split}_classification_report.txt")
    with open(report_txt, "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HC','FTD','AD'],
                yticklabels=['HC','FTD','AD'])
    plt.title(f'Confusion Matrix – {split}')
    plt.xlabel('Pred'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{split}_confusion_matrix.png"))
    plt.close()

print(f"\n[INFO] Tutti i risultati salvati in: {SAVE_DIR}")
PYCODE

echo ">>> Completato!"
