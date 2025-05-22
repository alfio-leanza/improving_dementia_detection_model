#!/usr/bin/env bash
set -euo pipefail

# ======= Paths =======================================================
DS_PARENT="/home/tom/dataset_eeg"
DS_NAME="miltiadous_deriv_uV_d1.0s_o0.0s"
MONITOR_DIR="/home/alfio/improving_dementia_detection_model/results_cnn_weighted"
CKPT_GNN="/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt"

ALPHA=0.3
INVERT=false         # true -> w_i = 1-goodness
LS_EPS=0.1          # label-smoothing ε

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="/home/alfio/improving_dementia_detection_model/results_reweight_ls"
OUT_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT_DIR}"

# ======= TRAIN =======================================================
python3 train_reweight.py \
  -n "${DS_NAME}" -p "${DS_PARENT}" -m "${MONITOR_DIR}" -k "${CKPT_GNN}" \
  --alpha ${ALPHA} $( $INVERT && echo "--invert" ) --ls_eps ${LS_EPS} \
  --device cuda:0 --num_epochs 50 --batch_size 64 --lr 2e-5 --seed 1234 \
  --weight_decay 1e-7 --p_drop 0.5

BEST_CKPT="best_reweight_ls.pt"
export BEST_CKPT OUT_DIR DS_PARENT DS_NAME MONITOR_DIR

# ======= REPORT ======================================================
python3 - << 'PYCODE'
import os, torch, pandas as pd, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from torch_geometric.loader import DataLoader

from reweight_dataset import ReweightCWTGraphDataset
from models           import GNNCWT2D_Mk11_1sec

BEST_CKPT=os.getenv('BEST_CKPT'); OUT_DIR=os.getenv('OUT_DIR')
DS_PARENT=os.getenv('DS_PARENT'); DS_NAME=os.getenv('DS_NAME')
MONITOR_DIR=os.getenv('MONITOR_DIR')

annot_fp=os.path.join(DS_PARENT,DS_NAME,'annot_all_hc-ftd-ad.csv')
crop_dir=os.path.join(DS_PARENT,DS_NAME,'cwt')
annot=pd.read_csv(annot_fp)

def sub(l): return [f"sub-{s:03d}"for s in l]
split_sub={'train':sub([37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]),
            'val':  sub([54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28]),
            'test': sub([60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36])}

device='cuda' if torch.cuda.is_available() else 'cpu'
model=GNNCWT2D_Mk11_1sec(19,(40,500),3).to(device)
model.load_state_dict(torch.load(BEST_CKPT,map_location=device)); model.eval()

for split,subj in split_sub.items():
    df=annot[annot.original_rec.isin(subj)].reset_index(drop=True)
    csv=os.path.join(MONITOR_DIR,f'{split}_predictions_detailed.csv')
    ds=ReweightCWTGraphDataset(df,crop_dir,csv)
    dl=DataLoader(ds,batch_size=64,shuffle=False,num_workers=4,pin_memory=False)

    rows,y_t,y_p=[],[],[]
    idx=0
    for batch in tqdm(dl,desc=split.upper()):
        batch=batch.to(device)
        with torch.no_grad():
            log=model(batch.x,batch.edge_index,batch.batch)
            soft=F.softmax(log,dim=1).cpu().numpy()
            act=log.cpu().numpy(); pred=soft.argmax(1)
        bs=pred.shape[0]
        for i in range(bs):
            row=df.iloc[idx+i]
            rows.append([row.crop_file,act[i].tolist(),split,soft[i].tolist(),
                         int(pred[i]),int(row.label),row.original_rec,
                         row.crop_start_sample,row.crop_end_sample])
        y_t.extend(df.label.iloc[idx:idx+bs]); y_p.extend(pred); idx+=bs

    pd.DataFrame(rows,columns=['crop_file','activation_values','dataset',
              'softmax_values','pred_label','true_label','original_rec',
              'crop_start_sample','crop_end_sample']
    ).to_csv(os.path.join(OUT_DIR,f'{split}_results.csv'),index=False)

    with open(os.path.join(OUT_DIR,f'{split}_classification_report.txt'),'w') as f:
        f.write(classification_report(y_t,y_p,digits=4))

    cm=confusion_matrix(y_t,y_p)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=['HC','FTD','AD'],yticklabels=['HC','FTD','AD'])
    plt.title(f'Confusion Matrix – {split}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f'{split}_confusion_matrix.png')); plt.close()

print(f'\n[INFO] Risultati salvati in: {OUT_DIR}')
PYCODE

echo -e "\n>>> Pipeline completata!"
