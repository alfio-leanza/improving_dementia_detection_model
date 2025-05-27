#!/usr/bin/env python
# monitor_good_bad.py
import os, json, random, time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # solo per la heat-map

# ————————————————————
# 1. PERCORSI & PARAMETRI
# ————————————————————
CWT_DIR   = Path("/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt")
CKPT_GNN  = Path("/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt")
CSV_INFER = Path("/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/detailed_inference/true_pred.csv")
OUT_DIR   = Path("/home/alfio/improving_dementia_detection_model/results_monitor_gnn");  OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
LR          = 1e-3
EPOCHS      = 25
PATIENCE    = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ————————————————————
# 2. DATASET & LABELS
# ————————————————————
from datasets import CWTGraphDataset   # import locale (file caricato)

df = pd.read_csv(CSV_INFER)
df["label"] = (df["pred_label"] == df["true_label"]).astype(int)   # 1 = Good, 0 = Bad

splits = {}
for split in ["training", "validation", "test"]:
    split_df = df[df["dataset"] == split].reset_index(drop=True)[["crop_file","label"]]
    splits[split] = CWTGraphDataset(annot_df = split_df,
                                    dataset_crop_path = CWT_DIR,
                                    norm_stats_path   = None)       # normalizzazione per-record

loaders = {s: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(s=="training"))
           for s,ds in splits.items()}

# ————————————————————
# 3. MODELLO GOOD/BAD
# ————————————————————
from models import GNNCWT2D_Mk11_1sec

def load_monitor_model():
    # 3a) istanzio la vecchia architettura (3 classi) per caricare i pesi
    model3 = GNNCWT2D_Mk11_1sec(n_electrodes=19, cwt_size=(40,500), num_classes=3)
    ckpt = torch.load(CKPT_GNN, map_location="cpu")
    if "state_dict" in ckpt:    # pl-lightning style
        ckpt = ckpt["state_dict"]
        ckpt = {k.replace("model.",""):v for k,v in ckpt.items()}
    model3.load_state_dict(ckpt, strict=True)

    # 3b) creo il nuovo modello 2-classi e copio TUTTO tranne lin6.*
    model2 = GNNCWT2D_Mk11_1sec(n_electrodes=19, cwt_size=(40,500), num_classes=2)
    sd2 = model2.state_dict()
    for k,v in model3.state_dict().items():
        if k.startswith("lin6."):   # shape (32,3) / (3,)
            continue
        sd2[k] = v
    model2.load_state_dict(sd2, strict=False)

    # 3c) congelo tutti i layer tranne lin6
    for name,param in model2.named_parameters():
        param.requires_grad = name.startswith("lin6")
    return model2.to(DEVICE)

model = load_monitor_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ————————————————————
# 4. TRAINING LOOP
# ————————————————————
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0,0,0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y.squeeze()
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            preds = logits.argmax(1)
            total   += y.size(0)
            correct += (preds == y).sum().item()
            loss_sum+= loss.item()*y.size(0)
    return loss_sum/total, correct/total

best_acc, patience = 0.0, PATIENCE
for epoch in range(1,EPOCHS+1):
    tr_loss, tr_acc = run_epoch(loaders["training"], train=True)
    val_loss, val_acc = run_epoch(loaders["validation"])

    print(f"Ep {epoch:02d} | tr_acc {tr_acc:.3f} | val_acc {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc; patience = PATIENCE
        torch.save(model.state_dict(), OUT_DIR/"best_monitor.pt")
    else:
        patience -= 1
        if patience == 0:
            print("Early-stopping!"); break

# ————————————————————
# 5. VALUTAZIONE FINALE
# ————————————————————
model.load_state_dict(torch.load(OUT_DIR/"best_monitor.pt"))
model.eval()

def evaluate_split(split):
    loader = loaders[split]
    y_true, y_pred, y_logits, y_softmax, crop_files = [],[],[],[],[]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            soft   = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)

            y_true  += batch.y.squeeze().cpu().tolist()
            y_pred  += preds.cpu().tolist()
            y_logits+= logits.cpu().tolist()
            y_softmax += soft.cpu().tolist()
            crop_files += batch.crop_file if hasattr(batch,"crop_file") else [""]*len(preds)

    # -- csv
    csv_path = OUT_DIR/f"detailed_inference_{split}.csv"
    out_df = pd.DataFrame({
        "crop_file": crop_files,
        "true_label": y_true,
        "pred_label": y_pred,
        "logits": [json.dumps(x) for x in y_logits],
        "softmax": [json.dumps(x) for x in y_softmax],
        "goodness": [x[1] for x in y_softmax]      # prob. classe Good
    })
    out_df.to_csv(csv_path, index=False)

    # -- classification report
    report_txt = classification_report(y_true, y_pred, target_names=["Bad","Good"], digits=4)
    with open(OUT_DIR/f"classification_report_{split}.txt","w") as fp:
        fp.write(report_txt)

    # -- confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Bad","Good"], yticklabels=["Bad","Good"])
    plt.xlabel("Predetto"); plt.ylabel("Reale"); plt.title(f"Confusion Matrix – {split}")
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"confusion_matrix_{split}.png", dpi=150)
    plt.close()

for split in ["training","validation","test"]:
    evaluate_split(split)

print(f"Risultati salvati in → {OUT_DIR.resolve()}")
