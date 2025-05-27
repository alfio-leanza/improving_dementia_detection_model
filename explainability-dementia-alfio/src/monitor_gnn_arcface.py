#!/usr/bin/env python
# monitor_good_bad_arcface.py
import os, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ————————————————————
# 1. PERCORSI & PARAMETRI
# ————————————————————
CWT_DIR   = Path("/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt")
CKPT_GNN  = Path("/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt")
CSV_INFER = Path("/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/detailed_inference/true_pred.csv")
OUT_DIR   = Path("/home/alfio/improving_dementia_detection_model/results_monitor_gnn_arcface");  OUT_DIR.mkdir(exist_ok=True)

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
from datasets import CWTGraphDataset

class CWTCropDataset(CWTGraphDataset):
    """Include crop_file dentro al Data PyG."""
    def get(self, idx):
        g = super().get(idx)
        g.crop_file = self.annot_df.iloc[idx]['crop_file']
        return g

df = pd.read_csv(CSV_INFER)
df["label"] = (df["pred_label"] == df["true_label"]).astype(int)   # 1 = Good, 0 = Bad

splits, loaders = {}, {}
for split in ["training", "validation", "test"]:
    split_df = df[df["dataset"] == split].reset_index(drop=True)[["crop_file","label"]]
    ds = CWTCropDataset(split_df, CWT_DIR, None)
    splits[split]  = ds
    loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split=="training"))

# ————————————————————
# 3. MODELLO (estrattore) + ARCLOSS
# ————————————————————
from models import GNNCWT2D_Mk11_1sec
from pytorch_metric_learning.losses import ArcFaceLoss         # <<<--- nuova import

def load_monitor_model():
    # 3a) carico pesi 3-classi
    model3 = GNNCWT2D_Mk11_1sec(19, (40,500), 3)
    ckpt = torch.load(CKPT_GNN, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    state = {k.replace("model.","").replace("module.",""):v for k,v in state.items()}
    model3.load_state_dict(state, strict=False)

    # 3b) creo modello 2-classi ma tolgo la testa -> embedding 32-D
    model2 = GNNCWT2D_Mk11_1sec(19, (40,500), 2)
    sd2 = model2.state_dict()
    for k,v in model3.state_dict().items():
        if not k.startswith("lin6."):
            sd2[k] = v
    model2.load_state_dict(sd2, strict=False)
    model2.lin6 = nn.Identity()                              # <<<--- testa rimossa

    # congelo l’estrattore
    for p in model2.parameters():
        p.requires_grad = False

    return model2.to(DEVICE)

model = load_monitor_model()

# --------- ArcFaceLoss come criterio unico ---------
criterion = ArcFaceLoss(num_classes=2,             # <<<---
                        embedding_size=32,
                        margin=0.50,  # rad ~28.6°
                        scale=30).to(DEVICE)

optimizer = torch.optim.Adam(criterion.parameters(), lr=LR)  # <<<---
# ---------------------------------------------------

# ————————————————————
# 4. TRAINING LOOP
# ————————————————————
def run_epoch(loader, train=False):
    model.eval()                              # estrattore sempre eval
    criterion.train() if train else criterion.eval()

    tot, corr, loss_sum = 0,0,0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            y = batch.y.squeeze()
            feats  = model(batch.x, batch.edge_index, batch.batch)     # (B,32)
            loss   = criterion(feats, y)                               # <<<---
            logits = criterion.get_logits(feats).detach()              # <<<---

            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            preds = logits.argmax(1)
            tot += y.size(0); corr += (preds==y).sum().item(); loss_sum += loss.item()*y.size(0)
    return loss_sum/tot, corr/tot

best_acc, patience = 0.0, PATIENCE
for ep in range(1,EPOCHS+1):
    tr_loss, tr_acc = run_epoch(loaders["training"], True)
    val_loss, val_acc = run_epoch(loaders["validation"])
    print(f"Ep {ep:02d} | tr_acc {tr_acc:.3f} | val_acc {val_acc:.3f}")
    if val_acc > best_acc:
        best_acc, patience = val_acc, PATIENCE
        torch.save({"feature_extractor": model.state_dict(),
                    "criterion": criterion.state_dict()}, OUT_DIR/"best_monitor.pt")
    else:
        patience -= 1
        if patience==0:
            print("Early-stopping!"); break

# ————————————————————
# 5. VALUTAZIONE & SALVATAGGIO
# ————————————————————
ckpt_best = torch.load(OUT_DIR/"best_monitor.pt", map_location=DEVICE)
model.load_state_dict(ckpt_best["feature_extractor"])
criterion.load_state_dict(ckpt_best["criterion"])
model.eval(); criterion.eval()

def evaluate_split(split):
    ldr = loaders[split]
    y_t, y_p, logit_l, soft_l, crops = [],[],[],[],[]
    with torch.no_grad():
        for batch in ldr:
            batch = batch.to(DEVICE)
            feats  = model(batch.x, batch.edge_index, batch.batch)
            logits = criterion.get_logits(feats)                      # <<<---
            soft   = torch.softmax(logits,1); preds = logits.argmax(1)

            y_t += batch.y.squeeze().cpu().tolist()
            y_p += preds.cpu().tolist()
            logit_l += logits.cpu().tolist()
            soft_l  += soft.cpu().tolist()
            crops += batch.crop_file

    pd.DataFrame({
        "crop_file": crops,
        "true_label": y_t,
        "pred_label": y_p,
        "logits": [json.dumps(x) for x in logit_l],
        "softmax": [json.dumps(x) for x in soft_l],
        "goodness": [x[1] for x in soft_l]
    }).to_csv(OUT_DIR/f"detailed_inference_{split}.csv", index=False)

    rep = classification_report(y_t,y_p, target_names=["Bad","Good"], digits=4)
    (OUT_DIR/f"classification_report_{split}.txt").write_text(rep)

    cm = confusion_matrix(y_t,y_p,labels=[0,1])
    plt.figure(figsize=(4,3))
    sns.color_palette(palette = 'Blues', as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Bad","Good"], yticklabels=["Bad","Good"])
    plt.xlabel("Predetto"); plt.ylabel("Reale"); plt.title(f"Confusion Matrix – {split}")
    plt.tight_layout(); plt.savefig(OUT_DIR/f"confusion_matrix_{split}.png", dpi=150); plt.close()

for s in ["training","validation","test"]:
    evaluate_split(s)

print(f"Risultati (ArcFaceLoss) salvati in → {OUT_DIR.resolve()}")
