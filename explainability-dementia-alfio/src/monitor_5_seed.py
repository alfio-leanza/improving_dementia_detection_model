#!/usr/bin/env python
# monitor_good_bad_multi.py — versione multi-seed monitor training

import os, json, random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ————————————————————
# 1. PERCORSI & PARAMETRI
# ————————————————————
CWT_DIR = Path("/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt")
BASE_DIR = Path("/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/5_seed")
OUT_BASE = Path("/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/5_seed/monitor")
OUT_BASE.mkdir(exist_ok=True)

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 25
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ————————————————————
# 2. DATASET & LABELS
# ————————————————————
from datasets import CWTGraphDataset

class CWTCropDataset(CWTGraphDataset):
    """Versione con crop_file nel Data."""
    def get(self, idx):
        g = super().get(idx)
        g.crop_file = self.annot_df.iloc[idx]['crop_file']
        return g

# ————————————————————
# 3. MODELLO GOOD/BAD
# ————————————————————
from models import GNNCWT2D_Mk11_1sec

def load_monitor_model(ckpt_path):
    model3 = GNNCWT2D_Mk11_1sec(19, (40,500), 3)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    state = {k.replace("model.","").replace("module.",""):v for k,v in state.items()}
    model3.load_state_dict(state, strict=False)

    model2 = GNNCWT2D_Mk11_1sec(19, (40,500), 2)
    sd2 = model2.state_dict()
    for k,v in model3.state_dict().items():
        if not k.startswith("lin6."):
            sd2[k] = v
    model2.load_state_dict(sd2, strict=False)

    for n,p in model2.named_parameters():
        p.requires_grad = n.startswith("lin6")
    return model2.to(DEVICE)

# ————————————————————
# 4. TRAINING LOOP
# ————————————————————
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, corr, loss_sum = 0,0,0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y.squeeze()
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(1)
            tot += y.size(0); corr += (preds==y).sum().item(); loss_sum += loss.item()*y.size(0)
    return loss_sum/tot, corr/tot

def evaluate_split(model, loader, split, out_dir):
    model.eval()
    y_t, y_p, logit_l, soft_l, crops = [],[],[],[],[]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch)
            soft = torch.softmax(logits, 1); preds = logits.argmax(1)
            y_t += batch.y.squeeze().cpu().tolist()
            y_p += preds.cpu().tolist()
            logit_l += logits.cpu().tolist()
            soft_l += soft.cpu().tolist()
            crops += batch.crop_file
    pd.DataFrame({
        "crop_file": crops,
        "true_label": y_t,
        "pred_label": y_p,
        "logits": [json.dumps(x) for x in logit_l],
        "softmax": [json.dumps(x) for x in soft_l],
        "goodness": [x[1] for x in soft_l]
    }).to_csv(out_dir/f"detailed_inference_{split}.csv", index=False)

    rep = classification_report(y_t, y_p, target_names=["Bad", "Good"], digits=4)
    (out_dir/f"classification_report_{split}.txt").write_text(rep)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
    plt.xlabel("Predetto"); plt.ylabel("Reale"); plt.title(f"Confusion Matrix – {split}")
    plt.tight_layout(); plt.savefig(out_dir/f"confusion_matrix_{split}.png", dpi=150); plt.close()

# ————————————————————
# 5. MAIN PER OGNI SEED
# ————————————————————
for seed_dir in sorted(BASE_DIR.glob("checkpoints/train_*")):
    seed_name = seed_dir.name  # es. train_20250510_172519
    ckpt_path = seed_dir / "best_val_acc.pt"
    inf_dir = BASE_DIR / "results" / seed_name
    out_dir = OUT_BASE / seed_name
    out_dir.mkdir(exist_ok=True)

    print(f"\n========================\nSeed: {seed_name}\n========================")
    dfs, splits, loaders = {}, {}, {}

    for split in ["trai", "val", "test"]:
        csv_path = inf_dir / f"{split}_inferences.csv"
        df = pd.read_csv(csv_path)
        df["label"] = (df["pred_label"] == df["true_label"]).astype(int)
        df = df[["crop_file", "label"]].reset_index(drop=True)
        ds = CWTCropDataset(df, CWT_DIR, None)
        splits[split] = ds
        loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split=="train"))

    model = load_monitor_model(ckpt_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_acc, patience = 0.0, PATIENCE
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion)
        print(f"[{seed_name}] Ep {ep:02d} | tr_acc {tr_acc:.3f} | val_acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc, patience = val_acc, PATIENCE
            torch.save(model.state_dict(), out_dir / "best_monitor.pt")
        else:
            patience -= 1
            if patience == 0:
                print("Early-stopping!")
                break

    model.load_state_dict(torch.load(out_dir / "best_monitor.pt"))
    for split in ["train", "val", "test"]:
        evaluate_split(model, loaders[split], split, out_dir)

    print(f"✓ Risultati salvati in → {out_dir.resolve()}")
