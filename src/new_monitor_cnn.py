#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training CNN-ChannelAttention con:
– 80 epoche
– AdamW lr=2e-3
– Warm-up (epoche 1-4) senza OHEM
– OHEM dinamico da epoca 5 (hard_ratio 0.10→0.30)
– Salvataggio checkpoint sul miglior val_acc
"""

# ===============================================================
# 1) IMPORT
# ===============================================================
import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax as sf
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

# ===============================================================
# 2) SEED
# ===============================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ===============================================================
# 3) MODEL
# ===============================================================
class CNN_ChannelAttention(nn.Module):
    def __init__(self, num_channels: int = 19, num_classes: int = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(256, 64, bias=False), nn.ReLU(inplace=True),
            nn.Linear(64, 256, bias=False), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x).flatten(1)
        x = x * self.channel_attention(x)
        return self.classifier(x)

# ===============================================================
# 4) DATA
# ===============================================================
# --- Percorsi ---
root_dir_cwt = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
annot_csv    = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv"
inference_dir= "/home/tom/dataset_eeg/inference_20250327_171717"

# --- Attivazioni già calcolate ---
train_act = np.load(os.path.join(inference_dir, "train_activations.npy"), allow_pickle=True).item()
val_act   = np.load(os.path.join(inference_dir, "val_activations.npy"),   allow_pickle=True).item()
test_act  = np.load(os.path.join(inference_dir, "test_activations.npy"),  allow_pickle=True).item()

def build_df(d: dict, split: str) -> pd.DataFrame:
    df = pd.DataFrame({"crop_file": list(d.keys()), "valore": list(d.values())})
    df["dataset"] = split
    df["valore_softmax"] = df["valore"].apply(sf)
    df["pred_label"] = df["valore_softmax"].apply(lambda x: int(np.argmax(x)))
    return df

train_df, val_df, test_df = (
    build_df(train_act, "train"),
    build_df(val_act,   "val"),
    build_df(test_act,  "test"),
)
all_df = pd.concat([train_df, val_df, test_df])

# --- Etichette vere ---
annot = pd.read_csv(annot_csv).rename(columns={"label": "true_label"})
true_pred = all_df.merge(annot, on="crop_file")
true_pred["train_label"] = (true_pred["pred_label"] == true_pred["true_label"]).astype(int)

# --- Split per soggetto (come da tuo codice) ---
splits = {
    "train": [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,
              1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    "val":   [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    "test":  [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36],
}

# --- Dataset & DataLoader ---
class CWT_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int: return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        data = np.load(os.path.join(root_dir_cwt, row.crop_file))  # (40,500,19)
        if self.augment:
            data += np.random.normal(0, 0.01, data.shape)
        tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # (19,40,500)
        return tensor, torch.tensor(row.train_label), row.crop_file

def make_loader(split: str, batch: int = 64, augment: bool = False, shuffle: bool = True):
    subset = true_pred[
        true_pred["original_rec"].isin([f"sub-{s:03d}" for s in splits[split]])
    ]
    return DataLoader(CWT_Dataset(subset, augment),
                      batch_size=batch,
                      shuffle=shuffle,
                      num_workers=4,
                      pin_memory=True)

train_loader = make_loader("train", augment=True)
val_loader   = make_loader("val",   shuffle=False)
test_loader  = make_loader("test",  shuffle=False)

# ===============================================================
# 5) TRAINING (warm-up + OHEM dinamico)
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CNN_ChannelAttention().to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

num_epochs   = 80
best_val_acc = 0.0
out_dir      = "/home/alfio/improving_dementia_detection_model/results_new_cnn"
os.makedirs(out_dir, exist_ok=True)
best_ckpt    = os.path.join(out_dir, "cnn_channelattention_best.pth")

ce_mean = nn.CrossEntropyLoss(reduction="mean")   # warm-up
ce_none = nn.CrossEntropyLoss(reduction="none")   # per-sample (OHEM)

for ep in range(1, num_epochs + 1):
    model.train()
    corr = tot = 0; running_loss = 0.0
    use_ohem = ep >= 5
    if use_ohem:
        hard_ratio = min(0.10 + 0.05 * ((ep - 5) // 10), 0.30)

    for x, y, _ in tqdm(train_loader, desc=f"Epoch {ep}/{num_epochs}", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)

        if use_ohem:
            losses = ce_none(logits, y)            # (B,)
            k = max(1, int(hard_ratio * len(losses)))
            loss = losses.topk(k).values.mean()    # OHEM
        else:
            loss = ce_mean(logits, y)              # warm-up

        loss.backward(); optimizer.step()
        running_loss += loss.item()
        pred = logits.argmax(1); tot += y.size(0); corr += pred.eq(y).sum().item()

    train_acc = corr / tot

    # ---- VALIDATION ----
    model.eval(); vcorr = vtot = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            preds = model(x).argmax(1)
            vcorr += preds.eq(y).sum().item(); vtot += y.size(0)
    val_acc = vcorr / vtot

    tag = f"(warm-up)" if not use_ohem else f"(OHEM {hard_ratio:.0%})"
    print(f"Epoch {ep:02d}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
          f"loss={running_loss/len(train_loader):.4f}  {tag}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_ckpt)
        print(f"[INFO] Nuovo best checkpoint salvato (val_acc={best_val_acc:.4f})")

# ===============================================================
# 6) TEST CON IL MIGLIOR MODELLO
# ===============================================================
print(f"[INFO] Carico best checkpoint con val_acc={best_val_acc:.4f}")
model.load_state_dict(torch.load(best_ckpt, map_location=device))
model.eval()

def collect(loader: DataLoader, split: str):
    rows = []
    with torch.no_grad():
        for x, y, fns in loader:
            logits = model(x.to(device)).cpu()
            sm = sf(logits, axis=1)
            preds = logits.argmax(1)
            goodness = sm[np.arange(len(sm)), preds]
            for f, t, p, l, s, g in zip(fns, y.cpu(), preds,
                                        logits.tolist(), sm.tolist(), goodness):
                rows.append([f, int(t), int(p), l, s, float(g)])

    pd.DataFrame(rows, columns=["crop_file", "true_label", "pred_label",
                                "logits", "softmax", "goodness"]) \
        .to_csv(os.path.join(out_dir, f"{split}_predictions_detailed.csv"), index=False)
    print(f"[INFO] Salvato {split}_predictions_detailed.csv")

for sp, ld in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
    collect(ld, sp)

# ---- Confusion matrix & classification report ----
yt, yp = [], []
with torch.no_grad():
    for x, y, _ in test_loader:
        yt.extend(y.numpy())
        yp.extend(model(x.to(device)).argmax(1).cpu().numpy())

cm = confusion_matrix(yt, yp)
report = classification_report(yt, yp, target_names=["Bad", "Good"])
with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
    f.write(report)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title("Confusion Matrix")
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
plt.close()

# ---- Copia finale modello ----
torch.save(model.state_dict(), os.path.join(out_dir, "cnn_channelattention_final.pth"))
print(f"[INFO] Training completato. Modello e risultati in: {out_dir}")
