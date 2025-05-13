import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast              # <— nuovo namespace AMP
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax as sf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ========= CNN Channel Attention (lievi modifiche: +dropout, stesso I/O) ==========
class CNN_ChannelAttention(nn.Module):
    """Idem a prima ma con dropout un po' più alto per regolarizzare."""

    def __init__(self, num_channels: int = 19, num_classes: int = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),                    # 0.2 -> 0.3
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.35),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.35),
            nn.AdaptiveAvgPool2d(1),
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 256, bias=False),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                      # 0.6 -> 0.5
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x).flatten(1)
        x = x * self.channel_attention(x)
        return self.classifier(x)


# ========= Data loading (unchanged salvo augmentazioni extra) ==========
train_act = np.load(
    "/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy",
    allow_pickle=True,
).item()
val_act = np.load(
    "/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy",
    allow_pickle=True,
).item()
test_act = np.load(
    "/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy",
    allow_pickle=True,
).item()


def build_df(d, split):
    df = pd.DataFrame({"crop_file": list(d.keys()), "valore": list(d.values())})
    df["dataset"] = split
    df["valore_softmax"] = df["valore"].apply(sf)
    df["pred_label"] = df["valore_softmax"].apply(lambda x: int(np.argmax(x)))
    return df


a, b, c = build_df(train_act, "train"), build_df(val_act, "val"), build_df(test_act, "test")
train_df, val_df, test_df = a, b, c
all_df = pd.concat([train_df, val_df, test_df])
annot = (
    pd.read_csv(
        "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv"
    )
    .rename(columns={"label": "true_label"})
)
true_pred = all_df.merge(annot, on="crop_file")
true_pred["train_label"] = (
    (true_pred["pred_label"] == true_pred["true_label"]).astype(int)
)

root_dir = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
# splits dict invariato
splits = {
    "train": [
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 66,
        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    ],
    "val": [
        54, 55, 56, 57, 58, 59, 79, 80, 81, 82, 83, 22, 23, 24, 25, 26, 27, 28,
    ],
    "test": [
        60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36,
    ],
}


# ---- piccole funzioni di augmentation -------------------------------------------------

def _time_shift(arr, max_shift_frac: float = 0.1):
    """Shift circolare lungo l'asse tempo (asse 1)"""
    T = arr.shape[1]
    shift = np.random.randint(-int(T * max_shift_frac), int(T * max_shift_frac))
    return np.roll(arr, shift, axis=1)


def _freq_mask(arr, max_width_frac: float = 0.15):
    F = arr.shape[0]
    width = np.random.randint(0, int(F * max_width_frac))
    f0 = np.random.randint(0, max(1, F - width))
    arr[f0 : f0 + width, :] = 0
    return arr


class CWT_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        data = np.load(os.path.join(root_dir, row.crop_file))  # shape (H, W, C)
        if self.augment:
            # Rumore gaussiano
            data = data + np.random.normal(0, 0.01, data.shape)
            # Time‑shift casuale
            data = _time_shift(data, 0.05)
            # Frequency masking
            data = _freq_mask(data, 0.1)
        tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # (C,H,W)
        return tensor, torch.tensor(row.train_label), row.crop_file


def make_loader(split: str, batch: int = 64, augment: bool = False, shuffle: bool = True):
    subset = true_pred[
        true_pred["original_rec"].isin([f"sub-{s:03d}" for s in splits[split]])
    ]
    return DataLoader(
        CWT_Dataset(subset, augment), batch_size=batch, shuffle=shuffle, num_workers=4
    )


train_loader = make_loader("train", augment=True)
val_loader = make_loader("val", shuffle=False)
test_loader = make_loader("test", shuffle=False)

# ========= Training (80 ep + regolarizzazioni "zero‑overfitting") ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_ChannelAttention().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

# LR scheduler: Cosine over epoche
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

# SWA setup (ultime 10 epoche)
swa_start_ep = 70
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)

# directory checkpoints
ckpt_dir = "/home/alfio/improving_dementia_detection_model/checkpoints_cnn"
os.makedirs(ckpt_dir, exist_ok=True)
best_val_acc = 0.0
best_epoch = 0
patience = 10  # early stopping se nessun miglioramento x epoche
scaler = GradScaler(enabled=torch.cuda.is_available())

for ep in range(1, 81):
    model.train()
    corr = tot = 0

    for x, y, _ in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = out.argmax(1)
        tot += y.size(0)
        corr += pred.eq(y).sum().item()

    # validation
    model.eval()
    vcorr = vtot = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                logits = model(x)
            vcorr += logits.argmax(1).eq(y).sum().item()
            vtot += y.size(0)

    train_acc = corr / tot
    val_acc = vcorr / vtot
    print(f"Epoch {ep:2d}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    # checkpointing
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = ep
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
            },
            os.path.join(ckpt_dir, "cnn_channelattention_best.pth"),
        )
        print(f"[CHECKPOINT] Nuovo best model salvato (val_acc={best_val_acc:.4f})")

    # LR schedule / SWA update
    if ep >= swa_start_ep:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    # early‑stopping
    if ep - best_epoch >= patience:
        print("Early stopping attivato: nessun improvement da", patience, "epoche")
        break

# Finito training: se abbiamo usato SWA, aggiorna BN e usa swa_model come finale
if swa_start_ep < ep:  # SWA effettivamente usato
    print("Aggiorno BatchNorm del modello SWA …")
    update_bn(train_loader, swa_model, device=device)
    final_model = swa_model
else:
    final_model = model

# ========= Inferenza & salvataggio (usiamo il *miglior* checkpoint) ==========

print("\n[INFO] Carico il best checkpoint per la fase di inferenza …")
best_ckpt = torch.load(os.path.join(ckpt_dir, "cnn_channelattention_best.pth"))
final_model.load_state_dict(best_ckpt["model_state_dict"])
final_model.to(device)
final_model.eval()

out_dir = "/home/alfio/improving_dementia_detection_model/results_cnn"
os.makedirs(out_dir, exist_ok=True)


def collect(loader, split):
    rows = []
    with torch.no_grad():
        for x, y, fns in loader:
            with autocast():
                logits = final_model(x.to(device)).cpu()
            sm = sf(logits, axis=1)
            preds = logits.argmax(1)
            goodness = sm[np.arange(len(sm)), preds]
            for f, t, p, l, s, g in zip(
                fns, y.cpu(), preds, logits.tolist(), sm.tolist(), goodness
            ):
                rows.append([f, int(t), int(p), l, s, float(g)])
    pd.DataFrame(
        rows,
        columns=[
            "crop_file",
            "true_label",
            "pred_label",
            "logits",
            "softmax",
            "goodness",
        ],
    ).to_csv(os.path.join(out_dir, f"{split}_predictions_detailed.csv"), index=False)


for sp, ld in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
    collect(ld, sp)

# Confusion matrix & classification report (su test)

yt, yp = [], []
with torch.no_grad():
    for x, y, _ in test_loader:
        yt.extend(y.numpy())
        with autocast():
            yp.extend(final_model(x.to(device)).argmax(1).cpu().numpy())

cm = confusion_matrix(yt, yp)
report = classification_report(yt, yp, target_names=["Bad", "Good"])

with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
    f.write(report)

plt.figure(figsize=(6, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Bad", "Good"],
    yticklabels=["Bad", "Good"],
)
plt.title("Confusion Matrix (best ckpt)")
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
plt.close()

# Salva state_dict finale (SWA o ultimo) per completezza

torch.save(final_model.state_dict(), os.path.join(out_dir, "cnn_channelattention_final.pth"))
print("\n[INFO] Results + modelli salvati in", out_dir)
