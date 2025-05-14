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
# 4) DATA (identica alla versione precedente)
# ===============================================================
# ... codice caricamento dataframe, split, dataset, dataloader invariato ...
# (Per brevità, la sezione è la stessa che hai già: true_pred, splits, CWT_Dataset, make_loader)

# ---------- copia breve (rimane invariata, sostituisci con la tua) ----------
train_loader = ...  # come in versione precedente
val_loader   = ...
test_loader  = ...
# ----------------------------------------------------------------------------

# ===============================================================
# 5) TRAINING con warm-up + OHEM dinamico
# ===============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CNN_ChannelAttention().to(device)

# Ottimizzatore
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

num_epochs   = 80
best_val_acc = 0.0
out_dir      = '/home/alfio/improving_dementia_detection_model/results_new_cnn'
os.makedirs(out_dir, exist_ok=True)
best_ckpt    = os.path.join(out_dir, 'cnn_channelattention_best.pth')

# Funzioni loss
ce_mean = nn.CrossEntropyLoss(reduction='mean')   # warm-up
ce_none = nn.CrossEntropyLoss(reduction='none')   # per OHEM

for ep in range(1, num_epochs + 1):
    model.train()
    corr = tot = 0; running_loss = 0.0

    # Definisci se usare OHEM e hard_ratio corrente
    use_ohem = ep >= 5
    if use_ohem:
        hard_ratio = min(0.10 + 0.05 * ((ep - 5) // 10), 0.30)

    for x, y, _ in tqdm(train_loader, desc=f"Epoch {ep}/{num_epochs}"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)                              # (B, 2)

        if use_ohem:
            losses = ce_none(logits, y)                # (B,)
            k = max(1, int(hard_ratio * len(losses)))
            loss = losses.topk(k).values.mean()        # OHEM select
        else:
            loss = ce_mean(logits, y)                  # warm-up

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
    print(f"Epoch {ep:02d}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f} "
          f"loss={running_loss/len(train_loader):.4f} "
          f"{'(OHEM {:.0%})'.format(hard_ratio) if use_ohem else '(warm-up)'}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_ckpt)
        print(f"[INFO] Nuovo best checkpoint salvato (val_acc={best_val_acc:.4f})")

# ===============================================================
# 6) TEST FINALE (carica best checkpoint) – sezione invariata
# ===============================================================
model.load_state_dict(torch.load(best_ckpt, map_location=device))
model.eval()

def collect(loader: DataLoader, split: str):
    rows = []
    with torch.no_grad():
        for x, y, fns in loader:
            logits = model(x.to(device)).cpu()
            sm     = sf(logits, axis=1)
            preds  = logits.argmax(1)
            goodness = sm[np.arange(len(sm)), preds]
            for f, t, p, l, s, g in zip(fns, y.cpu(), preds, logits.tolist(), sm.tolist(), goodness):
                rows.append([f, int(t), int(p), l, s, float(g)])

    pd.DataFrame(rows,
                 columns=['crop_file', 'true_label', 'pred_label',
                          'logits', 'softmax', 'goodness']) \
      .to_csv(os.path.join(out_dir, f'{split}_predictions_detailed.csv'), index=False)
    print(f"[INFO] Salvato {split}_predictions_detailed.csv")

for sp, ld in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
    collect(ld, sp)

# ----- Confusion matrix & classification report -----
yt, yp = [], []
with torch.no_grad():
    for x, y, _ in test_loader:
        yt.extend(y.numpy())
        yp.extend(model(x.to(device)).argmax(1).cpu().numpy())

cm = confusion_matrix(yt, yp)
report = classification_report(yt, yp, target_names=['Bad', 'Good'])
with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
plt.close()

# ----- Copia finale del best model -----
final_model_path = os.path.join(out_dir, 'cnn_channelattention_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"[INFO] Training completato. Miglior modello e risultati salvati in: {out_dir}")
