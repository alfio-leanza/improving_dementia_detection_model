import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax as sf
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

"""
Allenamento del **modello principale** (3 classi: HC, FTD, AD) che utilizza il monitor binario
pre‑allenato per fornire (pred_label, goodness) come feature aggiuntive.
Salva automaticamente:
  • history.csv  – train/val loss & accuracy per epoca
  • history_accuracy.png & history_loss.png (curve train/val)
  • confusion matrix e classification report per ciascuno split
  • csv con logits, softmax, pred_label
  • pesi del modello principale
"""

# ====== PATH & COSTANTI (adatta ai tuoi percorsi) ======
root_dir = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
annot_path = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv"
monitor_path = "/home/alfio/improving_dementia_detection/results_cnn/cnn_channelattention.pth" 
results_dir = "/home/alfio/improving_dementia_detection/results_main_model"
os.makedirs(results_dir, exist_ok=True)

# ====== DEFINIZIONE MONITOR (già presente nel tuo progetto) ======
class CNN_ChannelAttention(nn.Module):
    def __init__(self, num_channels=19, num_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.3), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.3), nn.AdaptiveAvgPool2d(1))
        self.channel_attention = nn.Sequential(nn.Linear(256, 64, bias=False), nn.ReLU(), nn.Linear(64, 256, bias=False), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.conv_block(x).flatten(1)
        x = x * self.channel_attention(x)
        return self.classifier(x)

# ====== MODELLO PRINCIPALE (3 classi) ======
class EEG_CWT_Main(nn.Module):
    def __init__(self, num_channels=19, num_classes=3):
        super().__init__()
        self.cnn_backbone = CNN_ChannelAttention(num_channels=num_channels, num_classes=256)  # output 256 feature
        self.fc_final = nn.Sequential(
            nn.Linear(256 + 2, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes))

    def forward(self, x_cwt, monitor_pred, monitor_goodness):
        feats = self.cnn_backbone.conv_block(x_cwt).flatten(1)
        feats = feats * self.cnn_backbone.channel_attention(feats)
        extra = torch.stack([monitor_pred.float(), monitor_goodness], dim=1)
        return self.fc_final(torch.cat([feats, extra], dim=1))

# ====== DATASET con monitor integrato ======
class MainDataset(Dataset):
    def __init__(self, df, monitor, device, augment=False):
        self.df = df.reset_index(drop=True)
        self.monitor = monitor.eval()
        self.device = device
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        data = np.load(os.path.join(root_dir, row.crop_file))
        if self.augment:
            data += np.random.normal(0, 0.01, data.shape)
        tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_m = self.monitor(tensor)
            probs_m = torch.softmax(logits_m, dim=1)
            pred_m = torch.argmax(probs_m, dim=1).item()
            good_m = probs_m.max().item()
        tensor = tensor.squeeze(0).cpu()
        return tensor, pred_m, good_m, row.true_label, row.crop_file

# ====== CARICAMENTO DATAFRAME & SPLIT SOGGETTI ======
annot = pd.read_csv(annot_path)
# --- sostituisci con i tuoi ID soggetto ---
splits = {
    'train': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78],
    'val'  : [22,23,24,25,26,27,28,54,55,56,57,58,59,79,80,81,82,83],
    'test' : [29,30,31,32,33,34,35,36,60,61,62,63,64,65,84,85,86,87,88],
}

def make_loader(split, monitor, device, augment=False, shuffle=True):
    subset = annot[annot.original_rec.isin([f'sub-{s:03d}' for s in splits[split]])]
    return DataLoader(MainDataset(subset, monitor, device, augment), batch_size=64, shuffle=shuffle, num_workers=4)

# ====== PREPARAZIONE TRAINING ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
monitor = CNN_ChannelAttention().to(device)
monitor.load_state_dict(torch.load(monitor_path, map_location=device))
monitor.eval()

model = EEG_CWT_Main().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

train_loader = make_loader('train', monitor, device, augment=True, shuffle=True)
val_loader   = make_loader('val',   monitor, device, augment=False, shuffle=False)
test_loader  = make_loader('test',  monitor, device, augment=False, shuffle=False)

# ====== LOOP DI TRAINING ======
history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}

for epoch in range(1, 21):
    # --- TRAIN ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, mp, mg, y, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d} [train]"):
        x, mp, mg, y = x.to(device), mp.to(device), mg.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, mp, mg)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item(); total += y.size(0)
    train_loss = running_loss / len(train_loader)
    train_acc  = correct / total

    # --- VALIDATION ---
    model.eval(); val_loss, vcorrect, vtotal = 0.0, 0, 0
    with torch.no_grad():
        for x, mp, mg, y, _ in tqdm(val_loader, desc=f"Epoch {epoch:02d} [val]  ", leave=False):
            x, mp, mg, y = x.to(device), mp.to(device), mg.to(device), y.to(device)
            out = model(x, mp, mg)
            val_loss += criterion(out, y).item()
            vcorrect += out.argmax(1).eq(y).sum().item(); vtotal += y.size(0)
    val_loss /= len(val_loader)
    val_acc  = vcorrect / vtotal

    history['train_loss'].append(train_loss)
    history['val_loss']  .append(val_loss)
    history['train_acc'] .append(train_acc)
    history['val_acc']   .append(val_acc)

    print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

# ====== SALVATAGGIO HISTORY & PLOT ======
pd.DataFrame(history).to_csv(os.path.join(results_dir, 'history.csv'), index=False)

# Accuracy plot
plt.figure(); plt.plot(history['train_acc'], label='train'); plt.plot(history['val_acc'], label='val'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy');
plt.savefig(os.path.join(results_dir, 'history_accuracy.png'))

# Loss plot
plt.figure(); plt.plot(history['train_loss'], label='train'); plt.plot(history['val_loss'], label='val'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss');
plt.savefig(os.path.join(results_dir, 'history_loss.png'))
plt.close('all')

# ====== FUNZIONE PER SALVARE PREDIZIONI & METRICHE ======
def save_preds(loader, split):
    rows, yt, yp = [], [], []
    with torch.no_grad():
        for x, mp, mg, y, fn in loader:
            logits = model(x.to(device), mp.to(device), mg.to(device)).cpu()
            sm = sf(logits, axis=1); pred = logits.argmax(1)
            for f, t, p, l, s in zip(fn, y, pred, logits.numpy(), sm):
                rows.append([f, int(t), int(p), l, s])
            yt.extend(y.numpy()); yp.extend(pred.numpy())

    pd.DataFrame(rows, columns=['crop_file', 'true_label', 'pred_label', 'logits', 'softmax']) \
      .to_csv(os.path.join(results_dir, f'{split}_predictions.csv'), index=False)

    cm = confusion_matrix(yt, yp)
    report = classification_report(yt, yp, target_names=['HC', 'FTD', 'AD'])
    with open(os.path.join(results_dir, f'{split}_classification_report.txt'), 'w') as f:
        f.write(report)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HC', 'FTD', 'AD'], yticklabels=['HC', 'FTD', 'AD'])
    plt.title(f'{split.capitalize()} Confusion Matrix')
    plt.savefig(os.path.join(results_dir, f'{split}_confusion_matrix.png'))
    plt.close()

# ====== SALVA PREDIZIONI PER OGNI SPLIT ======
for ld, sp in [(train_loader, 'train'), (val_loader, 'val'), (test_loader, 'test')]:
    save_preds(ld, sp)

# ====== SALVA MODELLI ======
torch.save(model.state_dict(), os.path.join(results_dir, 'main_model.pth'))
print(f"[INFO] Tutti i risultati sono stati salvati in → {results_dir}")
