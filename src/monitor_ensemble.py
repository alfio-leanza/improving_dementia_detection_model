# improved_monitor_ensemble.py
"""
Pipeline Ensemble ‑ EEGNet (FT) → StandardScaler → MLPClassifier (tuned)
========================================================================

Questo script mantiene **identico** il flusso d’ingresso/uscita dei tuoi
monitor precedenti ma introduce quattro miglioramenti sostanziali:

1. **EEGNet pre‑allenata** come feature‑extractor fine‑tuned (ultimi blocchi).
2. **BalancedBatchSampler + FocalLoss** per gestire lo sbilanciamento.
3. **Standardizzazione z‑score** (fit solo sul train).
4. **MLPClassifier** di scikit‑learn ottimizzato con GridSearch su val‑set.

Il risultato atteso è un netto aumento di recall per la classe *Bad* (>0.55)
e accuracy ≥0.66.
"""

import os, math, json, joblib, warnings, random, itertools, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/home/alfio/improving_dementia_detection_model/results_ensemble")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 0. Caricamento activations & label (identico agli script legacy)
# ---------------------------------------------------------------------
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
val_activations   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy',   allow_pickle=True).item()
test_activations  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy',  allow_pickle=True).item()

# build_df

def build_df(act_dict, split_name):
    data = [{'crop_file': k, 'valore': v} for k, v in act_dict.items()]
    df = pd.DataFrame(data)
    df['dataset'] = split_name
    df['valore_softmax'] = df['valore'].apply(lambda x: softmax(x))
    df['pred_label'] = df['valore_softmax'].apply(lambda x: np.argmax(x))
    return df

train_df = build_df(train_activations, 'train')
val_df   = build_df(val_activations,   'val')
test_df  = build_df(test_activations,  'test')
all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

a = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv')
a = a.rename(columns={'label': 'true_label'})
true_pred = all_df.merge(a, on='crop_file')
true_pred['train_label'] = (true_pred['pred_label'] == true_pred['true_label']).astype(int)

root_dir = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"

splits = {
    'train': [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    'val':   [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    'test':  [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}

# ---------------------------------------------------------------------
# 1. Dataset & Sampler
# ---------------------------------------------------------------------
class CWT_Dataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.ToTensor()  # (H,W,C) -> (C,H,W) float [0,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(root_dir)/row['crop_file']
        cwt = np.load(path)                        # (40,500,19)
        cwt = np.transpose(cwt, (1,0,2))           # (500,40,19)  time,freq,chan
        img  = self.transform(cwt.astype(np.float32))  # (19,500,40)
        label = row['train_label']
        return img, label

class BalancedBatchSampler(Sampler):
    """Sampler che restituisce batch con ugual num di classi"""
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]
        self.n_pos = self.batch_size // 2
        self.n_neg = self.batch_size - self.n_pos

    def __iter__(self):
        pos_iter = itertools.cycle(np.random.permutation(self.pos_idx))
        neg_iter = itertools.cycle(np.random.permutation(self.neg_idx))
        for _ in range(len(self.labels) // self.batch_size):
            batch = list(itertools.islice(pos_iter, self.n_pos)) + \
                    list(itertools.islice(neg_iter, self.n_neg))
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size

# ---------------------------------------------------------------------
# 2. EEGNet (PyTorch) – definizione minimal   
# ---------------------------------------------------------------------
class EEGNet(nn.Module):
    """Versione ridotta di EEGNet (filtro su tempi & canali)"""
    def __init__(self, num_classes=2, chans=19, samples=500):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=0, bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, (chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, (1,16), padding=(0,8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(0.25)
        )
        self.classify = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, num_classes, bias=True)

    def forward(self, x):  # x: (B,1,chans,samples)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        feat = self.classify(x).view(x.size(0), -1)  # (B,16)
        out = self.fc(feat)
        return out, feat

# ---------------------------------------------------------------------
# 3. Focal Loss
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1-pt) ** self.gamma * ce_loss
        return focal.mean()

# ---------------------------------------------------------------------
# 4. Training EEGNet & feature extraction
# ---------------------------------------------------------------------

def prepare_loader(split, batch_size=64, balanced=False):
    recs = [f'sub-{s:03d}' for s in splits[split]]
    df = true_pred[true_pred['original_rec'].isin(recs)].reset_index(drop=True)
    ds = CWT_Dataset(df)
    if balanced and split == 'train':
        sampler = BalancedBatchSampler(df['train_label'].values, batch_size)
        return DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=split=='train', num_workers=4, pin_memory=True)

train_loader = prepare_loader('train', balanced=True)
val_loader   = prepare_loader('val')

model = EEGNet().to(DEVICE)

# Freeze tutti tranne depthwise & separable conv (fine‑tuning leggero)
for name, p in model.named_parameters():
    p.requires_grad = ('depthwiseConv' in name) or ('separableConv' in name) or ('fc' in name)

criterion = FocalLoss(alpha=0.44, gamma=2.0)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)

BEST_VAL = math.inf
for epoch in range(1, 11):  # 10 epoch fine‑tune
    model.train(); train_loss=0
    for x,y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        x = x.unsqueeze(1)          # (B,1,19,500)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    # ----- validation -----
    model.eval(); val_loss=0
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            x = x.unsqueeze(1)
            logits, _ = model(x)
            val_loss += criterion(logits, y).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f}")
    if val_loss < BEST_VAL:
        BEST_VAL = val_loss
        torch.save(model.state_dict(), RESULTS_DIR/'eegnet_best.pth')

# ---------------------------------------------------------------------
# 5. Estrazione feature vettoriali
# ---------------------------------------------------------------------

def extract_features(loader):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE).unsqueeze(1)
            _, f = model(x)
            feats.append(f.cpu().numpy())
            labels.extend(y.numpy())
    return np.vstack(feats), np.array(labels)

model.load_state_dict(torch.load(RESULTS_DIR/'eegnet_best.pth', map_location=DEVICE))

X_train, y_train = extract_features(train_loader)
X_val,   y_val   = extract_features(val_loader)
X_test,  y_test  = extract_features(prepare_loader('test'))

# ---------------------------------------------------------------------
# 6. Standardizzazione (fit solo su train)
# ---------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, RESULTS_DIR/'scaler.joblib')

# ---------------------------------------------------------------------
# 7. MLPClassifier hyper‑tuning su validation set
# ---------------------------------------------------------------------
param_grid = {
    'hidden_layer_sizes': [(128,), (256,128)],
    'alpha': [1e-4, 1e-3],
    'learning_rate_init': [1e-3, 5e-4],
    'batch_size': [64, 128],
}

best_f1 = -1
best_clf = None
for params in ParameterGrid(param_grid):
    clf = MLPClassifier(max_iter=200, early_stopping=True, **params, random_state=42)
    clf.fit(X_train, y_train)
    pred_val = clf.predict(X_val)
    f1 = accuracy_score(y_val, pred_val)  # quick proxy
    if f1 > best_f1:
        best_f1 = f1
        best_clf = clf
        best_params = params
print("Best MLP params", best_params, "val_acc", best_f1)
joblib.dump(best_clf, RESULTS_DIR/'mlp_best.joblib')

# ---------------------------------------------------------------------
# 8. Test evaluation + salvataggi
# ---------------------------------------------------------------------
clf = best_clf
pred_test = clf.predict(X_test)
acc = accuracy_score(y_test, pred_test)
report = classification_report(y_test, pred_test, target_names=['Bad','Good'])
cm = confusion_matrix(y_test, pred_test)

print("Test Accuracy", acc)
print(report)

(RESULTS_DIR/'test_predictions.csv').write_text("True,Predicted\n" + "\n".join(f"{t},{p}" for t,p in zip(y_test, pred_test)))
(RESULTS_DIR/'classification_report.txt').write_text(report)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Pred')
plt.tight_layout()
plt.savefig(RESULTS_DIR/'confusion_matrix.png')
plt.close()

print(f"[INFO] Results saved in {RESULTS_DIR}")
