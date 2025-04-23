# improved_monitor_ensemble.py
"""
Pipeline Ensemble – EEGNet → StandardScaler → FT‑Transformer
===========================================================
Questo script replica _pari‑pari_ il flusso I/O dei tuoi vecchi monitor:
* **Caricamento activations & label** inalterato.
* **Split train/val/test** identico.
* Legge le CWT `.npy` (40 × 500 × 19) e ne estrae feature con **EEGNet** pre‑allenata.
* Standardizza le feature (z‑score).
* Classifica con **FT‑Transformer** (fallback MLP se `rtdl` non presente).
* Salva:
  * `test_predictions.csv`
  * `classification_report.txt`
  * `confusion_matrix.png`
nel path legacy `/home/alfio/improving_dementia_detection_model/results_ensemble`.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------
# SECTION A – Load activations & labels  (UNCHANGED)                           |
# -----------------------------------------------------------------------------
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
val_activations   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy',   allow_pickle=True).item()
test_activations  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy',  allow_pickle=True).item()

def build_df(act_dict, split_name):
    data = [{'crop_file': k, 'valore': v} for k, v in act_dict.items()]
    df = pd.DataFrame(data)
    df['dataset'] = split_name
    df['valore_softmax'] = df['valore'].apply(lambda x: softmax(x))
    df['pred_label']     = df['valore_softmax'].apply(lambda x: np.argmax(x))
    return df

train_df = build_df(train_activations, 'train')
val_df   = build_df(val_activations,   'val')
test_df  = build_df(test_activations,  'test')
all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

a = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv')
a = a.rename(columns={'label': 'true_label'})
true_pred = all_df.merge(a, on='crop_file')
true_pred['train_label'] = (true_pred['pred_label'] == true_pred['true_label']).astype(int)

# -----------------------------------------------------------------------------
# SECTION B – Split list (UNCHANGED)                                           |
# -----------------------------------------------------------------------------
splits = {
    'train': [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    'val':   [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    'test':  [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}

root_dir = '/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt'

# -----------------------------------------------------------------------------
# SECTION C – Dataset & DataLoader                                            |
# -----------------------------------------------------------------------------
class CWTDataset(Dataset):
    """Restituisce tensore (C=19, F=40, T=500) pronto per EEGNet."""
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        fp = os.path.join(root_dir, os.path.basename(self.file_list[idx]))
        cwt = np.load(fp)                          # (40,500,19)
        tensor = torch.tensor(cwt, dtype=torch.float32).permute(2,0,1)  # (19,40,500)
        return tensor, self.labels[idx]

def get_loader(split_name, batch_size=64):
    subset = true_pred[true_pred['original_rec'].isin([f'sub-{s:03d}' for s in splits[split_name]])]
    files  = subset['crop_file'].tolist()
    labels = subset['train_label'].tolist()
    return DataLoader(CWTDataset(files, labels), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

train_loader = get_loader('train')
val_loader   = get_loader('val')
test_loader  = get_loader('test')

# -----------------------------------------------------------------------------
# SECTION D – EEGNet feature extractor                                        |
# -----------------------------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=19):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0,32), bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, (n_channels,1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(0.25)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1,16), padding=(0,8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(0.25)
        )
    def forward(self, x):
        # x: (B,19,40,500) -> add dummy dim for Conv2d
        b, c, f, t = x.shape
        x = x.unsqueeze(1)               # (B,1,19,40,500)
        x = x.view(b,1,c,f*t)            # (B,1,19,20000)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x.flatten(start_dim=1)    # (B, features)

feature_extractor = EEGNet()
# Carica eventuali pesi pre‑trainati
# feature_extractor.load_state_dict(torch.load('/path/to/eegnet_pretrained.pt'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_extractor.to(device).eval()

@torch.no_grad()
def extract_features(loader):
    feats, labs = [], []
    for x, y in tqdm(loader, desc='Extracting'):
        x = x.to(device)
        f = feature_extractor(x)
        feats.append(f.cpu().numpy())
        labs.extend(y)
    return np.vstack(feats), np.array(labs)

X_train, y_train = extract_features(train_loader)
X_val,   y_val   = extract_features(val_loader)
X_test,  y_test  = extract_features(test_loader)

# -----------------------------------------------------------------------------
# SECTION E – Standardization                                                 |
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# SECTION F – Classifier (FT‑Transformer → fallback MLP)                     |
# -----------------------------------------------------------------------------
try:
    import rtdl
    from torch.nn.functional import cross_entropy
    from torch.optim import AdamW

    class FTWrapper(nn.Module):
        def __init__(self, dim_in, n_classes=2):
            super().__init__()
            self.backbone = rtdl.FTTransformer(
                d_in=dim_in,
                n_blocks=3,
                n_heads=8,
                d_out=n_classes,
                ffn_dropout=0.2,
                attention_dropout=0.2
            )
        def forward(self, x):
            return self.backbone(x)

    ft_model = FTWrapper(X_train.shape[1]).to(device)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long,  device=device)
    Xv  = torch.tensor(X_val,   dtype=torch.float32, device=device)
    yv  = torch.tensor(y_val,   dtype=torch.long,  device=device)

    opt = AdamW(ft_model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state, best_loss, patience = None, 1e9, 0
    for epoch in range(300):
        ft_model.train(); opt.zero_grad()
        loss = cross_entropy(ft_model(Xtr), ytr)
        loss.backward(); opt.step()
        ft_model.eval()
        with torch.no_grad():
            val_loss = cross_entropy(ft_model(Xv), yv).item()
        if val_loss < best_loss:
            best_loss = val_loss; best_state = ft_model.state_dict(); patience = 0
        else:
            patience += 1
        if patience > 20:
            break
    ft_model.load_state_dict(best_state)

    @torch.no_grad()
    def classifier_predict(model, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        return model(X_t).argmax(1).cpu().numpy()

    preds = classifier_predict(ft_model, X_test)

except ImportError:
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), activation='relu', solver='adam',
                        max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
    mlp.fit(X_train, y_train)
    preds = mlp.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")

# -----------------------------------------------------------------------------
# SECTION G – Save legacy‑style outputs                                       |
# -----------------------------------------------------------------------------
out_dir = '/home/alfio/improving_dementia_detection_model/results_ensemble'
os.makedirs(out_dir, exist_ok=True)

# 1) Test predictions CSV
pd.DataFrame({'True': y_test, 'Predicted': preds}).to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

# 2) Classification report
report = classification_report(y_test, preds, target_names=['Bad', 'Good'])
with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# 3) Confusion matrix plot
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
plt.close()

print(f"[INFO] Outputs salvati in {out_dir}")
