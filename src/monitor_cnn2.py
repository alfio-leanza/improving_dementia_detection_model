# monitor_cnn2d_residual.py
"""
CNN 2D con Residual + SE‑Attention, Focal Loss e Balanced Sampler
================================================================

✔ Caricamento activations & label **identico** agli script legacy.
✔ Data‑split invariato.
✔ Dataset con augment EEG‑aware (noise, masking, time‑warp, freq‑shift).
✔ Architettura: 4 Residual‑SE block (19→32→64→128→256) con Depthwise conv.
✔ Focal Loss (γ = 2, class_weight bilanciato) + BalancedBatchSampler.
✔ Optimizer AdamW + CosineAnnealingWarmRestarts + SWA.
✔ Early‑Stopping su recall classe Bad (patience 10).
✔ Salva test_predictions.csv, classification_report.txt, confusion_matrix.png.

Eseguire: `python monitor_cnn2d_residual.py` (richiede torch >=1.10).
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from scipy.special import softmax

# === Load activations and labels ===
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
test_activations  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy', allow_pickle=True).item()
val_activations   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy', allow_pickle=True).item()

def build_df(act_dict, split_name):
    data = [{'crop_file': k, 'valore': v} for k, v in act_dict.items()]
    df = pd.DataFrame(data)
    df['dataset'] = split_name
    df['valore_softmax'] = df['valore'].apply(lambda x: softmax(x))
    df['pred_label'] = df['valore_softmax'].apply(lambda x: np.argmax(x))
    return df

train_df = build_df(train_activations, 'train')
val_df   = build_df(val_activations,  'val')
test_df  = build_df(test_activations, 'test')
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

# === Utility: BalancedBatchSampler ===
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes=2, n_samples=32):
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.idxs = [np.where(self.labels == cls)[0] for cls in self.classes]
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        for idx in self.idxs:
            np.random.shuffle(idx)
        self.cursors = [0]*len(self.classes)
        self.length = len(self.labels) // self.batch_size
    def __iter__(self):
        for _ in range(self.length):
            batch = []
            for cls_idx, cls in enumerate(self.classes):
                start = self.cursors[cls_idx]
                end = start + self.n_samples
                idx_pool = self.idxs[cls_idx]
                if end > len(idx_pool):
                    np.random.shuffle(idx_pool)
                    start, end = 0, self.n_samples
                batch.extend(idx_pool[start:end])
                self.cursors[cls_idx] = end
            np.random.shuffle(batch)
            yield batch
    def __len__(self):
        return self.length

# === Dataset with EEG‑aware augmentation ===
class CWT_Dataset(Dataset):
    def __init__(self, df_subset, augment=False):
        self.files  = df_subset['crop_file'].tolist()
        self.labels = df_subset['train_label'].tolist()
        self.augment = augment
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = np.load(os.path.join(root_dir, self.files[idx]))  # (40,500,19)
        if self.augment:
            # Gaussian noise
            data += np.random.normal(0, 0.01, data.shape)
            # Random masking (time or freq)
            if np.random.rand() < 0.5:
                t = np.random.randint(0, data.shape[1]-50)
                data[:, t:t+50, :] = 0
            if np.random.rand() < 0.5:
                f = np.random.randint(0, data.shape[0]-5)
                data[f:f+5, :, :] = 0
            # Time‑warp (zoom)
            if np.random.rand() < 0.3:
                rate = np.random.uniform(0.9,1.1)
                data = np.interp(np.arange(0,500,rate), np.arange(500), data.reshape(40*19,500)).reshape(40,19,-1).transpose(0,2,1)
                data = data[:40,:500,:19]
            # Freq‑shift (roll)
            if np.random.rand() < 0.3:
                shift = np.random.randint(-3,4)
                data = np.roll(data, shift, axis=0)
        # ↓→ (19,40,500) reorder then to (19,40,500)
        data = np.transpose(data, (2,0,1)).astype('float32')  # (19,40,500)
        return torch.tensor(data), torch.tensor(self.labels[idx], dtype=torch.long)

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha if alpha is not None else torch.ones(2)
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha.to(logits.device))
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        return loss

# === Residual Block with SE and optional Depthwise ===
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, depthwise=False):
        super().__init__()
        g = out_c if depthwise else 1
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, groups=g, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, groups=g, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.short = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c!=out_c else nn.Identity()
        # SE‑block
        self.se_fc1 = nn.Linear(out_c, out_c//8, bias=False)
        self.se_fc2 = nn.Linear(out_c//8, out_c, bias=False)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.short(x)
        # SE attention
        w = F.adaptive_avg_pool2d(y,1).view(y.size(0), -1)
        w = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(w)))).view(y.size(0), y.size(1),1,1)
        y = y * w
        return F.relu(y)

# === CNN with 4 Residual blocks ===
class CNN2DResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            ResBlock(19, 32),
            nn.MaxPool2d(2),
            ResBlock(32, 64, depthwise=True),
            nn.MaxPool2d(2),
            ResBlock(64, 128),
            nn.MaxPool2d(2),
            ResBlock(128, 256, depthwise=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

# === Create DataLoaders ===

def get_loader(split, batch_size=64, augment=False, balanced=False):
    subset = true_pred[true_pred['original_rec'].isin([f'sub-{s:03d}' for s in splits[split]])]
    ds = CWT_Dataset(subset, augment=augment)
    if balanced and split=='train':
        sampler = BalancedBatchSampler(subset['train_label'].values, n_classes=2, n_samples=batch_size//2)
        return DataLoader(ds, batch_sampler=sampler, num_workers=4)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=4)

train_loader = get_loader('train', augment=True, balanced=True)
val_loader   = get_loader('val')
test_loader  = get_loader('test')

# === Training utilities ===

def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            all_pred.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())
    return np.array(all_true), np.array(all_pred)

# === Main ===

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN2DResidual().to(device)

    criterion = FocalLoss(alpha=torch.tensor([0.56,0.44]), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    swa_model = AveragedModel(model)
    swa_start = 48  # start SWA at 80% of 60 epochs
    swa_scheduler = SWALR(optimizer, anneal_epochs=5, anneal_strategy='cos')

    best_recall0, patience, wait = 0.0, 10, 0
    epochs = 60
    history = {'train_loss':[], 'val_recall0':[]}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # validation recall Bad (class 0)
        y_true, y_pred = evaluate(model, val_loader, device)
        recall0 = recall_score(y_true, y_pred, pos_label=0)
        val_loss = running_loss / len(train_loader)
        history['train_loss'].append(val_loss)
        history['val_recall0'].append(recall0)
        print(f"Epoch {epoch}: loss={val_loss:.4f} recall0={recall0:.4f}")

        if recall0 > best_recall0:
            best_recall0, wait = recall0, 0
            torch.save(model.state_dict(), 'best_cnn2d.pth')
        else:
            wait += 1
            if wait >= patience:
                print("[Early stopping]")
                break

    # SWA: swap model
    update_bn(train_loader, swa_model, device=device)
    model = swa_model

    # === Test ===
    y_true, y_pred = evaluate(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=['Bad','Good'])
    cm = confusion_matrix(y_true, y_pred)

    output_dir = "/home/alfio/improving_dementia_detection_model/results_cnn2d_residual"
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'True':y_true, 'Predicted':y_pred}).to_csv(os.path.join(output_dir,'test_predictions.csv'), index=False)
    with open(os.path.join(output_dir,'classification_report.txt'), 'w') as f:
        f.write(report)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    print("[INFO] Results saved in", output_dir)

if __name__ == '__main__':
    main()
