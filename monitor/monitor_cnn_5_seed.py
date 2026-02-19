import os, glob, re, ast, random
import numpy as np
import pandas as pd
from scipy.special import softmax as sf
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

# ================== CONFIG ==================
project_root = '/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio'  
cwt_root     = '/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt'                     
out_root     = os.path.join(project_root, 'monitor_cnn')                              
annot_csv    = None  

epochs       = 20
batch_size   = 64
lr           = 1e-3
weight_decay = 1e-3
num_workers  = 4
augment_sigma= 0.01    
seed_global  = 42      

os.makedirs(out_root, exist_ok=True)

# ================== MODEL ===================
class CNN_ChannelAttention(nn.Module):
    def __init__(self, num_channels=19, num_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.3), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.3), nn.AdaptiveAvgPool2d(1))
        self.channel_attention = nn.Sequential(nn.Linear(256,64,bias=False), nn.ReLU(), nn.Linear(64,256,bias=False), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(256,num_classes))
    def forward(self,x):
        x = self.conv_block(x).flatten(1)
        x = x * self.channel_attention(x)
        return self.classifier(x)

# ================== UTILS ===================
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def convert_to_array(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x).squeeze()
    if isinstance(x, str):
        s = x.strip()
        if not (s.startswith('[') and s.endswith(']')):
            s = re.sub(r'(?<=\d)\s+(?=\d)', ',', s)
            s = '[' + s + ']'
        return np.array(ast.literal_eval(s)).squeeze()
    return np.array(x).squeeze()

def read_inferences(csv_path, annot_df=None):
    """Read a *_inferences.csv and return a DF containing at least crop_file, pred_label, true_label."""
    df = pd.read_csv(csv_path)
    if 'crop_file' not in df.columns:
        raise KeyError(f"'crop_file' missing in {csv_path}")
    # pred_label: if missing, derivate it from softmax_values/logits
    if 'pred_label' not in df.columns:
        if 'softmax_values' in df.columns:
            tmp = df['softmax_values'].apply(convert_to_array)
            df['pred_label'] = tmp.apply(lambda a: int(np.argmax(a)))
        elif 'logits' in df.columns:
            tmp = df['logits'].apply(convert_to_array)
            df['pred_label'] = tmp.apply(lambda a: int(np.argmax(a)))
        else:
            raise KeyError(f"'pred_label' missing and no column softmax/logits found in {csv_path}")
    # true_label: use the CSV one if available, otherwise annot_df
    if 'true_label' not in df.columns:
        if annot_df is None:
            raise KeyError(f"'true_label' missing on {csv_path} and annot_df=None")
        df = df.merge(annot_df[['crop_file','true_label']], on='crop_file', how='left')
        if df['true_label'].isna().any():
            missing = df[df['true_label'].isna()].shape[0]
            raise ValueError(f"{missing} true_label not found via annot_df for {csv_path}")
    return df[['crop_file','pred_label','true_label']]

class CWT_Dataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        data = np.load(os.path.join(cwt_root, row.crop_file))  # shape (H,W,19)
        if self.augment:
            data = data + np.random.normal(0, augment_sigma, data.shape)
        tensor = torch.tensor(data, dtype=torch.float32).permute(2,0,1)  # (C,H,W) con C=19
        label = torch.tensor(row.train_label, dtype=torch.long)
        return tensor, label, row.crop_file

def make_loader(df, batch=batch_size, augment=False, shuffle=True):
    return DataLoader(CWT_Dataset(df, augment), batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def compute_class_weights(labels):
    # labels ∈ {0,1}; weights inversely proportional to frequency
    counts = np.bincount(labels, minlength=2).astype(float)
    counts[counts==0] = 1.0
    total = counts.sum()
    weights = total / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)

def collect_predictions(model, loader, device, out_csv):
    model.eval(); rows=[]
    with torch.no_grad():
        for x,y,fns in loader:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            sm = sf(logits, axis=1)
            preds = np.argmax(logits, axis=1)
            goodness = sm[np.arange(len(sm)), preds]
            for f, t, p, l, s, g in zip(fns, y.numpy(), preds, logits.tolist(), sm.tolist(), goodness.tolist()):
                rows.append([f, int(t), int(p), l, s, float(g)])
    pd.DataFrame(rows, columns=['crop_file','true_train_label','pred_train_label','logits','softmax','goodness']).to_csv(out_csv, index=False)

def plot_and_save_cm(y_true, y_pred, out_png, title='Confusion Matrix', labels=('Bad','Good')):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.xlabel('Pred'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    return cm

# ================== MAIN LOOP ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(seed_global)

# optional: load annotations if needed
annot_df = pd.read_csv(annot_csv) if (annot_csv and os.path.exists(annot_csv)) else None
if annot_csv and annot_df is not None and 'label' in annot_df.columns:
    annot_df = annot_df.rename(columns={'label':'true_label'})

experiments = sorted([p for p in glob.glob(os.path.join(project_root, '5_seed*')) if os.path.isdir(p)])

if not experiments:
    raise RuntimeError(f"No '5_seed*' folder found in {project_root}")

for exp_path in experiments:
    exp_name = os.path.basename(exp_path)
    results_dir = os.path.join(exp_path, 'results')
    if not os.path.isdir(results_dir):
        print(f"[WARN] No 'results' in {exp_name}, skip.")
        continue

    seed_dirs = sorted(glob.glob(os.path.join(results_dir, 'train_*')))
    if not seed_dirs:
        print(f"[WARN] No 'train_*' seed folder in {results_dir}, skip.")
        continue

    for seed_path in seed_dirs:
        seed_name = os.path.basename(seed_path)  # es. train_20250719_141127_seed1234
        # Path inference CSV
        csv_train = os.path.join(seed_path, 'train_inferences.csv')
        csv_val   = os.path.join(seed_path, 'val_inferences.csv')
        csv_test  = os.path.join(seed_path, 'test_inferences.csv')

        missing = [p for p in [csv_train, csv_val, csv_test] if not os.path.exists(p)]
        if missing:
            print(f"[WARN] CSV missed on {exp_name}/{seed_name}: {missing}. Skip this seed.")
            continue

        # --- DataFrame building with monitor target ---
        df_train = read_inferences(csv_train, annot_df=annot_df); df_train['dataset'] = 'train'
        df_val   = read_inferences(csv_val,   annot_df=annot_df); df_val['dataset']   = 'val'
        df_test  = read_inferences(csv_test,  annot_df=annot_df); df_test['dataset']  = 'test'

        for df in (df_train, df_val, df_test):
            df['train_label'] = (df['pred_label'] == df['true_label']).astype(int)

        # --- Loader ---
        train_loader = make_loader(df_train, augment=True, shuffle=True)
        val_loader   = make_loader(df_val,   augment=False, shuffle=False)
        test_loader  = make_loader(df_test,  augment=False, shuffle=False)

        # --- Model, loss, optimizer ---
        model = CNN_ChannelAttention().to(device)
        class_weights = compute_class_weights(df_train['train_label'].values).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # --- Training loop ---
        print(f"\n[INFO] Monitor training  for {exp_name} / {seed_name}")
        best_val_acc = -1.0
        best_state   = None

        for ep in range(1, epochs+1):
            model.train(); corr=tot=loss_sum=0.0
            for x,y,_ in DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True):
                x,y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()*y.size(0)
                pred = out.argmax(1)
                tot += y.size(0); corr += pred.eq(y).sum().item()

            # Validation
            model.eval(); vcorr=vtot=0
            with torch.no_grad():
                for x,y,_ in val_loader:
                    x,y = x.to(device), y.to(device)
                    pred = model(x).argmax(1)
                    vcorr += pred.eq(y).sum().item(); vtot += y.size(0)
            train_acc = corr/max(1,tot); val_acc = vcorr/max(1,vtot)
            print(f"Epoch {ep:02d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}

        # Restore best and evaluate on test
        if best_state is not None:
            model.load_state_dict(best_state)

        # --- Output directory ---
        out_dir = os.path.join(out_root, exp_name, seed_name)
        os.makedirs(out_dir, exist_ok=True)

        # --- Save detailed predictions ---
        collect_predictions(model, train_loader, device, os.path.join(out_dir, 'train_predictions_detailed.csv'))
        collect_predictions(model, val_loader,   device, os.path.join(out_dir, 'val_predictions_detailed.csv'))
        collect_predictions(model, test_loader,  device, os.path.join(out_dir, 'test_predictions_detailed.csv'))

        # --- Report and CM on test ---
        yt, yp = [], []
        model.eval()
        with torch.no_grad():
            for x,y,_ in test_loader:
                yt.extend(y.numpy())
                yp.extend(model(x.to(device)).argmax(1).cpu().numpy())
        report = classification_report(yt, yp, target_names=['Bad','Good'])
        with open(os.path.join(out_dir,'classification_report.txt'),'w') as f:
            f.write(report)
        plot_and_save_cm(yt, yp, os.path.join(out_dir,'confusion_matrix.png'), title='Confusion Matrix (Test)')

        # --- Salva modello ---
        torch.save(model.state_dict(), os.path.join(out_dir,'cnn_channelattention.pth'))
        print(f"[INFO] Saved on {out_dir}")