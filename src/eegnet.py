import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from scipy.special import softmax

# ================== EEGNet‑v4 implementation ==================
class EEGNet(nn.Module):
    """Minimal EEGNet‑v4 (Lawhern et al. 2018) for 19×500 inputs."""
    def __init__(self, num_classes=2, in_channels=19, input_samples=500,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),  # temporal
            nn.BatchNorm2d(F1)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(F2 * ((input_samples // 4) // 8), num_classes)

    def forward(self, x):          # x shape (B,1,19,500)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

# ================== Data loading ==================
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
val_activations   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy',   allow_pickle=True).item()
test_activations  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy',  allow_pickle=True).item()

def build_df(act_dict, split):
    df = pd.DataFrame([{'crop_file':k, 'valore':v} for k,v in act_dict.items()])
    df['dataset'] = split; df['valore_softmax'] = df['valore'].apply(lambda x: softmax(x));
    df['pred_label'] = df['valore_softmax'].apply(lambda x: np.argmax(x)); return df

train_df = build_df(train_activations,'train'); val_df = build_df(val_activations,'val'); test_df = build_df(test_activations,'test')
all_df   = pd.concat([train_df,val_df,test_df], ignore_index=True)
annot = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv').rename(columns={'label':'true_label'})
true_pred = all_df.merge(annot, on='crop_file'); true_pred['train_label'] = (true_pred['pred_label']==true_pred['true_label']).astype(int)

root_dir = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
splits = {
    'train':[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    'val':  [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    'test': [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}

class CWT_Dataset(Dataset):
    def __init__(self, df_subset):
        self.files  = df_subset['crop_file'].tolist(); self.labels = df_subset['train_label'].tolist()
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        cwt = np.load(os.path.join(root_dir, self.files[idx])).mean(axis=0).T.astype('float32')  # (19,500)
        tensor = torch.tensor(cwt).unsqueeze(0)  # (1,19,500)
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

def make_loader(split,batch=32,shuffle=True):
    subset = true_pred[true_pred['original_rec'].isin([f'sub-{s:03d}' for s in splits[split]])]
    return DataLoader(CWT_Dataset(subset), batch_size=batch, shuffle=shuffle, num_workers=4)

train_loader = make_loader('train'); val_loader = make_loader('val',shuffle=False); test_loader = make_loader('test',shuffle=False)

# ================== Training ==================
parser = argparse.ArgumentParser(); parser.add_argument('--weights',type=str,default=''); args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGNet(num_classes=2, in_channels=19, input_samples=500).to(device)
if args.weights and os.path.isfile(args.weights):
    print('[INFO] Loading pre‑trained weights:', args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.56,0.44], device=device))
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

epochs = 10
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
for ep in range(1, epochs+1):
    model.train(); tloss=0.0; correct=0; total=0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(); out = model(x); loss = criterion(out,y); loss.backward(); optimizer.step()
        tloss += loss.item(); preds = out.argmax(1); total += y.size(0); correct += preds.eq(y).sum().item()
    train_loss = tloss/len(train_loader); train_acc = correct/total

    model.eval(); vloss=0.0; vcorrect=0; vtotal=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device); out = model(x)
            vloss += criterion(out,y).item(); vcorrect += out.argmax(1).eq(y).sum().item(); vtotal += y.size(0)
    val_loss = vloss/len(val_loader); val_acc = vcorrect/vtotal; scheduler.step()

    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss);   history['val_acc'].append(val_acc)
    print(f"Epoch {ep}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

# ================== Test ==================
model.eval(); yt,yp=[],[]
with torch.no_grad():
    for x,y in test_loader:
        yt.extend(y.numpy()); yp.extend(model(x.to(device)).argmax(1).cpu().numpy())
cm = confusion_matrix(yt,yp); report = classification_report(yt,yp, target_names=['Bad','Good'])

out_dir = '/home/alfio/improving_dementia_detection_model/results_eegnet'; os.makedirs(out_dir, exist_ok=True)
pd.DataFrame({'True':yt,'Predicted':yp}).to_csv(os.path.join(out_dir,'test_predictions.csv'), index=False)
with open(os.path.join(out_dir,'classification_report.txt'),'w') as f: f.write(report)
plt.figure(figsize=(6,6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
plt.title('Confusion Matrix'); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'confusion_matrix.png')); plt.close()
print('[INFO] Finished. Results in', out_dir)
