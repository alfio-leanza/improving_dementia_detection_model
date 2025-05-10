import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from scipy.special import softmax

# =============== EEGNet‑v4 implementation ===============
class EEGNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=19, input_samples=500,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.classifier = nn.Linear(F2 * ((input_samples // 4) // 8), num_classes)
    def forward(self, x):
        x = self.firstconv(x); x = self.depthwise(x); x = self.separable(x)
        return self.classifier(x.flatten(start_dim=1))

# =============== Data loading (identico) ===============
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
val_activations   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy',   allow_pickle=True).item()
test_activations  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy',  allow_pickle=True).item()

def build_df(act_dict, split):
    df = pd.DataFrame([{'crop_file':k, 'valore':v} for k,v in act_dict.items()])
    df['dataset'] = split
    df['valore_softmax'] = df['valore'].apply(lambda x: softmax(x))
    df['pred_label'] = df['valore_softmax'].apply(lambda x: np.argmax(x))
    return df

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
        path = os.path.join(root_dir, self.files[idx])
        cwt  = np.load(path).mean(axis=0).T.astype('float32')  # (19,500)
        return torch.tensor(cwt).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long), self.files[idx]

def make_loader(split,batch=32,shuffle=True):
    subset = true_pred[true_pred['original_rec'].isin([f'sub-{s:03d}' for s in splits[split]])]
    return DataLoader(CWT_Dataset(subset), batch_size=batch, shuffle=shuffle, num_workers=4)

train_loader = make_loader('train'); val_loader = make_loader('val',shuffle=False); test_loader = make_loader('test',shuffle=False)

# =============== Training ===============================
parser = argparse.ArgumentParser(); parser.add_argument('--weights',type=str,default=''); args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGNet(num_classes=2, in_channels=19, input_samples=500).to(device)
if args.weights and os.path.isfile(args.weights):
    print('[INFO] Loading pre‑trained weights:', args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
EPOCHS = 25
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# freeze first blocks first 5 ep
for p in model.firstconv.parameters(): p.requires_grad = False
for p in model.depthwise.parameters(): p.requires_grad = False

history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
for ep in range(1, EPOCHS+1):
    if ep == 6:
        for p in model.firstconv.parameters(): p.requires_grad = True
        for p in model.depthwise.parameters(): p.requires_grad = True

    model.train(); tl, correct, total = 0.0,0,0
    for x,y,_ in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(); out = model(x); loss = criterion(out,y); loss.backward(); optimizer.step()
        tl += loss.item(); preds = out.argmax(1); total += y.size(0); correct += preds.eq(y).sum().item()
    train_loss = tl/len(train_loader); train_acc = correct/total

    model.eval(); vl, vc, vt = 0.0,0,0
    with torch.no_grad():
        for x,y,_ in val_loader:
            x,y = x.to(device), y.to(device); out = model(x)
            vl += criterion(out,y).item(); vc += out.argmax(1).eq(y).sum().item(); vt += y.size(0)
    val_loss = vl/len(val_loader); val_acc = vc/vt; scheduler.step()
    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss);   history['val_acc'].append(val_acc)
    print(f"Epoch {ep}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

# =============== Helper to collect predictions ==========

def collect_predictions(loader, split):
    rows = []
    model.eval()
    with torch.no_grad():
        for x,y,fn in loader:
            logits = model(x.to(device))
            preds  = logits.argmax(1).cpu().numpy()
            sm     = softmax(logits.cpu().numpy(), axis=1)
            for f,t,p,l,prob in zip(fn, y.numpy(), preds, logits.cpu().numpy(), sm):
                goodness = prob[p]
                rows.append([f, int(t), int(p), [float(l[0]), float(l[1])], [float(prob[0]), float(prob[1])], goodness])
    df = pd.DataFrame(rows, columns=['crop_file','true_label','pred_label','logits','softmax_values','goodness'])
    df.to_csv(os.path.join(out_dir, f"{split}_predictions_detailed.csv"), index=False)
    return df

# =============== Test & Save ============================
out_dir = '/home/alfio/improving_dementia_detection_model/results_eegnet2'; os.makedirs(out_dir, exist_ok=True)

# detailed CSVs
collect_predictions(train_loader, 'train')
collect_predictions(val_loader,   'val')
collect_predictions(test_loader,  'test')

# Confusion matrix & classification report on test
from sklearn.metrics import classification_report, confusion_matrix

df_test = pd.read_csv(os.path.join(out_dir,'test_predictions_detailed.csv'))
cm = confusion_matrix(df_test['true_label'], df_test['pred_label'])
report = classification_report(df_test['true_label'], df_test['pred_label'], target_names=['Bad','Good'])

with open(os.path.join(out_dir,'classification_report.txt'),'w') as f:
    f.write(report)

plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Bad','Good'],yticklabels=['Bad','Good'])
plt.title('Confusion Matrix'); plt.xlabel('Pred'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(out_dir,'confusion_matrix.png')); plt.close()

# Save model
torch.save(model.state_dict(), os.path.join(out_dir,'eegnet_finetuned.pth'))

print('[INFO] Finished. Results in', out_dir)
