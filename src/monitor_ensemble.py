# improved_monitor_ensemble.py
"""
Pipeline Ensemble – EEGNet → StandardScaler → MLPClassifier (tuned)
==================================================================
Fix: RuntimeError dato da input 5‑D nella first Conv2d.
Ora le CWT (.npy 40×500×19) sono **mediate lungo l’asse frequenza (40)**
e convertite in tensore (1, 19, 500) che corrisponde a (batch, channel=1,
channelsEEG=19, time=500) richiesto da EEGNet.

Flusso invariato (activations, split, salvataggi), con:
1. **Feature extraction fine‑tuned su EEGNet** (ultimi layer).
2. **BalancedBatchSampler + FocalLoss**.
3. **Scikit‑learn MLPClassifier** ottimizzato via grid.

É pronto per lanciare.
"""

import os, math, joblib, warnings, random, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
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
# 0. Load activations & labels (unchanged)
# ---------------------------------------------------------------------
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
val_activations   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy',   allow_pickle=True).item()
test_activations  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy',  allow_pickle=True).item()

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

a = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv').rename(columns={'label':'true_label'})
true_pred = all_df.merge(a, on='crop_file')
true_pred['train_label'] = (true_pred['pred_label'] == true_pred['true_label']).astype(int)

root_dir = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"

splits = {
    'train':[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    'val'  :[54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    'test' :[60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}

# ---------------------------------------------------------------------
# 1. Dataset & Balanced Sampler
# ---------------------------------------------------------------------
class CWT_Dataset(Dataset):
    """Restituisce tensore (1,19,500) + label"""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cwt = np.load(Path(root_dir)/row['crop_file'])  # (40,freq 500,time 19channels)
        cwt = cwt.mean(axis=0).astype(np.float32)       # media su freq → (500,19)
        cwt = cwt.T                                     # (19,500)
        tensor = torch.from_numpy(cwt).unsqueeze(0)      # (1,19,500)
        return tensor, row['train_label']

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_idx = np.where(self.labels==1)[0]
        self.neg_idx = np.where(self.labels==0)[0]
        self.half = batch_size//2

    def __iter__(self):
        pos_iter = itertools.cycle(np.random.permutation(self.pos_idx))
        neg_iter = itertools.cycle(np.random.permutation(self.neg_idx))
        for _ in range(len(self.labels)//self.batch_size):
            batch = list(itertools.islice(pos_iter, self.half)) + list(itertools.islice(neg_iter, self.batch_size-self.half))
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.labels)//self.batch_size

# ---------------------------------------------------------------------
# 2. EEGNet (ridotta) + Focal Loss
# ---------------------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, chans=19, samples=500):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1,64), bias=False),
            nn.BatchNorm2d(8))
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, (chans,1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(0.25))
        self.separable = nn.Sequential(
            nn.Conv2d(16,16,(1,16), padding=(0,8), bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1,8)), nn.Dropout(0.25))
        self.classify = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16,2)
    def forward(self,x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        feat = self.classify(x).flatten(1)
        return self.fc(feat), feat

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.44, gamma=2):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.ce=nn.CrossEntropyLoss(reduction='none')
    def forward(self,logits,targets):
        ce = self.ce(logits,targets); pt=torch.exp(-ce); return (self.alpha*(1-pt)**self.gamma*ce).mean()

# ---------------------------------------------------------------------
# 3. Loader helpers
# ---------------------------------------------------------------------

def make_loader(split,batch=64,balanced=False):
    recs=[f'sub-{s:03d}' for s in splits[split]]
    df=true_pred[true_pred['original_rec'].isin(recs)]
    ds=CWT_Dataset(df)
    if balanced and split=='train':
        sampler=BalancedBatchSampler(df['train_label'].values,batch)
        return DataLoader(ds,batch_sampler=sampler,num_workers=4,pin_memory=True)
    return DataLoader(ds,batch_size=batch,shuffle=split=='train',num_workers=4,pin_memory=True)

train_loader=make_loader('train',balanced=True)
val_loader  =make_loader('val')

eegnet=EEGNet().to(DEVICE)
for n,p in eegnet.named_parameters():
    p.requires_grad=('depthwise' in n) or ('separable' in n) or ('fc' in n)

criterion=FocalLoss()
opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,eegnet.parameters()),lr=1e-4,weight_decay=1e-3)

best_val=math.inf
for epoch in range(1,11):
    eegnet.train(); tl=0
    for x,y in train_loader:
        x=x.to(DEVICE); y=y.to(DEVICE)
        opt.zero_grad(); out,_=eegnet(x); loss=criterion(out,y); loss.backward(); opt.step(); tl+=loss.item()
    tl/=len(train_loader)
    eegnet.eval(); vl=0
    with torch.no_grad():
        for x,y in val_loader:
            x=x.to(DEVICE); y=y.to(DEVICE)
            vl+=criterion(eegnet(x)[0],y).item()
    vl/=len(val_loader)
    print(f"Epoch {epoch}: train {tl:.4f} val {vl:.4f}")
    if vl<best_val:
        best_val=vl; torch.save(eegnet.state_dict(),RESULTS_DIR/'eegnet_best.pth')

eegnet.load_state_dict(torch.load(RESULTS_DIR/'eegnet_best.pth',map_location=DEVICE))

def extract(loader):
    eegnet.eval(); feats,lab=[],[]
    with torch.no_grad():
        for x,y in loader:
            f=eegnet(x.to(DEVICE))[1].cpu().numpy(); feats.append(f); lab.extend(y.numpy())
    return np.vstack(feats), np.array(lab)

X_train,y_train=extract(train_loader)
X_val,y_val    =extract(val_loader)
X_test,y_test  =extract(make_loader('test'))

scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train); X_val=scaler.transform(X_val); X_test=scaler.transform(X_test)
joblib.dump(scaler,RESULTS_DIR/'scaler.joblib')

param_grid={'hidden_layer_sizes':[(128,),(256,128)],'alpha':[1e-4,1e-3],'learning_rate_init':[1e-3,5e-4],'batch_size':[64,128]}

best_acc=-1
for params in ParameterGrid(param_grid):
    clf=MLPClassifier(max_iter=200,early_stopping=True,random_state=42,**params)
    clf.fit(X_train,y_train)
    acc=accuracy_score(y_val,clf.predict(X_val))
    if acc>best_acc:
        best_acc, best_params, best_clf = acc, params, clf
print("Best",best_params,"val_acc",best_acc)
joblib.dump(best_clf,RESULTS_DIR/'mlp_best.joblib')

pred=best_clf.predict(X_test)
acc=accuracy_score(y_test,pred)
report=classification_report(y_test,pred,target_names=['Bad','Good'])
cm=confusion_matrix(y_test,pred)
print("Test acc",acc)
print(report)

pd.DataFrame({'True':y_test,'Predicted':pred}).to_csv(RESULTS_DIR/'test_predictions.csv',index=False)
Path(RESULTS_DIR/'classification_report.txt').write_text(report)
plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Bad','Good'],yticklabels=['Bad','Good'])
plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Pred');
plt.tight_layout(); plt.savefig(RESULTS_DIR/'confusion_matrix.png'); plt.close()
print("[INFO] Results saved in",RESULTS_DIR)
