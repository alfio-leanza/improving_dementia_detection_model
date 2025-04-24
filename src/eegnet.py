# monitor_eegnet_pretrained.py
"""
EEGNet‑v4 fine‑tuning – salvataggio uniforme (model + CSV per train/val/test)
============================================================================
• Architettura **invariata**.
• Aggiunti CSV dettagliati (logits, softmax list, goodness) per ciascun set.
• Salva anche modello finito `eegnet_finetuned.pth`.
"""
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from scipy.special import softmax as sf

# =========== EEGNet‑v4 (unchanged) ===========
class EEGNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=19, input_samples=500,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.firstconv = nn.Sequential(nn.Conv2d(1,F1,(1,64),padding=(0,32),bias=False), nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1,F1*D,(in_channels,1),groups=F1,bias=False), nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d((1,4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D,F2,(1,16),padding=(0,8),bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1,8)), nn.Dropout(dropout))
        self.classifier = nn.Linear(F2*((input_samples//4)//8), num_classes)
    def forward(self,x):
        x=self.firstconv(x); x=self.depthwise(x); x=self.separable(x); return self.classifier(x.flatten(1))

# =========== Data loading ===========
train_acts=np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy',allow_pickle=True).item()
val_acts  =np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy', allow_pickle=True).item()
test_acts =np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy',allow_pickle=True).item()

def build_df(act,split):
    df=pd.DataFrame({'crop_file':list(act.keys()),'valore':list(act.values())}); df['dataset']=split
    df['valore_softmax']=df['valore'].apply(sf); df['pred_label']=df['valore_softmax'].apply(lambda x:int(np.argmax(x)))
    return df
train_df, val_df, test_df = (build_df(train_acts,'train'), build_df(val_acts,'val'), build_df(test_acts,'test'))
all_df = pd.concat([train_df,val_df,test_df]);
annot = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv').rename(columns={'label':'true_label'})
true_pred = all_df.merge(annot,on='crop_file'); true_pred['train_label']=(true_pred['pred_label']==true_pred['true_label']).astype(int)

root_dir="/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
splits={'train':[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'val':[54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
        'test':[60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]}

class CWT_Dataset(Dataset):
    def __init__(self, df):
        self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.loc[idx]; cwt=np.load(os.path.join(root_dir,row.crop_file)).mean(0).T.astype('float32')
        return torch.tensor(cwt).unsqueeze(0), torch.tensor(row.train_label), row.crop_file

def make_loader(split,batch=32,shuffle=True):
    subset=true_pred[true_pred['original_rec'].isin([f'sub-{s:03d}'for s in splits[split]])]
    return DataLoader(CWT_Dataset(subset),batch_size=batch,shuffle=shuffle,num_workers=4)
train_loader, val_loader, test_loader = (make_loader('train'), make_loader('val',shuffle=False), make_loader('test',shuffle=False))

# =========== Training ===========
parser=argparse.ArgumentParser(); parser.add_argument('--weights',type=str,default=''); args=parser.parse_args()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=EEGNet().to(device)
if args.weights and os.path.isfile(args.weights):
    model.load_state_dict(torch.load(args.weights,map_location=device),strict=False)
criterion=nn.CrossEntropyLoss(weight=torch.tensor([0.56,0.44],device=device))
optimizer=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,eta_min=1e-6)

for ep in range(1,11):
    model.train(); corr=tot=tloss=0.0
    for x,y,_ in train_loader:
        x,y=x.to(device),y.to(device); optimizer.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
        tloss+=loss.item(); pred=out.argmax(1); tot+=y.size(0); corr+=pred.eq(y).sum().item()
    model.eval(); vcorr=vtot=vloss=0.0
    with torch.no_grad():
        for x,y,_ in val_loader:
            x,y=x.to(device),y.to(device); out=model(x); vloss+=criterion(out,y).item(); vcorr+=out.argmax(1).eq(y).sum().item(); vtot+=y.size(0)
    scheduler.step(); print(f"Epoch {ep}: train_acc={corr/tot:.4f} val_acc={vcorr/vtot:.4f}")

# =========== Collect preds & save ==========
out_dir='/home/alfio/improving_dementia_detection_model/results_eegnet'; os.makedirs(out_dir,exist_ok=True)

def collect(loader,split):
    model.eval(); rows=[]
    with torch.no_grad():
        for x,y,fn in loader:
            logits=model(x.to(device)).cpu(); sm=sf(logits,axis=1); preds=logits.argmax(1)
            goodness=sm[np.arange(len(sm)),preds]
            for f,t,p,l,s,g in zip(fn,y.cpu(),preds,logits.tolist(),sm.tolist(),goodness):
                rows.append([f,int(t),int(p),l,s,float(g)])
    pd.DataFrame(rows,columns=['crop_file','true_label','pred_label','logits','softmax','goodness']) \
      .to_csv(os.path.join(out_dir,f'{split}_predictions_detailed.csv'),index=False)

collect(train_loader,'train'); collect(val_loader,'val'); collect(test_loader,'test')

# Confusion matrix & report
yt,yp=[],[]
with torch.no_grad():
    for x,y,_ in test_loader:
        yt.extend(y.numpy()); yp.extend(model(x.to(device)).argmax(1).cpu().numpy())
cm=confusion_matrix(yt,yp); report=classification_report(yt,yp,target_names=['Bad','Good'])
with open(os.path.join(out_dir,'classification_report.txt'),'w') as f: f.write(report)
plt.figure(figsize=(6,6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Bad','Good'],yticklabels=['Bad','Good'])
plt.title('Confusion Matrix'); plt.savefig(os.path.join(out_dir,'confusion_matrix.png')); plt.close()

# Save model
torch.save(model.state_dict(), os.path.join(out_dir,'eegnet_finetuned.pth'))
print('[INFO] Results saved in', out_dir)
