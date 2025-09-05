import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax

"""
MobileNetV2 fine‑tuning per classificazione "predizione‑corretta" su CWT EEG
────────────────────────────────────────────────────────────────────────────
• Data‑augmentation sul solo training set (time‑shift, rumore, channel‑dropout)
• Class balancing (pesi inversi alla frequenza)
• Label smoothing 0.1
• Scheduler ReduceLROnPlateau (val loss) + Early‑Stopping (patience 5)
• **Salvataggio**: training_history.png, confusion_matrix.png / .csv, classification_report.txt, metrics_summary.csv
"""
# ======================================================
# 1. Caricamento attivazioni (predizioni modello base)
# ======================================================
train_acts = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
val_acts   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy',   allow_pickle=True).item()
test_acts  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy',  allow_pickle=True).item()

def _to_df(name,dic):
    df = pd.DataFrame([{'crop_file':k,'activation_values':v} for k,v in dic.items()])
    df['dataset']=name
    df['pred_label']=df['activation_values'].apply(lambda x: np.argmax(softmax(x)))
    return df

train_df,val_df,test_df=[_to_df(n,d) for n,d in zip(['train','val','test'],[train_acts,val_acts,test_acts])]
all_act_df=pd.concat([train_df,val_df,test_df],ignore_index=True)

# ======================================================
# 2. Annotazioni & label (1 = predizione corretta)
# ======================================================
annot = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv')
annot = annot.rename(columns={'label':'true_label'})

df_labels = all_act_df.merge(annot,on='crop_file')
df_labels['crop_file']=df_labels['crop_file'].apply(os.path.basename)
df_labels['train_label']=(df_labels['pred_label']==df_labels['true_label']).astype(int)

# ======================================================
# 3. Path CWT & split soggetti
# ======================================================
cwt_path='/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt'

split={'train':[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
       'val'  :[54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
       'test' :[60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]}

# ======================================================
# 4. Dataset con Augmentation
# ======================================================
class CWT_Dataset(Dataset):
    def __init__(self, files, labels, root, augment=False):
        self.files=files; self.labels=labels; self.root=root; self.augment=augment
    def __len__(self): return len(self.files)
    def _time_shift(self,x):
        if torch.rand(1)<0.5:
            s=torch.randint(1,x.shape[2],(1,)).item(); x=torch.roll(x,s,2)
        return x
    def _noise(self,x,sigma=0.05):
        return x+torch.randn_like(x)*sigma if torch.rand(1)<0.5 else x
    def _channel_drop(self,x):
        if torch.rand(1)<0.3:
            ch=torch.randint(0,x.shape[0],(1,)).item(); x[ch]=0
        return x
    def __getitem__(self,idx):
        x=np.load(os.path.join(self.root,self.files[idx]))
        x=torch.tensor(x,dtype=torch.float32).permute(2,0,1)
        if self.augment:
            x=self._channel_drop(self._noise(self._time_shift(x)))
        return x,self.labels[idx]

def loader(split_name,batch=32,augment=False):
    subs=[f'sub-{s:03d}' for s in split[split_name]]
    subdf=df_labels[df_labels['original_rec'].isin(subs)]
    ds=CWT_Dataset(list(subdf['crop_file']),list(subdf['train_label']),cwt_path,augment)
    return DataLoader(ds,batch_size=batch,shuffle=True,num_workers=4,pin_memory=True)

# ======================================================
# 5. MobileNetV2 a 19 canali
# ======================================================
class MobileNet19(nn.Module):
    def __init__(self,drop=0.5):
        super().__init__()
        m=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.features[0][0]=nn.Conv2d(19,32,kernel_size=3,stride=2,padding=1,bias=False)
        in_f=m.classifier[1].in_features
        m.classifier=nn.Sequential(nn.Dropout(drop),nn.Linear(in_f,2))
        self.net=m
    def forward(self,x): return self.net(x)

# ======================================================
# 6. Early‑Stopping helper
# ======================================================
class EarlyStop:
    def __init__(self,patience=5):
        self.patience=patience; self.best=float('inf'); self.counter=0; self.chk=None
    def step(self,val_loss,model):
        if val_loss<self.best-1e-4:
            self.best=val_loss; self.counter=0
            self.chk={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else:
            self.counter+=1
        return self.counter>self.patience
    def load_best(self,model):
        model.load_state_dict(self.chk)

# ======================================================
# 7. Train & Evaluate utilities
# ======================================================

def evaluate(model,dl,loss_fn):
    model.eval(); tot=0; corr=0; lsum=0; preds=[]; labels=[]
    with torch.no_grad():
        for x,y in dl:
            x,y=x.to(device),y.to(device)
            out=model(x); loss=loss_fn(out,y)
            _,p=out.max(1); tot+=y.size(0); corr+=p.eq(y).sum().item()
            lsum+=loss.item(); preds+=p.cpu().tolist(); labels+=y.cpu().tolist()
    return lsum/len(dl),100.*corr/tot,labels,preds

# ======================================================
# 8. Funzione di training
# ======================================================

def train(model,train_dl,val_dl,loss_fn,opt,sched,epochs=50):
    hist={'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    es=EarlyStop(patience=5)
    for ep in range(epochs):
        model.train(); tot=0; corr=0; lsum=0
        for x,y in tqdm(train_dl,desc=f'Epoch {ep+1}'):
            x,y=x.to(device),y.to(device)
            opt.zero_grad(); out=model(x); loss=loss_fn(out,y); loss.backward(); opt.step()
            _,p=out.max(1); tot+=y.size(0); corr+=p.eq(y).sum().item(); lsum+=loss.item()
        tr_loss, tr_acc = lsum/len(train_dl),100.*corr/tot
        val_loss,val_acc,_,_=evaluate(model,val_dl,loss_fn)
        sched.step(val_loss)
        hist['train_loss'].append(tr_loss); hist['train_acc'].append(tr_acc)
        hist['val_loss'].append(val_loss);   hist['val_acc'].append(val_acc)
        print(f'Ep{ep+1}: TL={tr_loss:.3f} TA={tr_acc:.2f}% VL={val_loss:.3f} VA={val_acc:.2f}%')
        if es.step(val_loss,model):
            print('[EarlyStop] no improvement for',es.patience,'epochs'); break
    es.load_best(model)
    return hist

# ======================================================
# 9. Salvataggio risultati
# ======================================================

def save_outputs(hist,metrics,outdir='/home/alfio/improving_dementia_detection_model/results_mobilenetv2'):
    os.makedirs(outdir,exist_ok=True)
    # history plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(hist['train_loss'],'-o'); plt.plot(hist['val_loss'],'-o'); plt.legend(['Train','Val']); plt.title('Loss');
    plt.subplot(1,2,2); plt.plot(hist['train_acc'],'-o'); plt.plot(hist['val_acc'],'-o'); plt.legend(['Train','Val']); plt.title('Accuracy');
    plt.tight_layout(); plt.savefig(os.path.join(outdir,'training_history.png')); plt.close()
    # confusion matrix
    cm=metrics['conf_matrix']; sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Pred0','Pred1'],yticklabels=['True0','True1'])
    plt.title('Confusion Matrix'); plt.savefig(os.path.join(outdir,'confusion_matrix.png')); plt.close()
    pd.DataFrame(cm).to_csv(os.path.join(outdir,'confusion_matrix.csv'),index=False)
    # classification report
    with open(os.path.join(outdir,'classification_report.txt'),'w') as f:
        f.write(metrics['report'])
    # summary csv
    pd.DataFrame([{
        'accuracy':metrics['accuracy'],'precision':metrics['precision'],
        'recall':metrics['recall'],'f1':metrics['f1']
    }]).to_csv(os.path.join(outdir,'metrics_summary.csv'),index=False)

# ======================================================
# 10. Main eseguibile
# ======================================================
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataloader
    train_dl=loader('train',batch=32,augment=True)
    val_dl  =loader('val'  ,batch=32,augment=False)
    test_dl =loader('test' ,batch=32,augment=False)
    # class weights
    counts=np.bincount([y for _,y in train_dl.dataset])
    w=torch.tensor(1.0/counts,dtype=torch.float32)
    # model and optimizer
    model=MobileNet19(drop=0.5).to(device)
    loss_fn=nn.CrossEntropyLoss(weight=w.to(device),label_smoothing=0.1)
    opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=2)
    # train
    history=train(model,train_dl,val_dl,loss_fn,opt,sched,epochs=50)
    # test
    test_loss,test_acc,test_lbls,test_preds=evaluate(model,test_dl,loss_fn)
    metrics={
        'accuracy':test_acc,
        'precision':precision_score(test_lbls,test_preds),
        'recall':recall_score(test_lbls,test_preds),
        'f1':f1_score(test_lbls,test_preds),
        'conf_matrix':confusion_matrix(test_lbls,test_preds),
        'report':classification_report(test_lbls,test_preds,target_names=['Class0','Class1'])
    }
    print(f"Test Acc {test_acc:.2f}% | P {metrics['precision']:.2f} R {metrics['recall']:.2f} F1 {metrics['f1']:.2f}")
    # save
    save_outputs(history,metrics)
