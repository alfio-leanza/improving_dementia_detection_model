#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grid-search CNN-ChannelAttention su CWT EEG.

• 64 combinazioni di hyper-parametri
• early-stopping (pazienza 10 epoche)
• salvataggio dei 3 migliori modelli
• per ciascun best: CSV di predizioni, classification report, confusion matrix
"""

# ======================================================================
# 1) IMPORT
# ======================================================================
import os, itertools, time, copy, joblib, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax as sf
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

# ------------------- cartelle output -------------------
OUT_DIR  = "/home/alfio/improving_dementia_detection_model/results_grid"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ======================================================================
# 2) MODELLO
# ======================================================================
class CNN_ChannelAttention(nn.Module):
    def __init__(self,
                 channels_block=(32,64,128),
                 act_fn=nn.ReLU,
                 dropout_block=0.1,
                 dropout_head=0.4,
                 num_channels=19,
                 num_classes=2):
        super().__init__()
        ch1,ch2,ch3 = channels_block
        act = act_fn(inplace=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels,ch1,3,padding=1,bias=False),
            nn.BatchNorm2d(ch1), act, nn.Dropout2d(dropout_block), nn.MaxPool2d(2),
            nn.Conv2d(ch1,ch2,3,padding=1,bias=False),
            nn.BatchNorm2d(ch2), copy.deepcopy(act), nn.Dropout2d(dropout_block), nn.MaxPool2d(2),
            nn.Conv2d(ch2,ch3,3,padding=1,bias=False),
            nn.BatchNorm2d(ch3), copy.deepcopy(act), nn.Dropout2d(dropout_block),
            nn.AdaptiveAvgPool2d(1),
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(ch3,ch3//4,bias=False), copy.deepcopy(act),
            nn.Linear(ch3//4,ch3,bias=False), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout_head),
                                        nn.Linear(ch3,num_classes))
    def forward(self,x):
        x = self.conv_block(x).flatten(1)
        x = x * self.channel_attention(x)
        return self.classifier(x)

# ======================================================================
# 3) SPAZIO IPER-PARAMETRI
# ======================================================================
grid = {
    "channels_block":[(16,32,64),(32,64,128)],
    "act_fn":[nn.ReLU, nn.SELU],
    "dropout_block":[0.2,0.3],
    "dropout_head":[0.5,0.6],
    "weight_decay":[1e-2,2e-2],
    "lr":[1e-3,2e-3],
}
keys, values = zip(*grid.items())
configs = [dict(zip(keys,v)) for v in itertools.product(*values)]
print("Totale combinazioni:", len(configs))           # 64

# ======================================================================
# 4) DATA (stesso schema usato finora)
# ======================================================================
root_dir_cwt = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
annot_csv    = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv"
inference_dir= "/home/tom/dataset_eeg/inference_20250327_171717"

def build_df(d, split):
    df = pd.DataFrame({"crop_file":list(d.keys()),"valore":list(d.values())})
    df["dataset"]=split
    df["valore_softmax"]=df["valore"].apply(sf)
    df["pred_label"]=df["valore_softmax"].apply(lambda x:int(np.argmax(x)))
    return df

train_act = np.load(os.path.join(inference_dir,"train_activations.npy"),allow_pickle=True).item()
val_act   = np.load(os.path.join(inference_dir,"val_activations.npy"),  allow_pickle=True).item()
test_act  = np.load(os.path.join(inference_dir,"test_activations.npy"), allow_pickle=True).item()

train_df,val_df,test_df = build_df(train_act,"train"),build_df(val_act,"val"),build_df(test_act,"test")
all_df = pd.concat([train_df,val_df,test_df])
annot  = pd.read_csv(annot_csv).rename(columns={"label":"true_label"})
true_pred = all_df.merge(annot,on="crop_file")
true_pred["train_label"]=(true_pred["pred_label"]==true_pred["true_label"]).astype(int)

splits = {
    "train":[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,
             70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
             17,18,19,20,21],
    "val":[54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    "test":[60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36],
}
def subset_df(split):
    return true_pred[true_pred["original_rec"].isin([f"sub-{s:03d}"for s in splits[split]])]

class CWT_Dataset(Dataset):
    def __init__(self, df, augment=False):
        self.df=df.reset_index(drop=True); self.augment=augment
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.loc[idx]
        data=np.load(os.path.join(root_dir_cwt,row.crop_file)).astype(np.float32)
        if self.augment: data+=np.random.normal(0,0.01,data.shape)
        tensor=torch.tensor(data).permute(2,0,1)      # (19,40,500)
        return tensor, torch.tensor(row.train_label), row.crop_file

def make_loader(split,batch=64,augment=False,shuffle=True):
    df = subset_df(split)
    return DataLoader(CWT_Dataset(df,augment),
                      batch_size=batch,shuffle=shuffle,
                      num_workers=4,pin_memory=True)

train_loader = make_loader("train",augment=True, shuffle=False)
val_loader   = make_loader("val",shuffle=False)
test_loader  = make_loader("test",shuffle=False)

# ======================================================================
# 5) TRAIN + EARLY-STOP
# ======================================================================
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
NUM_EPOCHS=50
PATIENCE  =10

def train_one(run_id,cfg):
    model=CNN_ChannelAttention(
        channels_block=cfg["channels_block"],
        act_fn=cfg["act_fn"],
        dropout_block=cfg["dropout_block"],
        dropout_head=cfg["dropout_head"]).to(device)

    optim=optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])

    best_val=0.0; patience=0; best_state=None
    for ep in range(1,NUM_EPOCHS+1):
        model.train(); corr=tot=tloss=0.0
        for x,y,_ in train_loader:
            x,y=x.to(device),y.to(device)
            optim.zero_grad(); out=model(x); loss=criterion(out,y)
            loss.backward(); optim.step()
            tloss+=loss.item(); tot+=y.size(0); corr+=out.argmax(1).eq(y).sum().item()
        train_acc=corr/tot

        model.eval(); vcorr=vtot=0
        with torch.no_grad():
            for x,y,_ in val_loader:
                x,y=x.to(device),y.to(device)
                vcorr+=model(x).argmax(1).eq(y).sum().item(); vtot+=y.size(0)
        val_acc=vcorr/vtot
        print(f"[{run_id}] Ep{ep:02d} train={train_acc:.3f} val={val_acc:.3f} "
              f"loss={tloss/len(train_loader):.3f}")

        if val_acc>best_val:
            best_val,val_epoch=val_acc,ep
            patience=0
            best_state=copy.deepcopy(model.state_dict())
        else:
            patience+=1
            if patience>=PATIENCE: break
    return best_val,best_state,cfg

# ======================================================================
# 6) GRID LOOP
# ======================================================================
results=[]
for idx,cfg in enumerate(configs,1):
    run_id=f"run{idx:02d}"
    print("\n"+"="*25,f"{run_id}/{len(configs)}","="*25)
    print(cfg)
    best_val,state,cfg_out=train_one(run_id,cfg)
    results.append((best_val,state,cfg_out))

# ======================================================================
# 7) SELEZIONE TOP-3 & VALUTAZIONE
# ======================================================================
results.sort(key=lambda x:x[0],reverse=True)
top3=results[:3]

def predict_and_save(model,loader,split,out_sub):
    rows=[]
    model.eval()
    with torch.no_grad():
        for x,y,fns in loader:
            logits=model(x.to(device)).cpu()
            sm=torch.softmax(logits,dim=1).numpy()
            preds=sm.argmax(1)
            goodness=sm[np.arange(len(sm)),preds]
            for f,t,p,l,s,g in zip(fns,y.numpy(),preds,logits.numpy(),sm,goodness):
                rows.append([f,int(t),int(p),l.tolist(),s.tolist(),float(g)])
    cols=["crop_file","true_label","pred_label","logits","softmax","goodness"]
    pd.DataFrame(rows,columns=cols).to_csv(os.path.join(out_sub,f"{split}_predictions_detailed.csv"),index=False)

def evaluate_and_save(rank,val_acc,state,cfg):
    subdir=os.path.join(OUT_DIR,f"best_{rank}")
    os.makedirs(subdir,exist_ok=True)
    torch.save(state,os.path.join(subdir,"model.pth"))
    joblib.dump(cfg,os.path.join(subdir,"config.pkl"))

    model=CNN_ChannelAttention().to(device)
    model.load_state_dict(state)

    for sp,ldr in [("train",train_loader),("val",val_loader),("test",test_loader)]:
        predict_and_save(model,ldr,sp,subdir)

    # classification report su test
    yt,yp=[],[]
    model.eval()
    with torch.no_grad():
        for x,y,_ in test_loader:
            yt.extend(y.numpy())
            yp.extend(model(x.to(device)).argmax(1).cpu().numpy())
    rep=classification_report(yt,yp,target_names=["Bad","Good"])
    with open(os.path.join(subdir,"classification_report.txt"),"w") as f: f.write(rep)

    cm=confusion_matrix(yt,yp)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["Bad","Good"],yticklabels=["Bad","Good"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(subdir,"confusion_matrix.png"))
    plt.close()

print("\n============= TOP-3 =============")
for rank,(val_acc,state,cfg) in enumerate(top3,1):
    evaluate_and_save(rank,val_acc,state,cfg)
    print(f"{rank}) val_acc={val_acc:.4f}  -> risultati salvati in best_{rank}/")

print("\nGrid completata.")