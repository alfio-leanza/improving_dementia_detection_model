# improved_monitor_3dcnn.py
"""
Monitor CWT – Deep 3D‑CNN con Channel Attention
==============================================
• 3D‑CNN profondo con Channel‑SE attention, BalancedSampler, Weighted CE/Focal loss opzionale, LLRD, cosine scheduler
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax

# ==== ↳ A) DATA LOADING (identico ai vecchi script) ============================
root_activ = "/home/tom/dataset_eeg/inference_20250327_171717"
annot_csv  = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv"
cwt_root   = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"

# 1. carica activations
train_activ = np.load(os.path.join(root_activ, "train_activations.npy"), allow_pickle=True).item()
val_activ   = np.load(os.path.join(root_activ, "val_activations.npy"),   allow_pickle=True).item()
test_activ  = np.load(os.path.join(root_activ, "test_activations.npy"),  allow_pickle=True).item()

# 2. converte in DataFrame + softmax + pred_label

def activ_dict_to_df(dct, split):
    df = pd.DataFrame([{"crop_file":k, "valore":v} for k,v in dct.items()])
    df["dataset"] = split
    df["valore_softmax"] = df["valore"].apply(lambda x: softmax(x))
    df["pred_label"]    = df["valore_softmax"].apply(lambda x: np.argmax(x))
    return df

train_df = activ_dict_to_df(train_activ, "train")
val_df   = activ_dict_to_df(val_activ,   "val")
test_df  = activ_dict_to_df(test_activ,  "test")
all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

# 3. merge con annotazioni e calcola train_label (1 se corretta)
annot = pd.read_csv(annot_csv).rename(columns={"label":"true_label"})
true_pred = all_df.merge(annot, on="crop_file")
true_pred["train_label"] = (true_pred["pred_label"] == true_pred["true_label"]).astype(int)
true_pred["crop_file"] = true_pred["crop_file"].apply(os.path.basename)

# 4. split identico agli altri script
DATA_SPLIT = {
    "train": [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    "val"  : [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    "test" : [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}

# helper to select subset rows by subject id
true_pred["subject_id"] = true_pred["original_rec"].str.extract(r"sub-(\d+)").astype(int)

def subset_split(split):
    return true_pred[true_pred["subject_id"].isin(DATA_SPLIT[split])]

# ==== ↳ B) DATASET & DATALOADER ==============================================
class CWT3D_Dataset(Dataset):
    """Return (1,19,40,500) tensor """
    def __init__(self, file_list, labels, root_dir, augment=False):
        self.file_list, self.labels, self.root_dir, self.augment = file_list, labels, root_dir, augment
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root_dir, self.file_list[idx]))  # (40,500,19)
        if self.augment:
            data += np.random.normal(0, 0.01, data.shape)
            if np.random.rand() < 0.5:
                # freq masking
                f = np.random.randint(1, data.shape[0]//5)
                s = np.random.randint(0, data.shape[0]-f)
                data[s:s+f,:,:] = 0
            if np.random.rand() < 0.5:
                # time masking
                t = np.random.randint(1, data.shape[1]//5)
                s = np.random.randint(0, data.shape[1]-t)
                data[:,s:s+t,:] = 0
        tensor = torch.tensor(data, dtype=torch.float32).permute(2,0,1).unsqueeze(0)  # (1,19,40,500)
        return tensor, self.labels[idx]


def make_loader(split, batch=16, augment=False):
    sub = subset_split(split)
    files  = sub["crop_file"].tolist()
    labels = sub["train_label"].tolist()

    # Weighted sampler (class‑balance per batch)
    if split == "train":
        class_counts = np.bincount(labels)
        weights = 1./class_counts
        sample_weights = [weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(CWT3D_Dataset(files, labels, cwt_root, augment), batch_size=batch, sampler=sampler)
    else:
        return DataLoader(CWT3D_Dataset(files, labels, cwt_root, augment=False), batch_size=batch, shuffle=False)

# ==== ↳ C) MODEL ==============================================================
class CNN3D_ChannelAttention(nn.Module):
    def __init__(self, in_channels=1, depth=19, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 32),    # (B,32,19,40,500) -> pool
            self._block(32,64),              # (B,64,19,20,250)
            self._block(64,128),             # (B,128,19,10,125)
            self._block(128,256, pool=False) # (B,256,19,10,125)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(256,64,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64,256,bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(256, num_classes)
    def _block(self, in_c, out_c, pool=True):
        layers = [nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                  nn.BatchNorm3d(out_c), nn.ReLU(inplace=True)]
        if pool:
            layers += [nn.MaxPool3d((1,2,2))]  # solo tempo×freq
        layers += [nn.Dropout3d(0.2)]
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.features(x)
        w = self.ca(x).view(x.size(0),-1,1,1,1)
        x = (x*w).mean(dim=[2,3,4])  # GAP su D,H,W
        return self.classifier(x)

# ==== ↳ D) TRAIN / EVAL =======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.55,0.45]).to(DEVICE))
    # Layer‑wise LR decay
    params = []
    gamma = 0.8
    for i,(name,param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            layer_lr = lr*(gamma**i)
            params.append({"params": param, "lr": layer_lr})
    optimizer = optim.Adam(params, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_recall0 = 0
    history = {k:[] for k in ["train_loss","train_acc","val_loss","val_acc"]}
    for epoch in range(1,epochs+1):
        model.train(); running_loss=correct=total=0
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(); out=model(x)
            loss=criterion(out,y); loss.backward(); optimizer.step()
            running_loss+=loss.item(); pred=out.argmax(1); total+=y.size(0); correct+=(pred==y).sum().item()
        train_loss=running_loss/len(train_loader); train_acc=correct/total
        val_loss,val_acc,recall0=_eval(model,val_loader,criterion)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss);   history["val_acc"].append(val_acc)
        scheduler.step()
        print(f"Ep {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} recall0={recall0:.3f}")
        if recall0>best_recall0:
            best_recall0=recall0; torch.save(model.state_dict(),"best_3dcnn.pt")
    return history

def _eval(model, loader, criterion):
    model.eval(); loss=correct=total=0; all_pred=[]; all_y=[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            out=model(x); loss+=criterion(out,y).item(); p=out.argmax(1)
            total+=y.size(0); correct+=(p==y).sum().item(); all_pred.extend(p.cpu().numpy()); all_y.extend(y.cpu().numpy())
    cm=confusion_matrix(all_y,all_pred)
    recall0 = cm[0,0]/cm[0].sum() if cm[0].sum()>0 else 0
    return loss/len(loader), correct/total, recall0

# ==== ↳ E) TEST & SAVE (identico ai vecchi script) ============================

def test_and_save(model, test_loader, out_dir="/home/alfio/improving_dementia_detection_model/results_3dcnn"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval(); preds=[]; truths=[]
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(DEVICE); out=model(x)
            preds.extend(out.argmax(1).cpu().numpy()); truths.extend(y.numpy())
    # 1. CSV con predizioni
    pd.DataFrame({"True":truths,"Predicted":preds}).to_csv(os.path.join(out_dir,"test_predictions.csv"),index=False)
    # 2. Classification report
    with open(os.path.join(out_dir,"classification_report.txt"),"w") as f:
        f.write(classification_report(truths,preds))
    # 3. Confusion matrix heatmap
    cm = confusion_matrix(truths,preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad","Good"], yticklabels=["Bad","Good"])
    plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"confusion_matrix.png")); plt.close()
    print(f"[INFO] Results saved in {out_dir}")

# ==== ↳ F) MAIN ==============================================================

def main():
    train_loader = make_loader("train", batch=16, augment=True)
    val_loader   = make_loader("val",   batch=16)
    test_loader  = make_loader("test",  batch=16)
    model = CNN3D_ChannelAttention().to(DEVICE)
    history = train_model(model, train_loader, val_loader, epochs=25)
    # salva history plot come nei vecchi script
    def plot_history(history, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        epochs = range(1,len(history["train_loss"])+1)
        plt.figure(); plt.plot(epochs,history["train_loss"],label="Train Loss")
        plt.plot(epochs,history["val_loss"],label="Val Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend();
        plt.savefig(os.path.join(out_dir,"training_history_loss.png")); plt.close()
        plt.figure(); plt.plot(epochs,history["train_acc"],label="Train Acc")
        plt.plot(epochs,history["val_acc"],label="Val Acc"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend();
        plt.savefig(os.path.join(out_dir,"training_history_acc.png")); plt.close()
    plot_history(history, "/home/alfio/improving_dementia_detection_model/results_3dcnn")

    # carica migliori pesi e testa
    model.load_state_dict(torch.load("best_3dcnn.pt"))
    test_and_save(model, test_loader)

if __name__ == "__main__":
    main()
