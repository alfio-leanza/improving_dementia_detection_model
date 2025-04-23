# monitor_cnn2d_residual.py
"""
CNN 2D con Residual‑SE, Focal Loss, BalancedSampler
Log: train_loss, train_acc, val_loss, val_acc, recall0.
Fix: augment ‑ time‑warp corretto, rimosso doppio return, import ndimage.
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.metrics import recall_score, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from scipy import ndimage  # <-- per zoom
from scipy.special import softmax
import matplotlib.pyplot as plt, seaborn as sns
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import torch.nn.functional as F

# === Load activations and labels ===
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
 'val':[54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
 'test':[60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]}

# === Dataset ===
class CWT_Dataset(Dataset):
    def __init__(self, files, labels, root, augment=False):
        self.files, self.labels, self.root, self.aug = files, labels, root, augment
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        x = np.load(os.path.join(self.root, self.files[idx]))  # (40,500,19)
        if self.aug:
            if np.random.rand()<0.5:  # noise
                x += np.random.normal(0,0.01,x.shape)
            # time‑warp (zoom along time axis)
            if np.random.rand()<0.3:
                rate = np.random.uniform(0.9,1.1)
                zoomed = ndimage.zoom(x, (1, rate, 1), order=1)
                if zoomed.shape[1] > 500:
                    x = zoomed[:, :500, :]
                else:
                    pad = 500 - zoomed.shape[1]
                    x = np.pad(zoomed, ((0,0),(0,pad),(0,0)), mode='wrap')
            # freq‑shift
            if np.random.rand()<0.3:
                s = np.random.randint(-3,4); x = np.roll(x,s,axis=0)
        x = np.transpose(x,(2,0,1)).astype('float32')  # (19,40,500)
        return torch.tensor(x), torch.tensor(self.labels[idx])

# === Loader helper ===

def make_loader(split,batch,aug=False):
    recs = [f'sub-{s:03d}' for s in splits[split]]
    subset = true_pred[true_pred['original_rec'].isin(recs)]
    files = subset['crop_file'].tolist(); labels = subset['train_label'].tolist()
    return DataLoader(CWT_Dataset(files,labels,root_dir,aug), batch_size=batch, shuffle=aug, num_workers=4, drop_last=aug)

train_loader = make_loader('train',64,aug=True); val_loader = make_loader('val',64); test_loader = make_loader('test',64)

# === Model ===
class ResBlock(nn.Module):
    def __init__(self,in_c,out_c,dw=False):
        super().__init__(); g = in_c if dw else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,1,1,groups=g,bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c,out_c,3,1,1,groups=g,bias=False), nn.BatchNorm2d(out_c))
        self.short = nn.Conv2d(in_c,out_c,1,bias=False) if in_c!=out_c else nn.Identity()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(out_c,out_c//8,bias=False), nn.ReLU(), nn.Linear(out_c//8,out_c,bias=False), nn.Sigmoid())
    def forward(self,x):
        y = self.conv(x)+self.short(x)
        w = self.se(y).view(y.size(0),y.size(1),1,1)
        return torch.relu_(y*w)

class CNN2D(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.stem = nn.Conv2d(19,32,3,1,1,bias=False)
        self.b1 = ResBlock(32,32); self.p1 = nn.MaxPool2d(2)
        self.b2 = ResBlock(32,64); self.p2 = nn.MaxPool2d(2)
        self.b3 = ResBlock(64,128,dw=True); self.p3 = nn.MaxPool2d(2)
        self.b4 = ResBlock(128,256,dw=True)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256,num_classes))
    def forward(self,x):
        x = self.p1(self.b1(self.stem(x)))
        x = self.p2(self.b2(x))
        x = self.p3(self.b3(x))
        x = self.b4(x)
        return self.head(x)

# === Loss, optimizer, scheduler ===
class FocalLoss(nn.Module):
    def __init__(self,alpha,gamma=2):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.ce=nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, targets):
        alpha_t = self.alpha.to(logits.device)[targets]      # alpha sullo stesso device
        ce = F.cross_entropy(logits, targets, reduction='none')  # niente weight qui
        pt = torch.exp(-ce)
        loss = alpha_t * ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


alpha=torch.tensor([0.56,0.44])
criterion = FocalLoss(alpha,2.0)
model = CNN2D().cuda(); epochs=60
optimizer = optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)
swa_model = AveragedModel(model); swa_start=int(0.8*epochs); swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=10, anneal_strategy='cos')

# === Train / Eval helpers ===
@torch.no_grad()
def evaluate(net,loader,device):
    net.eval(); y_true,y_pred = [],[]; running=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = net(x); loss = criterion(logits,y); running+=loss.item()
        y_true.extend(y.cpu().numpy()); y_pred.extend(logits.argmax(1).cpu().numpy())
    return running/len(loader), y_true, y_pred

best_recall0, patience, wait = 0.0, 10, 0
for epoch in range(1,epochs+1):
    model.train(); running_loss, correct, total = 0.0,0,0
    for x,y in tqdm(train_loader,desc=f"Epoch {epoch}"):
        x,y = x.cuda(),y.cuda(); optimizer.zero_grad(); logits = model(x); loss = criterion(logits,y); loss.backward(); optimizer.step()
        running_loss += loss.item(); preds = logits.argmax(1); correct += preds.eq(y).sum().item(); total += y.size(0)
    train_loss = running_loss/len(train_loader); train_acc = correct/total
    val_loss, y_true, y_pred = evaluate(model,val_loader,'cuda'); val_acc = accuracy_score(y_true,y_pred); recall0 = recall_score(y_true,y_pred,pos_label=0)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} recall0={recall0:.4f}")

    if recall0>best_recall0: best_recall0, wait = recall0, 0; torch.save(model.state_dict(),'best.pth')
    else: wait +=1
    if wait>=patience: print("Early stop"); break

    scheduler.step();
    if epoch>=swa_start: swa_model.update_parameters(model); swa_scheduler.step()

# === SWA BN update & Test ===
update_bn(train_loader,swa_model,device='cuda'); model=swa_model
_, y_true, y_pred = evaluate(model,test_loader,'cuda')
cm = confusion_matrix(y_true,y_pred); report = classification_report(y_true,y_pred,target_names=['Bad','Good'])
results_dir = '/home/alfio/improving_dementia_detection_model/results_cnn2d_residual'; os.makedirs(results_dir,exist_ok=True)
np.savetxt(os.path.join(results_dir,'test_predictions.csv'), np.vstack([y_true,y_pred]).T, delimiter=',', fmt='%d', header='True,Pred', comments='')
with open(os.path.join(results_dir,'classification_report.txt'),'w') as f: f.write(report)
plt.figure(figsize=(6,6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Bad','Good'],yticklabels=['Bad','Good']); plt.savefig(os.path.join(results_dir,'confusion_matrix.png')); plt.close()
