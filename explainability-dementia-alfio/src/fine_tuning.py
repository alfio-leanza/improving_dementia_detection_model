import os, copy, torch, pandas as pd, numpy as np
import torch.nn as nn, torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

# ---------------- percorsi ----------------
DATASET_ROOT   = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s"
CROP_DIR       = os.path.join(DATASET_ROOT, "cwt")      # cartella .npy
GRAPH_DIR      = "local/graphs"                         # grafi .pt usati dalla GNN
MONITOR_CSV    = "/home/alfio/improving_dementia_detection_model/results_cnn/train_predictions_detailed.csv"
PRETRAINED_GNN = "/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt"
OUT_DIR        = "/home/alfio/improving_dementia_detection_model/finetune_gnn_monitor"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- importa tuoi moduli -------------
from datasets import CWTGraphDataset          # già definito
from models   import GNNCWT2D_Mk11_1sec       # già definito

# ------------- carica annot completi -----------
annot_path = os.path.join(DATASET_ROOT, "annot_all_hc-ftd-ad.csv")
annot      = pd.read_csv(annot_path)

# split soggetti (come single_fold)
train_subj = [f"sub-{s:03d}" for s in
              [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
               66,67,68,69,70,71,72,73,74,75,76,77,78,
               1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
val_subj   = [f"sub-{s:03d}" for s in
              [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28]]
test_subj  = [f"sub-{s:03d}" for s in
              [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]]

train_df = annot[annot["original_rec"].isin(train_subj)]
val_df   = annot[annot["original_rec"].isin(val_subj)]
test_df  = annot[annot["original_rec"].isin(test_subj)]

# ---------- unione goodness e peso ------------
# ---------- 1) Unione goodness SOLO su train ----------
goodness_df = pd.read_csv(MONITOR_CSV)[["crop_file","goodness"]]

# train
train_df = train_df.merge(goodness_df, on="crop_file", how="left")
train_df["sample_weight"] = 1.0 - train_df["goodness"]

# validation & test
val_df["sample_weight"]   = 1.0
test_df["sample_weight"]  = 1.0

# ---------- Dataset adattato per grafi ----------
class GraphDatasetWithWeight(CWTGraphDataset):
    def __init__(self, df, crop_path):
        super().__init__(df, crop_path, None)   # None = scaler interno
    def get(self, idx):
        data = super().get(idx)
        data.w = torch.tensor([self.annot_df.iloc[idx]["sample_weight"]],
                              dtype=torch.float32)
        return data

train_set = GraphDatasetWithWeight(train_df, CROP_DIR)
val_set   = GraphDatasetWithWeight(val_df,   CROP_DIR)
test_set  = GraphDatasetWithWeight(test_df,  CROP_DIR)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=4)

# ---------- modello ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GNNCWT2D_Mk11_1sec(19, (40,500), num_classes=3).to(device)
state  = torch.load(PRETRAINED_GNN, map_location=device)
model.load_state_dict(state["model_state_dict"]
                      if "model_state_dict" in state else state)
print("Pesi pre-addestrati caricati.")

# ---------- training setup ----------
crit = nn.CrossEntropyLoss(reduction="none")
opt  = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched= optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
EPOCHS, PATIENCE = 10, 5
best_val, patience, best_state = 0., 0, None

for ep in range(1, EPOCHS+1):
    # ---- train ----
    model.train(); corr=tot=tl=0.
    for data in tqdm(train_loader, desc=f"Ep{ep:02d} - train"):
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = (crit(out, data.y) * data.w.to(device)).mean()
        loss.backward(); opt.step()
        tl += loss.item(); tot += data.y.size(0)
        corr += (out.argmax(1)==data.y).sum().item()
    train_acc = corr / tot

    # ---- val ----
    model.eval(); vcorr=vtot=0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            vcorr += (out.argmax(1)==data.y).sum().item()
            vtot  += data.y.size(0)
    val_acc = vcorr / vtot
    print(f"Ep{ep:02d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    sched.step()

    if val_acc > best_val:
        best_val, patience = val_acc, 0
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, os.path.join(OUT_DIR,"best_gnn.pth"))
        print(f"[✓] Nuovo best salvato (val_acc={best_val:.4f})")
    else:
        patience += 1
        if patience >= PATIENCE:
            print("Early-stopping.")
            break

# ---------- valutazione finale ----------
model.load_state_dict(best_state); model.eval()

def evaluate(loader, split):
    rows=[]; corr=tot=0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch).cpu()
            prob   = torch.softmax(logits,1)
            pred   = prob.argmax(1)
            corr  += (pred==data.y.cpu()).sum().item(); tot+=data.y.size(0)
            rows.extend([ [f,p.item(),t.item(),l.tolist(),prob[i].tolist()]
                          for f,p,t,l,i in zip(
                              data.crop_file,
                              pred, data.y.cpu(), logits, range(len(logits)) )])
    pd.DataFrame(rows,columns=["crop_file","pred_label","true_label",
                               "logits","softmax"]) \
      .to_csv(os.path.join(OUT_DIR,f"{split}_predictions.csv"),index=False)
    return corr/tot

acc_train = evaluate(train_loader,"train")
acc_val   = evaluate(val_loader,"val")
acc_test  = evaluate(test_loader,"test")
print(f"ACC train={acc_train:.3f}  val={acc_val:.3f}  test={acc_test:.3f}")

# report & confusion su test
y_true,y_pred=[],[]
with torch.no_grad():
    for d in test_loader:
        d=d.to(device); y_true.extend(d.y.cpu())
        y_pred.extend(model(d.x,d.edge_index,d.batch).argmax(1).cpu())
report = classification_report(y_true,y_pred,target_names=["hc","ftd","ad"])
with open(os.path.join(OUT_DIR,"classification_report.txt"),"w") as f:
    f.write(report)
cm = confusion_matrix(y_true,y_pred)
plt.figure(figsize=(6,6)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["hc","ftd","ad"], yticklabels=["hc","ftd","ad"])
plt.savefig(os.path.join(OUT_DIR,"confusion_matrix.png")); plt.close()
print("[INFO] Fine-tuning completato – artefatti salvati in", OUT_DIR)