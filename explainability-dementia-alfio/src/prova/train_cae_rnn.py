"""
Addestra la rete CAE-RNN usando l’encoder pre-trainato.
— Sequenza di 8 crop (≈8 s) → GRU → 3 classi.
"""

import os, torch, pandas as pd
from torch.utils.data import DataLoader
from dataset_seq import CWTSequenceDataset
from model_cae_rnn import CAE_RNN
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# ─────────────── percorsi dataset ────────────────
DS_PARENT_DIR = "/home/tom/dataset_eeg"
DS_NAME       = "miltiadous_deriv_uV_d1.0s_o0.5s"
CLASSES       = "hc-ftd-ad"

CWT_DIR   = '/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt'
ANNOT_CSV = '/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv'

# ---------- soggetti split identici a prima ----------
train_subjects = [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                  66,67,68,69,70,71,72,73,74,75,76,77,78,
                  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
val_subjects   = [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28]
test_subjects  = [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]

train_subjects = [f"sub-{s:03d}" for s in train_subjects]
val_subjects   = [f"sub-{s:03d}" for s in val_subjects]

annot_df = pd.read_csv(ANNOT_CSV)
train_df = annot_df[annot_df['original_rec'].isin(train_subjects)]
val_df   = annot_df[annot_df['original_rec'].isin(val_subjects)]

# ───────────── loader sequenziali ────────────────
seq_len     = 8
batch_size  = 64
num_workers = 8

train_ds = CWTSequenceDataset(train_df, CWT_DIR, seq_len)
val_ds   = CWTSequenceDataset(val_df,   CWT_DIR, seq_len)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)

# ───────────── modello + encoder pretrain ────────
device   = "cuda" if torch.cuda.is_available() else "cpu"
model    = CAE_RNN(emb_dim=128, freeze_encoder=True).to(device)
model.encoder.load_state_dict(torch.load("/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/src/prova/checkpoints/encoder_cae.pth",
                                         map_location=device))

criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0,1.2]).to(device))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()), lr=3e-4, weight_decay= 1e-4)

# ───────────── training loop semplice ────────────
epochs = 60
for ep in range(epochs):
    model.train()
    y_true_tr, y_pred_tr = [], []

    with tqdm(train_loader,
              total=len(train_loader),
              desc=f"[Train] Ep {ep+1}/{epochs}",
              unit="batch",
              dynamic_ncols=True) as pbar:
        for x_seq, y in pbar:
            x_seq, y = x_seq.to(device), y.to(device)

            logits = model(x_seq)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(1)
            y_true_tr.extend(y.cpu().tolist())
            y_pred_tr.extend(pred.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # ── validation ──
    model.eval()
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for x_seq, y in tqdm(val_loader,
                             desc=f"[Val ] Ep {ep+1}/{epochs}",
                             unit="batch",
                             leave=False,
                             dynamic_ncols=True):
            x_seq, y = x_seq.to(device), y.to(device)
            logits = model(x_seq)

            y_true_val.extend(y.cpu().tolist())
            y_pred_val.extend(logits.argmax(1).cpu().tolist())

    tr_acc  = accuracy_score(y_true_tr,  y_pred_tr)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    print(f"Epoch {ep+1:02d}/{epochs}  train-acc {tr_acc:.3f}  val-acc {val_acc:.3f}")
    if ep % 10 == 0:
        print(classification_report(y_true_val, y_pred_val,
                                    target_names=['hc','ftd','ad']))
