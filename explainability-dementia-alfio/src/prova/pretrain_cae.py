"""
Pre-training MSE del CAE (encoder+decoder) su TUTTI i crop CWT.
Salva l’encoder in encoder_cae.pth.
"""

import os, torch, pandas as pd
from torch.utils.data import DataLoader
from dataset_raw import CWTRawDataset 
from model_cae import CAE
from tqdm import tqdm

# ─────────────── percorsi dataset ────────────────
DS_PARENT_DIR = "/home/tom/dataset_eeg"
DS_NAME       = "miltiadous_deriv_uV_d1.0s_o0.5s"
CLASSES       = "hc-ftd-ad"    # oppure "hc-ad"

CWT_DIR   = '/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt'
ANNOT_CSV = '/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv'

# ─────────────── iperparametri ───────────────────
device      = "cuda" if torch.cuda.is_available() else "cpu"
batch_size  = 32
epochs      = 30
lr          = 1e-3
emb_dim     = 128

# ─────────────── dataset e loader ────────────────
annot_df = pd.read_csv(ANNOT_CSV)
dataset  = CWTRawDataset(annot_df, CWT_DIR)
loader   = DataLoader(dataset, batch_size=batch_size,
                       shuffle=True, num_workers=4,
                       pin_memory=False)

# ─────────────── modello CAE ─────────────────────
model = CAE(emb_dim=emb_dim).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

for ep in range(epochs):
    model.train()
    epoch_loss = 0

    with tqdm(loader,
              total=len(loader),
              desc=f"[CAE] Epoch {ep+1}/{epochs}",
              unit="batch",
              dynamic_ncols=True) as pbar:
        for data in pbar:
            x = data.to(device)

            optim.zero_grad()
            x_hat, _ = model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"mse": f"{loss.item():.4f}"})

    print(f"Epoch {ep+1:02d}/{epochs}  avg-MSE: {epoch_loss/len(loader):.4f}")

# ─────────────── salva encoder ───────────────────
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.encoder.state_dict(), "/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/src/prova/checkpoints/encoder_cae.pth")
print("Encoder salvato in checkpoints/encoder_cae.pth")
