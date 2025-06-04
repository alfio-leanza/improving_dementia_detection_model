"""
Pre-training MSE del CAE (encoder+decoder) su TUTTI i crop CWT.
Salva l’encoder in encoder_cae.pth.
"""

import os, torch, pandas as pd
from torch.utils.data import DataLoader
from datasets import CWTGraphDataset
from model_cae import CAE
from tqdm import tqdm

# ─────────────── percorsi dataset ────────────────
DS_PARENT_DIR = "local/datasets"
DS_NAME       = "miltiadous_deriv_uV_d1.0s_o0.5s"
CLASSES       = "hc-ftd-ad"    # oppure "hc-ad"

CWT_DIR   = os.path.join(DS_PARENT_DIR, DS_NAME, "cwt")
ANNOT_CSV = os.path.join(DS_PARENT_DIR, DS_NAME,
                         f"annot_all_{CLASSES}.csv")

# ─────────────── iperparametri ───────────────────
device      = "cuda" if torch.cuda.is_available() else "cpu"
batch_size  = 32
epochs      = 30
lr          = 1e-3
emb_dim     = 128

# ─────────────── dataset e loader ────────────────
annot_df = pd.read_csv(ANNOT_CSV)
dataset  = CWTGraphDataset(annot_df, CWT_DIR, norm_stats_path=None)
loader   = DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=4)

# ─────────────── modello CAE ─────────────────────
model = CAE(emb_dim=emb_dim).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

for ep in range(epochs):
    model.train()
    epoch_loss = 0
    for data in tqdm(loader, desc=f"Pretrain epoch {ep}"):
        x = data.x.view(-1, 19, 40, 500).to(device)   # (B,19,40,500)

        optim.zero_grad()
        x_hat, _ = model(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    print(f"Epoch {ep:02d}  MSE: {epoch_loss/len(loader):.4f}")

# ─────────────── salva encoder ───────────────────
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.encoder.state_dict(), "checkpoints/encoder_cae.pth")
print("Encoder salvato in checkpoints/encoder_cae.pth")
