"""
Multi-task head per la GNNCWT2D_Mk11_1sec.
Non richiede modifiche ai sorgenti originali: li importa e li
ri-usa “as-is”.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GNNCWT2D_Mk11_1sec   # ← file originale, invariato


class MultiTaskGNNCWT2D_Mk11_1sec(nn.Module):
    """
    Output:
        logits_main : [B, 3]  (HC-FTD-AD)
        logits_aux  : [B, 2]  (Good / Bad)
    """
    def __init__(
        self,
        n_electrodes: int = 19,
        cwt_size: tuple = (40, 500),
        feat_dim: int = 32
    ):
        super().__init__()
        # 1) backbone ← GNN originale, ma la facciamo terminare a 32 feat
        self.backbone = GNNCWT2D_Mk11_1sec(
            n_electrodes, cwt_size, num_classes=feat_dim
        )
        # disattivo la vecchia testa
        self.backbone.lin6 = nn.Identity()

        # 2) nuove teste supervisionate
        self.main_head = nn.Linear(feat_dim, 3)   # HC / FTD / AD
        self.aux_head  = nn.Linear(feat_dim, 2)   # Good / Bad

    # ----------------------------------------------------------
    def forward(self, x, edge_index, batch):
        feats = self.backbone(x, edge_index, batch)   # [B, 32]
        feats = F.relu(feats)
        return self.main_head(feats), self.aux_head(feats)

    # ----------------------------------------------------------
    @classmethod
    def from_pretrained(cls, ckpt_path: str, device: str = "cpu"):
        """
        Carica i pesi di un checkpoint .pt salvato da single_fold.py
        (la vecchia GNN a 3 classi) dentro il nuovo modello multi-task.
        """
        # 1) modello “vecchio” per leggere lo state-dict
        old = GNNCWT2D_Mk11_1sec(19, (40, 500), num_classes=3)
        old.load_state_dict(torch.load(ckpt_path, map_location=device))

        # 2) modello nuovo
        new = cls().to(device)
        sd_new = new.state_dict()
        for k, v in old.state_dict().items():
            if k.startswith('lin6'):       # saltiamo la head a 3 classi
                continue
            if k in sd_new and v.shape == sd_new[k].shape:
                sd_new[k] = v
        new.load_state_dict(sd_new)
        return new
