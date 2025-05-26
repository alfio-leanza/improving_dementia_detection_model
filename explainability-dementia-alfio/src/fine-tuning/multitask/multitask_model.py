"""
Multi-task head per la GNNCWT2D_Mk11_1sec.
Non richiede modifiche ai sorgenti originali: li importa e ri-usa “as-is”.
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
        Carica i pesi di un checkpoint .pt (contenente *model_state_dict*)
        della vecchia GNN a 3 classi e li trasferisce nel modello multi-task.
        """
        # 1) carica il file .pt: può essere già uno state-dict oppure un wrapper
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in ckpt:           # caso wrapper di single_fold.py
            ckpt = ckpt['model_state_dict']

        # 2) inizializza modello “vecchio” per leggere i pesi
        old = GNNCWT2D_Mk11_1sec(19, (40, 500), num_classes=3)
        # ignoriamo chiavi extra/mancanti (strict=False)
        old.load_state_dict(ckpt, strict=False)

        # 3) costruisce il nuovo modello multi-task
        new = cls().to(device)
        sd_new = new.state_dict()
        for k, v in old.state_dict().items():
            if k.startswith('lin6'):        # salta la testa a 3 classi
                continue
            if k in sd_new and v.shape == sd_new[k].shape:
                sd_new[k] = v
        # carichiamo nel nuovo modello (strict=False per eventuali teste)
        new.load_state_dict(sd_new, strict=False)
        return new
