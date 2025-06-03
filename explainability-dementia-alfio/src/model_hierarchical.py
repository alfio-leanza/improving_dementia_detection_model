# model_hierarchical.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_arcface import GNNCWT2D_Mk11_1sec_Arc           # ← tuo encoder
from pytorch_metric_learning.losses import ArcFaceLoss     # stesso loss che già usi


class HierarchicalEEGNet(nn.Module):
    """
    Head 1 : hc  vs  (ftd+ad)       [binaria]
    Head 2 : ftd vs  ad             [binaria – usata SOLO se head 1=‘demenza’]
    """
    def __init__(self,
                 n_electrodes: int = 19,
                 cwt_size: tuple    = (40, 500),
                 emb_dim: int       = 32,
                 lambda_dem: float  = 0.5,
                 margin: float      = 0.5,
                 scale:  float      = 40.0):
        super().__init__()
        self.lambda_dem = lambda_dem

        # ---------- encoder identico al tuo -------------------------------
        self.backbone = GNNCWT2D_Mk11_1sec_Arc(n_electrodes,
                                               cwt_size,
                                               emb_dim)

        # ---------- due ArcFaceLoss indipendenti --------------------------
        self.arc_bin = ArcFaceLoss(num_classes=2,
                                   embedding_size=emb_dim,
                                   margin=margin,
                                   scale=scale)

        self.arc_dem = ArcFaceLoss(num_classes=2,
                                   embedding_size=emb_dim,
                                   margin=margin,
                                   scale=scale)

    # ---------------------------------------------------------------------
    def forward(self, x, edge_index, batch,
                y_bin: torch.Tensor | None = None,
                y_dem: torch.Tensor | None = None):
        """
        Se y_* sono forniti ⇒ modalità TRAIN: ritorna anche la loss.
        Se None  ⇒ modalità EVAL: ritorna solo i logits per le due head.
        """
        embeds = self.backbone(x, edge_index, batch)     # L2-norm già fatta

        if self.training:
            assert y_bin is not None, "serve y_bin in training"
            loss_bin = self.arc_bin(embeds, y_bin)

            # loss_dem solo sui campioni demenza
            dem_mask = y_bin == 1
            if dem_mask.any():
                loss_dem = self.arc_dem(embeds[dem_mask],
                                        y_dem[dem_mask])
            else:
                loss_dem = torch.zeros(1, device=embeds.device)

            loss = loss_bin + self.lambda_dem * loss_dem
            return loss

        # ---------- modalità inference -----------------------------------
        logits_bin = self.arc_bin.get_logits(embeds)     # (B,2)
        logits_dem = self.arc_dem.get_logits(embeds)     # (B,2)
        return logits_bin, logits_dem                    # niente loss
