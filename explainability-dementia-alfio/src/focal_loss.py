import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-label focal loss (funziona come BCE, ma con fattore (1-pt)^γ).
    Se `alpha` è:
        • None            ➜ nessun bilanciamento di classe
        • float           ➜ stesso α su tutte le classi
        • tensor shape(C) ➜ α specifico per classe (positivi)
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE “per-sample per-classe”
        #bce = F.binary_cross_entropy_with_logits(
        #          logits, targets, reduction='none')
        bce = torch.nn.BCEWithLogitsLoss(
                  logits, targets, reduction='none')
        pt  = torch.exp(-bce)              # prob. del target corretto
        focal = (1 - pt) ** self.gamma * bce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            focal = alpha_t * targets * focal + (1 - alpha_t) * (1 - targets) * focal

        return focal.mean() if self.reduction == "mean" else focal.sum()
