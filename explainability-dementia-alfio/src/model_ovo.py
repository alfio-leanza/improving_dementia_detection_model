"""
model_ovo.py
------------
Backbone GNNCWT2D_Mk11_1sec  +  One-vs-One multi-head (3 teste binarie):
    • head 0 : HC  vs  FTD   (ignora AD)
    • head 1 : HC  vs  AD    (ignora FTD)
    • head 2 : FTD vs  AD    (ignora HC)
forward(return_all=True)  ➜  (final_logits, [logit_hc_ftd, logit_hc_ad, logit_ftd_ad])
forward()                 ➜  final_logits shape (B,3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_max_pool


# ------------------------------------------------------------------ #
# 1)  Edge-weighted GraphConv                                        #
# ------------------------------------------------------------------ #
class _EdgeWeightsGraphConvLayer(nn.Module):
    def __init__(self, n_edges, in_ch, out_ch):
        super().__init__()
        self.conv = GraphConv(in_ch, out_ch, aggr="add")
        self.edge_w = nn.Parameter(torch.zeros(n_edges))
        self.n_electrodes = 19

    def forward(self, x, edge_index):
        w = F.softplus(self.edge_w)
        w = w.repeat(x.size(0) // self.n_electrodes)
        return self.conv(x, edge_index, w)


# ------------------------------------------------------------------ #
# 2)  Backbone (identico al precedente)                              #
# ------------------------------------------------------------------ #
class GNNCWT2D_Mk11_1sec(nn.Module):
    def __init__(self, n_electrodes=19, cwt_size=(40, 500), feat_dim=64):
        super().__init__()
        self.ne = n_electrodes
        self.nf, self.nt = cwt_size
        self.win = 25
        self.nw = self.nt // self.win

        self.lin2 = nn.Linear(self.nf * self.nw, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)

        self.bn2 = nn.BatchNorm1d(n_electrodes)
        self.bn3 = nn.BatchNorm1d(n_electrodes)
        self.bn4 = nn.BatchNorm1d(n_electrodes)
        self.drop = nn.Dropout(0.25)

        self.g1 = _EdgeWeightsGraphConvLayer(60, 128, 64)
        self.g2 = _EdgeWeightsGraphConvLayer(60, 64, 64)
        self.bn_g1 = nn.BatchNorm1d(64)
        self.bn_g2 = nn.BatchNorm1d(64)

        self.lin_feat = nn.Linear(64, feat_dim)

    def forward(self, x, edge_index, batch, *, return_features=False):
        B = torch.unique(batch).numel()
        x = x.view(B, self.ne, self.nf, self.nt)
        x = x.view(B, self.ne, self.nf, self.nw, self.win).mean(dim=-1)
        x = x.reshape(B, self.ne, -1)

        x = F.relu(self.lin2(x)); x = self.bn2(x); x = self.drop(x)
        x = F.relu(self.lin3(x)); x = self.bn3(x); x = self.drop(x)
        x = F.relu(self.lin4(x)); x = self.bn4(x); x = self.drop(x)

        x = x.view(B * self.ne, -1)
        x = F.relu(self.g1(x, edge_index)); x = self.bn_g1(x); x = self.drop(x)
        x = F.relu(self.g2(x, edge_index)); x = self.bn_g2(x); x = self.drop(x)

        x = global_max_pool(x, batch)
        feat = F.relu(self.lin_feat(x))
        return feat if return_features else feat


# ------------------------------------------------------------------ #
# 3)  One-vs-One GNN                                                 #
# ------------------------------------------------------------------ #
class OneVsOneGNN(nn.Module):
    """
    heads:
        0 → HC  vs FTD
        1 → HC  vs AD
        2 → FTD vs AD
    """
    def __init__(self, backbone, feat_dim=64):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(3)])
        for h in self.heads:
            nn.init.xavier_uniform_(h.weight); nn.init.zeros_(h.bias)

    # -------------------------------------------------------------- #
    def _aggregate_probs(self, p0, p1, p2):
        """
        Combina le tre probabilità binarie in prob. finali a 3 classi.
        p0: P(hc|hc-ftd)
        p1: P(hc|hc-ad)
        p2: P(ftd|ftd-ad)
        """
        p_hc  = (p0 + p1) / 2
        p_ftd = ((1 - p0) + p2) / 2
        p_ad  = ((1 - p1) + (1 - p2)) / 2
        probs = torch.stack((p_hc, p_ftd, p_ad), dim=1)
        return torch.log(probs + 1e-6)          # final logits

    # -------------------------------------------------------------- #
    def forward(self, x, edge_index, batch, *, return_all=False):
        feat = self.backbone(x, edge_index, batch, return_features=True)
        log_hc_ftd = self.heads[0](feat).squeeze(1)
        log_hc_ad  = self.heads[1](feat).squeeze(1)
        log_ftd_ad = self.heads[2](feat).squeeze(1)

        if return_all:
            return None, [log_hc_ftd, log_hc_ad, log_ftd_ad]  # solo teste

        p0 = torch.sigmoid(log_hc_ftd)
        p1 = torch.sigmoid(log_hc_ad)
        p2 = torch.sigmoid(log_ftd_ad)
        final_logits = self._aggregate_probs(p0, p1, p2)      # (B,3)
        return final_logits
