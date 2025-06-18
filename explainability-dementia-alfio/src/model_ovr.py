"""
model_ovr.py
------------
Backbone GNNCWT2D_Mk11_1sec  +  One-Vs-Rest multi-head (tre uscite binarie).
• Ogni testa = Linear(feat_dim, 1) → logit per “classe vs resto”.
• forward() restituisce tensor (B,3) di logit; applicare sigmoid fuori.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_max_pool, GATConv


# ------------------------------------------------------------------ #
# 1) Edge-weighted GraphConv (softplus > 0)
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
# 2) Backbone GNNCWT2D_Mk11_1sec
# ------------------------------------------------------------------ #
class GNNCWT2D_Mk11_1sec(nn.Module):
    def __init__(self, n_electrodes=19, cwt_size=(40, 500), feat_dim=64):
        super().__init__()
        self.ne = n_electrodes
        self.nf, self.nt = cwt_size
        self.win = 25
        self.nw = self.nt // self.win                                     # 20

        self.lin2 = nn.Linear(self.nf * self.nw, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)

        self.bn2 = nn.BatchNorm1d(n_electrodes)
        self.bn3 = nn.BatchNorm1d(n_electrodes)
        self.bn4 = nn.BatchNorm1d(n_electrodes)
        self.drop = nn.Dropout(0.25)

        self.g1 = _EdgeWeightsGraphConvLayer(60, 128, 64)
        #self.g2 = _EdgeWeightsGraphConvLayer(60, 64, 64)
        self.g2 = GATConv(64, 64, heads=4, concat=False, dropout=0.2)  ### GAT MOD ###
        self.bn_g1 = nn.BatchNorm1d(64)
        self.bn_g2 = nn.BatchNorm1d(64)

        self.lin_feat = nn.Linear(32, feat_dim)

    def forward(self, x, edge_index, batch, *, return_features=False):
        B = torch.unique(batch).numel()
        x = x.view(B, self.ne, self.nf, self.nt)
        x = x.view(B, self.ne, self.nf, self.nw, self.win).mean(dim=-1)    # (B,19,40,20)
        x = x.reshape(B, self.ne, -1)                                      # (B,19,800)

        x = F.relu(self.lin2(x)); x = self.bn2(x); x = self.drop(x)
        x = F.relu(self.lin3(x)); x = self.bn3(x); x = self.drop(x)
        x = F.relu(self.lin4(x)); x = self.bn4(x); x = self.drop(x)

        x = x.view(B * self.ne, -1)
        x = F.relu(self.g1(x, edge_index)); x = self.bn_g1(x); x = self.drop(x)
        x = F.relu(self.g2(x, edge_index)); x = self.bn_g2(x); x = self.drop(x)

        x = global_max_pool(x, batch)
        feat = F.relu(self.lin_feat(x))                                     # (B,feat_dim)
        return feat if return_features else feat


# ------------------------------------------------------------------ #
# 3) One-vs-Rest multi-head
# ------------------------------------------------------------------ #
class OneVsRestGNN(nn.Module):
    """
    Tre teste binarie → logit (B,3)
    Durante l’inference: sigmoid → prob, argmax = classe predetta.
    """
    def __init__(self, backbone, feat_dim=64):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(3)])
        for h in self.heads:
            nn.init.xavier_uniform_(h.weight); nn.init.zeros_(h.bias)

    def forward(self, x, edge_index, batch):
        feat = self.backbone(x, edge_index, batch, return_features=True)    # (B,feat_dim)
        logits = torch.cat([head(feat) for head in self.heads], dim=1)      # (B,3)
        return logits
