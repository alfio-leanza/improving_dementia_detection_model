"""
model_ovr.py  –  One-vs-Rest con Graph Attention (GATConv) + SE-Block
--------------------------------------------------------------------
* Backbone GNNCWT2D_Mk11_1sec_segat:
    - MLP per-elettrodo (800 → 128)
    - SE-block (attenzione canale-wise)
    - 2 layer GATConv (heads=4, concat=False)
* Three binary heads (HC, FTD, AD) → logits (B,3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool


# ------------------------------------------------------------------ #
# 1)  SE-Block                                                       #
# ------------------------------------------------------------------ #
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation sul vettore feature (dimensione C)
    opera su tensor shape (B, 19, C)
    """
    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // r, bias=False)
        self.fc2 = nn.Linear(channels // r, channels, bias=False)

    def forward(self, x):
        # x: (B, 19, C)
        z = x.mean(dim=1)                       # squeeze sui nodi → (B, C)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))  # (B, C)
        return x * s.unsqueeze(1)               # re-scale canali


# ------------------------------------------------------------------ #
# 2)  Backbone con SE + GAT                                          #
# ------------------------------------------------------------------ #
class GNNCWT2D_Mk11_1sec_segat(nn.Module):
    def __init__(self, n_electrodes=19, cwt_size=(40, 500), feat_dim=64):
        super().__init__()
        self.ne = n_electrodes
        self.nf, self.nt = cwt_size
        self.win = 25
        self.nw = self.nt // self.win                           # 20

        # MLP per-elettrodo
        self.lin2 = nn.Linear(self.nf * self.nw, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)

        self.bn2 = nn.BatchNorm1d(self.ne)
        self.bn3 = nn.BatchNorm1d(self.ne)
        self.bn4 = nn.BatchNorm1d(self.ne)
        self.drop = nn.Dropout(0.25)

        # SE-Block canale-wise
        self.se = SEBlock(128)

        # Graph Attention
        self.g1 = GATConv(128, 64, heads=4, concat=False, dropout=0.2)
        self.g2 = GATConv(64,  64, heads=4, concat=False, dropout=0.2)
        self.bn_g1 = nn.BatchNorm1d(64)
        self.bn_g2 = nn.BatchNorm1d(64)

        # feature linear
        self.lin_feat = nn.Linear(64, feat_dim)

    # -------------------------------------------------------------- #
    def forward(self, x, edge_index, batch, *, return_features=False):
        B = torch.unique(batch).numel()

        # reshape input → (B, 19, 40, 500)
        x = x.view(B, self.ne, self.nf, self.nt)
        x = x.view(B, self.ne, self.nf, self.nw, self.win).mean(dim=-1)    # (B,19,40,20)
        x = x.reshape(B, self.ne, -1)                                      # (B,19,800)

        x = F.relu(self.lin2(x)); x = self.bn2(x); x = self.drop(x)
        x = F.relu(self.lin3(x)); x = self.bn3(x); x = self.drop(x)
        x = F.relu(self.lin4(x)); x = self.bn4(x); x = self.drop(x)

        # SE-block
        x = self.se(x)                                                     # (B,19,128)

        # GAT: porta a grafo (B·19, feat)
        x = x.view(B * self.ne, -1)
        x = F.elu(self.g1(x, edge_index)); x = self.bn_g1(x); x = self.drop(x)
        x = F.elu(self.g2(x, edge_index)); x = self.bn_g2(x); x = self.drop(x)

        # pooling e feature
        x = global_max_pool(x, batch)          # (B,64)
        feat = F.relu(self.lin_feat(x))        # (B,feat_dim)
        return feat if return_features else feat


# ------------------------------------------------------------------ #
# 3)  One-Vs-Rest con 3 teste binarie                                #
# ------------------------------------------------------------------ #
class OneVsRestGNN(nn.Module):
    def __init__(self, backbone, feat_dim=64):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(3)])
        for h in self.heads:
            nn.init.xavier_uniform_(h.weight); nn.init.zeros_(h.bias)

    def forward(self, x, edge_index, batch):
        feat = self.backbone(x, edge_index, batch, return_features=True)
        logits = torch.cat([head(feat) for head in self.heads], dim=1)   # (B,3)
        return logits
