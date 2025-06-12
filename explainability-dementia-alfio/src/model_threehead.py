"""
Modelli grafici per CWT EEG 1 s:
– backbone GNNCWT2D_Mk11_1sec (ritocco: opzione return_features)
– HierarchicalBinaryThreeHead: 3 teste binarie (hc-ad, hc-ftd, ftd-ad)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_max_pool

__all__ = [
    "_EdgeWeightsGraphConvLayer",
    "GNNCWT2D_Mk11_1sec",          # backbone compatibile con precedente
    "HierarchicalBinaryThreeHead"  # nuovo modello gerarchico
]


# --------------------------------------------------------------------- #
# 1.  GraphConv con pesi d’arco learnable e softplus (=> positivi)      #
# --------------------------------------------------------------------- #
class _EdgeWeightsGraphConvLayer(nn.Module):
    def __init__(self,
                 n_edges: int,
                 in_channels: int,
                 out_channels: int,
                 aggr: str = "add",
                 bias: bool = True):
        super().__init__()
        self.GConv = GraphConv(in_channels, out_channels, aggr=aggr, bias=bias)
        # pesi inizializzati a 0 (equivalente a tutti gli archi=1 dopo softplus)
        self.edge_weights = nn.Parameter(torch.zeros(n_edges))
        self.n_electrodes = 19            # fissato dal dataset

    def forward(self, x, edge_index):
        # softplus => pesi > 0 e gradiente ben condizionato
        w = F.softplus(self.edge_weights)
        # ripeti per ciascun grafo nel batch
        w_exp = w.repeat(x.size(0) // self.n_electrodes)
        return self.GConv(x, edge_index, w_exp)


# --------------------------------------------------------------------- #
# Backbone – GNNCWT2D_Mk11_1sec  (fix in_features = 800)                #
# --------------------------------------------------------------------- #
class GNNCWT2D_Mk11_1sec(nn.Module):
    """
    Backbone grafico per CWT EEG (1 s):
      • input:  (B, 19, 40, 500)
      • output: feature [B, feat_dim]   se return_features=True
                logit   [B, 3]         altrimenti
    """
    def __init__(self,
                 n_electrodes: int = 19,
                 cwt_size: tuple = (40, 500),
                 feat_dim: int = 32):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_freq, self.n_time = cwt_size

        # ---------- parametro finestra temporale ----------
        self.win_len = 25                          # campioni (0,05 s)
        self.n_windows = self.n_time // self.win_len  # 500//25 = 20

        # ---------- MLP per-elettrodo ----------
        self.lin2  = nn.Linear(self.n_freq * self.n_windows, 512)  # 40×20 = 800
        self.bn3   = nn.BatchNorm1d(n_electrodes)
        self.drop3 = nn.Dropout(0.2)
        self.lin3  = nn.Linear(512, 256)
        self.bn4   = nn.BatchNorm1d(n_electrodes)
        self.drop4 = nn.Dropout(0.2)
        self.lin4  = nn.Linear(256, 128)
        self.bn5   = nn.BatchNorm1d(n_electrodes)
        self.drop5 = nn.Dropout(0.2)

        # ---------- GraphConv ----------
        self.gconv1 = _EdgeWeightsGraphConvLayer(60, 128, 64)
        self.bn6    = nn.BatchNorm1d(64)
        self.drop6  = nn.Dropout(0.2)
        self.gconv2 = _EdgeWeightsGraphConvLayer(60, 64, 64)
        self.bn7    = nn.BatchNorm1d(64)
        self.drop7  = nn.Dropout(0.2)

        # ---------- Pooling & classifier ----------
        self.lin5 = nn.Linear(64, feat_dim)
        self.lin6 = nn.Linear(feat_dim, 3)

    # ----------------------------------------------------------------- #
    def forward(self, x, edge_index, batch, *, return_features: bool = False):
        B = torch.unique(batch).numel()

        # reshape → (B, 19, 40, 500)
        x = x.view(B, self.n_electrodes, self.n_freq, self.n_time)
        # media su finestre di 25 sample
        x = x.view(B, self.n_electrodes, self.n_freq,
                   self.n_windows, self.win_len).mean(dim=4)
        # flatten → (B, 19, 800)
        x = x.reshape(B, self.n_electrodes, -1)

        # MLP per-elettrodo
        x = F.relu(self.lin2(x));   x = self.bn3(x); x = self.drop3(x)
        x = F.relu(self.lin3(x));   x = self.bn4(x); x = self.drop4(x)
        x = F.relu(self.lin4(x));   x = self.bn5(x); x = self.drop5(x)

        # GraphConv (B·19, feat)
        x = x.view(B * self.n_electrodes, -1)
        x = F.relu(self.gconv1(x, edge_index)); x = self.bn6(x); x = self.drop6(x)
        x = F.relu(self.gconv2(x, edge_index)); x = self.bn7(x); x = self.drop7(x)

        # pooling max → feature
        x = global_max_pool(x, batch)
        features = F.relu(self.lin5(x))

        if return_features:
            return features
        return self.lin6(features)


    # ------------------------------------------------------------- #
    def forward(self, x, edge_index, batch, *, return_features: bool = False):
        """
        Parametri
        ---------
        x : torch.Tensor               (N_tot_nodes, 800)
        edge_index : torch.LongTensor  (2, N_edges_tot)
        batch : torch.LongTensor       (N_tot_nodes,)
        return_features : bool
            Se True, restituisce il vettore feature; altrimenti i logit.
        """
        B = torch.unique(batch).numel()

        # reshape (flat -> (B, 19, 40, 500))
        x = x.view(B, self.n_electrodes, self.n_freq, self.n_time)
        # media su finestre di 25 campioni (0,05 s)
        x = x.view(B, self.n_electrodes, self.n_freq, 20, -1).mean(dim=4)
        # -> (B, 19, 40, 20) -> flat a 800
        x = x.reshape(B, self.n_electrodes, -1)

        # MLP per-elettrodo
        x = F.relu(self.lin2(x));   x = self.bn3(x); x = self.drop3(x)
        x = F.relu(self.lin3(x));   x = self.bn4(x); x = self.drop4(x)
        x = F.relu(self.lin4(x));   x = self.bn5(x); x = self.drop5(x)

        # grafi: (B·19, feat)
        x = x.view(B * self.n_electrodes, -1)
        x = F.relu(self.gconv1(x, edge_index)); x = self.bn6(x); x = self.drop6(x)
        x = F.relu(self.gconv2(x, edge_index)); x = self.bn7(x); x = self.drop7(x)

        # pooling max su nodi → (B, 64)
        x = global_max_pool(x, batch)
        features = F.relu(self.lin5(x))       # (B, feat_dim)

        if return_features:
            return features                   # ---> usato dai modelli a teste multiple
        # classificatore “flat” (non più usato, ma resta per retro-compatibilità)
        return self.lin6(features)            # (B, 3)


# --------------------------------------------------------------------- #
# 3.  Modello gerarchico a 3 teste binarie                              #
# --------------------------------------------------------------------- #
class HierarchicalBinaryThreeHead(nn.Module):
    """
    Testa 0: HC vs AD
    Testa 1: HC vs FTD
    Testa 2: FTD vs AD

    Output finale: logit a 3 classi [HC, FTD, AD] ricostruito come
        p(HC)  = p0(HC) * p1(HC)
        p(AD)  = p0(AD) * p2(AD)
        p(FTD) = p0(HC) * p1(FTD) + p0(AD) * p2(FTD)

    Ritorna:
        – final_logits  [B, 3]
        – opzionalmente anche i logit delle 3 teste (return_all=True)
    """
    def __init__(self,
                 backbone: GNNCWT2D_Mk11_1sec,
                 feat_dim: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.dropout  = nn.Dropout(dropout)

        self.head_root   = nn.Linear(feat_dim, 2)   # HC vs AD
        self.head_hcftd  = nn.Linear(feat_dim, 2)   # HC vs FTD
        self.head_ftdad  = nn.Linear(feat_dim, 2)   # FTD vs AD

        for m in (self.head_root, self.head_hcftd, self.head_ftdad):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------- #
    def _combine_probs(self, p0, p1, p2):
        """
        p0: root (HC/AD)      [B, 2]
        p1: HC-FTD            [B, 2]
        p2: FTD-AD            [B, 2]

        Restituisce tensor [B, 3] con prob. (HC, FTD, AD)
        """
        p_hc  = p0[:, 0] * p1[:, 0]
        p_ad  = p0[:, 1] * p2[:, 1]
        p_ftd = p0[:, 0] * p1[:, 1] + p0[:, 1] * p2[:, 0]
        return torch.stack((p_hc, p_ftd, p_ad), dim=1)

    # ------------------------------------------------------------- #
    def forward(self, x, edge_index=None, batch=None, *, return_all=False):
        """
        Compatibile sia con chiamata (Data) sia con (x, edge_index, batch).
        """
        if edge_index is None:          # -> è stato passato Data
            data       = x
            x          = data.x
            edge_index = data.edge_index
            batch      = data.batch

        feat = self.backbone(x, edge_index, batch, return_features=True)
        feat = self.dropout(feat)

        log_root   = self.head_root(feat)
        log_hcftd  = self.head_hcftd(feat)
        log_ftdad  = self.head_ftdad(feat)

        # da logit a probabilità
        p_root  = F.softmax(log_root,  dim=1)
        p_hcftd = F.softmax(log_hcftd, dim=1)
        p_ftdad = F.softmax(log_ftdad, dim=1)

        final_prob   = self._combine_probs(p_root, p_hcftd, p_ftdad)
        final_logits = torch.log(final_prob + 1e-8)   # stabilità numerica

        if return_all:
            return final_logits, log_root, log_hcftd, log_ftdad
        return final_logits
