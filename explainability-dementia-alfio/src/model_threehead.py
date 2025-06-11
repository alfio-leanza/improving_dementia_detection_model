import ipdb
import torch
from typing import Union, Tuple
from torch.nn import (Module, Linear, Dropout, Parameter,
                      BatchNorm1d)
from torch.nn import functional as F
from torch_geometric.nn import (GraphConv, global_max_pool)
from torch_geometric.nn.norm import BatchNorm as PyGBatchNorm


__all__ = ["GNNCWT2D_Mk11_1sec_3H"]                 # >>> 3-HEAD


# ──────────────────────────────────────────────────────────────
class _EdgeWeightsGraphConvLayer(Module):
    """
    Custom GraphConv layer with learnable edge weights.
    """
    def __init__(self, n_edges: int, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, weights_init: str = 'ones',
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super().__init__()
        self.n_electrodes = 19
        self.GConv = GraphConv(in_channels, out_channels, aggr, bias, **kwargs)

        if weights_init == 'zeros':
            self.edge_weights = Parameter(torch.zeros([n_edges]))
        elif weights_init == 'ones':
            self.edge_weights = Parameter(torch.ones([n_edges]))
        elif weights_init == 'rand':
            self.edge_weights = Parameter(torch.rand([n_edges]))
        else:
            raise Exception('Invalid edge weights init type.')

    def forward(self, x, edge_index):
        rep = self.edge_weights.repeat(x.size(0) // self.n_electrodes)
        return self.GConv(x, edge_index, rep)


# ──────────────────────────────────────────────────────────────
class GNNCWT2D_Mk11_1sec_3H(Module):                # >>> 3-HEAD
    """
    Mk11 variant (1-sec crop) con **tre teste**:
      • head_main : 3 classi (hc / ftd / ad)
      • head_bin  : 2 classi (hc vs demenza)
      • head_dem  : 2 classi (ftd vs ad, valutata solo sui campioni demenza)
    """
    def __init__(self, n_electrodes: int, cwt_size: Tuple[int, int]):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_freq, self.n_time_samples = cwt_size

        # ---------- dense MLP 800→64 ----------
        self.lin2 = Linear(800, 256)
        self.bn3  = BatchNorm1d(n_electrodes); self.dp3 = Dropout(p=0.3)
        self.lin3 = Linear(256, 128)
        self.bn4  = BatchNorm1d(n_electrodes); self.dp4 = Dropout(p=0.3)
        self.lin4 = Linear(128, 64)
        self.bn5  = BatchNorm1d(n_electrodes); self.dp5 = Dropout(p=0.3)

        # ---------- GNN 64→32 ----------
        self.gconv1 = _EdgeWeightsGraphConvLayer(60, 64, 32)
        self.bn6  = BatchNorm1d(32); self.dp6 = Dropout(p=0.3)
        self.gconv2 = _EdgeWeightsGraphConvLayer(60, 32, 32)
        self.bn7  = BatchNorm1d(32); self.dp7 = Dropout(p=0.3)

        # ---------- embedding global ----------
        self.lin5 = Linear(32, 32)

        # ---------- TRE TESTE ----------
        self.head_main = Linear(32, 3)   # hc/ftd/ad
        self.head_bin  = Linear(32, 2)   # hc vs demenza
        self.head_dem  = Linear(32, 2)   # ftd vs ad

        # pesi loss ausiliarie
        self.lambda_bin = 0.3            # >>> 3-HEAD
        self.lambda_dem = 0.3

    # ------------------- encoder condiviso -------------------
    def encode(self, x, edge_index, batch):
        B = len(torch.unique(batch))

        # split 20×25 campioni | media su 25
        x = x.view(B, self.n_electrodes, 40, 20, -1).mean(dim=4)
        x = x.view(B, self.n_electrodes, -1)          # (B,19,800)

        x = self.dp3(self.bn3(F.relu(self.lin2(x))))
        x = self.dp4(self.bn4(F.relu(self.lin3(x))))
        x = self.dp5(self.bn5(F.relu(self.lin4(x))))
        x = x.view(B * self.n_electrodes, -1)         # (B*19,64)

        x = self.dp6(self.bn6(F.relu(self.gconv1(x, edge_index))))
        x = self.dp7(self.bn7(F.relu(self.gconv2(x, edge_index))))
        x = global_max_pool(x, batch)                 # (B,32)
        return F.relu(self.lin5(x))                   # (B,32)

    # ------------------- forward ------------------------------
    def forward(self, x, edge_index, batch):
        z = self.encode(x, edge_index, batch)
        return (self.head_main(z),
                self.head_bin(z),
                self.head_dem(z))
