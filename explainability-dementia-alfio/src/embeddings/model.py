"""
Implementazione completa di GNNCWT2D_Mk11_1sec
(con metodo .embed() che restituisce un vettore 32-D).
"""
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, BatchNorm1d, Parameter
from torch_geometric.nn import GraphConv, global_max_pool

__all__ = ["GNNCWT2D_Mk11_1sec"]


class _EdgeWeightsGraphConvLayer(Module):
    """GraphConv con pesi d'arco apprendibili."""
    def __init__(self, n_edges: int, weights_init: str,
                 in_channels: int, out_channels: int,
                 aggr: str = "add", bias: bool = True):
        super().__init__()
        self.n_electrodes = 19
        self.conv = GraphConv(in_channels, out_channels, aggr, bias=bias)

        if weights_init == "zeros":
            self.edge_weights = Parameter(torch.zeros(n_edges))
        elif weights_init == "ones":
            self.edge_weights = Parameter(torch.ones(n_edges))
        elif weights_init == "rand":
            self.edge_weights = Parameter(torch.rand(n_edges))
        else:
            raise ValueError("weights_init must be 'zeros', 'ones' o 'rand'")

    def forward(self, x, edge_index):
        repeat = x.size(0) // self.n_electrodes
        ew = self.edge_weights.repeat(repeat)
        return self.conv(x, edge_index, ew)


class GNNCWT2D_Mk11_1sec(Module):
    def __init__(self, n_electrodes: int, cwt_size: tuple[int, int], num_classes: int):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_freq, self.n_time = cwt_size  # 40,500

        # ---- MLP nodo-per-nodo ----------------------------------------------
        self.lin2 = Linear(800, 512)
        self.bn3  = BatchNorm1d(n_electrodes)
        self.d3   = Dropout(0.2)
        self.lin3 = Linear(512, 256)
        self.bn4  = BatchNorm1d(n_electrodes)
        self.d4   = Dropout(0.2)
        self.lin4 = Linear(256, 128)
        self.bn5  = BatchNorm1d(n_electrodes)
        self.d5   = Dropout(0.2)

        # ---- GNN -------------------------------------------------------------
        self.g1  = _EdgeWeightsGraphConvLayer(60, "ones", 128, 64)
        self.bn6 = BatchNorm1d(64)
        self.d6  = Dropout(0.2)
        self.g2  = _EdgeWeightsGraphConvLayer(60, "ones", 64, 64)
        self.bn7 = BatchNorm1d(64)
        self.d7  = Dropout(0.2)

        # ---- testa -----------------------------------------------------------
        self.lin5 = Linear(64, 32)
        self.lin6 = Linear(32, num_classes)

    # -------------------------------------------------------------------------
    def _stem(self, x, edge_index, batch):
        """Blocco condiviso da .forward() e .embed(); restituisce (B,32)."""
        B = len(torch.unique(batch))

        # reshape → media su 25 campioni (0.05 s) → (B,19,800)
        x = x.view(B, self.n_electrodes, self.n_freq, 20, -1).mean(dim=4)
        x = x.view(B, self.n_electrodes, -1)

        x = F.relu(self.lin2(x)); x = self.bn3(x); x = self.d3(x)
        x = F.relu(self.lin3(x)); x = self.bn4(x); x = self.d4(x)
        x = F.relu(self.lin4(x)); x = self.bn5(x); x = self.d5(x)
        x = x.view(B * self.n_electrodes, -1)

        x = F.relu(self.g1(x, edge_index)); x = self.bn6(x); x = self.d6(x)
        x = F.relu(self.g2(x, edge_index)); x = self.bn7(x); x = self.d7(x)

        x = global_max_pool(x, batch)      # (B,64)
        x = F.relu(self.lin5(x))           # (B,32)
        return x

    def embed(self, x, edge_index, batch):
        """Embedding 32-D senza layer di classificazione."""
        return self._stem(x, edge_index, batch)

    def forward(self, x, edge_index, batch):
        x = self._stem(x, edge_index, batch)
        return self.lin6(x)                # CrossEntropyLoss applicherà softmax
