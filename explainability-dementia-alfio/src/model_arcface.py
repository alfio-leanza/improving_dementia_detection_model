import ipdb
import torch
from typing import Union, Tuple
from torch.nn import Module, Linear, Dropout, Parameter, Conv2d, Conv1d, BatchNorm1d
from torch.nn import functional as F
from torch_geometric.nn import GraphConv, global_max_pool
from torch_geometric.nn.norm import BatchNorm as PyGBatchNorm


__all__ = ["GNNCWT2D_Mk11_1sec_Arc"]


class _EdgeWeightsGraphConvLayer_Arc(Module):
    """
    Custom GraphConv layer with learnable edge weights.
    """
    def __init__(self, n_edges: int, weights_init: str,
                 in_channels: Union[int, Tuple[int, int]], out_channels: int,
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super().__init__()
        self.n_electrodes = 19  # TODO hard-coded value
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
        # weights are free and unbounded; an activation could be added if needed
        return self.GConv(x, edge_index,
                          self.edge_weights.repeat(x.shape[0] // self.n_electrodes))


class GNNCWT2D_Mk11_1sec_Arc(Module):
    """
    Mk11 variant per crop di 1 s. Ora restituisce embedding L2-normalizzati
    (dimensione 'embedding_dim') adatti a loss metriche come ArcFace.
    """
    def __init__(self, n_electrodes: int, cwt_size: Tuple[int, int],
                 embedding_dim: int = 128):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_freq, self.n_time_samples = cwt_size

        self.lin2  = Linear(800, 512)
        self.bn3   = BatchNorm1d(n_electrodes)
        self.drop3 = Dropout(p=0.2)

        self.lin3  = Linear(512, 256)
        self.bn4   = BatchNorm1d(n_electrodes)
        self.drop4 = Dropout(p=0.2)

        self.lin4  = Linear(256, 128)
        self.bn5   = BatchNorm1d(n_electrodes)
        self.drop5 = Dropout(p=0.2)

        # graph operations on node-level features
        self.gconv1 = _EdgeWeightsGraphConvLayer_Arc(60, 'ones', 128, 64)
        self.bn6    = BatchNorm1d(64)
        self.drop6  = Dropout(p=0.2)

        self.gconv2 = _EdgeWeightsGraphConvLayer_Arc(60, 'ones', 64, 64)
        self.bn7    = BatchNorm1d(64)
        self.drop7  = Dropout(p=0.2)

        # graph-level projection to embedding space
        self.lin5 = Linear(64, 32)
        self.lin6 = Linear(32, embedding_dim)   # ← ora produce embeddings

    def forward(self, x, edge_index, batch):
        actual_bs = len(torch.unique(batch))

        # (B, 19, 40, 500)  →  split su 20 minicrops × avg
        x = x.view(actual_bs, self.n_electrodes, self.n_freq, 20, -1)
        x = torch.mean(x, dim=4)                            # (B, 19, 40, 20)
        x = x.view(actual_bs, self.n_electrodes, -1)        # (B, 19, 800)

        x = F.relu(self.lin2(x));  x = self.drop3(self.bn3(x))
        x = F.relu(self.lin3(x));  x = self.drop4(self.bn4(x))
        x = F.relu(self.lin4(x));  x = self.drop5(self.bn5(x))
        x = x.view(actual_bs * self.n_electrodes, -1)       # (B*19, 128)

        x = F.relu(self.gconv1(x, edge_index)); x = self.drop6(self.bn6(x))
        x = F.relu(self.gconv2(x, edge_index)); x = self.drop7(self.bn7(x))

        x = global_max_pool(x, batch)                       # (B, 64)

        x = F.relu(self.lin5(x))
        x = self.lin6(x)                                    # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1)                      # L2-norm -> unit-sphere

        return x                                            # embeddings pronti
