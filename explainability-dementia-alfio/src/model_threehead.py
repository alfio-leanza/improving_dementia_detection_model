# model_threehead.py
import torch, torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, BatchNorm1d, Parameter
from torch_geometric.nn import GraphConv, global_max_pool

__all__ = ["GNNCWT2D_Mk11_1sec_3H"]

class _EdgeWeightsGraphConvLayer(Module):
    def __init__(self, n_edges, in_ch, out_ch, init="ones"):
        super().__init__()
        self.n_electrodes = 19
        self.gconv = GraphConv(in_ch, out_ch, aggr="add", bias=True)
        val = 0. if init == "zeros" else 1.
        self.edge_weights = Parameter(torch.full((n_edges,), val))

    def forward(self, x, edge_index):
        rep = self.edge_weights.repeat(x.size(0)//self.n_electrodes)
        return self.gconv(x, edge_index, rep)

class GNNCWT2D_Mk11_1sec_3H(Module):
    """Encoder + tre teste."""
    def __init__(self, n_electrodes: int, cwt_size: tuple[int, int]):
        super().__init__()
        self.ne = n_electrodes
        # dense 800→64
        self.lin2 = Linear(800, 256)
        self.bn3 = BatchNorm1d(n_electrodes); self.dp3 = Dropout(0.3)
        self.lin3 = Linear(256, 128)
        self.bn4 = BatchNorm1d(n_electrodes); self.dp4 = Dropout(0.3)
        self.lin4 = Linear(128, 64)
        self.bn5 = BatchNorm1d(n_electrodes); self.dp5 = Dropout(0.3)
        # gnn 64→32
        self.gconv1 = _EdgeWeightsGraphConvLayer(60, 64, 32)
        self.bn6 = BatchNorm1d(32); self.dp6 = Dropout(0.3)
        self.gconv2 = _EdgeWeightsGraphConvLayer(60, 32, 32)
        self.bn7 = BatchNorm1d(32); self.dp7 = Dropout(0.3)
        # embedding + teste
        self.lin5 = Linear(32, 32)
        self.head_main = Linear(32, 3)
        self.head_bin  = Linear(32, 2)
        self.head_dem  = Linear(32, 2)
        self.lambda_bin, self.lambda_dem = 0.3, 0.3

    def encode(self, x, edge_index, batch):
        B = len(torch.unique(batch))
        x = x.view(B, self.ne, 40, 20, -1).mean(dim=4).view(B, self.ne, -1)
        x = self.dp5(self.bn5(F.relu(self.lin4(
                self.dp4(self.bn4(F.relu(self.lin3(
                self.dp3(self.bn3(F.relu(self.lin2(x))))))))))))
        x = x.view(B*self.ne, -1)
        x = self.dp7(self.bn7(F.relu(self.gconv2(
                self.dp6(self.bn6(F.relu(self.gconv1(x, edge_index))))))))
        return F.relu(self.lin5(global_max_pool(x, batch)))

    def forward(self, x, edge_index, batch):
        z = self.encode(x, edge_index, batch)
        return self.head_main(z), self.head_bin(z), self.head_dem(z)
