import ipdb
import torch
from typing import Union, Tuple
from torch.nn import Module, Linear, Dropout, Parameter, Conv2d, Conv1d, BatchNorm1d
from torch.nn import functional as F
from torch_geometric.nn import GraphConv, GCNConv, ChebConv, global_max_pool, global_mean_pool
from torch_geometric.nn.norm import BatchNorm as PyGBatchNorm


__all__ = ["GNNCWT2D_Mk11_1sec"]


class _EdgeWeightsGraphConvLayer(Module):
    """
    Custom GraphConv layer with learnable edge weights.
    """
    def __init__(self, n_edges: int, weights_init: str, in_channels: Union[int, Tuple[int, int]], out_channels: int, aggr: str = 'add', bias: bool = True, **kwargs):
        super().__init__()
        self.n_electrodes = 19  # TODO hardcoded value
        self.GConv = GraphConv(in_channels, out_channels, aggr, bias, **kwargs)  # original gconv
        if weights_init == 'zeros':
            self.edge_weights = Parameter(torch.zeros([n_edges]))
        elif weights_init == 'ones':
            self.edge_weights = Parameter(torch.ones([n_edges]))
        elif weights_init == 'rand':
            self.edge_weights = Parameter(torch.rand([n_edges]))
        else:
            raise Exception('Invalid edge weights init type.')


    def forward(self, x, edge_index):
        # TODO: weights are free and unbounded, consider an activation function.
        return self.GConv(x, edge_index, self.edge_weights.repeat(x.shape[0]//self.n_electrodes))


class GNNCWT2D_Mk11_1sec(Module):
    def __init__(self, n_electrodes, cwt_size, num_classes):
        """
        Variant of Mk11 accepting 1-sec-long crops.
        Temporal dimension is reduced by averaging the 25 samples
        belonging to the 20 0.05-sec-long minicrops.
        For each CWT: 40x500 -> 40x20.
        Constant dropout p=0.2.

        'cwt_size' is a tuple of n_freq x n_time_samples, e.g. (40, 500).
        """
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_freq = cwt_size[0]
        self.n_time_samples = cwt_size[1]

        self.lin2 = Linear(800, 256) # prima 800,512
        self.bn3 = BatchNorm1d(n_electrodes)
        self.drop3 = Dropout(p=0.3) # prima 0.2 
        self.lin3 = Linear(256, 128) # prima 512,256
        self.bn4 = BatchNorm1d(n_electrodes)
        self.drop4 = Dropout(p=0.3) # prima 0.2
        #self.lin4 = Linear(128, 64) # prima 256,128
        #self.bn5 = BatchNorm1d(n_electrodes)
        #self.drop5 = Dropout(p=0.3) # prima 0.2
        # graph operations on node-level features
        self.gconv1 = _EdgeWeightsGraphConvLayer(60, 'ones', 128, 64) # prima  128,64
        self.bn6 = BatchNorm1d(32) # prima 64
        self.drop6 = Dropout(p=0.3) # prima 0.2
        self.gconv2 = _EdgeWeightsGraphConvLayer(60, 'ones', 128, 64) # prima 64,64
        self.bn7 = BatchNorm1d(32) # prima 64
        self.drop7 = Dropout(p=0.3) # prima 0.2
        # graph operations on graph-level features
        self.lin5 = Linear(64, 32)
        self.lin6 = Linear(32, num_classes)


    def forward(self, x, edge_index, batch):
        actual_batch_size = len(torch.unique(batch))

        # torch.Size([1216, 34000])
        #x = x.view(actual_batch_size, self.n_electrodes, self.n_freq, self.n_time_samples)
        # torch.Size([64, 19, 40, 500])

        # split and avg samples of 0.05-sec-long chunks
        x = x.view(actual_batch_size, self.n_electrodes, self.n_freq, 20, -1)
        # torch.Size([64, 19, 40, 20, 25])
        x = torch.mean(x, dim=4, keepdim=False)
        # torch.Size([64, 19, 40, 20])

        x = x.view(actual_batch_size, self.n_electrodes, -1)
        # torch.Size([64, 19, 800])

        x = self.lin2(x)
        x = F.relu(x)
        # torch.Size([64, 19, 512])
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.lin3(x)
        x = F.relu(x)
        # torch.Size([64, 19, 256])
        x = self.bn4(x)
        x = self.drop4(x)
        #x = self.lin4(x)
        x = F.relu(x)
        # torch.Size([64, 19, 128])
        #x = self.bn5(x)
        #x = self.drop5(x)
        x = x.view(actual_batch_size*self.n_electrodes, -1)
        # torch.Size([1216, 128])

        x = self.gconv1(x, edge_index)
        x = F.relu(x)
        # torch.Size([1216, 64])
        x = self.bn6(x)
        x = self.drop6(x)
        x = self.gconv2(x, edge_index)
        x = F.relu(x)
        # torch.Size([1216, 64])
        x = self.bn7(x)
        x = self.drop7(x)
        x = global_max_pool(x, batch)
        # torch.Size([64, 64])

        x = self.lin5(x)
        x = F.relu(x)
        # torch.Size([64, 32])
        x = self.lin6(x)
        # torch.Size([64, 2])

        # automatic softmax by torch.nn.CrossEntropyLoss()
        return x

