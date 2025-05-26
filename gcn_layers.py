import numpy as np
import torch
import torch.nn as nn


class GCNLayerFirstOrder(nn.Module):
    def __init__(self, DAD, in_features, out_features):
        super(GCNLayerFirstOrder, self).__init__()
        # Propagation adjacency matrix
        self.DAD = torch.tensor(DAD, dtype=torch.float32)

        # Initialize weights
        W1_init = np.random.normal(0, 1, (in_features, out_features))
        self.W1 = nn.Parameter(torch.tensor(W1_init, dtype=torch.float32))  # Learnable weights
        W2_init = np.random.normal(0, 1, (in_features, out_features))
        self.W2 = nn.Parameter(torch.tensor(W2_init, dtype=torch.float32))  # Learnable weights

    def forward(self, x):
        x = x.squeeze(1)  # No feature dimension
        x = x * self.W1 + torch.einsum('ij,bj->bi', self.DAD, x) * self.W2
        return x


class GCNLayerRenorm(nn.Module):
    def __init__(self, DAD_tild, in_features, out_features, bias=False):
        super(GCNLayerRenorm, self).__init__()
        # Propagation adjacency matrix
        self.DAD_tild = torch.tensor(DAD_tild, dtype=torch.float32)

        # Initialize weights
        W_init = np.random.normal(0, 0.1, (in_features, out_features))
        self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        if len(x.shape) == 3:
            # With batches
            x = torch.transpose(x, 1, 2)  # Make time dimension the graph dimension
            x = torch.einsum('ij,bjm,ml->bil', self.DAD_tild, x, self.W)
            x = torch.transpose(x, 1, 2)
        elif len(x.shape) == 2:
            # No batches (only 1 graph)
            x = torch.transpose(x, 0, 1)
            x = torch.einsum('ij,jm,ml->il', self.DAD_tild, x, self.W)
            x = x + self.bias
            x = torch.transpose(x, 0, 1)
        return x
