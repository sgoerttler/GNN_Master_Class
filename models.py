import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

from gcn_layers import GCNLayerRenorm, GCNLayerFirstOrder
from utils import D_norm


class CNN1Layer(nn.Module):
    def __init__(self, num_classes):
        super(CNN1Layer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)

        # Classification layers
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class GCNEmulationCNN(nn.Module):
    def __init__(self, num_classes, N_s, N_c=16, kernel_size=5):
        super(GCNEmulationCNN, self).__init__()
        self.N_c = N_c
        self.kernel_size = kernel_size

        # Define linear time graph
        A = np.zeros((N_s, N_s))
        for i in range(N_s - 1):
            A[i, i + 1] = 1

        # Set up the first-order GCN layers
        self.gcn = nn.ModuleList()
        for k in range(kernel_size):
            row = nn.ModuleList()
            for c in range(N_c):
                layer = GCNLayerFirstOrder(DAD=A, in_features=1, out_features=1)
                row.append(layer)
            self.gcn.append(row)

        # Classification layers
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(N_c, num_classes)

    def forward(self, x):
        xs = []
        for c in range(self.N_c):
            x_i = self.gcn[0][c](x)
            for i in range(1, self.kernel_size):
                x_i = self.gcn[i][c](x_i)
            x_i = F.relu(x_i)
            xs.append(x_i)
        x = torch.stack(xs, dim=1)

        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class GCNRenormEmulation(nn.Module):
    def __init__(self, num_classes, N_s):
        super(GCNRenormEmulation, self).__init__()

        # Initialize the modified adjacency matrix for a linear graph
        I = np.eye(N_s)
        A = np.zeros((N_s, N_s))
        for i in range(N_s - 1):
            A[i, i + 1] = 1
        A_tild = A + I
        DAD_tild = D_norm(A_tild) @ A_tild @ D_norm(A_tild)

        # Define the GCN layers
        self.gcn1 = GCNLayerRenorm(DAD_tild, in_features=1, out_features=2)
        self.gcn2 = GCNLayerRenorm(DAD_tild, in_features=2, out_features=2)
        self.gcn3 = GCNLayerRenorm(DAD_tild, in_features=2, out_features=4)
        self.gcn4 = GCNLayerRenorm(DAD_tild, in_features=4, out_features=8)
        self.gcn5 = GCNLayerRenorm(DAD_tild, in_features=8, out_features=16)

        # Classification layers
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.gcn4(x)
        x = self.gcn5(x)
        x = F.relu(x)

        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class GCNRenormTwoLayers(nn.Module):
    def __init__(self, num_classes, N_s, N_f, N_c=16, edge_index=None, custom_implementation=True):
        super(GCNRenormTwoLayers, self).__init__()

        self.custom_implementation = custom_implementation
        if self.custom_implementation:
            # Fill a zero matrix: A[i, j] = 1 if there is an edge from i -> j
            A = np.zeros((N_s, N_s), dtype=np.float32)
            A[edge_index[0], edge_index[1]] = 1
            A_tild = A + np.eye(N_s, dtype=np.float32)
            DAD_tild = D_norm(A_tild) @ A_tild @ D_norm(A_tild)

            self.gcn1 = GCNLayerRenorm(DAD_tild, N_f, N_c, bias=True)
            self.gcn2 = GCNLayerRenorm(DAD_tild, N_c, num_classes, bias=True)
        else:
            # torch_geometric alternative: use GCNConv, pass edge index to forward function
            self.gcn1 = GCNConv(N_f, N_c)
            self.gcn2 = GCNConv(N_c, num_classes)
            self.edge_index = edge_index

    def forward(self, x):

        if self.custom_implementation:
            x = torch.transpose(x, 0, 1)
            x = self.gcn1(x)
        else:
            x = self.gcn1(x, edge_index=self.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        if self.custom_implementation:
            x = self.gcn2(x)
            x = torch.transpose(x, 0, 1)
        else:
            x = self.gcn2(x, edge_index=self.edge_index)

        return F.log_softmax(x, dim=1)


def get_model(model_name, print_params=True, num_classes=5, N_s=140, N_f=1, N_c=1, edge_index=None, custom_implementation=True):
    if model_name == 'CNN_1_layer':
        model = CNN1Layer(num_classes=num_classes)
    elif model_name == 'GCN_CNN_emulation':
        model = GCNEmulationCNN(num_classes=num_classes, N_s=N_s)
    elif model_name == 'GCN_renorm_CNN_emulation':
        model = GCNRenormEmulation(num_classes=num_classes, N_s=N_s)
    elif model_name == 'GCN_renorm_2_layers':
        model = GCNRenormTwoLayers(num_classes=num_classes, N_s=N_s, N_f=N_f, N_c=N_c,
                                   edge_index=edge_index, custom_implementation=custom_implementation)

    if print_params:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')

    return model
