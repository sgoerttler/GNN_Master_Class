import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch_geometric.datasets import Planetoid


def load_ecg5000(folder='data'):
    train_data = np.loadtxt(os.path.join(folder, 'ECG5000', 'ECG5000_TRAIN.txt'))
    test_data = np.loadtxt(os.path.join(folder, 'ECG5000', 'ECG5000_TEST.txt'))
    data = np.vstack([train_data, test_data])

    X = data[:, np.newaxis, 1:]
    y = data[:, 0].astype(int) - 1  # classes from 0 to 4

    X = torch.tensor(X, dtype=torch.float32)  # (N, 1, 140)
    y = torch.tensor(y, dtype=torch.long)

    return TensorDataset(X, y)


def get_loaders(dataset='ECG5000', folder='data', batch_size=64, train_split=0.8):
    if dataset == 'ECG5000':
        data = load_ecg5000(folder)
        train_size = int(train_split * len(data))
        val_size = len(data) - train_size
        train_set, val_set = random_split(data, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        return train_loader, val_loader

    elif dataset == 'CiteSeer':
        data = Planetoid(root=os.path.join('data', 'CiteSeer'), name='CiteSeer')
        graph_data = data[0]
        return data, graph_data
