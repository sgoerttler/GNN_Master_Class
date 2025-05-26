import numpy as np
import os
import argparse

from dataloader import get_loaders
from models import get_model
from training import train_model, train_graph_model


def experiment_1():
    # Loading data
    print('Loading ECG5000 dataset...')
    train_loader, val_loader = get_loaders(dataset='ECG5000')
    print()

    # print('***** Experiment 1a *****')
    # # Running emulation model and baseline model
    # for model_name in ['GCN_CNN_emulation', 'CNN_1_layer']:
    #     print(f'Running {model_name} model...')
    #     model = get_model(model_name, num_classes=5, N_s=140)
    #     train_model(model, train_loader, val_loader, epochs=50)
    #     print()

    input('Press Enter to continue with Experiment 1b...\n')

    print('***** Experiment 1b *****')
    # Running GCN model with renormalization trick
    model_name = 'GCN_renorm_CNN_emulation'
    print(f'Running {model_name} model...')
    model = get_model(model_name, num_classes=5, N_s=140)
    train_model(model, train_loader, val_loader, epochs=50)
    print()


def experiment_2():
    # Loading data
    print('Loading CiteSeer dataset...')
    data, graph_data = get_loaders(dataset='CiteSeer')

    # Running 2-layer GCN model with renormalization trick (custom implementation)
    model_name = 'GCN_renorm_2_layers'
    print(f'Running {model_name} model (custom implementation)...')
    model = get_model(model_name, num_classes=data.num_classes, N_s=graph_data.num_nodes, N_f=data.num_node_features,
                      N_c=16, edge_index=graph_data.edge_index, custom_implementation=True)
    train_graph_model(model, graph_data, epochs=200)
    print()

    # Running 2-layer GCN model with renormalization trick (torch-geometric implementation)
    print(f'Running {model_name} model (torch-geometric implementation)...')
    model = get_model(model_name, num_classes=data.num_classes, N_s=graph_data.num_nodes, N_f=data.num_node_features,
                      N_c=16, edge_index=graph_data.edge_index, custom_implementation=False)
    train_graph_model(model, graph_data, epochs=200)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GNN master class experiments')
    parser.add_argument('--experiment', type=int, default=1, help='which experiment to run')
    args = parser.parse_args()

    if args.experiment == 1:
        experiment_1()
    elif args.experiment == 2:
        experiment_2()
