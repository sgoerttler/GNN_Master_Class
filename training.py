import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import time


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, verbose=True):
    if verbose:
        time_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(yb)

        acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))

        if verbose:
            if epoch + 1 in [1, 5, 10, 20, 50, 75, 100, 150, 200]:
                print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")

    if verbose:
        print(f"Training completed in {time.time() - time_start:.2f} seconds.")


def train_graph_model(model, graph_data, epochs=100, verbose=True):
    if verbose:
        time_start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Run training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x)
        loss = F.nll_loss(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        out = model(graph_data.x)
        pred = out.argmax(dim=1)
        accs = []
        for mask in [graph_data.train_mask, graph_data.val_mask, graph_data.test_mask]:
            correct = pred[mask] == graph_data.y[mask]
            accs.append(int(correct.sum()) / int(mask.sum()))

        if epoch + 1 in [1, 5, 10, 20, 50, 75, 100, 150, 200]:
            print(f'Epoch {epoch + 1:3d}, Loss: {loss.item():.4f}, '
                  f'Train Acc: {accs[0]:.4f}, Val Acc: {accs[1]:.4f}, Test Acc: {accs[2]:.4f}')

    if verbose:
        print(f"Training completed in {time.time() - time_start:.2f} seconds.")