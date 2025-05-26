# 🧠 GNN Master Class (27. May 2025)

This repository contains code for simple demonstrations of Graph Convolutional Networks (GCNs). 
These demonstrations explore the link between 1D-Convolutional Neural Networks and GCNs, and compare a custom implementation to a torch-geometric implementation.

---

## 🚀 How to Run

### 1. Install Dependencies

Install the required Python packages. You will need at least:

- `torch`
- `torch_geometric`
- `numpy`
- `scipy`
- `scikit-learn`

### 2. Download Dataset
The code uses two datasets:
- ECG5000
- CiteSeer

The ECG5000 dataset can be downloaded from [Time Series Classification Website](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) and should be placed in a directory named `data`.
The CiteSeer dataset is downloaded automatically by the code.

### 3. Experiments
**Experiment 1**: ECG5000 (Time Series Classification)
- **Dataset**: ECG5000
- **Models**:
  - Emulate 1D-CNN using GCN with 1st order propagation
  - GCN with renormalization trick
  - 1D-CNN (baseline)
- **Objective**: Evaluate how GCNs can emulate CNN behavior for temporal signals, which "live" on linear time graph.

**Experiment 2**: CiteSeer (Node Classification)
- **Dataset**: CiteSeer citation network
- **Models**:
  - 2-layer GCN with renormalization (custom implementation)
  - 2-layer GCN with renormalization (torch-geometric implementation)
- **Objective**: Understand GCN with renormalization for node classification task. 

### 4. Run the Code
You can run the experiments by executing the main.py file. The experiment number can be specified using the `--experiment` argument, which accepts either `1` or `2`. The default is `1`.

## 📁 Resources
- https://distill.pub/2021/gnn-intro/
- https://distill.pub/2021/understanding-gnns/
- https://www.youtube.com/watch?v=GXhBEj1ZtE8
- [Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.](https://arxiv.org/abs/1609.02907)
- [Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017, July). Neural message passing for quantum chemistry. In International conference on machine learning (pp. 1263-1272). PMLR.](https://arxiv.org/abs/1704.01212)

## 📬 Contact

For questions or contributions, feel free to open an issue or pull request.
