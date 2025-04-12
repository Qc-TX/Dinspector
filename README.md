# Dinspector: APT Detection with T-GAT

This project implements a unified framework for detecting Advanced Persistent Threats (APT) using Graph Neural Networks with edge-aware attention (T-GAT).

---
## 📁 Directory Structure

`Dinspector/`
`├── models/`
`│   ├── tgat.py         # Edge-aware attention layer`
`│   └── apt_model.py    # Full GNN model (SAGE + T-GAT)`
`├── datasets/`
`│   ├── loader.py       # Generic loader for graph text files`
`│   └── parse_darpatc.py # Parser for DARPA CDM`
`├── configs/`
`│   └── default.yaml    # Default training parameters`
`├── train.py            # Training script`
`├── test.py             # Testing + alert script`
`└── README.md`


## 🚀 How to Use

### 1. Setup Environment

```bash
pip install torch torch-geometric

Note: PyG may require additional dependencies, see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

```

### 2. Prepare Dataset
Create a folder like:
    `data/`
`├── darpatc/`
`│   ├── cadets_train.txt`
`│   ├── cadets_test.txt`
`├── streamspot/`
`│   ├── train.txt`
`│   ├── test.txt`
`├── unicorn/`
`│   ├── train.txt`
`│   ├── test.txt`

Each .txt file should follow this tab-separated format:

`src_id    src_type    dst_id    dst_type    edge_type    timestamp`


### 3. Train & Test
1. Place your dataset under `data/` folder, e.g., `data/darpatc/cadets_train.txt` and `cadets_test.txt`
2. Train: `python train.py --dataset darpatc --scene cadets --gpu`
3. Test: `python test.py --dataset darpatc --scene cadets --gpu`


### 4.  Configuration (Optional)🧩

Edit configs/default.yaml to customize parameters:

`dataset: darpatc`
`scene: cadets`
`epochs: 30`
`lr: 0.01`
`hidden_dim: 128`
`use_tgat: true`
`heads: 8`
`gamma: 1.0`

## 📢 Citation

If you use this project in academic work, please consider citing or acknowledging it.