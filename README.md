# Block Sparse Attention for Power Flow Prediction in Distribution Networks

A self-supervised Graph Transformer framework with block sparse attention mechanisms for efficient power flow prediction and state estimation in large-scale electrical distribution networks.

## 📋 Overview

This project implements a **Self-Supervised Graph Neural Network (SSGNN)** architecture that combines:
- **Block sparse attention** for scalable graph representation learning
- **Dynamic Message Passing Networks (DyMPN)** for localized feature aggregation
- **Multi-decoder architecture** for simultaneous prediction of multiple electrical quantities
- **Self-supervised masked modeling** inspired by BERT-style pretraining

The model is designed to handle **variable-sized distribution networks** with up to thousands of nodes, using padding and nodal masking strategies to batch heterogeneous graph structures efficiently.

## 🏗️ Architecture

### Model Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                              │
│  Node Features (H): [P, Q, P_gen, Q_gen, V, θ] × N_nodes   │
│  Sparse Adjacency (A): N_nodes × N_nodes                   │
│  Nodal Mask: Binary indicator for valid nodes              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Shared Encoder (Graph Transformer)             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Embedding + BatchNorm                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  NodeGTransformer Blocks (×blockNum)                │   │
│  │  • DyMPN for Q, K, V extraction                     │   │
│  │  • Linear Attention with nodal masking              │   │
│  │  • Residual connections + LayerNorm                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Multi-Decoder MLP Heads                        │
│  6 independent decoders → [P, Q, P_gen, Q_gen, V, θ]       │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Block Sparse Attention**: Exploits natural clustering in distribution networks (by voltage level and feeder) to reduce attention complexity from O(N²) to O(N·k) where k ≪ N.

2. **Nodal Masking Strategy**: Enables batch processing of networks with varying sizes by padding to a maximum size and masking invalid nodes during attention computation.

3. **Self-Supervised Pretraining**: Randomly masks node features based on node type (PQ/PV/Pt) and trains the model to reconstruct them, learning robust representations without labeled data.

4. **Type-Aware Masking**: Different feature subsets are masked for different node types:
   - **PQ nodes**: Voltage magnitude (V) and angle (θ)
   - **PV nodes**: Reactive power (Q_gen) and angle (θ)
   - **Pt (slack) nodes**: Active power (P_gen) and reactive power (Q_gen)

## 📁 Project Structure

```
block_sparse_attn/
├── baseModel.py                      # Main model architecture & training loop
├── baseModelBlockedTest_746sys.py    # Blocked attention variant (test)
├── baseModelBlocked_746sys.py        # Blocked attention variant
├── baseModelTest.py                  # Model testing utilities
├── config746sys.py                   # Configuration for 746-node system
├── FQ_gen_pf.py                      # Power flow data generation
├── gen_pf_samples_746.py             # Sample generation script
├── Interactive-1.ipynb               # Jupyter notebook for interactive analysis
├── new746_system_v0713.py            # System initialization & validation
├── sys_init_and_blocking.py          # Network partitioning logic
├── yanshi1003.py                     # (Legacy script)
├── yantian_sys 说明书.txt            # System documentation (Chinese)
│
├── Utls/                             # Utility modules
│   ├── GTransformerSparseNodalmasksAddAttnUtls.py  # Graph Transformer layers
│   ├── GTransformerSparseNodalmasksAddAttnUtlsBlockSparse.py
│   ├── utls.py                       # Data loading & preprocessing
│   ├── yantian_sys_746sys.py         # 746-node system utilities
│   ├── yantian_sys_all_system.py     # Multi-system utilities
│   └── __pycache__/
│
├── system_file/746sys/               # System data & configuration
│   ├── 746sys_node_order.json        # Canonical node ordering
│   ├── bus2gisid_746sys.npz          # Bus to GIS ID mapping
│   ├── fcID2fcname_746sys.pkl        # Feeder circuit name mapping
│   ├── fc_base_net.xlsx              # Base network (pandapower format)
│   ├── nd2gisid.npz                  # Node to GIS ID mapping
│   ├── *_per_node.npy                # Normalization statistics
│   ├── parsed_cim_new402_v0716.xlsx  # CIM format network data
│   │
│   ├── blocking/                     # Network partitioning results
│   │   ├── blocks_clusters_*.csv     # Cluster assignments by voltage
│   │   ├── blocks_node2cluster_*.csv # Node-to-cluster mapping
│   │   └── partitions_summary.json   # Partitioning statistics
│   │
│   └── FeederFiles/                  # Individual feeder data
│       ├── branch_F02*.csv
│       ├── branch_F09*.csv
│       ├── branch_F14*.csv
│       └── branch_F31*.csv
│
├── Logger/                           # Training logs (git-ignored)
│   ├── training_log.csv
│   └── logs/ssgnn_*/train/
│
└── saved_models/                     # Pretrained weights (git-ignored)
    └── fc_foundation_model/
        ├── SSGNN_*_layer_*.npz
        └── ...
```

## 🔧 Dependencies

```python
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
pandapower>=2.11.0
seaborn>=0.11.0
matplotlib>=3.4.0
```

## 🚀 Usage

### 1. Configuration

Edit `config746sys.py` to set paths and hyperparameters:

```python
WORKPATH = '/path/to/block_sparse_attn'
DATAPATH = '/path/to/data'
CUDA_VISIBLE_DEVICES = '0'  # GPU selection

# Model architecture
units = 6           # Input feature dimension
d_model = 48        # Transformer embedding dimension
num_heads = 8       # Attention heads
blockNum = 2        # Number of transformer blocks
mlpNeuron = 128     # MLP hidden size
```

### 2. Data Preparation

The model expects data in the following format:
- **Node features (H)**: `.npy` files, shape `(batch_size, num_nodes, 6)`
- **Adjacency matrices (A)**: `.npz` sparse matrices, shape `(num_nodes, num_nodes)`

Features are ordered as: `[P_load, Q_load, P_gen, Q_gen, V_pu, theta_deg]`

### 3. Training

Run the main training script:

```bash
cd /path/to/block_sparse_attn
python baseModel.py
```

Key training parameters in `baseModel.py`:
```python
total_sample_num = 24576      # Total training samples
sample_for_each_iter = 2048   # Batch size
shuffle = False               # Data shuffling
retrainFlag = True            # Start from scratch vs. resume
firstEpoch = True             # First epoch flag
```

### 4. Inference

After training, load pretrained weights:

```python
# In baseModel.py, set:
retrainFlag = False
firstEpoch = False

# Weights are automatically loaded from:
# saved_models/fc_foundation_model/SSGNN_{param_count}_layer_{i}_weights.npz
```

## 📊 Model Specifications

| Component | Configuration |
|-----------|---------------|
| Input dimension | 6 (P, Q, P_gen, Q_gen, V, θ) |
| Embedding size (d_model) | 48 |
| Attention heads | 8 |
| Transformer blocks | 2 |
| MLP hidden neurons | 746 (system-size dependent) |
| Output decoders | 6 (one per feature) |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.004) |
| Loss function | Masked MSE (type-aware) |

## 🎯 Experimental Setup

### Test System: 746-Node Distribution Network

The model is evaluated on a real-world **746-node distribution system** from Yantian District, Shenzhen, China:

| Voltage Level | Nodes | Clusters | Description |
|---------------|-------|----------|-------------|
| 220 kV | 64 | 3 | Transmission substations |
| 110 kV | 137 | 2 | Primary distribution |
| 10 kV | 551 | 5 | Secondary feeders (F02, F09, F14, F31, ungrouped) |

### Network Partitioning

The system is partitioned using spectral clustering on the electrical adjacency graph:
- **220kV**: 3 clusters (C1: 31 nodes, C2: 3 nodes, C3: 30 nodes)
- **110kV**: 2 clusters (C1: 74 nodes, C2: 63 nodes)
- **10kV**: 5 clusters (4 feeder-based + 1 ungrouped)

See `system_file/746sys/blocking/partitions_summary.json` for detailed assignments.

## 📈 Training Strategy

### Self-Supervised Pretraining

1. **Masking**: Randomly mask 15-100% of node features (progressively increasing during training)
2. **Reconstruction**: Train to predict masked features from context
3. **Type-aware loss**: Only compute loss on masked positions, weighted by node type

### Progressive Training

```python
mask_prob = np.random.uniform(
    (i + start_iter_num) / total_iters,  # Increases over time
    1.0
)
```

This curriculum learning approach starts with easy examples (low mask ratio) and gradually increases difficulty.

## 🔬 Key Files Explained

| File | Purpose |
|------|---------|
| `GTransformerSparseNodalmasksAddAttnUtls.py` | Core model: DyMPN, LinearAttention, NodeGTransformer |
| `utls.py` | Data I/O: `load_H`, `load_A_sparse`, normalization utilities |
| `baseModel.py` | Training pipeline, model instantiation, loss functions |
| `new746_system_v0713.py` | System initialization, pandapower integration, validation |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{block_sparse_attn,
  title = {Block Sparse Attention for Power Systems},
  author = {Yuhong Zhu},
  year = {2026},
  url = {https://github.com/zyh1996saa/block_sparse_attn}
}
```

## 📄 License

This project is provided as-is for research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: Model weights and training logs are excluded from this repository due to size constraints. Use Git LFS or download separately if needed.
