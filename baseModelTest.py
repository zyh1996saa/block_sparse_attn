# In[]
"""
Refactored version of the original test_model_performance.py.
Every major stage is wrapped in a function, and informative log
messages announce the start of each step.
"""

import os
import sys
import logging
import time
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from scipy.sparse import load_npz

# --- Local utilities ---------------------------------------------------------
from config import CUDA_VISIBLE_DEVICES, WORKPATH, DATAPATH
from Utls.utls import (
    load_H,
    load_A_sparse,
    PV,
    PQ,
    Pt,
    norm_H,
    zscore_H,
    recover_H,
)

from baseModel import create_SSSGNN_multi_decoder_with_bn_complete,masked_mse_loss,pad_node_type,pad_adjacency_matrices_sparse,pad_node_features,create_NodalMask
from baseModel import create_self_supervised_data_multi_decoder

from Utls.GTransformerSparseNodalmasksAddAttnUtls import DyMPN, NodeGTransformer  # noqa: F401
from baseModel import (
    create_SSSGNN_multi_decoder_with_bn_complete,
    masked_mse_loss,
    create_NodalMask,
    create_self_supervised_data_multi_decoder,
    pad_node_type,
    pad_adjacency_matrices_sparse,
    pad_node_features,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def setup_environment() -> None:
    """Configure CUDA devices and working directory."""
    print("Setting up environment …")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.chdir(WORKPATH)

    # Ensure WORKPATH utilities are importable
    sys.path.append(os.path.join(WORKPATH, "Utls"))

# ---------------------------------------------------------------------------

def load_base_case() -> Tuple[dict, int]:
    """Load base system case and return dictionary and system size."""
    print("Loading base case …")
    case_path = os.path.join(WORKPATH, "system_file", "fc_base_case.npz")
    case_file = np.load(case_path, allow_pickle=True)
    base_case = {key: case_file[key] for key in case_file}
    sys_size = base_case["bus"].shape[0]
    print("Base case loaded with system size %d", sys_size)
    return base_case, sys_size

# ---------------------------------------------------------------------------

def build_feature_masks(base_case: dict, max_sys_size: int, sys_size: int) -> List[np.ndarray]:
    """Construct feature masks for each decoder output."""
    print("Building feature masks …")
    isPQ = PQ(base_case)
    isPV = PV(base_case)
    isPt = Pt(base_case)

    padded_isPQ = pad_node_type(isPQ, max_sys_size, sys_size)
    padded_isPV = pad_node_type(isPV, max_sys_size, sys_size)
    padded_isPt = pad_node_type(isPt, max_sys_size, sys_size)

    masks = [
        np.zeros(max_sys_size, dtype=np.float32),  # 0: No loss
        np.zeros(max_sys_size, dtype=np.float32),  # 1: No loss
        padded_isPt,  # 2: Only Pt
        np.maximum(padded_isPV, padded_isPt),  # 3: PV or Pt
        padded_isPQ,  # 4: Only PQ
        np.maximum(padded_isPQ, padded_isPV),  # 5: PQ or PV
    ]
    print("Feature masks built.")
    return masks


# ---------------------------------------------------------------------------

def build_model(max_sys_size: int, sys_size: int, num_outputs: int = 6):
    """Create the SSGNN model and load pretrained weights."""
    print("Building model …")
    # model = create_SSSGNN_multi_decoder_with_bn_complete(
    #     max_sys_size=max_sys_size,
    #     sys_size=sys_size,
    #     units=6,
    #     num_heads=8,
    #     d_model=24,
    #     blockNum=12,
    #     mlpNeuron=int(sys_size),
    #     num_outputs=num_outputs,
    # )
    model = create_SSSGNN_multi_decoder_with_bn_complete(
            max_sys_size=max_sys_size, 
            sys_size=sys_size, 
            units=6, 
            num_heads=8, 
            d_model=48,
            blockNum=2,
            mlpNeuron=int(sys_size),
            num_outputs=6
        )

    opt = tf.keras.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )
    loss_dict = {f"decoder_{i}_reshape": masked_mse_loss for i in range(num_outputs)}
    metrics_dict = {f"decoder_{i}_reshape": ["mean_absolute_error"] for i in range(num_outputs)}

    model.compile(optimizer=opt, loss=loss_dict, metrics=metrics_dict)

    # Load pretrained layer-wise weights
    print("Loading pretrained weights …")
    para_num = model.count_params()
    model_name = os.path.join(WORKPATH, "saved_models", "fc_foundation_model", f"SSGNN_{para_num}")
    for i, layer in enumerate(model.layers):
        weight_path = f"{model_name}_layer_{i}_weights.npz"
        weights = np.load(weight_path)
        layer.set_weights([weights[f"arr_{j}"] for j in range(len(weights))])
    print("Model built and weights loaded.")
    return model

# ---------------------------------------------------------------------------

def load_statistics():
    """Load saved statistics for normalization/denormalization."""
    print("Loading normalization statistics …")
    stats_dir = os.path.join(WORKPATH, "system_file")
    stats = {
        "mean_norm": np.load(os.path.join(stats_dir, "mean_norm.npy")),
        "std_norm": np.load(os.path.join(stats_dir, "std_norm.npy")),
        "mean_per_node": np.load(os.path.join(stats_dir, "mean_per_node.npy")),
        "std_per_node": np.load(os.path.join(stats_dir, "std_per_node.npy")),
        "max_per_node": np.load(os.path.join(stats_dir, "max_per_node.npy")),
        "min_per_node": np.load(os.path.join(stats_dir, "min_per_node.npy")),
    }
    print("Statistics loaded.")
    return stats


# ---------------------------------------------------------------------------

def prepare_dataset(
    sys_size: int,
    max_sys_size: int,
    total_samples: int,
    stats: dict,
    mode=0,
    start_label=0,
    end_label=0
):
    """Load and preprocess dataset for inference."""
    print("Preparing dataset …")

    # Load H and A
    if mode == 0:
        H_in = load_H(
            path=os.path.join(DATAPATH, "yantian_single_fc_dispatch"),
            start_label=0,
            end_label=total_samples,
            sys_size=sys_size,
            sample_for_each_iter=total_samples,
        )
        A_sparse_iter = load_A_sparse(
            start_label=0,
            end_label=total_samples,
            path=os.path.join(DATAPATH, "yantian_single_fc_dispatch"),
            sys_size=sys_size,
            sample_for_each_iter=total_samples,
        )
    elif mode == 1:
        H_in = load_H(
            path=os.path.join(DATAPATH, "yantian_single_fc_dispatch"),
            start_label=start_label,
            end_label=start_label+total_samples,
            sys_size=sys_size,
            sample_for_each_iter=total_samples,
        )
        A_sparse_iter = load_A_sparse(
            start_label=start_label,
            end_label=start_label+total_samples,
            path=os.path.join(DATAPATH, "yantian_single_fc_dispatch"),
            sys_size=sys_size,
            sample_for_each_iter=total_samples,
        )

    # Normalization
    H_z, _, _ = zscore_H(
        H_in,
        given_stat=True,
        mean_per_node=stats["mean_per_node"],
        std_per_node=stats["std_per_node"],
    )
    H_norm, _, _ = norm_H(
        H_z,
        given_stat=True,
        max_per_node=stats["max_per_node"],
        min_per_node=stats["min_per_node"],
    )

    # Pad features and adjacency matrices
    padded_H = pad_node_features(H_norm, max_sys_size, units=6)
    padded_A = pad_adjacency_matrices_sparse(
        A_sparse_iter,
        max_sys_size,
        start_label=0,
        sample_for_each_iter=total_samples,
    )
    nodal_masks = create_NodalMask(padded_H, max_sys_size)

    print("Dataset prepared.")
    return padded_H, padded_A, nodal_masks, H_norm


# ---------------------------------------------------------------------------

def apply_feature_masks(
    padded_H: np.ndarray,
    nodal_masks: np.ndarray,
    feature_masks: List[np.ndarray],
    isPQ: np.ndarray,
    isPV: np.ndarray,
    isPt: np.ndarray,
    mask_prob: float = 1.0,
):
    """Create self‑supervised masked inputs and corresponding targets."""
    print("Applying feature masks …")
    masked_H, original_splits, _ = create_self_supervised_data_multi_decoder(
        padded_H, isPQ, isPV, isPt, mask_prob
    )

    num_outputs = len(feature_masks)
    H_true_masked_splits = []
    batch_size = padded_H.shape[0]
    for j in range(num_outputs):
        feature_mask_batch = np.tile(feature_masks[j], (batch_size, 1))
        combined_mask = nodal_masks * feature_mask_batch
        masked_true = original_splits[j] * combined_mask[..., np.newaxis]
        H_true_masked_splits.append(masked_true)

    print("Feature masks applied.")
    return masked_H, H_true_masked_splits

# In[]
# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 环境配置
    start_time = time.time()
    setup_environment()

    base_case, sys_size = load_base_case()
    max_sys_size = sys_size
    # 特征掩码
    feature_masks = build_feature_masks(base_case, max_sys_size, sys_size)
    # 加载预训练模型
    model = build_model(max_sys_size, sys_size)
    stats = load_statistics()
    # 准备数据集
    total_samples = 2
    start_label = 0
    padded_H, padded_A, nodal_masks, H_norm = prepare_dataset(
        sys_size, max_sys_size, total_samples, stats, mode=1, start_label=0
    )
    # 构建掩码样本集
    masked_H, H_true_masked_splits = apply_feature_masks(
        padded_H,
        nodal_masks,
        feature_masks,
        PQ(base_case),
        PV(base_case),
        Pt(base_case),
    )

    # 模型推理
    print("Running inference …")
    _ = model([masked_H, padded_A, nodal_masks], training=True)  # Warm‑up
    H_pred_norm_list = model([masked_H, padded_A, nodal_masks], training=True)

    v_true, v_pred = H_pred_norm_list[-2],H_true_masked_splits[-2]
    theta_pre,theta_true = H_pred_norm_list[-1],H_true_masked_splits[-1]
    theta_pre_0 = H_pred_norm_list[-1][0,:,:]
    theta_true_0 = H_true_masked_splits[-1][0,:,:]
    H_pred_normalized = np.concatenate(H_pred_norm_list, axis=-1)  # 形状: (batch_size, max_sys_size, 6)
    H_pred_normalized_0 = H_pred_normalized[0,:,:]

    H_true_masked_splits_0 = np.concatenate(H_true_masked_splits, axis=-1)[0,:,:]

    H_pred_denormed = recover_H(H_pred_normalized, stats['mean_per_node'], stats['std_per_node'], stats['max_per_node'], stats['min_per_node'],)
    H_iter_denormed = recover_H(padded_H, stats['mean_per_node'], stats['std_per_node'], stats['max_per_node'], stats['min_per_node'],)
    H_pred_0 = H_pred_denormed[0,:,:]
    H_0 = H_iter_denormed[0,:,:]

    print(f'电压误差：{v_true[0,:,:]-v_pred[0,:,:]}')
    print(f'相角误差：{theta_true_0-theta_pre_0}')
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
# %%
