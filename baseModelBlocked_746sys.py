# In[]
import os, sys, json, time
os.chdir("/home/user/Desktop/zyh/block_sparse_attn")
import numpy as np
import pandas as pd
import tensorflow as tf

from config746sys import CUDA_VISIBLE_DEVICES, WORKPATH, DATAPATH
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.chdir(WORKPATH)
sys.path.append(WORKPATH + r'/Utls')

from Utls.utls import load_H, load_A_sparse, norm_H, zscore_H, recover_H
from Utls.utls import PV, PQ, Pt

# 分块注意力版本
from Utls.GTransformerSparseNodalmasksAddAttnUtlsBlockSparse import (
    masked_mse_loss, create_SSSGNN_multi_decoder_with_blocks
)

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
import pandapower as pp
# In[]
# ------------------------------ utils ------------------------------
def _get_bus_id_series(fc_base_net):
    bus = fc_base_net.bus
    if 'ID' in bus.columns:
        return bus['ID'].astype(str).to_list()
    if 'name' in bus.columns:
        return bus['name'].astype(str).to_list()
    return list(map(str, bus.index.to_list()))

def build_block_onehot_from_partitions(fc_base_net, vt_partitions):
    """
    根据分块结果（vt_partitions）构造节点×块的 onehot 指示矩阵。
    返回：block_onehot [N, G]，blocks_meta 列表（记录每个块包含哪些节点、所属电压层等）。
    """
    node_ids_in_order = _get_bus_id_series(fc_base_net)
    blocks_meta = []
    node2block = {}

    # 遍历各电压层的簇，组装块元数据
    for volt_key, info in vt_partitions.items():
        clusters = info.get("clusters", [])
        for ci, c in enumerate(clusters):
            nodes = [str(x) for x in c.get("nodes", [])]
            blocks_meta.append({
                "volt": volt_key,
                "cluster_id": c.get("id", f"{volt_key}-C{ci+1}"),
                "nodes": nodes
            })

    # 附加一个 UNK 块用于兜底
    blocks_meta.append({"volt": "UNK", "cluster_id": "UNK", "nodes": []})
    G = len(blocks_meta)

    # 建立 node → block 的映射（不包含 UNK）
    for bidx, meta in enumerate(blocks_meta[:-1]):
        for nid in meta["nodes"]:
            node2block[str(nid)] = bidx

    # 生成 onehot
    N = len(node_ids_in_order)
    block_onehot = np.zeros((N, G), dtype=np.float32)
    unk_idx = G - 1
    for i, nid in enumerate(node_ids_in_order):
        b = node2block.get(str(nid), unk_idx)
        block_onehot[i, b] = 1.0
    return block_onehot, blocks_meta

def get_node_type_vectors(net):
    """
    返回三个二进制向量：is_PQ, is_PV, is_Pt（长度均为 len(net.bus)）。
    互斥优先级：ext_grid(平衡) > gen(PV) > load/sgen(PQ)。
    若某个节点既不是负荷、也不是发电机和平衡节点，则默认按 PQ 处理。
    最终会校验 is_PQ + is_PV + is_Pt 是否为全 1 向量。
    """
    bus_order = list(net.bus.index)
    pos_of = {b: i for i, b in enumerate(bus_order)}
    n = len(bus_order)

    is_PQ = np.zeros(n, dtype=int)
    is_PV = np.zeros(n, dtype=int)
    is_Pt = np.zeros(n, dtype=int)

    def buses_of(attr):
        df = getattr(net, attr, None)
        if df is None or getattr(df, "empty", True):
            return set()
        return set(df.bus.values)

    pt_set = buses_of("ext_grid")          # 平衡
    pv_set = buses_of("gen")               # PV
    pq_set = buses_of("load") | buses_of("sgen")  # PQ 候选
    print(pt_set)
    # 先按优先级赋值（互斥）
    for b in pt_set:
        if b in pos_of:
            is_Pt[pos_of[b]] = 1
    for b in pv_set:
        if b in pos_of and not is_Pt[pos_of[b]]:
            is_PV[pos_of[b]] = 1
    for b in pq_set:
        if b in pos_of and not is_Pt[pos_of[b]] and not is_PV[pos_of[b]]:
            is_PQ[pos_of[b]] = 1

    # 将尚未被标记的节点归为 PQ
    unassigned = np.where((is_PQ + is_PV + is_Pt) == 0)[0]
    if unassigned.size > 0:
        is_PQ[unassigned] = 1

    # 校验互斥且覆盖：逐元素求和应为 1
    sum_vec = is_PQ + is_PV + is_Pt
    if not np.all(sum_vec == 1):
        bad_idx = np.where(sum_vec != 1)[0]
        raise ValueError(
            f"节点类型标记不一致：以下索引的和!=1 -> {bad_idx.tolist()}；"
            "请检查 net 中的 bus/元件关联是否有重复或缺失。"
        )

    return is_PQ, is_PV, is_Pt

def pad_node_type(is_type, max_sys_size, sys_size):
    padded = np.zeros(max_sys_size, dtype=np.float32)
    padded[:sys_size] = is_type
    return padded

def pad_adjacency_matrices_sparse(A_sparse, max_sys_size, start_label, sample_for_each_iter):
    """将批量稀疏邻接矩阵填充到 [sample_for_each_iter, max_sys_size, max_sys_size]，
    并把 indices 中的批次维重定位到当前 batch 内。
    """
    new_dense_shape = [sample_for_each_iter, max_sys_size, max_sys_size]
    indices = A_sparse.indices.numpy()
    values = A_sparse.values.numpy()

    # 仅保留当前 batch 的条目
    mask = (indices[:, 0] >= start_label) & (indices[:, 0] < start_label + sample_for_each_iter)
    indices = indices[mask]
    values = values[mask]

    # 将批次索引改为相对本 batch 的 [0..batch-1]
    indices[:, 0] = indices[:, 0] - start_label
    padded_A = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=new_dense_shape)
    padded_A = tf.sparse.reorder(padded_A)
    return padded_A

def pad_node_features(H, max_sys_size, units=6):
    padded_H = np.zeros((H.shape[0], max_sys_size, units), dtype=np.float32)
    for i in range(H.shape[0]):
        num_nodes = H[i].shape[0]
        padded_H[i, :num_nodes, :] = H[i]
    return padded_H

def create_NodalMask(H, max_sys_size):
    NodalMask = np.zeros((H.shape[0], max_sys_size), dtype=np.float32)
    for i in range(H.shape[0]):
        # 非零行计数作为有效节点数
        num_nodes = np.count_nonzero(np.sum(H[i], axis=-1))
        NodalMask[i, :num_nodes] = 1.0
    return NodalMask

def mask_features_by_type(features, isPQ, isPV, isPt, mask_prob=0.15):
    """按节点类型对指定特征列随机置零。"""
    batch_size, num_nodes, feature_dim = features.shape
    masks = np.zeros_like(features, dtype=bool)
    for node_type, feature_indices in zip([isPQ, isPV, isPt], [[4, 5], [3, 5], [2, 3]]):
        for i, is_type in enumerate(node_type):
            if is_type:
                masks[:, i, feature_indices] = np.random.rand(batch_size, len(feature_indices)) < mask_prob
    masked_features = features * (1 - masks)
    return masked_features, masks

def create_self_supervised_data_multi_decoder(H_iter, isPQ, isPV, isPt, mask_prob=0.15):
    masked_H_iter, masks = mask_features_by_type(H_iter, isPQ, isPV, isPt, mask_prob)
    H_iter_splits = [H_iter[:, :, i:i+1] for i in range(H_iter.shape[-1])]
    return masked_H_iter, H_iter_splits, masks

# ------------------------------ train ------------------------------
if __name__ == "__main__":
    # ---------- 基础设置 ----------
    fc_base_net = pp.from_excel(WORKPATH + "/system_file/746sys/fc_base_net.xlsx")
    sys_size = fc_base_net.bus.shape[0]
    max_sys_size = sys_size
    num_outputs = 6  # 对应 H 的 6 列

    # ===== 训练控制参数（保留原有功能） =====
    total_sample_num = 2048 * 16   # 支持多轮
    sample_for_each_iter = 2048    # 每次装载的样本数

    shuffle = False                # True 则每轮随机抽一个窗口
    retrainFlag = True             # 从头开始训练；若 False 则尝试加载权重
    firstEpoch = True              # 是否是第一个回合（配合 retrainFlag=False 使用）

    # ---------- 加载分块并构造 BlockOneHot ----------
    json_path = WORKPATH + "/system_file/746sys/blocking/partitions_summary.json"
    with open(json_path, "r", encoding="utf-8") as f:
        vt_partitions = json.load(f)
    BlockOneHot_np, blocks_meta = build_block_onehot_from_partitions(fc_base_net, vt_partitions)
    num_blocks = BlockOneHot_np.shape[1]

    # ---------- 节点类型掩码（损失用） ----------
    isPQ, isPV, isPt = get_node_type_vectors(fc_base_net)
    padded_isPQ = pad_node_type(isPQ, max_sys_size, sys_size)
    padded_isPV = pad_node_type(isPV, max_sys_size, sys_size)
    padded_isPt = pad_node_type(isPt, max_sys_size, sys_size)

    feature_masks = [
        np.zeros(max_sys_size, dtype=np.float32),                 # Feature 0: 不计算损失
        np.zeros(max_sys_size, dtype=np.float32),                 # Feature 1: 不计算损失
        padded_isPt,                                              # Feature 2: 仅 Pt
        np.maximum(padded_isPV, padded_isPt),                     # Feature 3: PV 或 Pt
        padded_isPQ,                                              # Feature 4: 仅 PQ
        np.maximum(padded_isPQ, padded_isPV)                      # Feature 5: PQ 或 PV
    ]


    # ---------- 构建 Block-Sparse 模型（增强版，包含跨块通讯） ----------
    SSGNN = create_SSSGNN_multi_decoder_with_blocks(
        max_sys_size=max_sys_size, 
        num_blocks=num_blocks, 
        units=6, 
        num_heads=8, 
        d_model=48, 
        blockNum=2, 
        mlpNeuron=int(sys_size), 
        num_outputs=num_outputs,
        num_routing_tokens=4,  # 每个块的路由令牌数量
        use_cross_block_comm=True  # 启用跨块通讯
    )

    opt = tf.keras.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )
    loss_dict = {f'dec_{i}': masked_mse_loss for i in range(num_outputs)}
    metrics_dict = {f'dec_{i}': ['MAE'] for i in range(num_outputs)}
    SSGNN.compile(optimizer=opt, loss=loss_dict, metrics=metrics_dict)

    # 用参数规模构造权重保存前缀
    para_num = SSGNN.count_params()
    model_name = WORKPATH + f'/saved_models/fc_foundation_model/SSGNN_{para_num}'

if __name__ == "__main__":
    # ---------- （可选）加载历史权重 ----------
    if not retrainFlag:
        if not firstEpoch:
            print(f"Loading Pre-trained Weights from {model_name}_layer_*_weights.npz")
            for i, layer in enumerate(SSGNN.layers):
                npy_path = model_name + f'_layer_{i}_weights.npz'
                if os.path.exists(npy_path):
                    weights = np.load(npy_path)
                    SSGNN.layers[i].set_weights([weights[f'arr_{j}'] for j in range(len(weights))])
            del weights


    
    
    # ---------- 数据标准化 ----------
    H_in = load_H(path=DATAPATH + r'/yantian752_251001',
                  start_label=0, end_label=total_sample_num,
                  sys_size=sys_size, sample_for_each_iter=total_sample_num)

    mean_per_node = np.load(WORKPATH + '/system_file/746sys/mean_per_node.npy')
    std_per_node  = np.load(WORKPATH + '/system_file/746sys/std_per_node.npy')
    max_per_node  = np.load(WORKPATH + '/system_file/746sys/max_per_node.npy')
    min_per_node  = np.load(WORKPATH + '/system_file/746sys/min_per_node.npy')

    H_z, _, _ = zscore_H(H_in, given_stat=True, mean_per_node=mean_per_node, std_per_node=std_per_node)
    H_norm, _, _ = norm_H(H_z, given_stat=True, max_per_node=max_per_node, min_per_node=min_per_node)

    # ---------- 回调 ----------
    csv_logger  = CSVLogger(WORKPATH + '/Logger/training_log.csv', append=True, separator=',')
    tensorboard = TensorBoard(log_dir=WORKPATH + '/Logger/logs/ssgnn_blocks', histogram_freq=1)
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)
    callbacks = [csv_logger, tensorboard, early_stopping]

    # ---------- 训练主循环（支持多轮与可选随机抽窗） ----------
    num_iters = int(total_sample_num / sample_for_each_iter)
    start_iter_num = 0

    for i in range(num_iters):
        iter_start_time = time.time()

        if not shuffle:
            setLabel = i
        else:
            setLabel = np.random.randint(0, num_iters)
        start_label = setLabel * sample_for_each_iter
        end_label   = (setLabel + 1) * sample_for_each_iter
        print(f"\n[Iter {i+1}/{num_iters}] 加载数据窗口: [{start_label}, {end_label})")

        # 取该窗口的特征与邻接
        H_iter = H_norm[start_label:end_label, :, :]
        A_sparse_iter = load_A_sparse(
            start_label=start_label,
            end_label=end_label,
            path=DATAPATH + '/yantian752_251001',
            sys_size=sys_size,
            sample_for_each_iter=sample_for_each_iter
        )

        # pad 到统一大小
        padded_A_iter = pad_adjacency_matrices_sparse(
            A_sparse_iter, max_sys_size,
            start_label=start_label, sample_for_each_iter=sample_for_each_iter
        )
        padded_H_iter = pad_node_features(H_iter, max_sys_size, units=6)
        NodalMasks    = create_NodalMask(padded_H_iter, max_sys_size)  # [B, N]

        # Block onehot 扩展到 batch 维
        B = padded_H_iter.shape[0]
        BlockOneHot = np.tile(BlockOneHot_np[None, :, :], (B, 1, 1)).astype(np.float32)  # [B, N, G]

        # 自监督遮盖（随迭代线性上升或固定，二择一）
        mask_prob = np.random.uniform((i + start_iter_num) / max(1, num_iters), 1.0)
        # mask_prob = 1.0  # 若需强遮盖，可直接置 1.0

        masked_H_iter, original_H_iter_splits, masks = create_self_supervised_data_multi_decoder(
            padded_H_iter, isPQ, isPV, isPt, mask_prob
        )

        # 目标掩码（按节点类型与有效节点）
        H_true_masked_splits = []
        for j in range(num_outputs):
            feature_mask = feature_masks[j]  # (N,)
            feature_mask_batch = np.tile(feature_mask, (B, 1))  # (B, N)
            combined_mask = NodalMasks * feature_mask_batch      # (B, N)
            H_true_masked = original_H_iter_splits[j] * combined_mask[..., np.newaxis]  # (B, N, 1)
            H_true_masked_splits.append(H_true_masked)

        # 训练一步
        history = SSGNN.fit(
            x=[masked_H_iter, padded_A_iter, NodalMasks, BlockOneHot],
            y=H_true_masked_splits,
            epochs=50,
            batch_size=128,
            verbose=1,
            callbacks=callbacks
        )

        # 保存当前权重（逐层保存为 npz）
        for layer_num, layer in enumerate(SSGNN.layers):
            weights = layer.get_weights()
            if len(weights) == 0:
                continue
            np.savez(model_name + f"_layer_{layer_num}_weights.npz", *weights)
        print(f"已保存第 {i} 轮训练后的权重 → {model_name}_layer_*_weights.npz")

        iter_end_time = time.time()
        print(f"本轮耗时: {iter_end_time - iter_start_time:.2f}s; mask_prob≈{mask_prob:.3f}")
# In[]
# ---------- （可选）一次前向与反归一化：仅取前 K 个样本，避免 OOM ----------
def slice_batched_sparse(A_sparse, start: int, count: int):
    """
    从 batched SparseTensor A_sparse（形状 [B, N, N]）中裁剪出
    [start, start+count) 这段 batch，返回形状 [count, N, N] 的 SparseTensor。
    """
    # 取出稀疏三元组
    idx = A_sparse.indices.numpy()     # [nnz, 3]，列 0 是 batch 维
    val = A_sparse.values.numpy()
    B, N1, N2 = A_sparse.dense_shape.numpy()

    # 仅保留目标 batch 范围内的条目
    m = (idx[:, 0] >= start) & (idx[:, 0] < start + count)
    idx = idx[m].copy()
    val = val[m].copy()

    # 将批次索引平移到 [0, count)
    idx[:, 0] = idx[:, 0] - start

    dense_shape = np.array([count, N1, N2], dtype=np.int64)
    A_slice = tf.sparse.SparseTensor(indices=idx, values=val, dense_shape=dense_shape)
    A_slice = tf.sparse.reorder(A_slice)
    return A_slice

try:
    # K = 32（如果最后一轮 batch 小于 32，就按实际 B 来）
    K = 32
    B_total = masked_H_iter.shape[0]
    K = min(K, B_total)

    # 取前 K 个样本
    masked_H_iter_K = masked_H_iter[:K]                     # [K, N, U]
    NodalMasks_K     = NodalMasks[:K]                       # [K, N]
    BlockOneHot_K    = BlockOneHot[:K]                      # [K, N, G]
    padded_A_iter_K  = slice_batched_sparse(padded_A_iter, 0, K)  # [K, N, N] (SparseTensor)

    # 前向
    H_pred_list = SSGNN([masked_H_iter_K, padded_A_iter_K, NodalMasks_K, BlockOneHot_K], training=False)

    # 取电压/相角两个通道的真值与预测（注意：H_true_masked_splits 是此前整 batch 的标签）
    v_true_K     = H_true_masked_splits[-2][:K]    # [K, N, 1]
    theta_true_K = H_true_masked_splits[-1][:K]    # [K, N, 1]
    v_pred_K     = H_pred_list[-2]                 # [K, N, 1]
    theta_pred_K = H_pred_list[-1]                 # [K, N, 1]

    # 反归一化
    H_pred_norm_K  = np.concatenate(H_pred_list, axis=-1)   # [K, N, 6]
    H_pred_denorm_K = recover_H(H_pred_norm_K, mean_per_node, std_per_node, max_per_node, min_per_node)
    H_iter_denorm_K = recover_H(padded_H_iter[:K], mean_per_node, std_per_node, max_per_node, min_per_node)

    print('电压误差（示例 K=32 batch 均值）:', np.mean(v_true_K - v_pred_K))
    print('相角误差（示例 K=32 batch 均值）:', np.mean(theta_true_K - theta_pred_K))
    
    # ---------- 有名值对比 ----------
    v_true_denorm   = H_iter_denorm_K[:, :, -2]   # 电压真实值
    v_pred_denorm   = H_pred_denorm_K[:, :, -2]   # 电压预测值
    theta_true_denorm = H_iter_denorm_K[:, :, -1] # 相角真实值
    theta_pred_denorm = H_pred_denorm_K[:, :, -1] # 相角预测值

    # 计算误差指标
    v_mae = np.mean(np.abs(v_true_denorm - v_pred_denorm))
    theta_mae = np.mean(np.abs(theta_true_denorm - theta_pred_denorm))

    print(f'电压（有名值）平均绝对误差: {v_mae:.6f}')
    print(f'相角（有名值）平均绝对误差: {theta_mae:.6f}')


except Exception as e:
    print("Sanity check（K=32）失败（可忽略）:", e)


# %%
