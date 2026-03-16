# test_simple.py
import os, sys, json
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
from Utls.GTransformerSparseNodalmasksAddAttnUtlsBlockSparse import create_SSSGNN_multi_decoder_with_blocks

import pandapower as pp

def main():
    print("=== 单样本模型测试 ===")
    
    # ---------- 基础设置 ----------
    fc_base_net = pp.from_excel(WORKPATH + "/system_file/746sys/fc_base_net.xlsx")
    sys_size = fc_base_net.bus.shape[0]
    max_sys_size = sys_size
    num_outputs = 6

    # ---------- 加载分块信息 ----------
    json_path = WORKPATH + "/system_file/746sys/blocking/partitions_summary.json"
    with open(json_path, "r", encoding="utf-8") as f:
        vt_partitions = json.load(f)
    
    def build_block_onehot_from_partitions(fc_base_net, vt_partitions):
        bus = fc_base_net.bus
        node_ids_in_order = list(map(str, bus.index.to_list()))
        blocks_meta = []
        node2block = {}

        for volt_key, info in vt_partitions.items():
            clusters = info.get("clusters", [])
            for ci, c in enumerate(clusters):
                nodes = [str(x) for x in c.get("nodes", [])]
                blocks_meta.append({
                    "volt": volt_key,
                    "cluster_id": c.get("id", f"{volt_key}-C{ci+1}"),
                    "nodes": nodes
                })

        blocks_meta.append({"volt": "UNK", "cluster_id": "UNK", "nodes": []})
        G = len(blocks_meta)

        for bidx, meta in enumerate(blocks_meta[:-1]):
            for nid in meta["nodes"]:
                node2block[str(nid)] = bidx

        N = len(node_ids_in_order)
        block_onehot = np.zeros((N, G), dtype=np.float32)
        unk_idx = G - 1
        for i, nid in enumerate(node_ids_in_order):
            b = node2block.get(str(nid), unk_idx)
            block_onehot[i, b] = 1.0
        return block_onehot, blocks_meta

    BlockOneHot_np, blocks_meta = build_block_onehot_from_partitions(fc_base_net, vt_partitions)
    num_blocks = BlockOneHot_np.shape[1]

    # ---------- 构建模型 ----------
    print("构建模型...")
    SSGNN = create_SSSGNN_multi_decoder_with_blocks(
        max_sys_size=max_sys_size, 
        num_blocks=num_blocks, 
        units=6, 
        num_heads=8, 
        d_model=48, 
        blockNum=2, 
        mlpNeuron=int(sys_size), 
        num_outputs=num_outputs,
        num_routing_tokens=4,
        use_cross_block_comm=True
    )

    # 加载权重
    para_num = SSGNN.count_params()
    model_name = WORKPATH + f'/saved_models/fc_foundation_model/SSGNN_{para_num}'
    
    weights_loaded = False
    for i in range(len(SSGNN.layers)):
        npy_path = model_name + f'_layer_{i}_weights.npz'
        if os.path.exists(npy_path):
            weights = np.load(npy_path)
            SSGNN.layers[i].set_weights([weights[f'arr_{j}'] for j in range(len(weights))])
            weights_loaded = True
    
    if not weights_loaded:
        print("警告：未找到权重文件")
        return

    # ---------- 加载单个测试样本 ----------
    print("加载测试样本...")
    sample_index = 100  # 可以修改这个索引来测试不同的样本
    
    H_test = load_H(
        path=DATAPATH + r'/yantian752_251001',
        start_label=sample_index,
        end_label=sample_index + 1,  # 只加载一个样本
        sys_size=sys_size,
        sample_for_each_iter=1
    )

    # 加载标准化参数
    mean_per_node = np.load(WORKPATH + '/system_file/746sys/mean_per_node.npy')
    std_per_node  = np.load(WORKPATH + '/system_file/746sys/std_per_node.npy')
    max_per_node  = np.load(WORKPATH + '/system_file/746sys/max_per_node.npy')
    min_per_node  = np.load(WORKPATH + '/system_file/746sys/min_per_node.npy')

    # 标准化
    H_z_test, _, _ = zscore_H(H_test, given_stat=True, mean_per_node=mean_per_node, std_per_node=std_per_node)
    H_norm_test, _, _ = norm_H(H_z_test, given_stat=True, max_per_node=max_per_node, min_per_node=min_per_node)

    # 加载邻接矩阵
    A_sparse_test = load_A_sparse(
        start_label=sample_index,
        end_label=sample_index + 1,
        path=DATAPATH + '/yantian752_251001',
        sys_size=sys_size,
        sample_for_each_iter=1
    )

    # ---------- 数据预处理 ----------
    # 填充到统一大小
    def pad_adjacency_matrices_sparse(A_sparse, max_sys_size, start_label, sample_for_each_iter):
        new_dense_shape = [sample_for_each_iter, max_sys_size, max_sys_size]
        indices = A_sparse.indices.numpy()
        values = A_sparse.values.numpy()

        mask = (indices[:, 0] >= start_label) & (indices[:, 0] < start_label + sample_for_each_iter)
        indices = indices[mask]
        values = values[mask]

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
            num_nodes = np.count_nonzero(np.sum(H[i], axis=-1))
            NodalMask[i, :num_nodes] = 1.0
        return NodalMask

    padded_A_test = pad_adjacency_matrices_sparse(
        A_sparse_test, max_sys_size,
        start_label=sample_index, sample_for_each_iter=1
    )
    padded_H_test = pad_node_features(H_norm_test, max_sys_size, units=6)
    NodalMasks_test = create_NodalMask(padded_H_test, max_sys_size)

    # Block onehot 扩展到 batch 维
    BlockOneHot_test = np.tile(BlockOneHot_np[None, :, :], (1, 1, 1)).astype(np.float32)

    # ---------- 模型预测 ----------
    print("进行模型预测...")
    predictions = SSGNN(
        [padded_H_test, padded_A_test, NodalMasks_test, BlockOneHot_test], 
        training=False
    )

    # ---------- 反归一化 ----------
    H_pred_norm = np.concatenate(predictions, axis=-1)
    H_pred_denorm = recover_H(H_pred_norm, mean_per_node, std_per_node, max_per_node, min_per_node)
    H_true_denorm = recover_H(padded_H_test, mean_per_node, std_per_node, max_per_node, min_per_node)

    # ---------- 结果分析 ----------
    print("\n" + "="*100)
    print(f"样本 #{sample_index} 有名值对比分析")
    print("="*100)
    
    feature_names = ['P_inj (MW)', 'Q_inj (MVar)', 'P_gen (MW)', 'Q_gen (MVar)', 'V_mag (p.u.)', 'V_angle (rad)']
    
    # 打印前10个节点的详细对比
    print("\n前10个节点的详细对比:")
    print("-"*120)
    print(f"{'节点':<4} {'特征':<15} {'真实值':<12} {'预测值':<12} {'绝对误差':<12} {'相对误差(%)':<12}")
    print("-"*120)
    
    for node_idx in range(min(10, sys_size)):
        print(f"{node_idx:<4} {'':<15} {'':<12} {'':<12} {'':<12} {'':<12}")
        for feat_idx, feat_name in enumerate(feature_names):
            true_val = H_true_denorm[0, node_idx, feat_idx]
            pred_val = H_pred_denorm[0, node_idx, feat_idx]
            abs_error = pred_val - true_val
            rel_error = (abs_error / (abs(true_val) + 1e-8)) * 100
            
            print(f"{'':<4} {feat_name:<15} {true_val:>10.6f} {pred_val:>10.6f} {abs_error:>11.6f} {rel_error:>11.2f}%")
        print("-"*120)

    # 统计所有节点的误差
    print(f"\n所有节点误差统计 ({sys_size}个节点):")
    print("-"*80)
    print(f"{'特征':<15} {'平均绝对误差':<15} {'最大绝对误差':<15} {'平均相对误差(%)':<15}")
    print("-"*80)
    
    for feat_idx, feat_name in enumerate(feature_names):
        true_vals = H_true_denorm[0, :, feat_idx]
        pred_vals = H_pred_denorm[0, :, feat_idx]
        abs_errors = np.abs(pred_vals - true_vals)
        rel_errors = np.abs((pred_vals - true_vals) / (np.abs(true_vals) + 1e-8)) * 100
        
        mae = np.mean(abs_errors)
        max_ae = np.max(abs_errors)
        mean_re = np.mean(rel_errors)
        
        print(f"{feat_name:<15} {mae:>13.6f} {max_ae:>13.6f} {mean_re:>13.2f}%")

    # 重点关注电压和相角
    print(f"\n重点关注 - 电压和相角:")
    print("-"*80)
    v_true = H_true_denorm[0, :, 4]  # 电压
    v_pred = H_pred_denorm[0, :, 4]
    v_errors = np.abs(v_pred - v_true)
    
    theta_true = H_true_denorm[0, :, 5]  # 相角
    theta_pred = H_pred_denorm[0, :, 5]
    theta_errors = np.abs(theta_pred - theta_true)
    
    print(f"电压平均绝对误差: {np.mean(v_errors):.6f} p.u.")
    print(f"电压最大绝对误差: {np.max(v_errors):.6f} p.u.")
    print(f"相角平均绝对误差: {np.mean(theta_errors):.6f} rad")
    print(f"相角最大绝对误差: {np.max(theta_errors):.6f} rad")

    # 找出误差最大的节点
    print(f"\n误差最大的节点:")
    print("-"*80)
    max_v_error_node = np.argmax(v_errors)
    max_theta_error_node = np.argmax(theta_errors)
    
    print(f"电压误差最大 - 节点 {max_v_error_node}:")
    print(f"  真实值: {v_true[max_v_error_node]:.6f} p.u., 预测值: {v_pred[max_v_error_node]:.6f} p.u., 误差: {v_errors[max_v_error_node]:.6f} p.u.")
    
    print(f"相角误差最大 - 节点 {max_theta_error_node}:")
    print(f"  真实值: {theta_true[max_theta_error_node]:.6f} rad, 预测值: {theta_pred[max_theta_error_node]:.6f} rad, 误差: {theta_errors[max_theta_error_node]:.6f} rad")

    print("\n" + "="*100)
    print("测试完成！")

if __name__ == "__main__":
    main()