# In[]
import os
from config746sys import CUDA_VISIBLE_DEVICES,WORKPATH,DATAPATH
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 让编号与 nvidia‑smi 一致
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES  
os.chdir(WORKPATH)
import sys
sys.path.append(WORKPATH + r'/Utls')

#from Utls.utls import PQ,PV,Pt,load_A,refresh_busnum
from Utls.utls import load_H, load_A_sparse
from Utls.GTransformerSparseNodalmasksAddAttnUtls import DyMPN,NodeGTransformer
from Utls.utls import load_H, norm_H, zscore_H, recover_H
from Utls.utls import PV, PQ, Pt

import numpy as np
import pandas as pd
import copy


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
from tensorflow.keras import layers, activations
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from scipy.sparse import load_npz
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.losses import MeanSquaredError

import time
import pandapower as pp

def pad_node_type(is_type, max_sys_size, sys_size):
    padded = np.zeros(max_sys_size, dtype=np.float32)
    padded[:sys_size] = is_type
    return padded

def create_SSSGNN_multi_decoder_with_bn_complete(max_sys_size, sys_size, units=6, num_heads=8, d_model=24, blockNum=4, mlpNeuron=24000, num_outputs=6):
    hin = Input(shape=(max_sys_size, units), dtype=tf.float32, name='input1')
    NodalMask = Input(shape=(max_sys_size,), dtype=tf.float32, name='nodalMask')
    A_sparse_input = Input(shape=(max_sys_size, max_sys_size), dtype=tf.float32, sparse=True, name='input2')
    A_sparse = Lambda(lambda x: tf.stop_gradient(x))(A_sparse_input)

    # 共享编码器部分
    h_embeded = Dense(d_model, activation='relu', name='embedding_layer')(hin)
    h_embeded = BatchNormalization(name='embedding_bn')(h_embeded)

    # 使用 NodeGTransformer 处理节点特征和邻接矩阵
    h_shared = NodeGTransformer(units=d_model, num_heads=num_heads)([h_embeded, A_sparse], NodalMask=NodalMask)
    if blockNum > 1:
        for block_iter in range(blockNum-1):
            h_shared = NodeGTransformer(units=d_model, num_heads=num_heads)([h_shared, A_sparse], NodalMask=NodalMask)

    h_deEmbeded = Dense(units, activation='relu', name='deEmbedding_layer')(h_shared)
    h_deEmbeded = BatchNormalization(name='deEmbedding_bn')(h_deEmbeded)

    # 将初始特征和处理后的特征进行拼接
    h_concat = Concatenate()([h_deEmbeded, hin])
    # 应用掩码，将填充节点的特征置零
    h_masked = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))([h_concat, NodalMask])

    h_masked_flat = Flatten()(h_masked)  # [batch_size, max_sys_size * (d_model + units)]

    # 为每个输出创建独立的MLP和解码器
    decoder_outputs = []
    for i in range(num_outputs):
        # 定义该解码器的独立MLP层
        mlp_hidden = Dense(mlpNeuron, activation='relu', name=f'decoder_{i}_dense1')(h_masked_flat)
        mlp_hidden = Dense(mlpNeuron, activation='relu', name=f'decoder_{i}_dense2')(mlp_hidden)
        # 定义解码器输出层
        decoder = Dense(max_sys_size * 1, name=f'decoder_{i}_dense_output')(mlp_hidden)
        decoder = Reshape((max_sys_size, 1), name=f'dec_{i}')(decoder)
        decoder_outputs.append(decoder)

    # 创建和编译模型，输出多个预测值
    model = Model([hin, A_sparse_input, NodalMask], decoder_outputs)
    return model

# 定义自定义损失函数
def masked_mse_loss(y_true, y_pred):
    """
    自定义的掩码均方误差损失函数。

    参数:
    - y_true: [batch_size, max_sys_size, units]，真实值已根据掩码被置零
    - y_pred: [batch_size, max_sys_size, units]，预测值

    返回:
    - masked_mse: 标量，掩码后的均方误差
    """
    # 获取掩码：mask = 1.0 if y_true 不为0，else 0.0
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)  # [batch_size, max_sys_size, units]

    # 计算平方误差
    mse = tf.square(y_true - y_pred)  # [batch_size, max_sys_size, units]

    # 计算每个节点的平均MSE
    mse_per_node = tf.reduce_mean(mse, axis=-1)  # [batch_size, max_sys_size]

    # 应用掩码
    masked_mse = mse_per_node * tf.reduce_max(mask, axis=-1)  # [batch_size, max_sys_size]
    # 使用 tf.reduce_max 来确保掩码为1的节点被正确处理

    # 计算平均损失，避免除以零
    return tf.reduce_sum(masked_mse) / (tf.reduce_sum(tf.reduce_max(mask, axis=-1)) + 1e-8)

def pad_adjacency_matrices_sparse(A_sparse, max_sys_size, start_label, sample_for_each_iter):
    """
    将批量稀疏邻接矩阵填充到指定的最大系统大小，并调整批次索引。

    参数：
    - A_sparse: tf.sparse.SparseTensor，形状为 [total_sample_num, sys_size, sys_size]
    - max_sys_size: int，填充后的系统大小
    - start_label: int，当前批次的起始索引
    - sample_for_each_iter: int，当前批次的样本数量

    返回：
    - padded_A: tf.sparse.SparseTensor，形状为 [sample_for_each_iter, max_sys_size, max_sys_size]
    """
    # 获取原始 dense_shape
    original_dense_shape = A_sparse.dense_shape.numpy()
    total_samples, sys_size_1, sys_size_2 = original_dense_shape

    # 创建新的 dense_shape
    new_dense_shape = [sample_for_each_iter, max_sys_size, max_sys_size]

    # 调整批次索引
    indices = A_sparse.indices.numpy()
    # 仅选择当前批次的索引
    mask = (indices[:, 0] >= start_label) & (indices[:, 0] < start_label + sample_for_each_iter)
    indices = indices[mask]
    values = A_sparse.values.numpy()[mask]

    # 调整批次索引为相对于当前批次
    indices[:, 0] = indices[:, 0] - start_label

    padded_A = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=new_dense_shape
    )
    padded_A = tf.sparse.reorder(padded_A)

    return padded_A

def pad_node_features(H, max_sys_size, units=6):
    """
    将节点特征填充到指定的最大系统大小。
    
    参数:
    - H: 原始特征，形状为 (batch_size, num_nodes, units)
    - max_sys_size: int，填充后的系统大小
    - units: int，特征维度
    
    返回:
    - padded_H: 填充后的特征，形状为 (batch_size, max_sys_size, units)
    """
    padded_H = np.zeros((H.shape[0], max_sys_size, units), dtype=np.float32)
    for i in range(H.shape[0]):
        num_nodes = H[i].shape[0]
        padded_H[i, :num_nodes, :] = H[i]
    return padded_H

def create_NodalMask(H, max_sys_size):
    NodalMask = np.zeros((H.shape[0], max_sys_size), dtype=np.float32)
    for i in range(H.shape[0]):
        num_nodes = np.count_nonzero(np.sum(H[i], axis=-1))
        NodalMask[i, :num_nodes+1] = 1.0
    return NodalMask

def mask_features_by_type(features, isPQ, isPV, isPt, mask_prob=0.15):
    """
    根据节点类型随机掩盖输入特征的一部分。

    参数:
    - features: 原始特征，维度为(batch_size, num_nodes, feature_dim)
    - isPQ: 标记每个节点是否为PQ节点的向量，维度为(num_nodes,)
    - isPV: 标记每个节点是否为PV节点的向量，维度为(num_nodes,)
    - isPt: 标记每个节点是否为Pt节点的向量，维度为(num_nodes,)
    - mask_prob: 每个特征被掩盖的概率

    返回值:
    - masked_features: 掩盖后的特征
    - masks: 实际掩盖的位置标志，用于损失计算
    """
    batch_size, num_nodes, feature_dim = features.shape
    # 初始化掩码为全0，表示不掩盖
    masks = np.zeros_like(features, dtype=bool)
    
    # 根据节点类型设置掩码
    for node_type, feature_indices in zip([isPQ, isPV, isPt], [[4, 5], [3, 5], [2, 3]]):
        for i, is_type in enumerate(node_type):
            if is_type:  # 如果节点属于当前类型
                # 根据mask_prob随机决定是否掩盖对应特征
                masks[:, i, feature_indices] = np.random.rand(batch_size, len(feature_indices)) < mask_prob
    
    # 生成掩盖后的特征，掩盖部分设为0
    masked_features = features * (1 - masks)
    return masked_features, masks

def create_self_supervised_data_multi_decoder(H_iter, isPQ, isPV, isPt, mask_prob=0.15,):
    """
    创建自监督任务的输入数据和多个输出数据。
    
    参数:
    - H_iter: 原始数据，维度为(batch_size, num_nodes, feature_dim)
    - mask_prob: 掩码概率
    
    返回值:
    - masked_H_iter: 掩盖后的输入特征
    - H_iter_splits: 分割后的原始未掩盖的特征，作为多个目标输出
    - masks: 掩码，指示哪些特征被掩盖，用于损失计算时忽略未掩盖的特征
    """
    masked_H_iter, masks = mask_features_by_type(H_iter, isPQ, isPV, isPt, mask_prob)
    
    # 分割 H_iter 为多个目标，每个目标对应一个特征列
    H_iter_splits = [H_iter[:, :, i:i+1] for i in range(H_iter.shape[-1])]
    
    return masked_H_iter, H_iter_splits, masks

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

if __name__=="__main__":
    fc_base_net = pp.from_excel(WORKPATH + "/system_file/746sys/fc_base_net.xlsx")
    sys_size = fc_base_net.bus.shape[0]
    max_sys_size = sys_size
    num_outputs = 6  # 对应 H 的 6 列
    #total_sample_num = 50000
    total_sample_num = 2048 * 12
    sample_for_each_iter = 2048 

    shuffle = False #True
    retrainFlag = True # 从头开始训练
    firstEpoch = True # 是否是第一个回合

    isPQ, isPV, isPt = get_node_type_vectors(fc_base_net)
    # 填充
    padded_isPQ = pad_node_type(isPQ, max_sys_size, sys_size)
    padded_isPV = pad_node_type(isPV, max_sys_size, sys_size)
    padded_isPt = pad_node_type(isPt, max_sys_size, sys_size)

    feature_masks = [
        np.zeros(max_sys_size, dtype=np.float32),                  # Feature 0: 不计算损失
        np.zeros(max_sys_size, dtype=np.float32),                  # Feature 1: 不计算损失
        padded_isPt,                                               # Feature 2: 仅Pt
        np.maximum(padded_isPV, padded_isPt),                      # Feature 3: PV或Pt
        padded_isPQ,                                               # Feature 4: 仅PQ
        np.maximum(padded_isPQ, padded_isPV)                       # Feature 5: PQ或PV
    ]

    # SSGNN_with_bn = create_SSSGNN_multi_decoder_with_bn_complete(
    #         max_sys_size=max_sys_size, 
    #         sys_size=sys_size, 
    #         units=6, 
    #         num_heads=8, 
    #         d_model=24,
    #         blockNum=12,
    #         mlpNeuron=int(sys_size),
    #         num_outputs=6
    #     )
    SSGNN_with_bn = create_SSSGNN_multi_decoder_with_bn_complete(
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
    # 编译模型，定义损失和优化器
    loss_dict = {f'dec_{i}': masked_mse_loss for i in range(num_outputs)}
    metrics_dict = {f'dec_{i}': ['MAE'] for i in range(num_outputs)}

    SSGNN_with_bn.compile(optimizer=opt, loss=loss_dict, metrics=metrics_dict)


    para_num = SSGNN_with_bn.count_params()
    model_name = WORKPATH+r'/saved_models/fc_foundation_model/SSGNN_%s'%para_num
    # 加载历史模型权重（可选）
    if not retrainFlag:
        if not firstEpoch: 
            
            print("Loading Pre-trained Weights from %s"%(WORKPATH+r'/saved_models/fc_foundation_model'))   
            for i, layer in enumerate(SSGNN_with_bn.layers):   
                weights = np.load(model_name+f'_layer_{i}_weights.npz')
                layer.set_weights([weights[f'arr_{j}'] for j in range(len(weights))])
            del weights

    # 加载全部数据
    H_in = load_H(path=DATAPATH+r'/yantian752_251001', start_label=0, end_label=total_sample_num, sys_size=sys_size, 
            sample_for_each_iter=total_sample_num)
    # 加载已保存的统计数据
    #mean_norm = np.load(WORKPATH+'/system_file/mean_norm.npy')
    #std_norm = np.load(WORKPATH+'/system_file/std_norm.npy')
    mean_per_node = np.load(WORKPATH+'/system_file/746sys/mean_per_node.npy')
    std_per_node = np.load(WORKPATH+'/system_file/746sys/std_per_node.npy')
    max_per_node = np.load(WORKPATH+'/system_file/746sys/max_per_node.npy')
    min_per_node = np.load(WORKPATH+'/system_file/746sys/min_per_node.npy')
    # 数据标准化
    H_zscored, mean_per_node, std_per_node = zscore_H(H_in,given_stat=True, mean_per_node=mean_per_node, std_per_node=std_per_node)
    H_norm, max_per_node, min_per_node = norm_H(H_zscored, given_stat=True, max_per_node=max_per_node, min_per_node=min_per_node)


    # 创建回调实例
    csv_logger = CSVLogger(WORKPATH+'/Logger/training_log.csv', append=True, separator=',')
    tensorboard = TensorBoard(log_dir=WORKPATH+'/Logger/logs/ssgnn_training', histogram_freq=1)
    early_stopping = EarlyStopping(
        monitor='loss',  # 或根据需要选择其他指标
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    callbacks = [csv_logger, tensorboard, early_stopping]
# In[]
if __name__=="__main__":
    """
    开始训练
    """
    start_iter_num = 0
    # 主循环
    for i in range(int(total_sample_num / sample_for_each_iter)):
        iter_start_time = time.time()

        if not shuffle:
            print(f"正加载第{i}批数据")
            H_iter = H_norm[i * sample_for_each_iter : (i + 1) * sample_for_each_iter, :, :]
            A_sparse_iter = load_A_sparse(
                start_label=i * sample_for_each_iter,
                end_label=(i + 1) * sample_for_each_iter,
                path=DATAPATH+'/yantian752_251001',
                sys_size=sys_size,
                sample_for_each_iter=sample_for_each_iter
            )
        else:
            setLabel = np.random.randint(0, int(total_sample_num / sample_for_each_iter))
            #setLabel = 0
            print(f"正加载第{i}批数据，数据范围{setLabel * sample_for_each_iter}-{(setLabel + 1) * sample_for_each_iter}")
            H_iter = H_norm[setLabel * sample_for_each_iter : (setLabel + 1) * sample_for_each_iter, :, :]
            A_sparse_iter = load_A_sparse(
                start_label=setLabel * sample_for_each_iter,
                end_label=(setLabel + 1) * sample_for_each_iter,
                path=DATAPATH+'/yantian752_251001',
                sys_size=sys_size,
                sample_for_each_iter=sample_for_each_iter
            )

        # 调用调整后的 pad_adjacency_matrices_sparse
        padded_A_iter = pad_adjacency_matrices_sparse(
            A_sparse_iter,
            max_sys_size,
            start_label=i * sample_for_each_iter if not shuffle else setLabel * sample_for_each_iter,
            sample_for_each_iter=sample_for_each_iter
        )
        # 将节点特征填充至指定大小
        padded_H_iter = pad_node_features(H_iter, max_sys_size, units=6)
        # 创建节点掩码矩阵
        NodalMasks = create_NodalMask(padded_H_iter, max_sys_size)  # [batch_size, max_sys_size]
        
        mask_prob = np.random.uniform(
            (i + start_iter_num) / int(total_sample_num / sample_for_each_iter), 1
        )
        #mask_prob = 1
        #print(f"掩码率：{mask_prob}")

        masked_H_iter, original_H_iter_splits, masks = create_self_supervised_data_multi_decoder(
            padded_H_iter, isPQ, isPV, isPt, mask_prob
        )

        # 应用特征掩码到 y_true
        H_true_masked_splits = []
        for j in range(num_outputs):
            feature_mask = feature_masks[j]  # shape (max_sys_size,)
            # 将特征掩码扩展到批量大小
            feature_mask_batch = np.tile(feature_mask, (sample_for_each_iter, 1))  # (batch_size, max_sys_size)
            # 结合节点掩码
            combined_mask = NodalMasks * feature_mask_batch  # (batch_size, max_sys_size)
            # 应用掩码到 y_true
            H_true_masked = original_H_iter_splits[j] * combined_mask[..., np.newaxis]  # (batch_size, max_sys_size, 1)
            H_true_masked_splits.append(H_true_masked)

        # 统一集中训练模型
        SSGNN_with_bn.fit(
            x=[masked_H_iter, padded_A_iter, NodalMasks],
            y=H_true_masked_splits,
            epochs=50,  
            batch_size=128,
            verbose=1,
            callbacks=callbacks  # 添加回调
        )
        
        # 保存权重（可选）
        for layer_num, layer in enumerate(SSGNN_with_bn.layers):
            weights = layer.get_weights()  # 获取层的参数
            para_num = SSGNN_with_bn.count_params()
            np.savez(
                model_name+f"_layer_{layer_num}_weights.npz",
                    *weights
                )
        print(f"已保存第{i}批数据训练后的权重")

        iter_end_time = time.time()
        print(f"第{i}批数据训练耗时{iter_end_time - iter_start_time}s")
        #del masked_H_iter, padded_A_iter, NodalMasks, H_true_masked_splits, original_H_iter_splits, masks

    #H_pred_normalized_list = SSGNN_with_bn.predict([masked_H_iter, padded_A_iter, NodalMasks])
    H_pred_normalized_list = SSGNN_with_bn([masked_H_iter, padded_A_iter, NodalMasks], training=True)
    v_true, v_pred = H_pred_normalized_list[-2],H_true_masked_splits[-2]
    theta_pre,theta_true = H_pred_normalized_list[-1],H_true_masked_splits[-1]
    theta_pre_0 = H_pred_normalized_list[-1][0,:,:]
    theta_true_0 = H_true_masked_splits[-1][0,:,:]
    H_pred_normalized = np.concatenate(H_pred_normalized_list, axis=-1)  # 形状: (batch_size, max_sys_size, 6)
    H_pred_normalized_0 = H_pred_normalized[0,:,:]
    H_iter_0 = H_iter[0,:,:]
    H_true_masked_splits_0 = np.concatenate(H_true_masked_splits, axis=-1)[0,:,:]

    H_pred_denormed = recover_H(H_pred_normalized, mean_per_node, std_per_node, max_per_node, min_per_node)
    H_iter_denormed = recover_H(H_iter,            mean_per_node, std_per_node, max_per_node, min_per_node)
    H_pred_0 = H_pred_denormed[0,:,:]
    H_0 = H_iter_denormed[0,:,:]

    print(f'电压误差：{v_true[0,:,:]-v_pred[0,:,:]}')
    print(f'相角误差：{theta_true_0-theta_pre_0}')
# %%
