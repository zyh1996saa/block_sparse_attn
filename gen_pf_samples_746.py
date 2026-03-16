# In[]
import os
from config746sys import WORKPATH,DATAPATH,sub_402_bus_system_node_order_file,DistTfPath
from Utls.yantian_sys import *
from Utls.utls import get_network_matrices
import pandapower as pp
import numpy as np
from multiprocessing import Pool
import json
from new746_system_v0713 import init_feeder_net, enforce_node_order, built_ppnet_for_pfcal


def set_fc_state_with_acts(feeder_cluster,fc_base_net,actions):
    new_net = copy.deepcopy(fc_base_net)
    #if not self.pp_pf_cal_obj.is_scan_feasible_switch_states:
    #    self.pp_pf_cal_obj.scan_feasible_switch_states
    
    closed_switches = [] #断开的开关组
    open_switches = [] #打开的开关组
    changed_switches = 0 #开关动作数
    for act_num,act in enumerate(actions):
        switch_condition_in_group = feeder_cluster.feasible_switch_states[act]
        closed_switches += switch_condition_in_group['1']
        open_switches += switch_condition_in_group['0']
        
    for line in open_switches:
        line_to_delete = new_net.line[(new_net.line['from_bus'] == line.I_nd.bus) & (new_net.line['to_bus'] == line.J_nd.bus)].index
        line.closed = '0'
        if not line_to_delete.empty:
            pp.drop_lines(new_net, line_to_delete)
            changed_switches += 1
            
    for line in closed_switches:
        line.closed = '1'
        line_exists = not new_net.line[(new_net.line['from_bus'] == line.I_nd.bus) & (new_net.line['to_bus'] == line.J_nd.bus)].empty
        if not line_exists:
            pp.create_line_from_parameters(new_net,
                from_bus=line.I_nd.bus,
                to_bus=line.J_nd.bus,
                length_km=1,
                r_ohm_per_km=float(line.r),
                x_ohm_per_km=float(line.x),
                c_nf_per_km=float(line.b) ,
                max_i_ka=float(line.max_i_ka) ,
                name=line.device_type + line.name)
            changed_switches += 1
    return new_net



def generate_and_save(sample_idx):
    new_net = sample_a_new_net(feeder_cluster, fc_base_net)
    new_net_cal = built_ppnet_for_pfcal(new_net)
    #pp.runpp(new_net)
    Hi, Yi, bus_map = get_network_matrices(new_net, new_net_cal)
    for i in range(fc_base_net.line.shape[0]):
        line = fc_base_net.line.loc[i,:]
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        mapped_from = bus_map[from_bus]
        mapped_to = bus_map[to_bus]
        Y_from_to = Yi[mapped_from, mapped_to]
        #print(f"Line from {from_bus} to {to_bus} has Ybus value: {Y_from_to}")
    sparse_Yi = csr_matrix(Yi)

    #print( "sample_idx:",sample_idx, "Yi[1,15]",Yi[1,15])
    np.save(os.path.join(DATAPATH, f'yantian_752fc_20250715/H_{sample_idx}.npy'), Hi)
    save_npz(os.path.join(DATAPATH, f'yantian_752fc_20250715/Y_{sample_idx}'), sparse_Yi)
    print(f"{sample_idx}/{datasetSize}", )
    
    return Hi.shape, Yi.shape  # 不是必须，只是看情况要不要返回

if __name__ == "__main__":

    parsed_cim = CimEParser(PfDataPath)

    pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim,slack_nd='703002137')    
    #pp_pf_calculator.scan_feasible_feeders_switch_states()
    #save_feasible_feeders_switch_states(pp_pf_calculator, path=WORKPATH + '/system_file/')
    
    feeder_cluster, fc_base_net = init_feeder_net(pp_pf_calculator)
    fc_base_net_cal = built_ppnet_for_pfcal(fc_base_net)

    pp.runpp(fc_base_net_cal)
    
    new_net = sample_a_new_net(feeder_cluster, fc_base_net)
    new_net_cal = built_ppnet_for_pfcal(new_net)
    pp.runpp(new_net_cal)
    
    # for i in range(402):
    #     for j in range(402):
    #         if Yi[i,j] != 0 and i!=j:
    #             print(f"Yi[{i},{j}] = {Yi[i,j]}")
    # fc_base_case = to_ppc(fc_base_net)
    # np.savez(WORKPATH + '/system_file/fc_base_case.npz', **fc_base_case)
    # fc_base_case = np.load(WORKPATH + '/system_file/fc_base_case.npz',allow_pickle=True)
    # fc_base_case = {key: fc_base_case[key] for key in fc_base_case}
    # fc_base_net = pp.converter.from_ppc(fc_base_case, f_hz=50)
# In[]
# if __name__== "__main__":
#     # 随机采样一个子系统工况
#     new_net = sample_a_new_net(feeder_cluster, fc_base_net)
#     pp.runpp(new_net)
#     new_case_i_unref = to_ppc(new_net)
#     new_case_i = refresh_busnum(new_case_i_unref)
#     Hi, Yi, Si = case2AandH(new_case_i)

#     # 记录I_nd、J_nd在子系统中的编号
#     #I_nd.bus_in_sub_sys=150,J_nd.bus_in_sub_sys=33
#     # 记录I_nd、J_nd在总系统中的编号
#     #I_nd.bus=3671,J_nd.bus=3610

#     A = (Yi != 0).astype(int)  # 创建稀疏矩阵的二进制版本（非零为1）
#     # 将A转为稠密的numpy数组
#     A_dense = A.toarray()  # 转为稠密矩阵
#     print('I_nd.bus_in_sub_sys-J_nd.bus_in_sub_sys:',A_dense[33,150])

#     non_zero_indices = np.argwhere(A_dense != 0)
# In[]
if __name__ == "__main__":
    new_net = sample_a_new_net(feeder_cluster, fc_base_net)
    #fig = pp_pf_calculator.plotly_colored_by_vlevel_and_load(new_net,fig_size=(800, 450),bus_size=3,)

    #start_num = 2048 * 24
    #datasetSize = 2048 * 128
    start_num = 2048 * 16 # 0
    datasetSize = 2048 * 128 #2048 * 16
    # availableDataNum = start_num

    num_workers = 128

    # 创建一个进程池
    with Pool(processes=num_workers) as pool:
       results = pool.map(generate_and_save, range(start_num, datasetSize))
    # for sample_idx in range(start_num, datasetSize):
    #     Hi_shape, Yi_shape = generate_and_save(sample_idx)
# In[]
"""
校验生成的Yi矩阵中支路与feeder_cluster中支路的一致性
"""
from scipy.sparse import load_npz
import tensorflow as tf
    
if __name__ == "__main__":
    
    sample_idx = 1
    Y = load_npz(os.path.join(DATAPATH, f'yantian_single_fc_20250703/Y_{sample_idx}.npz'))
    
    # 直接处理稀疏矩阵，避免将其转为稠密矩阵
    A = (Y != 0).astype(int)  # 创建稀疏矩阵的二进制版本（非零为1）
    # 将A转为稠密的numpy数组
    A_dense = A.toarray()  # 转为稠密矩阵
    # 记录A_dense中非零元素的索引
    non_zero_indices = np.argwhere(A_dense != 0)
    # 排除non_zero_indices中对角元素的索引
    non_zero_indices = non_zero_indices[non_zero_indices[:, 0] != non_zero_indices[:, 1]]
    # 逐一校核non_zero_indices中的支路是否存在于feeder_cluster中
    for i, j in non_zero_indices:
        # 获取节点编号
        if i == len(feeder_cluster.nodes) or j == len(feeder_cluster.nodes):
            print(f"支路为{i}和{j}中存在虚拟连接点，跳过校验。")
            continue
        i_bus = feeder_cluster.nodes[i].bus
        j_bus = feeder_cluster.nodes[j].bus
        line_exist = False
        for line in feeder_cluster.lines:
            if (line.I_nd.bus == i_bus and line.J_nd.bus == j_bus) or (line.I_nd.bus == j_bus and line.J_nd.bus == i_bus):
                #print(f"Line {line.name} exists between nodes {i} and {j}.")
                line_exist = True
                break
        if not line_exist:
            # 如果没有找到对应的支路，打印信息
            print(f"Line does not exist between nodes {i} and {j}.")
            


    
    



# %%
