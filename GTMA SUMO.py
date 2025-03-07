import os
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
import math
import time
from datetime import datetime


###############################################
# 1) 解析 SiouxFalls.net.xml -> 构建 edge_map
###############################################
def build_edge_map_from_net(net_xml_path):
    """
    从 net.xml 解析所有 <edge id="..." from="xxx" to="yyy" ... />
    并把 (int(xxx), int(yyy)) => edge_id 存入字典。
    如果 from/to 不是纯数字，需要做其他映射（比如 strip('N')）。
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    edge_map = {}
    for edge_elem in root.findall('edge'):
        eid = edge_elem.get('id')     # SUMO中边 id
        fr  = edge_elem.get('from')   # 例如 "22"
        to  = edge_elem.get('to')     # 例如 "15"

        # 跳过 function="internal" 那种内部虚拟边:
        if edge_elem.get('function') == 'internal':
            continue

        if eid and fr and to:
            # 如果 net.xml 中节点是纯数字，这里可直接 int()
            # 若是 "N22" 之类，需要自己做 strip('N'), int(...) 等
            try:
                fr_i = int(fr)
                to_i = int(to)
                edge_map[(fr_i, to_i)] = eid
            except ValueError:
                pass
    return edge_map


###############################################
# 2) 读取 XLSX 中的 network + OD
###############################################
def read_data(path):
    data = pd.read_excel(path)
    if 'Source' not in data.columns or 'Target' not in data.columns:
        data['Source'] = data['init_node']
        data['Target'] = data['term_node']
    return data

def add_three_col(network, od, total_time):
    # 在网络表中插入 flow, flow_time 列；在 OD 表中插入 min_cost
    network["flow"] = [np.zeros(total_time) for _ in range(len(network))]
    network["flow_time"] = [np.zeros(total_time) for _ in range(len(network))]
    od.loc[:, "min_cost"] = np.zeros(len(od))


###############################################
# 3) 原先 cost & shortest path 逻辑 (简化保留)
###############################################
def gamma_t(iteration, lambda_=1.0, xi=1.0, c=1.0):
    val = lambda_ * iteration + xi
    if val <= 1:
        val = 2
    return (1.0 / c) * math.log(val)

def psi_t(flow, capacity, psi0=1.0, eta=0.5):
    if capacity <= 0:
        return psi0
    ratio = flow / capacity
    return psi0 * (1.0 + eta * ratio)

def external_cost_for_edge(idx, network, current_time, user_flow=1.0):
    flow_with_i = network.at[idx, "flow"][current_time]
    flow_wo_i   = max(flow_with_i - user_flow, 0)
    capacity    = network.at[idx, "capacity"]
    fft         = network.at[idx, "free_flow_time"]
    if capacity is None or capacity <= 1e-9 or pd.isna(flow_with_i):
        return 0.0
    alpha = 0.15
    beta  = 4
    def bpr_t(flow):
        return fft * (1.0 + alpha * (flow / capacity)**beta)
    t_with = bpr_t(flow_with_i)
    t_wo   = bpr_t(flow_wo_i)
    return max(t_with - t_wo, 0)

def calculate_edge_cost(source, target, network, current_time, iteration,
                        psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0,
                        event=False, user_flow=1.0):
    edge_info = network[(network['init_node'] == source) & (network['term_node'] == target)]
    if edge_info.empty:
        return float('inf')

    idx = edge_info.index[0]
    ltm_time = network.at[idx, "flow_time"][current_time]
    if ltm_time <= 0:
        ltm_time = network.at[idx, "free_flow_time"]
    flow_val = network.at[idx, "flow"][current_time]
    cap_val  = network.at[idx, "capacity"]

    gamma_val = gamma_t(iteration, lambda_, xi, c)
    psi_val   = psi_t(flow_val, cap_val, psi0=1.0, eta=0.5)
    dyn_factor = 1.0 + gamma_val / (1.0 + psi_val)
    cost_base  = ltm_time * dyn_factor

    e_i = external_cost_for_edge(idx, network, current_time, user_flow)
    cost_with_e = cost_base + e_i
    if event:
        cost_with_e *= 1.2

    return cost_with_e


def get_shortestpath_with_spao(inode, tnode_list, network, index_table,
                               current_time, iteration,
                               psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0,
                               event=False, user_flow=1.0):
    """
    Dijkstra，求出 (inode->tnode) 的最短路
    """
    import heapq
    nodes = set(network['init_node']).union(set(network['term_node']))
    dist = {n: float('inf') for n in nodes}
    dist[inode] = 0
    pred = {n: None for n in nodes}

    visited = set()
    pq = [(0, inode)]
    while pq:
        cur_dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if set(tnode_list).issubset(visited):
            break
        edges_from_node = network[network['init_node'] == node]
        for idx2, row2 in edges_from_node.iterrows():
            v = row2['term_node']
            if v in visited:
                continue
            cost_edge = calculate_edge_cost(
                node, v, network, current_time, iteration,
                psi0=psi0, eta=eta, lambda_=lambda_, xi=xi, c=c,
                event=event, user_flow=user_flow
            )
            alt = cur_dist + cost_edge
            if alt < dist[v]:
                dist[v] = alt
                pred[v] = node
                heapq.heappush(pq, (alt, v))

    result = []
    for tnode in tnode_list:
        if dist[tnode] == float('inf'):
            result.append([inode, tnode, None, float('inf')])
            continue
        path = []
        cur = tnode
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path.reverse()
        if path and path[0] == inode:
            result.append([inode, tnode, path, dist[tnode]])
        else:
            result.append([inode, tnode, None, float('inf')])
    return result


###############################################
# 4) 生成 <routes>：在这里插入 <vType id="car"/>
###############################################
def build_sumo_route_file(od_assignments_by_t, network_df, edge_map, out_route_file):
    """
    在 <routes> 标签下，先声明一个 <vType id="car" ... >，
    然后再依次写 <vehicle type="car" ...>。
    """
    vehicle_id = 0
    with open(out_route_file, "w", encoding="utf-8") as f:
        f.write('<routes>\n')
        # 在 routes 文件里声明一个车种 "car"
        f.write('  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="70" color="1,0,0"/>\n\n')

        for t, lst in od_assignments_by_t.items():
            for (flow_val, path_nodes) in lst:
                # 将节点序列转成 SUMO edgeID 序列
                edge_seq = []
                for i in range(len(path_nodes)-1):
                    u = path_nodes[i]
                    v = path_nodes[i+1]
                    if (u,v) in edge_map:
                        edge_seq.append(edge_map[(u,v)])
                    else:
                        print(f"Warning: no edge map for {u}->{v}")
                        edge_seq = []
                        break
                if not edge_seq:
                    continue
                edges_str = " ".join(edge_seq)
                for _ in range(int(flow_val)):
                    vehicle_id += 1
                    f.write(f'  <vehicle id="veh_{vehicle_id}" type="car" depart="{t}" >\n')
                    f.write(f'    <route edges="{edges_str}" />\n')
                    f.write('  </vehicle>\n')

        f.write('</routes>\n')


###############################################
# 5) 调用 SUMO, 解析 tripinfo
###############################################
def run_sumo_once(sumo_cfg, route_file, output_tripinfo="myTripInfo.xml"):
    cmd = [
        "sumo",
        "-c", sumo_cfg,
        "--route-files", route_file,
        "--tripinfo-output", output_tripinfo,
        "--no-step-log", "true",
        "--time-to-teleport", "-1"
    ]
    print("Running SUMO:", " ".join(cmd))
    # 如果 sumo.exe 不在系统PATH里，可写绝对路径, 或在os.environ['PATH']里添加
    subprocess.run(cmd, check=True)
    print("SUMO finished.")

def parse_tripinfo_xml(tripinfo_path):
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()
    durations = []
    for ti in root.findall('tripinfo'):
        dur = float(ti.get('duration', 0.0))
        durations.append(dur)
    if len(durations)==0:
        return 0.0
    return sum(durations)/len(durations)


###############################################
# 6) 迭代框架: 动态分配 + 调用 SUMO
###############################################
def solve_nash_equilibrium_with_spao_SUMO(
    network, od,
    start_term_rel,
    index_table,
    total_time, max_iterations=3,
    sumo_cfg="SiouxFalls.sumocfg",
    edge_map=None,
    event=False,
    psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0,
    user_flow=1.0
):

    conv_info = []
    prev_tt = None

    # 先把 flow_time = free_flow_time
    for i in range(len(network)):
        fft = network.at[i, 'free_flow_time']
        network.at[i, 'flow_time'] = np.full(total_time, fft)
        network.at[i, 'flow']      = np.zeros(total_time)

    for iteration in range(1, max_iterations+1):
        print(f"\n=== Iteration {iteration} ===")
        start_t = time.time()

        # (A) 构造 od_assignments_by_t
        od_assignments_by_t = {}
        for t in range(total_time):
            all_paths = []
            for item in start_term_rel:
                sp_res = get_shortestpath_with_spao(
                    inode=item['start_node'],
                    tnode_list=item['term_node_list'],
                    network=network,
                    index_table=index_table,
                    current_time=t,
                    iteration=iteration,
                    psi0=psi0, eta=eta, lambda_=lambda_, xi=xi, c=c,
                    event=event, user_flow=user_flow
                )
                all_paths += sp_res

            sp_dict = {(p[0], p[1]): p for p in all_paths}
            assign_list = []
            for i2 in range(len(od)):
                st = od.iloc[i2]['init_node']
                ed = od.iloc[i2]['term_node']
                dm = od.iloc[i2]['demand']
                if (st, ed) in sp_dict:
                    pathinfo = sp_dict[(st, ed)]
                    chosen_path = pathinfo[2]
                    if chosen_path is not None:
                        assign_list.append((dm, chosen_path))
            if assign_list:
                od_assignments_by_t[t] = assign_list

        # (B) 生成 routes.xml
        route_file = f"routes_iter_{iteration}.rou.xml"
        build_sumo_route_file(od_assignments_by_t, network, edge_map, out_route_file=route_file)

        # (C) 调用 SUMO
        tripinfo_file = f"tripinfo_iter_{iteration}.xml"
        run_sumo_once(sumo_cfg, route_file, tripinfo_file)

        # (D) 解析 tripinfo
        avg_tt = parse_tripinfo_xml(tripinfo_file)
        print(f"  - Average Travel Time = {avg_tt:.2f}")

        # (E) 解析 <edgedata> 更新cost;
        for i_net in range(len(network)):
            fft = network.at[i_net, "free_flow_time"]
            cap = network.at[i_net, "capacity"]
            if cap>1e-9:
                ftest=500  # 假设每条边都500流量
                alpha=0.15
                beta=4
                cost_bpr = fft*(1+alpha*(ftest/cap)**beta)
                network.at[i_net, "flow_time"] = np.full(total_time, cost_bpr)
                network.at[i_net, "flow"]      = np.full(total_time, ftest)

        # (F) 计算 GAP
        if prev_tt is not None:
            gap = abs(avg_tt - prev_tt)/max(prev_tt,1e-9)
        else:
            gap = None
        prev_tt = avg_tt

        elapsed = time.time() - start_t
        conv_info.append({
            "Iteration": iteration,
            "AvgTT": avg_tt,
            "GAP": gap,
            "Elapsed_s": elapsed
        })
        print(f"  => GAP={gap}, iteration time={elapsed:.2f}s")

    return conv_info


###############################################
# 7) main
###############################################
if __name__ == "__main__":
    # 1) 文件名配置
    network_path = "network1.xlsx"
    od_path      = "od_data1.xlsx"
    net_xml      = "SiouxFalls.net.xml"   # 你的 net.xml
    sumo_cfg     = "SiouxFalls.sumocfg"   # 你的 sumocfg

    # 2) 读取 XLSX
    network_df = read_data(network_path)
    od_df      = read_data(od_path)
    od_df.dropna(subset=['init_node','term_node','demand'], inplace=True)
    od_df['init_node'] = od_df['init_node'].astype(int)
    od_df['term_node'] = od_df['term_node'].astype(int)
    od_df['demand'] = pd.to_numeric(od_df['demand'], errors='coerce').fillna(0)
    od_df = od_df[od_df['demand']>0].reset_index(drop=True)

    # 3) 预处理 network
    network_df['init_node'] = network_df['init_node'].astype(int)
    network_df['term_node'] = network_df['term_node'].astype(int)
    network_df['capacity']  = network_df['capacity'].astype(float)
    network_df['free_flow_time'] = network_df['free_flow_time'].astype(float)
    if 'length' not in network_df.columns:
        network_df['length'] = 1.0

    total_time = 3
    add_three_col(network_df, od_df, total_time)

    # start_term_rel
    start_term_rel = []
    for st in od_df['init_node'].unique():
        tlist = od_df[od_df['init_node']==st]['term_node'].unique().tolist()
        start_term_rel.append({"start_node": st, "term_node_list": tlist})

    # 4) 构建 (u,v)->edge_id 映射
    edge_map = build_edge_map_from_net(net_xml)

    from collections import defaultdict
    index_table = defaultdict(int)

    # 5) 迭代求解 + SUMO
    conv_info = solve_nash_equilibrium_with_spao_SUMO(
        network=network_df,
        od=od_df,
        start_term_rel=start_term_rel,
        index_table=index_table,
        total_time=total_time,
        max_iterations=3,
        sumo_cfg=sumo_cfg,
        edge_map=edge_map,
        event=False,
        psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0
    )

    # 6) 输出结果
    conv_df = pd.DataFrame(conv_info)
    conv_df.to_excel("SUMO_nash_results.xlsx", index=False)
    print("Done. See SUMO_nash_results.xlsx for iteration info.")

