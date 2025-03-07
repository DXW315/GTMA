import pandas as pd
import numpy as np
import os
import math
import networkx as nx
import pickle
import time
from datetime import datetime

#########################################
# 1. 数据读写
#########################################
def read_data(path):
    data = pd.read_excel(path)
    if 'Source' not in data.columns or 'Target' not in data.columns:
        data['Source'] = data['init_node']
        data['Target'] = data['term_node']
    return data

def read_node_data(path):
    data = pd.read_csv(path)
    return data

output_dir = 'traffic_flow_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#########################################
# 2. GAP计算函数
#########################################
def calculate_gap(current_flows, previous_flows):
    total_difference = np.abs(current_flows - previous_flows).sum()
    total_previous = np.abs(previous_flows).sum()
    if total_previous == 0:
        return 0
    return total_difference / total_previous


#########################################
# 3. 保留的数据初始化
#########################################
def add_three_col(network, od, total_time):
    network["flow"] = [np.zeros(total_time) for _ in range(len(network))]
    network["auxiliary_flow"] = [np.zeros(total_time) for _ in range(len(network))]
    network["flow_time"] = [np.zeros(total_time) for _ in range(len(network))]
    od.loc[:, "min_cost"] = np.zeros(len(od))

def construct_index_table(network):
    init_node_list = sorted(list(set(network["init_node"])))
    index_table = {}
    for node in init_node_list:
        matching_indices = network.index[network["init_node"] == node].tolist()
        if matching_indices:
            index_table[node] = matching_indices[0]
        else:
            print(f"Warning: init_node {node} not found in network.")
    return index_table

#########################################
# 4. 时间变学习率 γ(t) 与 拥堵敏感因子 ψ(t)
#########################################
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

#########################################
# [新增] 外部成本函数：对单条边的 E_i
#########################################
def external_cost_for_edge(idx, network, current_time, user_flow=1.0):
    """
    给定边索引 idx, 返回当前时段对“一个单位流量”的外部成本 E_i。
    """
    flow_with_i = network.at[idx, "flow"][current_time]
    flow_wo_i   = max(flow_with_i - user_flow, 0)
    capacity    = network.at[idx, "capacity"]
    fft         = network.at[idx, "free_flow_time"]

    # 如果容量<=0 或无效, 则外部成本记为0
    if capacity is None or capacity <= 1e-9 or pd.isna(flow_with_i):
        return 0.0

    alpha = 0.15
    beta  = 4

    def bpr_t(flow):
        return fft * (1.0 + alpha * (flow / capacity) ** beta)

    t_with  = bpr_t(flow_with_i)
    t_without = bpr_t(flow_wo_i)

    e_i = max(t_with - t_without, 0)
    return e_i

#########################################
# 5. calculate_edge_cost: 引入外部事件与 γ(t), ψ(t) + E_i
#########################################
def calculate_edge_cost(source, target, network, current_time, iteration,
                        psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0,
                        event=False,
                        user_flow=1.0):
    edge_info = network[(network['init_node'] == source) & (network['term_node'] == target)]
    if edge_info.empty:
        return float('inf')

    idx = edge_info.index[0]
    # 使用上一轮 LTM 更新的时变出行时间 flow_time[t]
    ltm_time = network.at[idx, "flow_time"][current_time]
    if ltm_time <= 0:
        # 如果是第一轮，可用 free_flow_time 作为后备
        ltm_time = network.at[idx, "free_flow_time"]

    # 若要考虑拥堵惩罚，需要用到 flow[t]
    flow_val = network.at[idx, "flow"][current_time]
    cap_val  = network.at[idx, "capacity"]

    gamma_val = gamma_t(iteration, lambda_, xi, c)
    psi_val   = psi_t(flow_val, cap_val, psi0=psi0, eta=eta)

    # （A）原有的动态修正系数
    dyn_factor = 1.0 + gamma_val / (1.0 + psi_val)
    cost_base  = ltm_time * dyn_factor

    # （B）外部成本
    e_i = external_cost_for_edge(idx, network, current_time, user_flow)

    # （C）合并：这里是加法叠加
    cost_with_e = cost_base + e_i

    # 外部事件放大
    if event:
        cost_with_e *= 1.2

    return cost_with_e

#########################################
# 6. 随机Log-Linear选路 (调用 cost)
#########################################
def log_linear_path_choice(candidate_paths, iteration, net, current_time,
                           psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0,
                           event=False, kappa=1.0, user_flow=1.0):
    costs = []
    for path_nodes in candidate_paths:
        cpath = 0.0
        for i in range(len(path_nodes) - 1):
            cpath += calculate_edge_cost(path_nodes[i], path_nodes[i + 1],
                                         net, current_time, iteration,
                                         psi0, eta, lambda_, xi, c,
                                         event=event,
                                         user_flow=user_flow)
        costs.append(cpath)

    # utility = -cost
    utilities = [-val for val in costs]
    gamma_val = gamma_t(iteration, lambda_, xi, c)

    tmp_scores = []
    for u in utilities:
        exponent_val = math.exp((gamma_val / kappa) * u)
        tmp_scores.append(exponent_val)
    denom = sum(tmp_scores)

    if denom < 1e-15:
        probs = [1 / len(tmp_scores)] * len(tmp_scores)
    else:
        probs = [x / denom for x in tmp_scores]

    rand_num = np.random.rand()
    cum = 0.0
    for idx, p in enumerate(probs):
        cum += p
        if rand_num <= cum:
            return candidate_paths[idx], probs
    return candidate_paths[-1], probs

#########################################
# 7. get_shortestpath_
#########################################
def get_shortestpath_with_spao(inode, tnode_list, network, index_table,
                               current_time, iteration,
                               psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0,
                               event=False,
                               use_loglinear=False,
                               user_flow=1.0):
    if not use_loglinear:
        # Dijkstra式搜索, cost来自 calculate_edge_cost
        S = []
        nodes = set(network['init_node']).union(set(network['term_node']))
        S_non = list(nodes)
        M = float('inf')
        dist = dict.fromkeys(S_non, M)
        pred = dict.fromkeys(S_non, None)
        dist[inode] = 0

        while S_non:
            min_node = min(S_non, key=lambda node: dist[node])
            if dist[min_node] == M:
                break
            S_non.remove(min_node)
            S.append(min_node)
            if set(tnode_list).issubset(set(S)):
                break
            edges_from_min_node = network[network['init_node'] == min_node]
            for idx_2, row in edges_from_min_node.iterrows():
                node = row['term_node']
                if node not in S_non:
                    continue
                cost_edge = calculate_edge_cost(
                    min_node, node, network, current_time, iteration,
                    psi0=psi0, eta=eta, lambda_=lambda_, xi=xi, c=c,
                    event=event,
                    user_flow=user_flow
                )
                if dist[min_node] + cost_edge < dist[node]:
                    dist[node] = dist[min_node] + cost_edge
                    pred[node] = min_node

        result = []
        for tnode in tnode_list:
            if dist[tnode] == M:
                result.append([inode, tnode, None, M])
                continue
            path = []
            current_node = tnode
            while current_node != inode:
                path.append(current_node)
                current_node = pred[current_node]
                if current_node is None:
                    break
            if current_node is None:
                result.append([inode, tnode, None, M])
                continue
            path.append(inode)
            path.reverse()
            result.append([inode, tnode, path, dist[tnode]])
        return result

    else:
        # 用最短路 + log-linear 挑选
        result = []
        for tnode in tnode_list:
            sp_result = get_shortestpath_with_spao(
                inode, [tnode], network, index_table, current_time, iteration,
                psi0, eta, lambda_, xi, c, event, use_loglinear=False,
                user_flow=user_flow
            )
            if sp_result and sp_result[0][2] is not None:
                candidate_paths = [sp_result[0][2]]  # 这里仅取最短路为候选
                chosen_path, _probs = log_linear_path_choice(
                    candidate_paths, iteration, network, current_time,
                    psi0, eta, lambda_, xi, c, event, kappa=1.0, user_flow=user_flow
                )
                cost_val = 0.0
                for i in range(len(chosen_path) - 1):
                    cost_val += calculate_edge_cost(
                        chosen_path[i], chosen_path[i + 1],
                        network, current_time, iteration,
                        psi0, eta, lambda_, xi, c, event,
                        user_flow=user_flow
                    )
                result.append([inode, tnode, chosen_path, cost_val])
            else:
                result.append([inode, tnode, None, float('inf')])
        return result

#########################################
# 8.LTM排队传播
#########################################
class MinimalLTM:
    def __init__(self, network_df, total_time):
        self.network_df = network_df
        self.total_time = total_time
        self.G = nx.DiGraph()
        self._build_graph()
        self.time_interval = 1.0
        self.total_travel_time_records = []

    def _build_graph(self):
        for idx, row in self.network_df.iterrows():
            o = row["init_node"]
            d = row["term_node"]
            cap = row["capacity"]
            fft = row["free_flow_time"]
            lng = row.get("length", 1.0)
            self.G.add_node(o)
            self.G.add_node(d)
            self.G.add_edge(o, d, capacity=cap, free_flow_time=fft, length=lng)

        for (u, v) in self.G.edges:
            e = self.G[u][v]
            e["N_up"] = np.zeros(self.total_time + 1)
            e["N_down"] = np.zeros(self.total_time + 1)
            e["Q_in"] = np.zeros(self.total_time + 1)
            e["Q_out"] = np.zeros(self.total_time + 1)
            e["queue"] = []
            e["buffer"] = []

            cap = e["capacity"]
            fft = e["free_flow_time"]
            lng = e["length"]
            f_speed = lng / fft if fft > 0 else 1.0
            jam_density = 0.2
            c_dens = cap / f_speed if f_speed > 0 else cap
            if (jam_density - c_dens) > 1e-5:
                b_speed = cap / (jam_density - c_dens)
            else:
                b_speed = f_speed
            e["forward_speed"] = f_speed
            e["backward_speed"] = b_speed
            e["critical_density"] = c_dens

    def load_vehicles(self, od_assignments, t):
        for (flow_val, path_nodes) in od_assignments:
            if len(path_nodes) < 2:
                continue
            first_edge = self.G[path_nodes[0]][path_nodes[1]]
            first_edge["queue"].append({
                "unit": flow_val,
                "start_time": t,
                "route_nodes": path_nodes,
                "cur_index": 0
            })
            first_edge["Q_in"][t] += flow_val
            first_edge["N_up"][t + 1] += flow_val

    def pre_update(self, t):
        if t > 0:
            for (u, v) in self.G.edges:
                e = self.G[u][v]
                e["N_up"][t] += e["N_up"][t - 1]
                e["N_down"][t] += e["N_down"][t - 1]

    def update(self, t):
        for (u, v) in self.G.edges:
            e = self.G[u][v]
            supply = e["capacity"]
            new_queue = []
            while e["queue"]:
                agent = e["queue"][0]
                if agent["unit"] <= supply:
                    supply -= agent["unit"]
                    e["queue"].pop(0)
                    e["Q_out"][t] += agent["unit"]
                    e["N_down"][t + 1] += agent["unit"]
                    route_nodes = agent["route_nodes"]
                    idx = agent["cur_index"]
                    if idx + 2 < len(route_nodes):
                        nxt_e = self.G[route_nodes[idx + 1]][route_nodes[idx + 2]]
                        nxt_e["queue"].append({
                            "unit": agent["unit"],
                            "start_time": agent["start_time"],
                            "route_nodes": route_nodes,
                            "cur_index": idx + 1
                        })
                        nxt_e["Q_in"][t] += agent["unit"]
                        nxt_e["N_up"][t + 1] += agent["unit"]
                    else:
                        trip_t = (t - agent["start_time"]) * self.time_interval
                        self.total_travel_time_records.append([trip_t, agent["unit"]])
                else:
                    partial = supply
                    agent["unit"] -= partial
                    supply = 0
                    e["Q_out"][t] += partial
                    e["N_down"][t + 1] += partial
                    route_nodes = agent["route_nodes"]
                    idx = agent["cur_index"]
                    if idx + 2 < len(route_nodes):
                        nxt_e = self.G[route_nodes[idx + 1]][route_nodes[idx + 2]]
                        nxt_e["queue"].append({
                            "unit": partial,
                            "start_time": agent["start_time"],
                            "route_nodes": route_nodes,
                            "cur_index": idx + 1
                        })
                        nxt_e["Q_in"][t] += partial
                        nxt_e["N_up"][t + 1] += partial
                    else:
                        trip_t = (t - agent["start_time"]) * self.time_interval
                        self.total_travel_time_records.append([trip_t, partial])
                    new_queue.append(agent)
                    break
            e["queue"] = new_queue

    def pro_update(self):
        for (u, v) in self.G.edges:
            e = self.G[u][v]
            if e["buffer"]:
                e["queue"].extend(e["buffer"])
                e["buffer"] = []

    def run_simulation(self, od_assignments_by_t):
        self.total_travel_time_records.clear()
        for t in range(self.total_time):
            self.pre_update(t)
            self.update(t)
            self.pro_update()
            if t in od_assignments_by_t:
                self.load_vehicles(od_assignments_by_t[t], t)
        return self._export_travel_time()

    def _export_travel_time(self):
        results = {}
        for (u, v) in self.G.edges:
            e = self.G[u][v]
            leng = e["length"]
            nu = e["N_up"]
            nd = e["N_down"]
            fsp = e["forward_speed"]
            bsp = e["backward_speed"]
            cd  = e["critical_density"]

            arr_tt = np.zeros(self.total_time)
            for t in range(self.total_time):
                dens = (nu[t] - nd[t]) / (leng + 1e-5)
                spd = fsp if dens < cd else bsp
                arr_tt[t] = leng / (spd + 1e-5)
            results[(u, v)] = arr_tt
        return results

    def get_total_trip_times(self):
        if len(self.total_travel_time_records) == 0:
            return 0.0
        arr = np.array(self.total_travel_time_records)
        wtt = np.sum(arr[:, 0] * arr[:, 1])
        total_flow = np.sum(arr[:, 1])
        return wtt / total_flow if total_flow > 1e-9 else 0.0

    def export_flows(self):
        flows = {}
        for (u, v) in self.G.edges:
            e = self.G[u][v]
            flows[(u, v)] = e["Q_in"][:self.total_time].copy()
        return flows

#########################################
# 9. 用LTM + 路径选择 来迭代求解
#########################################
def solve_nash_equilibrium_with_spao_LTM(network, od, start_term_rel, index_table,
                                         total_time, max_iterations=10000,
                                         event=False,
                                         use_loglinear=False,
                                         psi0=1.0, eta=0.5, lambda_=1.0, xi=1.0, c=1.0,
                                         user_flow=1.0):
    convergence_info = []
    prev_avg_tt = None

    # 初始化：先将 network["flow_time"] = free_flow_time
    for i in range(len(network)):
        fft = network.at[i, "free_flow_time"]
        network.at[i, "flow_time"] = np.full(total_time, fft)

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Iteration: {iteration} ===")
        iter_start_time = time.time()

        # (A) 构建每时段的 OD -> path -> flow
        od_assignments_by_t = {}
        for t in range(total_time):
            all_shortest_path = []
            for item in start_term_rel:
                result = get_shortestpath_with_spao(
                    inode=item["start_node"],
                    tnode_list=item["term_node_list"],
                    network=network,
                    index_table=index_table,
                    current_time=t,
                    iteration=iteration,
                    psi0=psi0, eta=eta, lambda_=lambda_, xi=xi, c=c,
                    event=event,
                    use_loglinear=use_loglinear,
                    user_flow=user_flow
                )
                all_shortest_path += result

            # 转为 (flow, path_nodes) 格式; 简单假设每时段都出发 od["demand"]
            shortest_path_dict = {(p[0], p[1]): p for p in all_shortest_path}
            od_assign_list = []
            for i2 in range(len(od)):
                st = od.iloc[i2]["init_node"]
                ed = od.iloc[i2]["term_node"]
                dm = od.iloc[i2]["demand"]
                if (st, ed) in shortest_path_dict:
                    path_info = shortest_path_dict[(st, ed)]
                    chosen_path = path_info[2]
                    if chosen_path is not None:
                        od_assign_list.append((dm, chosen_path))
            if od_assign_list:
                od_assignments_by_t[t] = od_assign_list

        # (B) 运行 LTM
        ltm_runner = MinimalLTM(network, total_time)
        td_times = ltm_runner.run_simulation(od_assignments_by_t)
        avg_tt = ltm_runner.get_total_trip_times()

        # (C) 更新 network["flow"], network["flow_time"]
        link_flows = ltm_runner.export_flows()
        for i_net in range(len(network)):
            o_ = network.at[i_net, "init_node"]
            d_ = network.at[i_net, "term_node"]
            if (o_, d_) in link_flows:
                arr_flow = link_flows[(o_, d_)]
                network.at[i_net, "flow"] = arr_flow

        for i_net in range(len(network)):
            o_ = network.at[i_net, "init_node"]
            d_ = network.at[i_net, "term_node"]
            if (o_, d_) in td_times:
                arr_tt = td_times[(o_, d_)]
                network.at[i_net, "flow_time"] = arr_tt

        # (D) GAP 或其他收敛指标
        if prev_avg_tt is not None:
            gap_val = abs(avg_tt - prev_avg_tt) / max(prev_avg_tt, 1e-9)
        else:
            gap_val = None
        prev_avg_tt = avg_tt

        iter_end_time = time.time()
        elapsed_time = iter_end_time - iter_start_time
        print(f"  Iteration {iteration} => AvgTT={avg_tt:.4f}, GAP={gap_val}, Time={elapsed_time:.2f}s")

        convergence_info.append({
            "Iteration": iteration,
            "AvgTT": avg_tt,
            "GAP": gap_val,
            "Time_s": elapsed_time
        })

    # 导出各边最终流量
    final_flows = []
    for (u, v) in ltm_runner.G.edges:
        e = ltm_runner.G[u][v]
        total_f = e["Q_out"].sum()
        final_flows.append({
            "Edge": f"{u}->{v}",
            "From": u,
            "To": v,
            "TotalFlow": total_f
        })

    return convergence_info, final_flows

#########################################
# 10. 主函数示例
#########################################
if __name__ == '__main__':
    # (1) 读取网络、OD、节点数据
    network_path = 'network1.xlsx'
    od_path = 'od_data1.xlsx'
    node_path = 'Node_Data.csv'

    network_1 = read_data(network_path)
    od_1 = read_data(od_path)
    if os.path.exists(node_path):
        node_data = read_node_data(node_path)
        network_1 = pd.merge(network_1, node_data, left_on='init_node', right_on='Node', how='left')
        network_1 = pd.merge(network_1, node_data, left_on='term_node', right_on='Node',
                             suffixes=('_init', '_term'), how='left')

    # (2) 基本预处理
    network_1['init_node'] = network_1['init_node'].astype(int)
    network_1['term_node'] = network_1['term_node'].astype(int)
    od_1 = od_1.dropna(subset=['init_node', 'term_node'])
    od_1['init_node'] = od_1['init_node'].astype(int)
    od_1['term_node'] = od_1['term_node'].astype(int)
    od_1['demand'] = pd.to_numeric(od_1['demand'], errors='coerce').fillna(0)
    od_1_new = od_1[od_1["demand"] > 0].copy().reset_index(drop=True)

    network_1['capacity'] = network_1['capacity'].astype(float)
    network_1['free_flow_time'] = network_1['free_flow_time'].astype(float)
    if 'length' not in network_1.columns:
        network_1['length'] = 1.0

    # (3) 初始化
    total_time = 5
    add_three_col(network_1, od_1_new, total_time)
    index_table_network = construct_index_table(network_1)

    start_term_rel = []
    for st in od_1_new['init_node'].unique():
        tlist = od_1_new[od_1_new['init_node'] == st]['term_node'].unique().tolist()
        start_term_rel.append({"start_node": st, "term_node_list": tlist})

    # (4) 迭代
    max_iterations = 10
    (conv_info, final_flow_list) = solve_nash_equilibrium_with_spao_LTM(
        network=network_1.copy(deep=True),
        od=od_1_new.copy(deep=True),
        start_term_rel=start_term_rel,
        index_table=index_table_network,
        total_time=total_time,
        max_iterations=max_iterations,
        event=True,
        use_loglinear=True,
        psi0=1.0, eta=0.5,
        lambda_=1.0, xi=1.0, c=1.0,
        user_flow=1.0  # 用于外部成本计算的一单位流量
    )

    conv_df = pd.DataFrame(conv_info)
    final_flow_df = pd.DataFrame(final_flow_list)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_excel = f"traffic_assignment_results_LTM_{timestamp_str}.xlsx"
    with pd.ExcelWriter(output_excel) as writer:
        conv_df.to_excel(writer, sheet_name="Convergence_Info", index=False)
        final_flow_df.to_excel(writer, sheet_name="Final_Flows", index=False)

    print(f"\nDone. Convergence & flow results => {output_excel}")
