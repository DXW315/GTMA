import os
import math
import time
import heapq
import copy
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import defaultdict

# ========== SUMO ==========
import sumolib
import traci

########################################################################
# 0. Time-Varying Parameter Class
########################################################################
class TimeVaryingParams:
    def __init__(self,
                 psi0=1.0, eta=0.5,
                 lambda_=1.0, xi=1.0, c_=1.0,
                 base_lr=1.0):
        self.psi0 = psi0
        self.eta = eta
        self.lambda_ = lambda_
        self.xi = xi
        self.c_ = c_
        self.base_lr = base_lr

    def get_gamma(self, iteration):
        val = self.lambda_ * iteration + self.xi
        if val <= 1:
            val = 2.0
        return (1.0 / self.c_) * math.log(val)

    def get_psi(self, flow, capacity):
        if capacity <= 1e-9:
            return 0.0
        ratio = flow / capacity
        return self.psi0 * (1.0 + self.eta * ratio)

    def get_learning_rate(self, iteration, t):
        lr = self.base_lr / (1 + 0.1 * iteration)
        return lr


########################################################################
# 1. Read network from Excel and build adjacency list
########################################################################
def read_data_to_adjlist(df_network):
    adj_list = defaultdict(dict)
    for idx, row in df_network.iterrows():
        o = int(row['init_node'])
        d = int(row['term_node'])
        cap = float(row['capacity'])
        fft = float(row['free_flow_time'])
        lng = float(row.get('length', 1.0))
        adj_list[o][d] = {
            'capacity': cap,
            'free_flow_time': fft,
            'length': lng,
            'flow': np.zeros(1),
            'flow_time': np.zeros(1)
        }
    return dict(adj_list)


########################################################################
# 2. External BPR cost difference (with caching)
########################################################################
_external_cost_cache = {}

def external_cost_for_edge(o, d, t, flow_val, capacity, fft,
                           user_flow=1.0, alpha=0.15, beta=4.0):
    global _external_cost_cache
    cache_key = (o, d, t, float(flow_val))
    if cache_key in _external_cost_cache:
        return _external_cost_cache[cache_key]

    if capacity <= 1e-9:
        _external_cost_cache[cache_key] = 0.0
        return 0.0

    def bpr_t(f):
        return fft * (1.0 + alpha * (f / capacity)**beta)

    t_with = bpr_t(flow_val)
    t_wo = bpr_t(max(flow_val - user_flow, 0))
    e_i = max(t_with - t_wo, 0)
    _external_cost_cache[cache_key] = e_i
    return e_i


########################################################################
# 3. Multi-Agent Path Strategy
########################################################################
class AgentPolicy:
    def __init__(self, paths):
        self.paths = paths
        if len(paths) > 0:
            self.probs = np.ones(len(paths)) / len(paths)
        else:
            self.probs = np.array([])

    def choose_path(self):
        if len(self.probs) == 0:
            return None
        r = np.random.rand()
        cum = 0
        for i, p in enumerate(self.probs):
            cum += p
            if r <= cum:
                return self.paths[i]
        return self.paths[-1]

    def update_policy(self, costs, learning_rate=1.0):
        if len(self.probs) == 0:
            return
        exps = np.zeros_like(self.probs)
        for i, cst in enumerate(costs):
            exps[i] = self.probs[i] * math.exp(-learning_rate * cst)
        denom = exps.sum()
        if denom < 1e-15:
            self.probs = np.ones_like(exps) / len(exps)
        else:
            self.probs = exps / denom


class MultiAgentRouteChoice:
    def __init__(self, adj_list, od_df, K_paths=3):
        self.agent_policies = {}
        self.od_demands = {}

        for idx, row in od_df.iterrows():
            o = int(row['init_node'])
            d = int(row['term_node'])
            dm = float(row['demand'])
            if dm <= 1e-9:
                continue

            key = (o, d)
            # BFS cost function
            def tmp_fun(oo, dd):
                return adj_list[oo][dd]['free_flow_time']

            kpaths = k_shortest_paths(adj_list, o, d, K_paths, cost_fun=tmp_fun)
            # Debug: print BFS path count
            print(f"OD({o}->{d}), demand={dm}, found {len(kpaths)} paths:", kpaths)

            self.agent_policies[key] = AgentPolicy(kpaths)
            self.od_demands[key] = dm

    def choose_paths_for_all_agents(self, iteration, current_time):
        """
        Returns in the form of: [(flow, path_nodes), ...]
        """
        results = []
        for key, policy in self.agent_policies.items():
            dm = self.od_demands[key]
            chosen_path = policy.choose_path()
            if chosen_path is not None:
                results.append((dm, chosen_path))
        # Debug: print assignment results
        if len(results) > 0:
            print(f"[Iter={iteration}, T={current_time}] Assigned {len(results)} OD flows: {results}")
        return results

    def update_all_policies(self, path_costs, learning_rate=1.0):
        for od_pair, cost_list in path_costs.items():
            if od_pair in self.agent_policies:
                self.agent_policies[od_pair].update_policy(cost_list, learning_rate)


########################################################################
# k shortest paths
########################################################################
def k_shortest_paths(adj_list, source, target, K=3, cost_fun=None):
    if source not in adj_list or target not in adj_list:
        return []
    if cost_fun is None:
        def cost_fun(o_, d_):
            return adj_list[o_][d_]['free_flow_time']

    heap = [(0.0, [source])]
    paths = []
    visited_set = set()

    while heap and len(paths) < K:
        cur_cost, path = heapq.heappop(heap)
        last_node = path[-1]
        if last_node == target:
            paths.append((cur_cost, path))
            continue
        if last_node in adj_list:
            for nxt in adj_list[last_node]:
                if nxt in path:
                    continue
                ecost = cost_fun(last_node, nxt)
                new_path = path + [nxt]
                new_cost = cur_cost + ecost
                signature = (last_node, nxt, len(path))
                if signature in visited_set:
                    continue
                visited_set.add(signature)
                heapq.heappush(heap, (new_cost, new_path))

    # Only return paths (excluding cost)
    return [p[1] for p in paths]


########################################################################
# 4. Compute edge/path cost
########################################################################
def calculate_edge_cost(o, d, t, iteration, adj_list,
                        tv_params, event=False, user_flow=1.0,
                        alpha=0.15, beta=4.0):
    data = adj_list[o][d]
    fft = data['free_flow_time']
    cap = data['capacity']

    flow_arr = data['flow']
    flow_ti = flow_arr[t] if t < len(flow_arr) else 0.0

    flow_time_arr = data['flow_time']
    ltm_time = flow_time_arr[t] if t < len(flow_time_arr) else fft
    if ltm_time <= 0:
        ltm_time = fft

    gamma_val = tv_params.get_gamma(iteration)
    psi_val = tv_params.get_psi(flow_ti, cap)
    dyn_factor = 1.0 + gamma_val / (1.0 + psi_val)
    base_time = ltm_time * dyn_factor

    ext_cost = external_cost_for_edge(o, d, t, flow_ti, cap, fft,
                                      user_flow=user_flow,
                                      alpha=alpha, beta=beta)
    total_cost = base_time + ext_cost
    if event:
        total_cost *= 1.2
    return total_cost


def calculate_path_cost(path_nodes, adj_list, iteration,
                        tv_params, total_time,
                        event=False, user_flow=1.0,
                        alpha=0.15, beta=4.0):
    if len(path_nodes) < 2:
        return 1e9
    if total_time <= 0:
        return 1e9

    sum_cost = 0.0
    for t in range(total_time):
        c_t = 0.0
        for i in range(len(path_nodes)-1):
            o = path_nodes[i]
            d = path_nodes[i+1]
            c_edge = calculate_edge_cost(o, d, t, iteration,
                                         adj_list=adj_list,
                                         tv_params=tv_params,
                                         event=event,
                                         user_flow=user_flow,
                                         alpha=alpha, beta=beta)
            c_t += c_edge
        sum_cost += c_t
    return sum_cost / total_time


########################################################################
# 5. Build node_edge_map + edge_length_map
########################################################################
def build_node_edge_dicts_from_sumocfg(sumo_config_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(sumo_config_file)
    root = tree.getroot()
    net_file_path = None
    for child in root:
        if child.tag == "input":
            for sub in child:
                if sub.tag == "net-file":
                    net_file_path = sub.attrib.get("value", None)
    if net_file_path is None:
        raise RuntimeError("No <net-file> found in sumocfg.")
    cfg_dir = os.path.dirname(os.path.abspath(sumo_config_file))
    net_file_full = os.path.join(cfg_dir, net_file_path)
    if not os.path.exists(net_file_full):
        raise FileNotFoundError(f"Cannot find net file: {net_file_full}")

    net = sumolib.net.readNet(net_file_full)
    node_edge_map = {}
    edge_length_map = {}

    for e in net.getEdges():
        e_id = e.getID()
        length = e.getLength()
        fNode = e.getFromNode().getID()
        tNode = e.getToNode().getID()
        try:
            o = int(fNode)
            d = int(tNode)
            node_edge_map[(o, d)] = e_id
            edge_length_map[(o, d)] = length
        except ValueError:
            pass

    return node_edge_map, edge_length_map


########################################################################
# 6. Use "vehicle departure-arrival time" to calculate RealAvgTT (with debug prints)
########################################################################
def simulate_with_sumo(
    sumo_config_file,
    od_assignments_by_t,
    total_time,
    iteration,
    node_edge_map,
    edge_length_map,
    use_gui=False
):
    """
    In each iteration:
      1) Generate a temporary .rou.xml using (node_edge_map) for routes
      2) Start sumo or sumo-gui, simulate for total_time steps
      3) Calculate actual travel time (arrival_time - departure_time)
      4) Meanwhile record link_flows, link_times for updating adj_list
    """

    route_file = f"temp_iter_{iteration}.rou.xml"
    with open(route_file, "w", encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write('  <vType id="car" accel="2.9" decel="7.5" sigma="0.5" length="5" maxSpeed="25"/>\n')
        veh_id_counter = 0
        total_veh_written = 0

        for t, assign_list in od_assignments_by_t.items():
            for (flow_val, path_nodes) in assign_list:
                num_veh = int(round(flow_val))
                if num_veh < 1:
                    continue
                edge_list = []
                for i in range(len(path_nodes) - 1):
                    o = path_nodes[i]
                    d = path_nodes[i + 1]
                    if (o, d) not in node_edge_map:
                        raise ValueError(f"node_edge_map missing: ({o},{d})!")
                    edge_list.append(node_edge_map[(o, d)])
                edge_str = " ".join(edge_list)

                for _ in range(num_veh):
                    veh_id = f"veh_iter{iteration}_t{t}_{veh_id_counter}"
                    depart_sec = t
                    veh_id_counter += 1
                    f.write(f'  <vehicle id="{veh_id}" type="car" depart="{depart_sec}" route="{edge_str}"/>\n')
                    total_veh_written += 1
        f.write('</routes>\n')

    # Debug: how many vehicles written
    print(f"[simulate_with_sumo] Iter={iteration} => route file={route_file}, total vehicles={total_veh_written}")

    sumo_bin = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_bin,
        "-c", sumo_config_file,
        "--step-length", "1.0",
        "--quit-on-end", "true"
    ]
    # Start SUMO
    traci.start(sumo_cmd)

    edge2od = {}
    for od_pair, eName in node_edge_map.items():
        edge2od[eName] = od_pair

    link_flows = defaultdict(lambda: np.zeros(total_time))
    link_times = defaultdict(lambda: np.zeros(total_time))

    vehicle_start_times = {}
    arrived_trip_times = []

    for step in range(total_time):
        traci.simulationStep()

        # (a) newly loaded vehicles
        loaded_ids = traci.simulation.getLoadedIDList()
        if loaded_ids:
            print(f"[simulate] step={step}, loaded_ids={loaded_ids}")
        for vid in loaded_ids:
            vehicle_start_times[vid] = step

        # (b) arrived vehicles
        arrived_ids = traci.simulation.getArrivedIDList()
        if arrived_ids:
            print(f"[simulate] step={step}, arrived_ids={arrived_ids}")
        for vid in arrived_ids:
            st = vehicle_start_times.get(vid, step)
            trip_time = (step - st)
            arrived_trip_times.append(trip_time)

        # (c) record flow/time on each edge
        for eid in traci.edge.getIDList():
            veh_num = traci.edge.getLastStepVehicleNumber(eid)
            mean_spd = traci.edge.getLastStepMeanSpeed(eid)
            if mean_spd < 0.1:
                mean_spd = 0.1
            if eid in edge2od:
                (o, d) = edge2od[eid]
                length = edge_length_map.get((o, d), 1.0)
                travel_time = length / mean_spd
                link_flows[(o, d)][step] = veh_num
                link_times[(o, d)][step] = travel_time

    traci.close()

    if len(arrived_trip_times) > 0:
        avg_tt_real = sum(arrived_trip_times) / len(arrived_trip_times)
    else:
        avg_tt_real = 0.0
        if total_veh_written > 0:
            print(f"[simulate_with_sumo] Warning: {total_veh_written} vehicles written, but none arrived within {total_time} steps => RealAvgTT=0.0")

    return dict(link_flows), dict(link_times), avg_tt_real


########################################################################
# 7. Multi-iteration DTA
########################################################################
def solve_multi_agent_dta_timevarying(
    adj_list, df_od,
    total_time=300,
    max_iterations=3,
    K_paths=3,
    event=False,
    user_flow=1.0,
    tv_params=None,
    alpha=0.15,
    beta=4.0,
    collect_convergence=True,
    sumo_config_file="SiouxFalls.sumocfg",
    node_edge_map=None,
    edge_length_map=None,
    use_gui=False
):
    if tv_params is None:
        tv_params = TimeVaryingParams()
    if node_edge_map is None:
        node_edge_map = {}
    if edge_length_map is None:
        edge_length_map = {}

    # Initialize flow/time
    for o in adj_list:
        for d in adj_list[o]:
            fft = adj_list[o][d]['free_flow_time']
            adj_list[o][d]['flow'] = np.zeros(total_time)
            adj_list[o][d]['flow_time'] = np.full(total_time, fft)

    route_choice_manager = MultiAgentRouteChoice(adj_list, df_od, K_paths=K_paths)

    if collect_convergence:
        convergence_info = []
    else:
        convergence_info = None

    prev_avg_tt = None

    for iteration in range(1, max_iterations + 1):
        # Clear BPR cache
        _external_cost_cache.clear()

        # A) Path assignment
        od_assignments_by_t = {}
        for t in range(total_time):
            assigned = route_choice_manager.choose_paths_for_all_agents(iteration, t)
            if assigned:
                od_assignments_by_t[t] = assigned

        # B) SUMO simulation to get actual arrival times
        link_flows, link_times, avg_tt_real = simulate_with_sumo(
            sumo_config_file=sumo_config_file,
            od_assignments_by_t=od_assignments_by_t,
            total_time=total_time,
            iteration=iteration,
            node_edge_map=node_edge_map,
            edge_length_map=edge_length_map,
            use_gui=use_gui
        )

        # C) Update adj_list
        for (o, d), arr_flow in link_flows.items():
            if o in adj_list and d in adj_list[o]:
                adj_list[o][d]['flow'] = arr_flow
        for (o, d), arr_time in link_times.items():
            if o in adj_list and d in adj_list[o]:
                adj_list[o][d]['flow_time'] = arr_time

        # D) Calculate path costs & update strategy
        path_costs_for_agents = {}
        for od_pair, policy in route_choice_manager.agent_policies.items():
            csts = []
            for path_nodes in policy.paths:
                cost_val = calculate_path_cost(
                    path_nodes, adj_list,
                    iteration=iteration,
                    tv_params=tv_params,
                    total_time=total_time,
                    event=event,
                    user_flow=user_flow,
                    alpha=alpha,
                    beta=beta
                )
                csts.append(cost_val)
            path_costs_for_agents[od_pair] = csts

        lr_current = tv_params.get_learning_rate(iteration, 0)
        route_choice_manager.update_all_policies(path_costs_for_agents, learning_rate=lr_current)

        # E) Convergence info
        if collect_convergence:
            if prev_avg_tt is not None:
                gap_val = abs(avg_tt_real - prev_avg_tt) / max(prev_avg_tt, 1e-9)
            else:
                gap_val = None
            prev_avg_tt = avg_tt_real
            print(f"Iteration={iteration}, RealAvgTT={avg_tt_real:.4f}, GAP={gap_val}")
            convergence_info.append({"Iteration": iteration, "RealAvgTT": avg_tt_real, "GAP": gap_val})

    # Collect final flows
    final_flows = []
    for o in adj_list:
        for d in adj_list[o]:
            arr_flow = adj_list[o][d]['flow']
            fl = arr_flow[-1] if len(arr_flow) > 0 else 0.0
            final_flows.append([o, d, fl])
    df_flow = pd.DataFrame(final_flows, columns=["Init Node", "Term Node", "Flow_TLast"])

    if convergence_info is not None:
        df_conv = pd.DataFrame(convergence_info)
    else:
        df_conv = None

    return df_conv, df_flow


########################################################################
# 8. main()
########################################################################
def main():
    network_path = 'network1.xlsx'
    od_path = 'od_data1.xlsx'
    sumo_cfg_path = 'SiouxFalls.sumocfg'

    if not os.path.exists(network_path) or not os.path.exists(od_path):
        print("Error: Excel not found.")
        return
    if not os.path.exists(sumo_cfg_path):
        print("Error: Sumo cfg not found.")
        return

    # 1) Read Excel
    df_network = pd.read_excel(network_path)
    df_od = pd.read_excel(od_path)

    # Field compatibility
    if 'Source' in df_network.columns and 'Target' in df_network.columns:
        df_network.rename(columns={'Source': 'init_node', 'Target': 'term_node'}, inplace=True)
    if 'Source' in df_od.columns and 'Target' in df_od.columns:
        df_od.rename(columns={'Source': 'init_node', 'Target': 'term_node'}, inplace=True)

    df_network['capacity'] = pd.to_numeric(df_network['capacity'], errors='coerce').fillna(0)
    df_network['free_flow_time'] = pd.to_numeric(df_network['free_flow_time'], errors='coerce').fillna(0)
    if 'length' not in df_network.columns:
        df_network['length'] = 1.0

    df_od['demand'] = pd.to_numeric(df_od['demand'], errors='coerce').fillna(0)
    df_od = df_od[df_od['demand'] > 0].copy()

    # 2) Build adjacency list
    adj_list = read_data_to_adjlist(df_network)

    # 3) Build node_edge_map / edge_length_map from SUMO cfg
    node_edge_map, edge_length_map = build_node_edge_dicts_from_sumocfg(sumo_cfg_path)

    # 4) Set time-varying parameters
    tvp = TimeVaryingParams(base_lr=2.0)

    # 5) Run multi-iteration DTA
    total_time = 300   # adjust as needed
    max_iters = 100      # number of iterations

    start_t = time.time()
    conv_df, flow_df = solve_multi_agent_dta_timevarying(
        adj_list=adj_list,
        df_od=df_od,
        total_time=total_time,
        max_iterations=max_iters,
        K_paths=3,
        event=False,
        user_flow=1.0,
        tv_params=tvp,
        alpha=0.15,
        beta=4.0,
        collect_convergence=True,
        sumo_config_file=sumo_cfg_path,
        node_edge_map=node_edge_map,
        edge_length_map=edge_length_map,
        use_gui=True  # set to False if you don't want GUI
    )
    end_t = time.time()

    print(f"\nExecution finished, elapsed time: {end_t - start_t:.2f}s")

    # 6) Write results
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"multi_agent_sumo_gui_{stamp}.xlsx"
    with pd.ExcelWriter(out_file) as writer:
        if conv_df is not None:
            conv_df.to_excel(writer, sheet_name="Convergence_Info", index=False)
        flow_df.to_excel(writer, sheet_name="LastTime_Flow", index=False)

    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
