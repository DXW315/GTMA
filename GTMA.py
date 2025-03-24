import pandas as pd
import numpy as np
import math
import os
import time
import networkx as nx
from collections import defaultdict
from datetime import datetime
import heapq
import copy


########################################################################
# 0. Time-varying parameters
########################################################################
class TimeVaryingParams:
    """
    Manages time-varying parameters: gamma_t, psi_t, learning rate, etc.,
    to reduce repeated computations.

    - get_gamma(iteration) : gamma_t = (1/c) * ln(lambda_ * iteration + xi)
    - get_psi(flow, capacity): psi_t = psi0 * [1 + eta * (flow / capacity)]
    - get_learning_rate(iteration, t): returns the learning rate for logit choice,
      which can vary by iteration/time period.
    """

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
        """
        gamma_t = (1/c_) * ln(lambda_ * iteration + xi)
        iteration >= 1 to avoid ln(0).
        """
        val = self.lambda_ * iteration + self.xi
        if val <= 1:
            val = 2.0
        return (1.0 / self.c_) * math.log(val)

    def get_psi(self, flow, capacity):
        """
        psi_t = psi0 * [1 + eta*(flow/capacity)]
        """
        if capacity <= 1e-9:
            return 0.0
        ratio = flow / capacity
        return self.psi0 * (1.0 + self.eta * ratio)

    def get_learning_rate(self, iteration, t):
        """
        Example of a learning rate that can decay with iterations/time:
          lr = base_lr / (1 + 0.1*iteration)
        """
        lr = self.base_lr / (1 + 0.1 * iteration)
        return lr


########################################################################
# 1. Read data and build an adjacency list
########################################################################
def read_data_to_adjlist(df_network):
    """
    Convert the network DataFrame into a dict-of-dict adjacency list:
       adj_list[o][d] = {
         'capacity': ...,
         'free_flow_time': ...,
         'length': ...,
         'flow': np.zeros(T),   # flow per time period
         'flow_time': np.zeros(T) # travel time per time period (from LTM)
       }
    """
    adj_list = defaultdict(dict)
    for idx, row in df_network.iterrows():
        o = int(row['init_node'])
        d = int(row['term_node'])
        adj_list[o][d] = {
            'capacity': float(row['capacity']),
            'free_flow_time': float(row['free_flow_time']),
            'length': float(row.get('length', 1.0)),
            'flow': np.zeros(1),       # placeholder initialization
            'flow_time': np.zeros(1)   # placeholder initialization
        }
    return dict(adj_list)


########################################################################
# 2. BPR external cost calculation
########################################################################
_external_cost_cache = {}


def external_cost_for_edge(o, d, t, flow_val, capacity, fft,
                           user_flow=1.0, alpha=0.15, beta=4.0):
    """
    Calculate the external cost of a single edge (o->d) at time period t.
    External cost e_i = BPR(flow_val) - BPR(flow_val - user_flow),
    approximated by the BPR function, with caching for faster retrieval.
    """
    global _external_cost_cache
    cache_key = (o, d, t, float(flow_val))
    if cache_key in _external_cost_cache:
        return _external_cost_cache[cache_key]

    if capacity <= 1e-9:
        _external_cost_cache[cache_key] = 0.0
        return 0.0

    def bpr_t(f):
        return fft * (1.0 + alpha * (f / capacity) ** beta)

    t_with = bpr_t(flow_val)
    t_wo = bpr_t(max(flow_val - user_flow, 0))
    e_i = max(t_with - t_wo, 0)
    _external_cost_cache[cache_key] = e_i
    return e_i


########################################################################
# 3. Multi-agent path strategy class
########################################################################
class AgentPolicy:
    """
    Each OD corresponds to one policy (multiple feasible paths + probabilities).
    """

    def __init__(self, paths):
        self.paths = paths
        if len(paths) > 0:
            self.probs = np.ones(len(paths)) / len(paths)
        else:
            self.probs = np.array([])

    def choose_path(self):
        """
        Randomly pick one path according to the current strategy probabilities.
        """
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
        """
        Logit choice update formula:
          p'_i = p_i * exp(-lr * cost_i) / sum_j [ p_j * exp(-lr * cost_j) ]
        """
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
    """
    (o, d) -> AgentPolicy
    Manages multiple OD pairs, each OD has several paths + probabilities.
    """

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

            def tmp_fun(oo, dd):
                return adj_list[oo][dd]['free_flow_time']

            kpaths = k_shortest_paths(adj_list, o, d, K=K_paths, cost_fun=tmp_fun)
            self.agent_policies[key] = AgentPolicy(kpaths)
            self.od_demands[key] = dm

    def choose_paths_for_all_agents(self, iteration, current_time):
        """
        For all ODs, pick one path according to the current policy, returning (demand, path).
        """
        results = []
        for key, policy in self.agent_policies.items():
            dm = self.od_demands[key]
            path_ = policy.choose_path()
            if path_ is not None:
                results.append((dm, path_))
        return results

    def update_all_policies(self, path_costs, learning_rate=1.0):
        """
        path_costs: { (o,d): [cost1, cost2, ...] }
        """
        for key, costs_list in path_costs.items():
            if key in self.agent_policies:
                self.agent_policies[key].update_policy(costs_list, learning_rate)


def k_shortest_paths(adj_list, source, target, K=3, cost_fun=None):
    """
    K-shortest-paths (based on a Dijkstra-like BFS + heapq).
    cost_fun(o, d) gives the cost of edge (o->d); by default, uses free_flow_time.
    """
    if source not in adj_list or target not in adj_list:
        return []
    if cost_fun is None:
        def cost_fun(o_, d_):
            return adj_list[o_][d_]['free_flow_time']

    heap = [(0, [source])]
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
                # Avoid loops
                if nxt in path:
                    continue
                edge_cost = cost_fun(last_node, nxt)
                new_path = path + [nxt]
                new_cost = cur_cost + edge_cost
                if (last_node, nxt, len(path)) in visited_set:
                    continue
                visited_set.add((last_node, nxt, len(path)))
                heapq.heappush(heap, (new_cost, new_path))
    return [p[1] for p in paths]


########################################################################
# 4. Dynamic cost calculation (integrating time-varying parameters)
########################################################################
def calculate_edge_cost(o, d, t, iteration, adj_list,
                        tv_params, event=False, user_flow=1.0,
                        alpha=0.15, beta=4.0):
    """
    At time period t, iteration 'iteration', calculate the comprehensive cost of edge (o->d):
      cost = base_time + external_cost
      base_time = ltm_time * [1 + gamma_t / (1 + psi_t)]
    """
    data = adj_list[o][d]
    fft = data['free_flow_time']
    capacity = data['capacity']

    flow_arr = data['flow']
    flow_ti = flow_arr[t] if t < len(flow_arr) else 0.0

    flow_time_arr = data['flow_time']
    ltm_time = flow_time_arr[t] if t < len(flow_time_arr) else fft
    if ltm_time <= 0:
        ltm_time = fft

    # Time-varying gamma, psi
    gamma_val = tv_params.get_gamma(iteration)
    psi_val = tv_params.get_psi(flow_ti, capacity)

    dyn_factor = 1.0 + gamma_val / (1.0 + psi_val)
    base_time = ltm_time * dyn_factor

    # External cost
    ext_cost = external_cost_for_edge(o, d, t, flow_ti, capacity, fft,
                                      user_flow=user_flow, alpha=alpha, beta=beta)
    total_cost = base_time + ext_cost

    # If event occurs, multiply by 1.2
    if event:
        total_cost *= 1.2

    return total_cost


def calculate_path_cost(path_nodes, adj_list, iteration,
                        tv_params, total_time, event=False, user_flow=1.0,
                        alpha=0.15, beta=4.0):
    """
    Calculate the cost of a path over time periods 0..(total_time-1).
    Example here: take the average of costs across all time periods.
    """
    if len(path_nodes) < 2:
        return 999999.0
    if total_time <= 0:
        return 999999.0

    sum_cost = 0.0
    for t in range(total_time):
        c_t = 0.0
        for i in range(len(path_nodes) - 1):
            o = path_nodes[i]
            d = path_nodes[i + 1]
            c_edge = calculate_edge_cost(o, d, t, iteration,
                                         adj_list,
                                         tv_params=tv_params,
                                         event=event,
                                         user_flow=user_flow,
                                         alpha=alpha,
                                         beta=beta)
            c_t += c_edge
        sum_cost += c_t
    avg_cost = sum_cost / total_time
    return avg_cost


########################################################################
# 5. MinimalLTM: A queue propagation model
########################################################################
class MinimalLTM:
    """
    A simplified LTM model using discrete time steps (Q_in / Q_out) to simulate flow propagation.
    """

    def __init__(self, adj_list, total_time):
        self.adj_list = adj_list
        self.total_time = total_time
        self.G = nx.DiGraph()
        self._build_graph()

        self.time_interval = 1.0
        self.total_travel_time_records = []

    def _build_graph(self):
        """
        Load the information from adj_list into a networkx.DiGraph
        and initialize discrete flow arrays.
        """
        for o in self.adj_list:
            for d, data in self.adj_list[o].items():
                self.G.add_node(o)
                self.G.add_node(d)
                self.G.add_edge(o, d,
                                capacity=data['capacity'],
                                free_flow_time=data['free_flow_time'],
                                length=data.get('length', 1.0))

        for (u, v) in self.G.edges:
            e = self.G[u][v]
            e["N_up"] = np.zeros(self.total_time + 1)
            e["N_down"] = np.zeros(self.total_time + 1)
            e["Q_in"] = np.zeros(self.total_time + 1)
            e["Q_out"] = np.zeros(self.total_time + 1)
            e["queue"] = []

            cap = e["capacity"]
            fft = e["free_flow_time"]
            lng = e["length"]
            f_speed = lng / fft if fft > 0 else 1.0
            jam_density = 0.2  # simplified assumption
            c_dens = cap / f_speed if f_speed > 0 else cap
            if (jam_density - c_dens) > 1e-5:
                b_speed = cap / (jam_density - c_dens)
            else:
                b_speed = f_speed

            e["forward_speed"] = f_speed
            e["backward_speed"] = b_speed
            e["critical_density"] = c_dens

    def load_vehicles(self, od_assignments, t):
        """
        At time t, place the assigned (flow_val, path_nodes) vehicles into the first edge's queue.
        """
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
        """
        Process queues on each edge with capacity constraints.
        """
        for (u, v) in self.G.edges:
            e = self.G[u][v]
            supply = e["capacity"]
            new_queue = []
            while e["queue"]:
                agent = e["queue"][0]
                if agent["unit"] <= supply:
                    # Enough capacity to let these vehicles pass
                    supply -= agent["unit"]
                    e["queue"].pop(0)
                    e["Q_out"][t] += agent["unit"]
                    e["N_down"][t + 1] += agent["unit"]

                    route_nodes = agent["route_nodes"]
                    idx = agent["cur_index"]
                    if idx + 2 < len(route_nodes):
                        nxt_u = route_nodes[idx + 1]
                        nxt_v = route_nodes[idx + 2]
                        nxt_e = self.G[nxt_u][nxt_v]
                        nxt_e["queue"].append({
                            "unit": agent["unit"],
                            "start_time": agent["start_time"],
                            "route_nodes": route_nodes,
                            "cur_index": idx + 1
                        })
                        nxt_e["Q_in"][t] += agent["unit"]
                        nxt_e["N_up"][t + 1] += agent["unit"]
                    else:
                        # Reached destination
                        trip_t = (t - agent["start_time"]) * self.time_interval
                        self.total_travel_time_records.append([trip_t, agent["unit"]])
                else:
                    # Only part of the vehicles can pass
                    partial = supply
                    agent["unit"] -= partial
                    supply = 0
                    e["Q_out"][t] += partial
                    e["N_down"][t + 1] += partial

                    route_nodes = agent["route_nodes"]
                    idx = agent["cur_index"]
                    if idx + 2 < len(route_nodes):
                        nxt_u = route_nodes[idx + 1]
                        nxt_v = route_nodes[idx + 2]
                        nxt_e = self.G[nxt_u][nxt_v]
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
        """
        Optional: for more complex post-update operations, if needed.
        """
        pass

    def run_simulation(self, od_assignments_by_t):
        """
        Main entry point: clear travel_time_records, then run updates and load vehicles by time step.
        """
        self.total_travel_time_records.clear()
        for t in range(self.total_time):
            self.pre_update(t)
            self.update(t)
            self.pro_update()
            if t in od_assignments_by_t:
                self.load_vehicles(od_assignments_by_t[t], t)
        return self._export_travel_time()

    def _export_travel_time(self):
        """
        Export the travel time of each edge per time period, based on density and wave speed.
        """
        results = {}
        for (u, v) in self.G.edges:
            e = self.G[u][v]
            leng = e["length"]
            nu = e["N_up"]
            nd = e["N_down"]
            fsp = e["forward_speed"]
            bsp = e["backward_speed"]
            cd = e["critical_density"]

            arr_tt = np.zeros(self.total_time)
            for t in range(self.total_time):
                dens = (nu[t] - nd[t]) / (leng + 1e-5)
                spd = fsp if dens < cd else bsp
                arr_tt[t] = leng / (spd + 1e-5)
            results[(u, v)] = arr_tt
        return results

    def get_total_trip_times(self):
        """
        Compute the average travel time of all vehicles that have reached their destination.
        """
        if len(self.total_travel_time_records) == 0:
            return 0.0
        arr = np.array(self.total_travel_time_records)
        wtt = np.sum(arr[:, 0] * arr[:, 1])  # weighted total travel time
        total_flow = np.sum(arr[:, 1])
        return wtt / total_flow if total_flow > 1e-9 else 0.0

    def export_flows(self):
        """
        Export the flow on each edge (Q_in) per time period.
        """
        flows = {}
        for (u, v) in self.G.edges:
            e = self.G[u][v]
            flows[(u, v)] = e["Q_in"][:self.total_time].copy()
        return flows


########################################################################
# 6. Main DTA solver (time-varying parameters + performance optimization)
########################################################################
def solve_multi_agent_dta_timevarying(adj_list, df_od,
                                      total_time=5, max_iterations=10, K_paths=3,
                                      event=False, user_flow=1.0,
                                      tv_params=None,  # an instance of TimeVaryingParams
                                      alpha=0.15, beta=4.0,
                                      collect_convergence=True):
    """
    Main function for multi-agent DTA with time-varying parameters:
      - In each iteration, assign paths to OD pairs based on the policy,
        then run the MinimalLTM simulation to get travel time and flow by edge/time period.
      - Then compute the average path cost and update the policies with a time-varying learning rate.
    """
    if tv_params is None:
        # Use default fixed values if none provided
        tv_params = TimeVaryingParams()

    # Initialize flow and flow_time
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

    # -- Begin iterations
    for iteration in range(1, max_iterations + 1):
        # Clear external cost cache to avoid leftover from previous iteration
        _external_cost_cache.clear()

        # A) Assign flows by policy (for each time period)
        od_assignments_by_t = {}
        for t in range(total_time):
            assignments_for_t = route_choice_manager.choose_paths_for_all_agents(iteration, t)
            if assignments_for_t:
                od_assignments_by_t[t] = assignments_for_t

        # B) LTM simulation
        ltm_runner = MinimalLTM(adj_list, total_time)
        td_times = ltm_runner.run_simulation(od_assignments_by_t)
        avg_tt = ltm_runner.get_total_trip_times()

        # C) Update adj_list flow and flow_time
        link_flows = ltm_runner.export_flows()
        for (o, d) in link_flows:
            adj_list[o][d]['flow'] = link_flows[(o, d)]
        for (o, d) in td_times:
            adj_list[o][d]['flow_time'] = td_times[(o, d)]

        # D) Calculate path costs and update policies
        path_costs_for_agents = {}
        for key, policy in route_choice_manager.agent_policies.items():
            costs_list = []
            for path_nodes in policy.paths:
                cost_val = calculate_path_cost(path_nodes, adj_list,
                                               iteration=iteration,
                                               tv_params=tv_params,
                                               total_time=total_time,
                                               event=event,
                                               user_flow=user_flow,
                                               alpha=alpha,
                                               beta=beta)
                costs_list.append(cost_val)
            path_costs_for_agents[key] = costs_list

        # Use time-varying learning rate (example: depends on iteration, t=0)
        lr_current = tv_params.get_learning_rate(iteration, 0)
        route_choice_manager.update_all_policies(path_costs_for_agents, learning_rate=lr_current)

        # E) Convergence info
        if collect_convergence:
            if prev_avg_tt is not None:
                gap_val = abs(avg_tt - prev_avg_tt) / max(prev_avg_tt, 1e-9)
            else:
                gap_val = None
            prev_avg_tt = avg_tt
            print(f"Iteration={iteration}, AvgTT={avg_tt:.4f}, GAP={gap_val}")
            convergence_info.append({"Iteration": iteration,
                                     "AvgTT": avg_tt,
                                     "GAP": gap_val})

    # Collect final flows at the last time period
    final_flow_output = []
    for o in adj_list:
        for d in adj_list[o]:
            flow_arr = adj_list[o][d]['flow']
            flow_last = flow_arr[-1] if len(flow_arr) > 0 else 0.0
            final_flow_output.append([o, d, flow_last])

    if convergence_info is not None:
        conv_df = pd.DataFrame(convergence_info)
    else:
        conv_df = None

    df_last_time = pd.DataFrame(final_flow_output, columns=["Init Node", "Term Node", "Flow_TLast"])
    return conv_df, df_last_time


########################################################################
# 7. Main function
########################################################################
def main():
    network_path = 'network1.xlsx'
    od_path = 'od_data1.xlsx'

    if not os.path.exists(network_path) or not os.path.exists(od_path):
        print("Please ensure network1.xlsx and od_data1.xlsx exist in the current directory.")
        return

    # 1) Read data
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

    # 3) Define time-varying parameters (you can adjust them as needed)
    tvp = TimeVaryingParams(
        psi0=1.0,
        eta=0.5,
        lambda_=1.0,
        xi=1.0,
        c_=1.0,
        base_lr=2.0  # a bit larger initial learning rate
    )

    # 4) Run the multi-agent DTA
    t_start = time.time()
    conv_df, last_time_df = solve_multi_agent_dta_timevarying(
        adj_list, df_od,
        total_time=5,
        max_iterations=100,
        K_paths=3,
        event=True,
        user_flow=1.0,
        tv_params=tvp,
        alpha=0.15,  # BPR parameter
        beta=4.0,    # BPR parameter
        collect_convergence=True
    )
    t_end = time.time()
    print(f"\nExecution finished, time elapsed: {(t_end - t_start):.2f} seconds")

    # 5) Write results to Excel
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"multi_agent_dta_timevarying_{timestamp_str}.xlsx"
    with pd.ExcelWriter(out_file) as writer:
        if conv_df is not None:
            conv_df.to_excel(writer, sheet_name="Convergence_Info", index=False)
        last_time_df.to_excel(writer, sheet_name="LastTime_Flow", index=False)
    print(f"Convergence info & last-time flow have been written to -> {out_file}")


if __name__ == '__main__':
    main()
