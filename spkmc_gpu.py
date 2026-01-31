##### GPU-ACCELERATED VERSION - SPKMC SIMULATION #####

import json
import os

import cudf
import cugraph
import cupy as cp
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


def get_dist_gpu(N, edges, sources, params):
    # 1. Transfer edges to the GPU
    edges_gpu = cp.asarray(edges, dtype=cp.int32)

    # 2. Sample recovery and infection times on the GPU
    if params["distribution"] == "gamma":
        recovery_times = cp.random.gamma(
            params["shape"], params["scale"], size=N
        )  # Gamma on the GPU :contentReference[oaicite:5]{index=5}
        times = cp.random.gamma(params["shape"], params["scale"], size=edges_gpu.shape[0])
    else:
        recovery_times = cp.random.exponential(
            1 / params["mu"], size=N
        )  # Exponential on the GPU :contentReference[oaicite:6]{index=6}
        times = cp.random.exponential(params["lmbd"], size=edges_gpu.shape[0])

    # 3. Infection vs. recovery condition
    u = edges_gpu[:, 0]
    infection_times = cp.where(times >= recovery_times[u], cp.inf, times)

    # 4. Super-node for multiple sources
    super_node = N
    super_src = cp.full_like(sources, super_node, dtype=cp.int32)
    dummy_edges = cp.stack([super_src, sources], axis=1)
    dummy_weights = cp.zeros_like(sources, dtype=cp.float32)

    # 5. Concatenate original edges + super-node
    src = cp.concatenate([edges_gpu[:, 0], dummy_edges[:, 0]])
    dst = cp.concatenate([edges_gpu[:, 1], dummy_edges[:, 1]])
    weight = cp.concatenate([infection_times.astype(cp.float32), dummy_weights])

    # 6. Build cuDF DataFrame directly on the GPU
    df = cudf.DataFrame({"src": src, "dst": dst, "weight": weight})

    # 7. Build the graph and run SSSP on the GPU
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="weight"
    )  # efficient construction :contentReference[oaicite:7]{index=7}
    result = cugraph.sssp(
        G, source=super_node, weight="weight"
    )  # Dijkstra on the GPU :contentReference[oaicite:8]{index=8}

    # 8. Extract distances back to the host
    dist_gpu = result["distance"].to_numpy()[:N]
    return dist_gpu, recovery_times.get()


def get_states_gpu(time_to_infect, recovery_times, t):
    S = time_to_infect > t
    I = (~S) & (time_to_infect + recovery_times > t)
    R = (~S) & (~I)
    return S, I, R


def calculate_gpu(N, time_to_infect, recovery_times, time_steps):
    steps = len(time_steps)
    S_time = cp.zeros(steps)
    I_time = cp.zeros(steps)
    R_time = cp.zeros(steps)
    for idx, t in enumerate(time_steps):
        S, I, R = get_states_gpu(cp.asarray(time_to_infect), cp.asarray(recovery_times), t)
        S_time[idx] = cp.sum(S) / N
        I_time[idx] = cp.sum(I) / N
        R_time[idx] = cp.sum(R) / N
    return S_time.get(), I_time.get(), R_time.get()


def spkmc_gpu(N, edges, sources, params, time_steps):
    d_inf, rec = get_dist_gpu(N, edges, sources, params)
    return calculate_gpu(N, d_inf, rec, time_steps)


def spkmc_avg_gpu(G, sources, params, time_steps, samples, use_tqdm=True):
    N = G.number_of_nodes()
    steps = len(time_steps)
    S_vals = cp.zeros((samples, steps))
    I_vals = cp.zeros((samples, steps))
    R_vals = cp.zeros((samples, steps))
    # Collect NetworkX edges for the GPU
    edges = cp.asarray(list(G.edges()), dtype=cp.int32)
    iterator = tqdm(range(samples)) if use_tqdm else range(samples)
    for i in iterator:
        S, I, R = spkmc_gpu(N, edges, sources, params, time_steps)
        S_vals[i, :] = cp.asarray(S)
        I_vals[i, :] = cp.asarray(I)
        R_vals[i, :] = cp.asarray(R)
    return (
        cp.mean(S_vals, axis=0).get(),
        cp.mean(I_vals, axis=0).get(),
        cp.mean(R_vals, axis=0).get(),
    )


def multiple_erdos_renyi_gpu(
    num, params, time_steps, N=3000, k_avg=10, samples=100, initial_perc=0.01, overwrite=False
):
    # Same on-disk cache check logic...
    S_list, I_list, R_list = [], [], []
    for _ in tqdm(range(num)):
        p = k_avg / (N - 1)
        G = nx.erdos_renyi_graph(N, p, directed=True)
        init_infect = max(1, int(N * initial_perc))
        sources = cp.random.randint(0, N, size=init_infect, dtype=cp.int32)
        S, I, R = spkmc_avg_gpu(G, sources, params, time_steps, samples, use_tqdm=False)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    # Final aggregations as in the CPU version...
    # (mean, standard error, JSON output, etc.)


def plot_result_with_error_gpu(S, I, R, S_err, I_err, R_err, time_steps):
    plt.figure(figsize=(10, 6))
    plt.errorbar(time_steps, R, yerr=R_err, label="Recovered", capsize=2)
    plt.errorbar(time_steps, I, yerr=I_err, label="Infected", capsize=2)
    plt.errorbar(time_steps, S, yerr=S_err, label="Susceptible", capsize=2)
    plt.xlabel("Time")
    plt.ylabel("Proportion of Individuals")
    plt.title("SIR Dynamics with Error Bars (GPU)")
    plt.legend()
    plt.show()
