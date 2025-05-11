##### FINAL VERSION - GAMMA IMPLEMENTED #####

import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads, config
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import os
import json

# Configurando Numba para usar o máximo nível de paralelismo
config.THREADING_LAYER = 'omp'
set_num_threads(8)  # Ajuste este número conforme necessário para o seu sistema

@njit
def gamma_sampling(shape, scale, size=None):
    return np.random.gamma(shape, scale, size)

@njit
def get_weight_exponential(param, size=None):
    return np.random.exponential(1 / param, size)

@njit(parallel=True)
def compute_infection_times_gamma(shape, scale, recovery_times, edges):
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in prange(num_edges):
        u = edges[i, 0]
        infection_time = gamma_sampling(shape, scale)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times

@njit(parallel=True)
def compute_infection_times_exponential(beta, recovery_times, edges):
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in prange(num_edges):
        u = edges[i, 0]
        infection_time = get_weight_exponential(beta)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times

def get_dist_sparse(N, edges, sources, params):
    if params['distribution'] == 'gamma':
        recovery_weights = gamma_sampling(params['shape'], params['scale'], N)
    elif params['distribution'] == 'exponential':
        recovery_weights = get_weight_exponential(params['mu'], N)
    else:
        raise ValueError("Unsupported distribution type")

    infection_times = compute_infection_times_exponential(params['lmbd'], recovery_weights, edges)
    
    row_indices = edges[:, 0]
    col_indices = edges[:, 1]
    
    graph_matrix = csr_matrix((infection_times, (row_indices, col_indices)), shape=(N, N))
    
    dist_matrix = dijkstra(csgraph = graph_matrix, directed = True, indices = sources, return_predecessors = False)
    dist = np.min(dist_matrix, axis = 0)

    if np.sum(dist == np.inf) > 0:
        print(dist)
        raise
        
    return dist, recovery_weights

@njit
def get_states(time_to_infect, time_to_recover, time):
    S = time_to_infect > time
    I = ~S & (time_to_infect + time_to_recover > time)
    R = ~S & ~I
    return S, I, R

@njit(parallel=True)
def calculate(N, time_to_infect, recovery_times, time_steps, steps):
    S_time = np.zeros(steps)
    I_time = np.zeros(steps)
    R_time = np.zeros(steps)
    
    for idx in prange(steps):
        time = time_steps[idx]
        S, I, R = get_states(time_to_infect, recovery_times, time)
        S_time[idx] = np.sum(S) / N
        I_time[idx] = np.sum(I) / N
        R_time[idx] = np.sum(R) / N

    return S_time, I_time, R_time

def spkmc(N, edges, sources, params, time, steps):
    time_to_infect, recovery_times = get_dist_sparse(N, edges, sources, params)
    return calculate(N, time_to_infect, recovery_times, time, steps)

def spkmc_avg(G, sources, params, time, samples, _tqdm):
    steps = time.shape[0]
    
    S_values = np.zeros((samples, steps))
    I_values = np.zeros((samples, steps))
    R_values = np.zeros((samples, steps))

    edges = np.array(G.edges())
    N = G.number_of_nodes()
    
    if _tqdm:
        sample_items = tqdm(range(samples))
    else:
        sample_items = range(samples)

    for sample in sample_items:
        S, I, R = spkmc(N, edges, sources, params, time, steps)
        S_values[sample, :] = S
        I_values[sample, :] = I
        R_values[sample, :] = R

    S_mean = np.mean(S_values, axis=0)
    I_mean = np.mean(I_values, axis=0)
    R_mean = np.mean(R_values, axis=0)

    return S_mean, I_mean, R_mean

def multiple_erdos_renyi(num, params, time, N=3000, k_avg=10, samples=100, initial_perc=0.01):
    S_list, I_list, R_list = [], [], []

    if params["distribution"] == "gamma":
        if f'results_{N}_{samples}_{params["shape"]}.json' in os.listdir(f'data/spkmc/gamma/ER/'):
            result = json.load(open(f'data/spkmc/gamma/ER/results_{N}_{samples}_{params["shape"]}.json'))
            return result.get('S_val', []), result.get('I_val', []), result.get('R_val', []), result.get('S_err', []), result.get('I_err', []), result.get('R_err', [])
    elif params["distribution"] == "exponential":        
        if f'results_{N}_{samples}.json' in os.listdir(f'data/spkmc/exponential/ER/'):
            result = json.load(open(f'data/spkmc/exponential/ER/results_{N}_{samples}.json'))
            return result.get('S_val', []), result.get('I_val', []), result.get('R_val', []), result.get('S_err', []), result.get('I_err', []), result.get('R_err', [])
    
    for run in tqdm(range(num)):
        p = k_avg / (N - 1)
        G = nx.erdos_renyi_graph(N, p, directed=True)
        init_infect = int(N * initial_perc)
        
        if init_infect < 1:
            raise Exception(f"Source is smaller than 1: N * initial_perc = {init_infect}")

        sources = np.random.randint(0, N, init_infect)
        
        S, I, R = spkmc_avg(G, sources, params, time, samples, _tqdm=False)
        
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

    S_avg = np.mean(np.array(S_list), axis=0)
    I_avg = np.mean(np.array(I_list), axis=0)
    R_avg = np.mean(np.array(R_list), axis=0)
    
    S_err = np.std(np.array(S_list) / np.sqrt(N), axis=0)
    I_err = np.std(np.array(I_list) / np.sqrt(N), axis=0)
    R_err = np.std(np.array(R_list) / np.sqrt(N), axis=0)

    result = {
        "S_val": list(S_avg),
        "S_err": list(S_err),
        "I_val": list(I_avg),
        "I_err": list(I_err),
        "R_val": list(R_avg),
        "R_err": list(R_err),
        "time": list(time),
    }
    
    if params["distribution"] == "gamma":
        with open(f'data/spkmc/gamma/ER/results_{N}_{samples}_{params["shape"]}.json', 'w') as f:
            f.write(json.dumps(result))
    elif params["distribution"] == "exponential":        
        with open(f'data/spkmc/exponential/ER/results_{N}_{samples}.json', 'w') as f:
            f.write(json.dumps(result))
        
    return S_avg, I_avg, R_avg, S_err, I_err, R_err

def generate_discrete_power_law(n, alpha, xmin, xmax):
    rand_nums = np.random.uniform(size = n)

    power_law_seq = (xmax**(1-alpha) - xmin**(1-alpha)) * rand_nums + xmin**(1-alpha)
    power_law_seq = power_law_seq**(1/(1-alpha))
    
    power_law_seq = power_law_seq.astype(int)

    if sum(power_law_seq) % 2 == 0:
        return power_law_seq
    else:
        power_law_seq[0] = power_law_seq[0] + 1
        return power_law_seq
        
def get_complex_network(N, exponent, k_avg):
    degree_sequence = generate_discrete_power_law(N, exponent, 2, np.sqrt(N))
    degree_sequence = np.round(degree_sequence * (k_avg / np.mean(degree_sequence))).astype(int)
    
    if sum(degree_sequence) % 2 != 0:
        degree_sequence[np.argmin(degree_sequence)] += 1
    
    G = nx.configuration_model(degree_sequence)
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def multiple_complex_network(num, params, exponent, time, N = 3000, k_avg = 10, samples = 100, initial_perc = 0.01):
    S_list = []
    I_list = []
    R_list = []
    
    if params["distribution"] == "gamma":
        if f'results_{str(exponent).replace(".", "")}_{N}_{samples}_{params["shape"]}.json' in os.listdir(f'data/spkmc/gamma/CN/'):
            result = json.load(open(f'data/spkmc/gamma/CN/results_{str(exponent).replace(".", "")}_{N}_{samples}_{params["shape"]}.json'))
            return result.get('S_val', []), result.get('I_val', []), result.get('R_val', []), result.get('S_err', []), result.get('I_err', []), result.get('R_err', [])
    elif params["distribution"] == "exponential":
        if f'results_{str(exponent).replace(".", "")}_{N}_{samples}.json' in os.listdir(f'data/spkmc/exponential/CN/'):
            result = json.load(open(f'data/spkmc/exponential/CN/results_{str(exponent).replace(".", "")}_{N}_{samples}.json'))
            return result.get('S_val', []), result.get('I_val', []), result.get('R_val', []), result.get('S_err', []), result.get('I_err', []), result.get('R_err', [])
        
    for run in tqdm(range(num), desc=f"{exponent}"):
        G           = get_complex_network(N, exponent, k_avg)
        init_infect = int(N * initial_perc)
        sources     = np.random.randint(0, N, init_infect)
        S, I, R     = spkmc_avg(G, sources, params, time, samples, _tqdm=False)
        
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

    S_avg = np.mean(np.array(S_list), axis = 0)
    I_avg = np.mean(np.array(I_list), axis = 0)
    R_avg = np.mean(np.array(R_list), axis = 0)
    
    S_err = np.std(np.array(S_list) / np.sqrt(N), axis=0)
    I_err = np.std(np.array(I_list) / np.sqrt(N), axis=0)
    R_err = np.std(np.array(R_list) / np.sqrt(N), axis=0)
    
    result = {
        "S_val": list(S_avg),
        "S_err": list(S_err),
        "I_val": list(I_avg),
        "I_err": list(I_err),
        "R_val": list(R_avg),
        "R_err": list(R_err),
        "time": list(time),
    }
    
    if params["distribution"] == "gamma":
        with open(f'data/spkmc/gamma/CN/results_{str(exponent).replace(".", "")}_{N}_{samples}_{params["shape"]}.json', 'w') as f:
            f.write(json.dumps(result))
    elif params["distribution"] == "exponential":
        with open(f'data/spkmc/exponential/CN/results_{str(exponent).replace(".", "")}_{N}_{samples}.json', 'w') as f:
            f.write(json.dumps(result))
            
    return S_avg, I_avg, R_avg, S_err, I_err, R_err

def complete_graph(params, time, N = 3000, samples = 100, initial_perc = 0.01, overwrite = False):

    if not overwrite:
        if params["distribution"] == "gamma":
            if f'results_{N}_{samples}_{params["shape"]}.json' in os.listdir(f'data/spkmc/gamma/CG/'):
                result = json.load(open(f'data/spkmc/gamma/CG/results_{N}_{samples}_{params["shape"]}.json'))
                return result.get('S_val', []), result.get('I_val', []), result.get('R_val', [])
        elif params["distribution"] == "exponential":
            if f'results_{N}_{samples}.json' in os.listdir(f'data/spkmc/exponential/CG/'):
                result = json.load(open(f'data/spkmc/exponential/CG/results_{N}_{samples}.json'))
                return result.get('S_val', []), result.get('I_val', []), result.get('R_val', [])

    G           = nx.complete_graph(N, create_using=nx.DiGraph())
    init_infect = int(N * initial_perc)
    sources     = np.random.randint(0, N, init_infect)
    S, I, R     = spkmc_avg(G, sources, params, time, samples, _tqdm = True)
    
    result = {
        "S_val": list(S),
        "I_val": list(I),
        "R_val": list(R),
        "time": list(time),
    }
    
    if params["distribution"] == "gamma":
        with open(f'data/spkmc/gamma/CG/results_{N}_{samples}_{params["shape"]}.json', 'w') as f:
            f.write(json.dumps(result))
    elif params["distribution"] == "exponential":
        with open(f'data/spkmc/exponential/CG/results_{N}_{samples}.json', 'w') as f:
            f.write(json.dumps(result))
            
    return S, I, R


# Plot com barras de erro
def plot_result_with_error(S, I, R, S_err, I_err, R_err, time):
    plt.figure(figsize=(10, 6))
    plt.errorbar(time, R, yerr=R_err, label='Recovered', capsize=2)
    plt.errorbar(time, I, yerr=I_err, label='Infected', capsize=2)
    plt.errorbar(time, S, yerr=S_err, label='Susceptible', capsize=2)
    
    plt.xlabel('Time')
    plt.ylabel('Proportion of Individuals')
    plt.title('SIR Model Dynamics Over Time with Error Bars')
    plt.legend()
    plt.show()