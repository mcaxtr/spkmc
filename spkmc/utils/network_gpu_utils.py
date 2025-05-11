"""
Utilitários para criação e manipulação de redes usando GPU no SPKMC.

Este módulo contém funções para criar e manipular redes diretamente usando GPU
através da biblioteca cugraph.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List

# Importações diretas das bibliotecas GPU
import cupy as cp
import cudf
import cugraph

def create_erdos_renyi_gpu(N: int, k_avg: float) -> Tuple[int, np.ndarray]:
    """
    Cria uma rede Erdos-Renyi diretamente na GPU.
    
    Args:
        N: Número de nós
        k_avg: Grau médio
        
    Returns:
        Tupla com (número de nós, arestas como matriz numpy)
    """
    import time
    from spkmc.cli.formatting import log_debug
    
    start_time = time.time()
    
    # Calcula a probabilidade de conexão
    p = k_avg / (N - 1)
    
    # Estima o número de arestas esperado
    expected_edges = int(p * N * (N-1))
    
    # Abordagem mais eficiente: gerar pares de nós diretamente
    # Gera mais pares do que o necessário para garantir que tenhamos o suficiente
    # após a filtragem pela probabilidade
    safety_factor = 1.2
    num_pairs_to_generate = int(expected_edges * safety_factor)
    
    # Gera pares de nós aleatórios
    sources = cp.random.randint(0, N, num_pairs_to_generate, dtype=cp.int32)
    targets = cp.random.randint(0, N, num_pairs_to_generate, dtype=cp.int32)
    
    # Remove auto-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    
    # Aplica a probabilidade de conexão
    random_values = cp.random.random(sources.shape)
    mask = random_values < p
    sources = sources[mask]
    targets = targets[mask]
    
    # Limita ao número esperado de arestas se tivermos mais
    if len(sources) > expected_edges:
        sources = sources[:expected_edges]
        targets = targets[:expected_edges]
    
    # Combina as fontes e alvos em uma matriz de arestas
    edges = cp.stack([sources, targets], axis=1)
    
    # Converte para NumPy para compatibilidade com o restante do código
    edges_np = edges.get()
    
    end_time = time.time()
    log_debug(f"GPU: Tempo total create_erdos_renyi_gpu: {(end_time-start_time)*1000:.2f}ms")
    
    return N, edges_np

def generate_power_law_sequence_gpu(N: int, exponent: float, k_avg: float) -> cp.ndarray:
    """
    Gera uma sequência de graus seguindo uma lei de potência na GPU.
    
    Args:
        N: Número de nós
        exponent: Expoente da lei de potência
        k_avg: Grau médio desejado
        
    Returns:
        Sequência de graus como array CuPy
    """
    # Gera a sequência de graus seguindo uma lei de potência
    rand_nums = cp.random.uniform(size=N)
    xmin = 2
    xmax = cp.sqrt(N)
    power_law_seq = (xmax**(1-exponent) - xmin**(1-exponent)) * rand_nums + xmin**(1-exponent)
    power_law_seq = power_law_seq**(1/(1-exponent))
    power_law_seq = cp.floor(power_law_seq).astype(cp.int32)
    
    # Ajusta para garantir que a soma seja par
    if cp.sum(power_law_seq) % 2 != 0:
        power_law_seq[0] += 1
    
    # Ajusta para o grau médio desejado
    power_law_seq = cp.round(power_law_seq * (k_avg / cp.mean(power_law_seq))).astype(cp.int32)
    
    return power_law_seq

def configuration_model_gpu(degree_sequence: cp.ndarray) -> cp.ndarray:
    """
    Implementação GPU do modelo de configuração para criar grafos com uma sequência de graus específica.
    
    Args:
        degree_sequence: Sequência de graus como array CuPy
        
    Returns:
        Matriz de arestas como array CuPy
    """
    import time
    from spkmc.cli.formatting import log_debug
    
    start_time = time.time()
    
    # Cria uma lista de "stubs" (meias-arestas) para cada nó
    # Cada nó i aparece degree_sequence[i] vezes na lista
    total_stubs = cp.sum(degree_sequence).item()
    stubs = cp.zeros(total_stubs, dtype=cp.int32)
    
    # Abordagem mais eficiente: usar operações vetorizadas em vez de loop Python
    # Cria um array de índices cumulativos
    cum_degrees = cp.cumsum(degree_sequence)
    start_indices = cp.zeros_like(degree_sequence)
    start_indices[1:] = cum_degrees[:-1]
    
    # Cria um kernel para preencher o array de stubs
    kernel_code = """
    extern "C" __global__
    void fill_stubs(int* stubs, int* degrees, int* start_indices, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) {
            int degree = degrees[i];
            int start = start_indices[i];
            for (int j = 0; j < degree; j++) {
                stubs[start + j] = i;
            }
        }
    }
    """
    
    # Compila e executa o kernel
    try:
        import cupy.cuda.compiler as compiler
        kernel = compiler.RawKernel(kernel_code, 'fill_stubs')
        
        # Executa o kernel
        threads_per_block = 256
        blocks_per_grid = (len(degree_sequence) + threads_per_block - 1) // threads_per_block
        kernel((blocks_per_grid,), (threads_per_block,),
               (stubs, degree_sequence, start_indices, len(degree_sequence)))
    except Exception as e:
        # Fallback para implementação Python se o kernel falhar
        log_debug(f"SPKMC: Erro ao executar kernel CUDA: {e}. Usando implementação Python.")
        idx = 0
        for i in range(len(degree_sequence)):
            degree = degree_sequence[i].item()
            stubs[idx:idx+degree] = i
            idx += degree
    
    # Embaralha os stubs
    cp.random.shuffle(stubs)
    
    # Cria as arestas conectando pares de stubs
    num_edges = len(stubs) // 2
    sources = stubs[:num_edges]
    targets = stubs[num_edges:]
    
    # Embaralha os alvos para evitar padrões
    cp.random.shuffle(targets)
    
    # Combina as fontes e alvos em uma matriz de arestas
    edges = cp.stack([sources, targets], axis=1)
    
    # Remove auto-loops (arestas que conectam um nó a ele mesmo)
    mask = sources != targets
    edges = edges[mask]
    
    end_time = time.time()
    log_debug(f"GPU: Tempo total configuration_model_gpu: {(end_time-start_time)*1000:.2f}ms")
    
    return edges

def create_complex_network_gpu(N: int, exponent: float, k_avg: float) -> Tuple[int, np.ndarray]:
    """
    Cria uma rede complexa com distribuição de grau seguindo lei de potência usando GPU.
    
    Args:
        N: Número de nós
        exponent: Expoente da lei de potência
        k_avg: Grau médio
        
    Returns:
        Tupla com (número de nós, arestas como matriz numpy)
    """
    import time
    from spkmc.cli.formatting import log_debug
    
    start_time = time.time()
    
    # Gera a sequência de graus na GPU
    degree_sequence_gpu = generate_power_law_sequence_gpu(N, exponent, k_avg)
    
    # Cria o grafo usando o modelo de configuração na GPU
    edges_gpu = configuration_model_gpu(degree_sequence_gpu)
    
    # Converte para NumPy para compatibilidade com o restante do código
    edges_np = edges_gpu.get()
    
    end_time = time.time()
    log_debug(f"GPU: Tempo total create_complex_network_gpu: {(end_time-start_time)*1000:.2f}ms")
    
    return N, edges_np

def create_complete_graph_gpu(N: int) -> Tuple[int, np.ndarray]:
    """
    Cria um grafo completo usando GPU.
    
    Args:
        N: Número de nós
        
    Returns:
        Tupla com (número de nós, arestas como matriz numpy)
    """
    import time
    from spkmc.cli.formatting import log_debug
    
    start_time = time.time()
    
    # Número total de arestas em um grafo completo direcionado sem auto-loops
    num_edges = N * (N - 1)
    
    # Abordagem mais eficiente: gerar todas as combinações possíveis de nós
    # sem criar uma matriz de adjacência completa
    sources = cp.zeros(num_edges, dtype=cp.int32)
    targets = cp.zeros(num_edges, dtype=cp.int32)
    
    # Preenche as fontes e alvos
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j:  # Evita auto-loops
                sources[idx] = i
                targets[idx] = j
                idx += 1
    
    # Combina as fontes e alvos em uma matriz de arestas
    edges = cp.stack([sources, targets], axis=1)
    
    # Converte para NumPy para compatibilidade com o restante do código
    edges_np = edges.get()
    
    end_time = time.time()
    log_debug(f"GPU: Tempo total create_complete_graph_gpu: {(end_time-start_time)*1000:.2f}ms")
    
    return N, edges_np