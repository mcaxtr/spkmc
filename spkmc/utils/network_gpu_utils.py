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
    
    # Cria uma matriz de adjacência aleatória na GPU
    random_matrix = cp.random.random((N, N))
    adj_matrix = random_matrix < p
    
    # Zera a diagonal (sem auto-loops)
    cp.fill_diagonal(adj_matrix, 0)
    
    # Extrai as arestas da matriz de adjacência
    edges_indices = cp.where(adj_matrix)
    sources = edges_indices[0]
    targets = edges_indices[1]
    
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
    # Cria uma lista de "stubs" (meias-arestas) para cada nó
    # Cada nó i aparece degree_sequence[i] vezes na lista
    stubs = cp.zeros(cp.sum(degree_sequence).item(), dtype=cp.int32)
    
    # Preenche a lista de stubs
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
    
    # Cria uma matriz de adjacência completa na GPU
    adj_matrix = cp.ones((N, N), dtype=cp.bool_)
    
    # Zera a diagonal (sem auto-loops)
    cp.fill_diagonal(adj_matrix, 0)
    
    # Extrai as arestas da matriz de adjacência
    edges_indices = cp.where(adj_matrix)
    sources = edges_indices[0]
    targets = edges_indices[1]
    
    # Combina as fontes e alvos em uma matriz de arestas
    edges = cp.stack([sources, targets], axis=1)
    
    # Converte para NumPy para compatibilidade com o restante do código
    edges_np = edges.get()
    
    end_time = time.time()
    log_debug(f"GPU: Tempo total create_complete_graph_gpu: {(end_time-start_time)*1000:.2f}ms")
    
    return N, edges_np