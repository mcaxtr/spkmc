"""
Utilitários para aceleração GPU no SPKMC.

Este módulo contém funções utilitárias para aceleração GPU no algoritmo SPKMC,
incluindo verificação de disponibilidade de GPU e implementações GPU das funções principais.
"""

import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
from spkmc.cli.formatting import log_debug

def check_gpu_dependencies():
    """
    Verifica se as dependências para GPU estão instaladas.
    
    Returns:
        bool: True se todas as dependências estão disponíveis, False caso contrário
    """
    try:
        import cupy
        import cudf
        import cugraph
        return True
    except ImportError:
        return False

def is_gpu_available():
    """
    Verifica se a GPU está disponível para uso.
    
    Returns:
        bool: True se a GPU está disponível, False caso contrário
    """
    if not check_gpu_dependencies():
        return False
    
    try:
        import cupy as cp
        # Tenta alocar um pequeno array na GPU
        x = cp.array([1, 2, 3])
        return True
    except:
        return False

# Importações condicionais para evitar erros se as dependências não estiverem disponíveis
try:
    import cupy as cp
    import cudf
    import cugraph
    
    def get_dist_gpu(N, edges, sources, params):
        """
        Calcula as distâncias mínimas dos nós de origem para todos os outros nós usando GPU.
        
        Args:
            N: Número de nós
            edges: Arestas do grafo como matriz (u, v)
            sources: Nós de origem
            params: Parâmetros da distribuição
            
        Returns:
            Tupla com (distâncias, tempos de recuperação)
        """
        import time
        start_time = time.time()
        
        # 1. Transferência das arestas para GPU - uma única vez
        edges_gpu = cp.asarray(edges, dtype=cp.int32)
        
        # 2. Amostragem de tempos de recuperação e infecção na GPU
        if params['distribution'] == 'gamma':
            recovery_times = cp.random.gamma(
                params['shape'], params['scale'], size=N
            )
            times = cp.random.gamma(
                params['shape'], params['scale'], size=edges_gpu.shape[0]
            )
        else:
            recovery_times = cp.random.exponential(
                1/params['mu'], size=N
            )
            times = cp.random.exponential(
                params['lambda'], size=edges_gpu.shape[0]
            )
        
        # 3. Condição de infecção vs. recuperação
        u = edges_gpu[:, 0]
        infection_times = cp.where(times >= recovery_times[u], cp.inf, times)
        
        # 4. Super-nó para múltiplas fontes - otimizado para reduzir transferências
        super_node = N
        sources_gpu = cp.asarray(sources, dtype=cp.int32)
        
        # 5. Preparação dos dados para o grafo - otimizado para usar operações vetorizadas
        # Criar arrays diretamente na GPU
        num_edges = edges_gpu.shape[0]
        num_sources = sources_gpu.shape[0]
        total_edges = num_edges + num_sources
        
        # Alocar arrays de uma vez
        all_src = cp.empty(total_edges, dtype=cp.int32)
        all_dst = cp.empty(total_edges, dtype=cp.int32)
        all_weights = cp.empty(total_edges, dtype=cp.float32)
        
        # Preencher com dados existentes
        all_src[:num_edges] = edges_gpu[:, 0]
        all_dst[:num_edges] = edges_gpu[:, 1]
        all_weights[:num_edges] = infection_times.astype(cp.float32)
        
        # Adicionar super-nó
        all_src[num_edges:] = super_node
        all_dst[num_edges:] = sources_gpu
        all_weights[num_edges:] = 0
        
        # 6. Construção do DataFrame cuDF diretamente na GPU
        df = cudf.DataFrame({
            'src': all_src,
            'dst': all_dst,
            'weight': all_weights
        })
        
        # 7. Criação do grafo e execução de SSSP em GPU
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
        result = cugraph.sssp(G, source=super_node, edge_attr='weight')
        
        # 8. Extração das distâncias - manter na GPU
        dist_gpu = result['distance'].to_cupy()[:N]
        
        end_time = time.time()
        log_debug(f"GPU: Tempo total get_dist_gpu: {(end_time-start_time)*1000:.2f}ms")
        
        # Retornar sem converter para CPU
        return dist_gpu, recovery_times
    
    def get_states_gpu(time_to_infect, recovery_times, t):
        """
        Calcula os estados (S, I, R) para cada nó em um determinado tempo usando GPU.
        
        Args:
            time_to_infect: Tempo para infecção de cada nó (array CuPy)
            recovery_times: Tempo para recuperação de cada nó (array CuPy)
            t: Tempo atual da simulação
            
        Returns:
            Tupla com arrays booleanos (S, I, R) indicando o estado de cada nó
        """
        # Assumimos que os arrays já são CuPy - sem verificações ou conversões
        S = time_to_infect > t
        I = (~S) & (time_to_infect + recovery_times > t)
        R = (~S) & (~I)
        
        return S, I, R
    
    def calculate_gpu(N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray, time_steps: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula a proporção de indivíduos em cada estado (S, I, R) para cada passo de tempo usando GPU.
        
        Args:
            N: Número de nós no grafo
            time_to_infect: Tempo para infecção de cada nó (array NumPy ou CuPy)
            recovery_times: Tempo para recuperação de cada nó (array NumPy ou CuPy)
            time_steps: Array com os passos de tempo
            steps: Número de passos de tempo
            
        Returns:
            Tupla com arrays (S_time, I_time, R_time) contendo a proporção de indivíduos em cada estado
        """
        import time
        start_time = time.time()
        
        # Verificar se os arrays já estão na GPU para evitar transferências desnecessárias
        if isinstance(time_to_infect, cp.ndarray):
            time_to_infect_gpu = time_to_infect
        else:
            time_to_infect_gpu = cp.asarray(time_to_infect)
            
        if isinstance(recovery_times, cp.ndarray):
            recovery_times_gpu = recovery_times
        else:
            recovery_times_gpu = cp.asarray(recovery_times)
            
        if isinstance(time_steps, cp.ndarray):
            time_steps_gpu = time_steps
        else:
            time_steps_gpu = cp.asarray(time_steps)
        
        # Implementação vetorizada: processa todos os passos de tempo de uma vez
        # Reshape time_steps para permitir broadcasting com arrays de nós
        t_reshaped = time_steps_gpu.reshape(-1, 1)  # [steps, 1]
        
        # Calcular estados para todos os passos de tempo de uma vez
        # time_to_infect_gpu tem shape [N]
        # Resultado terá shape [steps, N]
        S_matrix = time_to_infect_gpu > t_reshaped  # [steps, N]
        I_matrix = (~S_matrix) & ((time_to_infect_gpu + recovery_times_gpu) > t_reshaped)  # [steps, N]
        R_matrix = (~S_matrix) & (~I_matrix)  # [steps, N]
        
        # Calcular proporções para cada passo de tempo
        S_time = cp.sum(S_matrix, axis=1) / N  # [steps]
        I_time = cp.sum(I_matrix, axis=1) / N  # [steps]
        R_time = cp.sum(R_matrix, axis=1) / N  # [steps]
        
        # Manter os resultados na GPU se possível
        # Só converter para CPU se necessário
        if cp.cuda.get_current_stream().done:
            # Se a stream atual estiver concluída, podemos retornar os arrays GPU diretamente
            result = S_time, I_time, R_time
        else:
            # Caso contrário, converter para CPU
            result = S_time.get(), I_time.get(), R_time.get()
        
        end_time = time.time()
        log_debug(f"GPU: Tempo total calculate_gpu: {(end_time-start_time)*1000:.2f}ms")
        
        return result
    
except ImportError:
    # Funções stub para quando as dependências não estão disponíveis
    def get_dist_gpu(N, edges, sources, params):
        raise ImportError("Dependências GPU (cupy-cuda12x, cudf-cu12, cugraph-cu12) não estão instaladas")
    
    def get_states_gpu(time_to_infect, recovery_times, t):
        raise ImportError("Dependências GPU (cupy-cuda12x) não estão instaladas")
    
    def calculate_gpu(N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray, time_steps: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise ImportError("Dependências GPU (cupy-cuda12x) não estão instaladas")