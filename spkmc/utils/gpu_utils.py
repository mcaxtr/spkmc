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
        log_debug(f"GPU: get_dist_gpu - Entrada: N={N}, edges.shape={edges.shape}, sources.shape={sources.shape}")
        log_debug(f"GPU: get_dist_gpu - Tipos: edges={type(edges)}, sources={type(sources)}")
        
        # 1. Transferência das arestas para GPU
        log_debug(f"GPU: get_dist_gpu - Convertendo edges para GPU")
        edges_gpu = cp.asarray(edges, dtype=cp.int32)
        log_debug(f"GPU: get_dist_gpu - edges_gpu={type(edges_gpu)}, shape={edges_gpu.shape}")

        # 2. Amostragem de tempos de recuperação e infecção na GPU
        log_debug(f"GPU: get_dist_gpu - Amostrando tempos de recuperação e infecção")
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
        log_debug(f"GPU: get_dist_gpu - recovery_times={type(recovery_times)}, times={type(times)}")

        # 3. Condição de infecção vs. recuperação
        log_debug(f"GPU: get_dist_gpu - Calculando condição de infecção vs. recuperação")
        u = edges_gpu[:, 0]
        log_debug(f"GPU: get_dist_gpu - u={type(u)}, shape={u.shape}")
        infection_times = cp.where(times >= recovery_times[u], cp.inf, times)
        log_debug(f"GPU: get_dist_gpu - infection_times={type(infection_times)}")

        # 4. Super-nó para múltiplas fontes
        log_debug(f"GPU: get_dist_gpu - Criando super-nó para múltiplas fontes")
        super_node = N
        # Converter sources para GPU se ainda não for um array CuPy
        if not isinstance(sources, cp.ndarray):
            log_debug(f"GPU: get_dist_gpu - Convertendo sources de {type(sources)} para cp.ndarray")
            sources_gpu = cp.asarray(sources, dtype=cp.int32)
        else:
            sources_gpu = sources
        super_src = cp.full_like(sources_gpu, super_node, dtype=cp.int32)
        log_debug(f"GPU: get_dist_gpu - super_src={type(super_src)}")
        dummy_edges = cp.stack([super_src, sources_gpu], axis=1)
        log_debug(f"GPU: get_dist_gpu - dummy_edges={type(dummy_edges)}")
        dummy_weights = cp.zeros_like(sources_gpu, dtype=cp.float32)
        log_debug(f"GPU: get_dist_gpu - dummy_weights={type(dummy_weights)}")

        # 5. Concatenação das arestas originais + super-nó
        log_debug(f"GPU: get_dist_gpu - Concatenando arestas")
        src = cp.concatenate([edges_gpu[:, 0], dummy_edges[:, 0]])
        dst = cp.concatenate([edges_gpu[:, 1], dummy_edges[:, 1]])
        weight = cp.concatenate([infection_times.astype(cp.float32), dummy_weights])
        log_debug(f"GPU: get_dist_gpu - src={type(src)}, dst={type(dst)}, weight={type(weight)}")

        # 6. Construção do DataFrame cuDF diretamente na GPU
        log_debug(f"GPU: get_dist_gpu - Construindo DataFrame cuDF")
        df = cudf.DataFrame({'src': src, 'dst': dst, 'weight': weight})
        log_debug(f"GPU: get_dist_gpu - df={type(df)}")

        # 7. Criação do grafo e execução de SSSP em GPU
        log_debug(f"GPU: get_dist_gpu - Criando grafo e executando SSSP")
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
        log_debug(f"GPU: get_dist_gpu - Executando SSSP com super_node={super_node}")
        # Usar o parâmetro correto edge_attr em vez de weight
        result = cugraph.sssp(G, source=super_node, edge_attr='weight')
        log_debug(f"GPU: get_dist_gpu - result={type(result)}")

        # 8. Extração das distâncias
        # Manter os dados na GPU como arrays CuPy
        dist_gpu = cp.asarray(result['distance'].to_array()[:N])
        log_debug(f"GPU: get_dist_gpu - Saída: dist_gpu={type(dist_gpu)}, recovery_times={type(recovery_times)}")
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
        log_debug(f"GPU: get_states_gpu - Entrada: t={t}")
        log_debug(f"GPU: get_states_gpu - Tipos: time_to_infect={type(time_to_infect)}, recovery_times={type(recovery_times)}")
        
        # Verificar se os arrays são do tipo CuPy
        if not isinstance(time_to_infect, cp.ndarray):
            log_debug(f"GPU: get_states_gpu - Convertendo time_to_infect de {type(time_to_infect)} para cp.ndarray")
            time_to_infect = cp.asarray(time_to_infect)
        if not isinstance(recovery_times, cp.ndarray):
            log_debug(f"GPU: get_states_gpu - Convertendo recovery_times de {type(recovery_times)} para cp.ndarray")
            recovery_times = cp.asarray(recovery_times)
            
        S = time_to_infect > t
        I = (~S) & (time_to_infect + recovery_times > t)
        R = (~S) & (~I)
        
        log_debug(f"GPU: get_states_gpu - Saída: S={type(S)}, I={type(I)}, R={type(R)}")
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
        log_debug(f"GPU: calculate_gpu - Entrada: N={N}, steps={steps}")
        log_debug(f"GPU: calculate_gpu - Tipos: time_to_infect={type(time_to_infect)}, recovery_times={type(recovery_times)}, time_steps={type(time_steps)}")
        if isinstance(time_to_infect, np.ndarray):
            log_debug(f"GPU: calculate_gpu - time_to_infect.shape={time_to_infect.shape}")
        if isinstance(recovery_times, np.ndarray):
            log_debug(f"GPU: calculate_gpu - recovery_times.shape={recovery_times.shape}")
        
        S_time = cp.zeros(steps)
        I_time = cp.zeros(steps)
        R_time = cp.zeros(steps)
        
        # Verificar se os arrays já são CuPy, caso contrário, converter
        if not isinstance(time_to_infect, cp.ndarray):
            log_debug(f"GPU: calculate_gpu - Convertendo time_to_infect de {type(time_to_infect)} para cp.ndarray")
            time_to_infect_gpu = cp.asarray(time_to_infect)
        else:
            time_to_infect_gpu = time_to_infect
            
        if not isinstance(recovery_times, cp.ndarray):
            log_debug(f"GPU: calculate_gpu - Convertendo recovery_times de {type(recovery_times)} para cp.ndarray")
            recovery_times_gpu = cp.asarray(recovery_times)
        else:
            recovery_times_gpu = recovery_times
            
        # Converter time_steps para CuPy
        log_debug(f"GPU: calculate_gpu - Convertendo time_steps de {type(time_steps)} para cp.ndarray")
        time_steps_gpu = cp.asarray(time_steps)
        
        for idx, t in enumerate(time_steps_gpu):
            S, I, R = get_states_gpu(
                time_to_infect_gpu, recovery_times_gpu, t
            )
            S_time[idx] = cp.sum(S) / N
            I_time[idx] = cp.sum(I) / N
            R_time[idx] = cp.sum(R) / N
            
        # Converter de volta para NumPy apenas no final
        log_debug(f"GPU: calculate_gpu - Saída: convertendo resultados de CuPy para NumPy")
        return S_time.get(), I_time.get(), R_time.get()
    
except ImportError:
    # Funções stub para quando as dependências não estão disponíveis
    def get_dist_gpu(N, edges, sources, params):
        raise ImportError("Dependências GPU (cupy-cuda12x, cudf-cu12, cugraph-cu12) não estão instaladas")
    
    def get_states_gpu(time_to_infect, recovery_times, t):
        raise ImportError("Dependências GPU (cupy-cuda12x) não estão instaladas")
    
    def calculate_gpu(N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray, time_steps: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise ImportError("Dependências GPU (cupy-cuda12x) não estão instaladas")