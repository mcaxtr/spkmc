"""
Funções auxiliares otimizadas com CuPy para execução em GPU no algoritmo SPKMC.

Este módulo contém funções auxiliares que são otimizadas usando a biblioteca CuPy
para execução em GPU, melhorando o desempenho das simulações SPKMC.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import time

# Importação condicional de CuPy
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.csgraph import dijkstra as cp_dijkstra
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# Função para medir o tempo de execução
def timing_decorator(func):
    """
    Decorador para medir o tempo de execução de uma função.
    
    Args:
        func: Função a ser medida
        
    Returns:
        Função decorada que imprime o tempo de execução
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Função {func.__name__} executada em {end_time - start_time:.6f} segundos")
        return result
    return wrapper


def is_gpu_available() -> bool:
    """
    Verifica se CuPy está instalado e se há GPUs disponíveis.
    
    Returns:
        True se CuPy estiver instalado e houver GPUs disponíveis, False caso contrário
    """
    if not CUPY_AVAILABLE:
        print("CuPy não está instalado. GPU não disponível.")
        return False
    
    try:
        # Verificar se o CUDA está disponível
        cuda_available = cp.cuda.is_available()
        if not cuda_available:
            print("CUDA não está disponível no sistema.")
            return False
        
        # Verificar se há dispositivos CUDA disponíveis
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            print("Nenhum dispositivo CUDA encontrado.")
            return False
        
        # Se chegou até aqui, pelo menos um dispositivo CUDA está disponível
        print(f"Encontrados {device_count} dispositivo(s) CUDA.")
        
        # Verificar a memória disponível em pelo menos um dispositivo
        for i in range(device_count):
            try:
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                if device_props['totalGlobalMem'] > 0:
                    print(f"GPU disponível: {device_props['name'].decode()} com {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
                    # Definir o dispositivo atual para garantir que ele seja usado
                    cp.cuda.Device(i).use()
                    # Testar uma operação simples para confirmar que a GPU está funcionando
                    test_array = cp.array([1, 2, 3])
                    test_result = cp.sum(test_array)
                    if test_result == 6:
                        print("Teste de operação GPU bem-sucedido.")
                        return True
                    else:
                        print("Teste de operação GPU falhou.")
            except Exception as e:
                print(f"Erro ao verificar propriedades do dispositivo {i}: {e}")
        
        print("Nenhum dispositivo CUDA com memória disponível encontrado.")
        return False
    except Exception as e:
        print(f"Erro ao verificar disponibilidade da GPU: {e}")
        return False


@timing_decorator
def print_gpu_info():
    """
    Imprime informações detalhadas sobre a GPU disponível.
    
    Returns:
        bool: True se a GPU estiver disponível, False caso contrário
    """
    if not CUPY_AVAILABLE:
        print("CuPy não está instalado. Não é possível obter informações da GPU.")
        return False
    
    try:
        if cp.cuda.is_available():
            print("\n===== INFORMAÇÕES DA GPU =====")
            print(f"CUDA disponível: Sim")
            
            # Versão do CUDA
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            print(f"Versão CUDA: {major}.{minor} (build {cuda_version})")
            
            # Informações do driver
            try:
                driver_version = cp.cuda.runtime.driverGetVersion()
                driver_major = driver_version // 1000
                driver_minor = (driver_version % 1000) // 10
                print(f"Versão do driver: {driver_major}.{driver_minor} (build {driver_version})")
            except:
                print("Não foi possível obter a versão do driver")
            
            # Número de dispositivos
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"Número de dispositivos: {device_count}")
            
            # Informações detalhadas de cada dispositivo
            for i in range(device_count):
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"\nDispositivo {i}: {device_props['name'].decode()}")
                print(f"  Memória total: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
                print(f"  Multiprocessadores: {device_props['multiProcessorCount']}")
                print(f"  Capacidade de computação: {device_props['major']}.{device_props['minor']}")
                print(f"  Clock máximo: {device_props['clockRate'] / 1000:.2f} MHz")
                print(f"  Largura de banda de memória: {2 * device_props['memoryClockRate'] * (device_props['memoryBusWidth'] / 8) / 1.0e6:.2f} GB/s")
                print(f"  Registradores por bloco: {device_props['regsPerBlock']}")
                print(f"  Memória compartilhada por bloco: {device_props['sharedMemPerBlock'] / 1024:.2f} KB")
                print(f"  Tamanho máximo de grid: ({device_props['maxGridSize'][0]}, {device_props['maxGridSize'][1]}, {device_props['maxGridSize'][2]})")
                print(f"  Tamanho máximo de bloco: ({device_props['maxThreadsDim'][0]}, {device_props['maxThreadsDim'][1]}, {device_props['maxThreadsDim'][2]})")
                print(f"  Threads máximas por bloco: {device_props['maxThreadsPerBlock']}")
                
                # Definir o dispositivo atual para obter informações de uso
                cp.cuda.Device(i).use()
                
                # Mostrar uso atual de memória
                mempool = cp.get_default_memory_pool()
                print(f"  Uso atual de memória: {mempool.used_bytes() / (1024**3):.2f} GB")
                print(f"  Memória total alocada: {mempool.total_bytes() / (1024**3):.2f} GB")
                
                # Verificar se há operações em execução
                try:
                    cp.cuda.Stream.null.synchronize()
                    print(f"  Status: Pronto para operações")
                except Exception as e:
                    print(f"  Status: Ocupado - {str(e)}")
            
            print("\n==============================")
            return True
        else:
            print("CUDA está instalado, mas não está disponível.")
            return False
    except Exception as e:
        print(f"Erro ao obter informações da GPU: {e}")
        return False


def to_numpy(array: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
    """
    Converte um array CuPy para NumPy, se necessário.
    
    Args:
        array: Array CuPy ou NumPy
        
    Returns:
        Array NumPy
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return array.get()
    return array


def to_cupy(array: Union[np.ndarray, "cp.ndarray"]) -> "cp.ndarray":
    """
    Converte um array NumPy para CuPy, se necessário.
    
    Args:
        array: Array NumPy ou CuPy
        
    Returns:
        Array CuPy
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    if isinstance(array, np.ndarray):
        return cp.array(array)
    return array


def gamma_sampling_gpu(shape: float, scale: float, size: Optional[int] = None) -> "cp.ndarray":
    """
    Amostragem da distribuição Gamma usando CuPy para execução em GPU.
    
    Args:
        shape: Parâmetro de forma da distribuição Gamma
        scale: Parâmetro de escala da distribuição Gamma
        size: Tamanho da amostra a ser gerada
        
    Returns:
        Amostra da distribuição Gamma como array CuPy
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    return cp.random.gamma(shape, scale, size)


def get_weight_exponential_gpu(param: float, size: Optional[int] = None) -> "cp.ndarray":
    """
    Amostragem da distribuição Exponencial usando CuPy para execução em GPU.
    
    Args:
        param: Parâmetro da distribuição Exponencial (mu)
        size: Tamanho da amostra a ser gerada
        
    Returns:
        Amostra da distribuição Exponencial como array CuPy
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    return cp.random.exponential(1 / param, size)


def compute_infection_times_gamma_gpu(shape: float, scale: float, recovery_times: "cp.ndarray", edges: "cp.ndarray") -> "cp.ndarray":
    """
    Calcula os tempos de infecção usando a distribuição Gamma em GPU.
    
    Args:
        shape: Parâmetro de forma da distribuição Gamma
        scale: Parâmetro de escala da distribuição Gamma
        recovery_times: Tempos de recuperação para cada nó (array CuPy)
        edges: Arestas do grafo como matriz (u, v) (array CuPy)
        
    Returns:
        Tempos de infecção para cada aresta como array CuPy
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    num_edges = edges.shape[0]
    infection_times = cp.empty(num_edges, dtype=cp.float64)
    
    # Gerar todos os tempos de infecção de uma vez
    infection_samples = cp.random.gamma(shape, scale, num_edges)
    
    # Extrair os índices dos nós de origem
    u_indices = edges[:, 0].astype(cp.int32)
    
    # Obter os tempos de recuperação correspondentes
    recovery_times_u = recovery_times[u_indices]
    
    # Comparar e definir infinito quando necessário
    mask = infection_samples >= recovery_times_u
    infection_times = cp.where(mask, cp.inf, infection_samples)
    
    return infection_times


def compute_infection_times_exponential_gpu(beta: float, recovery_times: "cp.ndarray", edges: "cp.ndarray") -> "cp.ndarray":
    """
    Calcula os tempos de infecção usando a distribuição Exponencial em GPU.
    
    Args:
        beta: Parâmetro da distribuição Exponencial (lambda)
        recovery_times: Tempos de recuperação para cada nó (array CuPy)
        edges: Arestas do grafo como matriz (u, v) (array CuPy)
        
    Returns:
        Tempos de infecção para cada aresta como array CuPy
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    num_edges = edges.shape[0]
    infection_times = cp.empty(num_edges, dtype=cp.float64)
    
    # Gerar todos os tempos de infecção de uma vez
    infection_samples = cp.random.exponential(1 / beta, num_edges)
    
    # Extrair os índices dos nós de origem
    u_indices = edges[:, 0].astype(cp.int32)
    
    # Obter os tempos de recuperação correspondentes
    recovery_times_u = recovery_times[u_indices]
    
    # Comparar e definir infinito quando necessário
    mask = infection_samples >= recovery_times_u
    infection_times = cp.where(mask, cp.inf, infection_samples)
    
    return infection_times


def get_states_gpu(time_to_infect: "cp.ndarray", time_to_recover: "cp.ndarray", time: float) -> Tuple["cp.ndarray", "cp.ndarray", "cp.ndarray"]:
    """
    Calcula os estados (S, I, R) para cada nó em um determinado tempo usando GPU.
    
    Args:
        time_to_infect: Tempo para infecção de cada nó (array CuPy)
        time_to_recover: Tempo para recuperação de cada nó (array CuPy)
        time: Tempo atual da simulação
        
    Returns:
        Tupla com arrays booleanos (S, I, R) indicando o estado de cada nó (arrays CuPy)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    S = time_to_infect > time
    I = ~S & (time_to_infect + time_to_recover > time)
    R = ~S & ~I
    return S, I, R


@timing_decorator
def calculate_gpu(N: int, time_to_infect: "cp.ndarray", recovery_times: "cp.ndarray", time_steps: "cp.ndarray", steps: int) -> Tuple["cp.ndarray", "cp.ndarray", "cp.ndarray"]:
    """
    Calcula a proporção de indivíduos em cada estado (S, I, R) para cada passo de tempo usando GPU.
    
    Args:
        N: Número de nós no grafo
        time_to_infect: Tempo para infecção de cada nó (array CuPy)
        recovery_times: Tempo para recuperação de cada nó (array CuPy)
        time_steps: Array com os passos de tempo (array CuPy)
        steps: Número de passos de tempo
        
    Returns:
        Tupla com arrays (S_time, I_time, R_time) contendo a proporção de indivíduos em cada estado (arrays CuPy)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    S_time = cp.zeros(steps, dtype=cp.float64)
    I_time = cp.zeros(steps, dtype=cp.float64)
    R_time = cp.zeros(steps, dtype=cp.float64)
    
    for idx in range(steps):
        time = time_steps[idx]
        S, I, R = get_states_gpu(time_to_infect, recovery_times, time)
        S_time[idx] = cp.sum(S) / N
        I_time[idx] = cp.sum(I) / N
        R_time[idx] = cp.sum(R) / N
    
    return S_time, I_time, R_time


@timing_decorator
def calculate_gpu_vectorized(N: int, time_to_infect: "cp.ndarray", recovery_times: "cp.ndarray",
                           time_steps: "cp.ndarray") -> Tuple["cp.ndarray", "cp.ndarray", "cp.ndarray"]:
    """
    Versão otimizada e vetorizada da função calculate_gpu que evita loops sequenciais.
    Utiliza operações de broadcasting para calcular os estados para todos os passos de tempo de uma vez,
    reduzindo significativamente o tempo de processamento.
    
    Args:
        N: Número de nós no grafo
        time_to_infect: Tempo para infecção de cada nó (array CuPy)
        recovery_times: Tempo para recuperação de cada nó (array CuPy)
        time_steps: Array com os passos de tempo (array CuPy)
        
    Returns:
        Tupla com arrays (S_time, I_time, R_time) contendo a proporção de indivíduos em cada estado (arrays CuPy)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    # Verificar e converter tipos de dados para garantir compatibilidade
    if time_to_infect.dtype != cp.float64:
        time_to_infect = time_to_infect.astype(cp.float64)
    
    if recovery_times.dtype != cp.float64:
        recovery_times = recovery_times.astype(cp.float64)
    
    if time_steps.dtype != cp.float64:
        time_steps = time_steps.astype(cp.float64)
    
    steps = len(time_steps)
    
    # Pré-alocar memória para os resultados
    S_time = cp.empty(steps, dtype=cp.float64)
    I_time = cp.empty(steps, dtype=cp.float64)
    R_time = cp.empty(steps, dtype=cp.float64)
    
    # Usar reshape para preparar os arrays para broadcasting
    # Isso permite operações vetorizadas em vez de loops
    expanded_time_to_infect = time_to_infect.reshape(-1, 1)  # Coluna: (N, 1)
    expanded_recovery_times = recovery_times.reshape(-1, 1)  # Coluna: (N, 1)
    expanded_time_steps = time_steps.reshape(1, -1)          # Linha: (1, steps)
    
    # Calcular a matriz de recuperação uma vez para reutilização
    recovery_end_times = expanded_time_to_infect + expanded_recovery_times  # (N, 1)
    
    # Calcular estados para todos os passos de tempo de uma vez usando broadcasting
    # Isso cria matrizes booleanas de tamanho (N, steps)
    S_matrix = expanded_time_to_infect > expanded_time_steps
    I_matrix = (~S_matrix) & (recovery_end_times > expanded_time_steps)
    
    # Não precisamos calcular R_matrix explicitamente, podemos usar a propriedade S + I + R = 1
    # R_matrix = (~S_matrix) & (~I_matrix)
    
    # Calcular proporções para cada passo de tempo usando soma ao longo do eixo dos nós
    # e dividindo pelo número total de nós
    S_time = cp.sum(S_matrix, axis=0, dtype=cp.float64) / N
    I_time = cp.sum(I_matrix, axis=0, dtype=cp.float64) / N
    
    # Calcular R_time usando a propriedade S + I + R = 1
    # Isso é mais eficiente do que calcular R_matrix explicitamente
    R_time = 1.0 - S_time - I_time
    
    # Liberar memória intermediária
    del S_matrix, I_matrix, expanded_time_to_infect, expanded_recovery_times, recovery_end_times
    
    # Sincronizar para garantir que todas as operações foram concluídas
    cp.cuda.Stream.null.synchronize()
    
    return S_time, I_time, R_time


@timing_decorator
def get_dist_sparse_gpu(edges: "cp.ndarray", infection_times: "cp.ndarray", sources: "cp.ndarray", N: int) -> "cp.ndarray":
    """
    Calcula as distâncias mínimas dos nós de origem para todos os outros nós usando GPU.
    Implementação otimizada do algoritmo de Dijkstra que mantém os dados na GPU durante todo o processamento.
    
    Args:
        edges: Arestas do grafo como matriz (u, v) (array CuPy)
        infection_times: Tempos de infecção para cada aresta (array CuPy)
        sources: Nós de origem (array CuPy)
        N: Número de nós no grafo
        
    Returns:
        Array com as distâncias mínimas (array CuPy)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy não está disponível")
    
    # Verificar e converter tipos de dados para garantir compatibilidade
    if edges.dtype != cp.int32:
        edges = edges.astype(cp.int32)
    
    if infection_times.dtype != cp.float64:
        infection_times = infection_times.astype(cp.float64)
    
    if sources.dtype != cp.int32:
        sources = sources.astype(cp.int32)
    
    # Criar a matriz esparsa do grafo de forma otimizada
    row_indices = edges[:, 0]
    col_indices = edges[:, 1]
    
    # Usar o construtor otimizado da matriz esparsa
    graph_matrix = cp_csr_matrix(
        (infection_times, (row_indices, col_indices)),
        shape=(N, N),
        dtype=cp.float64
    )
    
    # Otimizar a matriz esparsa para melhorar o desempenho do algoritmo de Dijkstra
    graph_matrix.sort_indices()
    graph_matrix.eliminate_zeros()
    
    # Calcular as distâncias mínimas usando a implementação do CuPy
    # Configurar parâmetros para melhor desempenho
    dist_matrix = cp_dijkstra(
        csgraph=graph_matrix,
        directed=True,
        indices=sources,
        return_predecessors=False,
        limit=cp.inf,  # Sem limite de distância
        min_only=True  # Otimização quando só precisamos do mínimo
    )
    
    # Calcular o mínimo das distâncias para cada nó de forma eficiente
    dist = cp.min(dist_matrix, axis=0)
    
    # Sincronizar para garantir que todas as operações foram concluídas
    cp.cuda.Stream.null.synchronize()
    
    return dist