"""
Funções auxiliares otimizadas com Numba para o algoritmo SPKMC.

Este módulo contém funções auxiliares que são otimizadas usando a biblioteca Numba
para melhorar o desempenho das simulações SPKMC.
"""

import numpy as np
from numba import njit, prange, set_num_threads, config
from typing import Optional, Tuple

# Configurando Numba para usar o máximo nível de paralelismo
config.THREADING_LAYER = 'omp'
set_num_threads(8)  # Ajuste este número conforme necessário para o seu sistema


@njit
def gamma_sampling(shape: float, scale: float, size: Optional[int] = None) -> np.ndarray:
    """
    Amostragem da distribuição Gamma usando Numba para otimização.
    
    Args:
        shape: Parâmetro de forma da distribuição Gamma
        scale: Parâmetro de escala da distribuição Gamma
        size: Tamanho da amostra a ser gerada
        
    Returns:
        Amostra da distribuição Gamma
    """
    return np.random.gamma(shape, scale, size)


@njit
def get_weight_exponential(param: float, size: Optional[int] = None) -> np.ndarray:
    """
    Amostragem da distribuição Exponencial usando Numba para otimização.
    
    Args:
        param: Parâmetro da distribuição Exponencial (mu)
        size: Tamanho da amostra a ser gerada
        
    Returns:
        Amostra da distribuição Exponencial
    """
    return np.random.exponential(1 / param, size)


@njit(parallel=True)
def compute_infection_times_gamma(shape: float, scale: float, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Calcula os tempos de infecção usando a distribuição Gamma.
    
    Args:
        shape: Parâmetro de forma da distribuição Gamma
        scale: Parâmetro de escala da distribuição Gamma
        recovery_times: Tempos de recuperação para cada nó
        edges: Arestas do grafo como matriz (u, v)
        
    Returns:
        Tempos de infecção para cada aresta
    """
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in prange(num_edges):
        u = edges[i, 0]
        infection_time = gamma_sampling(shape, scale)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times


@njit(parallel=True)
def compute_infection_times_exponential(beta: float, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Calcula os tempos de infecção usando a distribuição Exponencial.
    
    Args:
        beta: Parâmetro da distribuição Exponencial (lambda)
        recovery_times: Tempos de recuperação para cada nó
        edges: Arestas do grafo como matriz (u, v)
        
    Returns:
        Tempos de infecção para cada aresta
    """
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in prange(num_edges):
        u = edges[i, 0]
        infection_time = get_weight_exponential(beta)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times


@njit
def get_states(time_to_infect: np.ndarray, time_to_recover: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula os estados (S, I, R) para cada nó em um determinado tempo.
    
    Args:
        time_to_infect: Tempo para infecção de cada nó
        time_to_recover: Tempo para recuperação de cada nó
        time: Tempo atual da simulação
        
    Returns:
        Tupla com arrays booleanos (S, I, R) indicando o estado de cada nó
    """
    S = time_to_infect > time
    I = ~S & (time_to_infect + time_to_recover > time)
    R = ~S & ~I
    return S, I, R


@njit(parallel=True)
def calculate(N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray, time_steps: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a proporção de indivíduos em cada estado (S, I, R) para cada passo de tempo.
    
    Args:
        N: Número de nós no grafo
        time_to_infect: Tempo para infecção de cada nó
        recovery_times: Tempo para recuperação de cada nó
        time_steps: Array com os passos de tempo
        steps: Número de passos de tempo
        
    Returns:
        Tupla com arrays (S_time, I_time, R_time) contendo a proporção de indivíduos em cada estado
    """
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