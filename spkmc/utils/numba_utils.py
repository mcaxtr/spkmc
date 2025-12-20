"""
Funções auxiliares otimizadas com Numba para o algoritmo SPKMC.

Este módulo contém funções auxiliares que são otimizadas usando a biblioteca Numba
para melhorar o desempenho das simulações SPKMC.

The parallel implementation pre-generates random numbers before parallel loops
to avoid thread-safety issues with RNG inside prange.
"""

import os

# Suppress OpenMP warnings - must be set before Numba imports OpenMP
os.environ.setdefault('KMP_WARNINGS', '0')
os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

import numpy as np
from numba import njit, prange, get_num_threads
from typing import Tuple


# =============================================================================
# CORE COMPUTATION FUNCTIONS (parallelized where beneficial)
# =============================================================================

@njit
def get_states(
    time_to_infect: np.ndarray, time_to_recover: np.ndarray, time: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def compute_infection_times_gamma(
    shape: float, scale: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """
    Calcula os tempos de infecção usando a distribuição Gamma.

    Random numbers are pre-generated, then parallel loop does deterministic work.

    Args:
        shape: Parâmetro de forma da distribuição Gamma
        scale: Parâmetro de escala da distribuição Gamma
        recovery_times: Tempos de recuperação para cada nó
        edges: Arestas do grafo como matriz (u, v)

    Returns:
        Tempos de infecção para cada aresta
    """
    num_edges = edges.shape[0]

    # Pre-generate all random numbers (sequential, thread-safe)
    random_times = np.random.gamma(shape, scale, num_edges)

    # Parallel loop for deterministic comparison
    infection_times = np.empty(num_edges)
    for i in prange(num_edges):
        u = edges[i, 0]
        if random_times[i] >= recovery_times[u]:
            infection_times[i] = np.inf
        else:
            infection_times[i] = random_times[i]

    return infection_times


@njit(parallel=True)
def compute_infection_times_exponential(
    beta: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """
    Calcula os tempos de infecção usando a distribuição Exponencial.

    Random numbers are pre-generated, then parallel loop does deterministic work.

    Args:
        beta: Parâmetro da distribuição Exponencial (lambda)
        recovery_times: Tempos de recuperação para cada nó
        edges: Arestas do grafo como matriz (u, v)

    Returns:
        Tempos de infecção para cada aresta
    """
    num_edges = edges.shape[0]

    # Pre-generate all random numbers (sequential, thread-safe)
    random_times = np.random.exponential(1.0 / beta, num_edges)

    # Parallel loop for deterministic comparison
    infection_times = np.empty(num_edges)
    for i in prange(num_edges):
        u = edges[i, 0]
        if random_times[i] >= recovery_times[u]:
            infection_times[i] = np.inf
        else:
            infection_times[i] = random_times[i]

    return infection_times


@njit(parallel=True)
def calculate(
    N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray,
    time_steps: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a proporção de indivíduos em cada estado (S, I, R) para cada passo de tempo.

    Uses parallel loop over time steps for efficient computation.

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


# =============================================================================
# ARRAY SAMPLING FUNCTIONS (for recovery weights)
# =============================================================================

@njit
def gamma_sampling(shape: float, scale: float, size: int) -> np.ndarray:
    """
    Sample an array from Gamma distribution.

    Args:
        shape: Shape parameter of Gamma distribution
        scale: Scale parameter of Gamma distribution
        size: Number of samples

    Returns:
        Array of samples from Gamma distribution
    """
    return np.random.gamma(shape, scale, size)


@njit
def get_weight_exponential(param: float, size: int) -> np.ndarray:
    """
    Sample an array from Exponential distribution.

    Args:
        param: Rate parameter (mu) of Exponential distribution
        size: Number of samples

    Returns:
        Array of samples from Exponential distribution
    """
    return np.random.exponential(1.0 / param, size)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_numba_thread_count() -> int:
    """Return the number of threads Numba is using."""
    return get_num_threads()
