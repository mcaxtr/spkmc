"""
Funções auxiliares otimizadas com Numba para o algoritmo SPKMC.

Este módulo contém funções auxiliares que são otimizadas usando a biblioteca Numba
para melhorar o desempenho das simulações SPKMC.

Environment Variables:
    SPKMC_DISABLE_NUMBA_PARALLEL: Set to "1" to disable parallel Numba execution.
                                   Useful when experiencing threading conflicts on Linux.
"""

import os

# Suppress OpenMP warnings - must be set before Numba imports OpenMP
os.environ.setdefault('KMP_WARNINGS', '0')
os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

import numpy as np
from numba import njit, prange
from typing import Tuple

# Check if parallel execution should be disabled
# Set SPKMC_DISABLE_NUMBA_PARALLEL=1 to disable parallel Numba functions
_DISABLE_PARALLEL = os.environ.get('SPKMC_DISABLE_NUMBA_PARALLEL', '0') == '1'

# Track if parallel execution has failed (auto-fallback)
_parallel_failed = False

# NOTE: Thread count is set via NUMBA_NUM_THREADS environment variable
# Do NOT call set_num_threads() or set THREADING_LAYER here - it causes conflicts


# =============================================================================
# SEQUENTIAL VERSIONS (always work)
# =============================================================================

@njit
def _gamma_sampling_single(shape: float, scale: float) -> float:
    """Sample a single value from Gamma distribution."""
    return np.random.gamma(shape, scale)


@njit
def _exponential_sampling_single(param: float) -> float:
    """Sample a single value from Exponential distribution."""
    return np.random.exponential(1.0 / param)


@njit
def _compute_infection_times_gamma_seq(
    shape: float, scale: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Compute infection times using Gamma distribution (sequential)."""
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in range(num_edges):
        u = edges[i, 0]
        infection_time = _gamma_sampling_single(shape, scale)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times


@njit
def _compute_infection_times_exponential_seq(
    beta: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Compute infection times using Exponential distribution (sequential)."""
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in range(num_edges):
        u = edges[i, 0]
        infection_time = _exponential_sampling_single(beta)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times


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


@njit
def _calculate_seq(
    N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray,
    time_steps: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate SIR proportions over time (sequential)."""
    S_time = np.zeros(steps)
    I_time = np.zeros(steps)
    R_time = np.zeros(steps)

    for idx in range(steps):
        time = time_steps[idx]
        S, I, R = get_states(time_to_infect, recovery_times, time)
        S_time[idx] = np.sum(S) / N
        I_time[idx] = np.sum(I) / N
        R_time[idx] = np.sum(R) / N

    return S_time, I_time, R_time


# =============================================================================
# PARALLEL VERSIONS (may fail on some systems)
# =============================================================================

@njit(parallel=True)
def _compute_infection_times_gamma_par(
    shape: float, scale: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Compute infection times using Gamma distribution (parallel)."""
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in prange(num_edges):
        u = edges[i, 0]
        infection_time = _gamma_sampling_single(shape, scale)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times


@njit(parallel=True)
def _compute_infection_times_exponential_par(
    beta: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Compute infection times using Exponential distribution (parallel)."""
    num_edges = edges.shape[0]
    infection_times = np.empty(num_edges)

    for i in prange(num_edges):
        u = edges[i, 0]
        infection_time = _exponential_sampling_single(beta)
        infection_times[i] = np.inf if infection_time >= recovery_times[u] else infection_time

    return infection_times


@njit(parallel=True)
def _calculate_par(
    N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray,
    time_steps: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate SIR proportions over time (parallel)."""
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
# PUBLIC API (with automatic fallback)
# =============================================================================

def compute_infection_times_gamma(
    shape: float, scale: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """
    Calcula os tempos de infecção usando a distribuição Gamma.

    Tenta usar execução paralela primeiro, cai para sequencial se falhar.

    Args:
        shape: Parâmetro de forma da distribuição Gamma
        scale: Parâmetro de escala da distribuição Gamma
        recovery_times: Tempos de recuperação para cada nó
        edges: Arestas do grafo como matriz (u, v)

    Returns:
        Tempos de infecção para cada aresta
    """
    global _parallel_failed

    if _DISABLE_PARALLEL or _parallel_failed:
        return _compute_infection_times_gamma_seq(shape, scale, recovery_times, edges)

    try:
        return _compute_infection_times_gamma_par(shape, scale, recovery_times, edges)
    except Exception:
        _parallel_failed = True
        return _compute_infection_times_gamma_seq(shape, scale, recovery_times, edges)


def compute_infection_times_exponential(
    beta: float, recovery_times: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """
    Calcula os tempos de infecção usando a distribuição Exponencial.

    Tenta usar execução paralela primeiro, cai para sequencial se falhar.

    Args:
        beta: Parâmetro da distribuição Exponencial (lambda)
        recovery_times: Tempos de recuperação para cada nó
        edges: Arestas do grafo como matriz (u, v)

    Returns:
        Tempos de infecção para cada aresta
    """
    global _parallel_failed

    if _DISABLE_PARALLEL or _parallel_failed:
        return _compute_infection_times_exponential_seq(beta, recovery_times, edges)

    try:
        return _compute_infection_times_exponential_par(beta, recovery_times, edges)
    except Exception:
        _parallel_failed = True
        return _compute_infection_times_exponential_seq(beta, recovery_times, edges)


def calculate(
    N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray,
    time_steps: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a proporção de indivíduos em cada estado (S, I, R) para cada passo de tempo.

    Tenta usar execução paralela primeiro, cai para sequencial se falhar.

    Args:
        N: Número de nós no grafo
        time_to_infect: Tempo para infecção de cada nó
        recovery_times: Tempo para recuperação de cada nó
        time_steps: Array com os passos de tempo
        steps: Número de passos de tempo

    Returns:
        Tupla com arrays (S_time, I_time, R_time) contendo a proporção de indivíduos em cada estado
    """
    global _parallel_failed

    if _DISABLE_PARALLEL or _parallel_failed:
        return _calculate_seq(N, time_to_infect, recovery_times, time_steps, steps)

    try:
        return _calculate_par(N, time_to_infect, recovery_times, time_steps, steps)
    except Exception:
        _parallel_failed = True
        return _calculate_seq(N, time_to_infect, recovery_times, time_steps, steps)


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
