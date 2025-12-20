"""
Funções auxiliares otimizadas com Numba para o algoritmo SPKMC.

Este módulo contém funções auxiliares que são otimizadas usando a biblioteca Numba
para melhorar o desempenho das simulações SPKMC.

The parallel implementation pre-generates random numbers before parallel loops
to avoid thread-safety issues with RNG inside prange.
"""

import os
import sys
import traceback

# Suppress OpenMP warnings - must be set before Numba imports OpenMP
os.environ.setdefault('KMP_WARNINGS', '0')
os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

import numpy as np
from numba import njit, prange, get_num_threads, config
from typing import Tuple

# Debug flag - set SPKMC_DEBUG=1 for verbose logging
_DEBUG = os.environ.get('SPKMC_DEBUG', '0') == '1'


def _log(msg: str) -> None:
    """Print debug message if debugging is enabled."""
    if _DEBUG:
        print(f"[NUMBA DEBUG] {msg}", file=sys.stderr)


def get_numba_info() -> dict:
    """Get information about Numba configuration."""
    info = {
        'num_threads': get_num_threads(),
        'threading_layer': config.THREADING_LAYER,
        'parallel': True,
    }
    _log(f"Numba info: {info}")
    return info


def clear_numba_cache() -> None:
    """Clear Numba's compilation cache."""
    import shutil
    from pathlib import Path

    # Clear __pycache__ directories with .nbc/.nbi files
    project_root = Path(__file__).parent.parent.parent
    cache_dirs = list(project_root.rglob('__pycache__'))

    cleared = 0
    for cache_dir in cache_dirs:
        for ext in ['*.nbc', '*.nbi']:
            for f in cache_dir.glob(ext):
                try:
                    f.unlink()
                    cleared += 1
                except Exception:
                    pass

    _log(f"Cleared {cleared} Numba cache files")
    print(f"[INFO] Cleared {cleared} Numba cache files")


# =============================================================================
# CORE COMPUTATION FUNCTIONS (parallelized where beneficial)
# =============================================================================

@njit(cache=False)  # Disable cache to avoid stale compilations
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


@njit(parallel=True, cache=False)
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


@njit(parallel=True, cache=False)
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


@njit(parallel=True, cache=False)
def _calculate_parallel(
    N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray,
    time_steps: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel implementation of calculate."""
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


@njit(cache=False)
def _calculate_sequential(
    N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray,
    time_steps: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential implementation of calculate (fallback)."""
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


# Track if parallel execution has failed
_parallel_failed = False


def calculate(
    N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray,
    time_steps: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a proporção de indivíduos em cada estado (S, I, R) para cada passo de tempo.

    Tries parallel execution first, falls back to sequential if it fails.

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

    _log(f"calculate() called: N={N}, steps={steps}, parallel_failed={_parallel_failed}")
    _log(f"  time_to_infect: shape={time_to_infect.shape}, dtype={time_to_infect.dtype}")
    _log(f"  recovery_times: shape={recovery_times.shape}, dtype={recovery_times.dtype}")
    _log(f"  time_steps: shape={time_steps.shape}, dtype={time_steps.dtype}")

    # Ensure all arrays are contiguous float64
    time_to_infect = np.ascontiguousarray(time_to_infect, dtype=np.float64)
    recovery_times = np.ascontiguousarray(recovery_times, dtype=np.float64)
    time_steps = np.ascontiguousarray(time_steps, dtype=np.float64)

    if _parallel_failed:
        _log("Using sequential execution (parallel previously failed)")
        return _calculate_sequential(N, time_to_infect, recovery_times, time_steps, steps)

    try:
        _log(f"Attempting parallel execution with {get_num_threads()} threads")
        result = _calculate_parallel(N, time_to_infect, recovery_times, time_steps, steps)
        _log("Parallel execution succeeded")
        return result
    except Exception as e:
        _parallel_failed = True
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"\n[WARNING] Parallel Numba execution failed: {error_msg}", file=sys.stderr)
        print(f"[WARNING] Traceback:\n{tb}", file=sys.stderr)
        print("[WARNING] Falling back to sequential execution\n", file=sys.stderr)
        _log(f"Parallel failed, using sequential: {error_msg}")
        return _calculate_sequential(N, time_to_infect, recovery_times, time_steps, steps)


# =============================================================================
# ARRAY SAMPLING FUNCTIONS (for recovery weights)
# =============================================================================

@njit(cache=False)
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


@njit(cache=False)
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


# Log initialization
_log(f"numba_utils loaded: threads={get_num_threads()}, threading_layer={config.THREADING_LAYER}")
