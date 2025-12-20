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

# Debug flag - set SPKMC_DEBUG=1 for verbose logging
_DEBUG = os.environ.get('SPKMC_DEBUG', '0') == '1'


def _log(msg):
    """Print debug message if debugging is enabled."""
    if _DEBUG:
        print(f"[NUMBA DEBUG] {msg}", file=sys.stderr)


def get_numba_info():
    """Get information about Numba configuration."""
    info = {
        'num_threads': get_num_threads(),
        'threading_layer': config.THREADING_LAYER,
        'parallel': True,
    }
    _log(f"Numba info: {info}")
    return info


def clear_numba_cache():
    """Clear Numba's compilation cache."""
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
# CORE COMPUTATION FUNCTIONS
# Note: No type hints on njit functions to avoid Numba typing issues
# =============================================================================

@njit(cache=False)
def _get_states_at_time(time_to_infect, time_to_recover, time):
    """Calculate S, I, R counts at a specific time."""
    n = len(time_to_infect)
    s_count = 0
    i_count = 0
    r_count = 0

    for i in range(n):
        if time_to_infect[i] > time:
            s_count += 1
        elif time_to_infect[i] + time_to_recover[i] > time:
            i_count += 1
        else:
            r_count += 1

    return s_count, i_count, r_count


@njit(parallel=True, cache=False)
def compute_infection_times_gamma(shape, scale, recovery_times, edges):
    """Compute infection times using Gamma distribution."""
    num_edges = edges.shape[0]

    # Pre-generate all random numbers (thread-safe)
    random_times = np.random.gamma(shape, scale, num_edges)

    # Parallel loop for deterministic comparison
    infection_times = np.empty(num_edges, dtype=np.float64)
    for i in prange(num_edges):
        u = edges[i, 0]
        if random_times[i] >= recovery_times[u]:
            infection_times[i] = np.inf
        else:
            infection_times[i] = random_times[i]

    return infection_times


@njit(parallel=True, cache=False)
def compute_infection_times_exponential(beta, recovery_times, edges):
    """Compute infection times using Exponential distribution."""
    num_edges = edges.shape[0]

    # Pre-generate all random numbers (thread-safe)
    random_times = np.random.exponential(1.0 / beta, num_edges)

    # Parallel loop for deterministic comparison
    infection_times = np.empty(num_edges, dtype=np.float64)
    for i in prange(num_edges):
        u = edges[i, 0]
        if random_times[i] >= recovery_times[u]:
            infection_times[i] = np.inf
        else:
            infection_times[i] = random_times[i]

    return infection_times


@njit(cache=False)
def _calculate_sequential_impl(N, time_to_infect, recovery_times, time_steps):
    """Sequential implementation - no prange, no parallel."""
    steps = len(time_steps)
    S_time = np.empty(steps, dtype=np.float64)
    I_time = np.empty(steps, dtype=np.float64)
    R_time = np.empty(steps, dtype=np.float64)

    n_float = float(N)

    for idx in range(steps):
        t = time_steps[idx]
        s, i, r = _get_states_at_time(time_to_infect, recovery_times, t)
        S_time[idx] = s / n_float
        I_time[idx] = i / n_float
        R_time[idx] = r / n_float

    return S_time, I_time, R_time


@njit(parallel=True, cache=False)
def _calculate_parallel_impl(N, time_to_infect, recovery_times, time_steps):
    """Parallel implementation using prange over time steps."""
    steps = len(time_steps)
    S_time = np.empty(steps, dtype=np.float64)
    I_time = np.empty(steps, dtype=np.float64)
    R_time = np.empty(steps, dtype=np.float64)

    n_float = float(N)

    for idx in prange(steps):
        t = time_steps[idx]
        s, i, r = _get_states_at_time(time_to_infect, recovery_times, t)
        S_time[idx] = s / n_float
        I_time[idx] = i / n_float
        R_time[idx] = r / n_float

    return S_time, I_time, R_time


# Track if parallel execution has failed
_parallel_failed = False


def calculate(N, time_to_infect, recovery_times, time_steps, steps):
    """
    Calculate S, I, R proportions over time.

    Tries parallel first, falls back to sequential if it fails.
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
        return _calculate_sequential_impl(N, time_to_infect, recovery_times, time_steps)

    try:
        _log(f"Attempting parallel execution with {get_num_threads()} threads")
        result = _calculate_parallel_impl(N, time_to_infect, recovery_times, time_steps)
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
        return _calculate_sequential_impl(N, time_to_infect, recovery_times, time_steps)


# =============================================================================
# ARRAY SAMPLING FUNCTIONS (for recovery weights)
# =============================================================================

@njit(cache=False)
def gamma_sampling(shape, scale, size):
    """Sample an array from Gamma distribution."""
    return np.random.gamma(shape, scale, size)


@njit(cache=False)
def get_weight_exponential(param, size):
    """Sample an array from Exponential distribution."""
    return np.random.exponential(1.0 / param, size)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_numba_thread_count():
    """Return the number of threads Numba is using."""
    return get_num_threads()


# Log initialization
_log(f"numba_utils loaded: threads={get_num_threads()}, threading_layer={config.THREADING_LAYER}")
