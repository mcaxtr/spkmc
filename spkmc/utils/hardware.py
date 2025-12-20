"""
Hardware detection and parallelization configuration for SPKMC.

This module provides automatic hardware detection (CPU cores, GPU availability)
and optimal parallelization strategy configuration for maximum performance.
"""

import os
import multiprocessing
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class HardwareInfo:
    """Container for detected hardware information."""
    cpu_count: int
    cpu_count_physical: int
    numba_threads: int
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    cuda_version: Optional[str] = None


@dataclass
class ParallelizationStrategy:
    """Configuration for multi-level parallelization."""
    scenario_workers: int      # Level 1: multiprocessing for scenarios
    simulation_workers: int    # Level 2: joblib for samples/runs
    numba_threads: int         # Level 3: Numba OpenMP threads
    use_gpu: bool              # GPU acceleration flag

    @classmethod
    def auto_configure(
        cls,
        hardware: HardwareInfo,
        num_scenarios: int = 1
    ) -> 'ParallelizationStrategy':
        """
        Automatically configure parallelization based on hardware and workload.

        Strategy to avoid nested parallelism conflicts:
        - If many scenarios: parallelize at scenario level, sequential simulations
        - If few scenarios: sequential scenarios, parallel simulations
        - Numba threads always reserved for inner loops

        Args:
            hardware: Detected hardware information
            num_scenarios: Number of scenarios to execute

        Returns:
            Optimally configured ParallelizationStrategy
        """
        available_cores = hardware.cpu_count_physical

        # Reserve cores for Numba inner loops (25% of cores, min 2, max 8)
        numba_threads = max(2, min(available_cores // 4, 8))
        remaining_cores = max(1, available_cores - numba_threads)

        if num_scenarios >= 4 and num_scenarios >= remaining_cores:
            # Many scenarios: parallelize at scenario level
            scenario_workers = min(num_scenarios, remaining_cores)
            simulation_workers = 1
        elif num_scenarios > 1:
            # Few scenarios: balance between levels
            scenario_workers = min(num_scenarios, max(1, remaining_cores // 2))
            simulation_workers = max(1, remaining_cores // (scenario_workers * 2))
        else:
            # Single scenario: all parallelism at simulation level
            scenario_workers = 1
            simulation_workers = remaining_cores

        return cls(
            scenario_workers=scenario_workers,
            simulation_workers=simulation_workers,
            numba_threads=numba_threads,
            use_gpu=hardware.gpu_available
        )


def detect_cpu_cores() -> Tuple[int, int]:
    """
    Detect available CPU cores dynamically.

    Returns:
        Tuple of (logical_cores, physical_cores)
    """
    logical_cores = os.cpu_count() or 1

    # Try to get physical core count using psutil
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or logical_cores
    except ImportError:
        # Estimate physical cores as half of logical (assuming hyperthreading)
        physical_cores = max(1, logical_cores // 2)

    return logical_cores, physical_cores


def detect_gpu() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Detect GPU availability and capabilities.

    Returns:
        Tuple of (is_available, gpu_info_dict)
    """
    gpu_info = None

    try:
        import cupy as cp

        # Try to access GPU
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)

        gpu_info = {
            'name': props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name'],
            'memory_mb': props['totalGlobalMem'] // (1024 * 1024),
            'cuda_version': f"{cp.cuda.runtime.runtimeGetVersion() // 1000}.{(cp.cuda.runtime.runtimeGetVersion() % 1000) // 10}",
            'compute_capability': f"{props['major']}.{props['minor']}"
        }

        # Verify cudf and cugraph are also available for full GPU support
        import cudf
        import cugraph

        return True, gpu_info

    except ImportError:
        return False, None
    except Exception as e:
        warnings.warn(f"GPU detection failed: {e}")
        return False, None


def get_hardware_info() -> HardwareInfo:
    """
    Collect all hardware information.

    Returns:
        HardwareInfo dataclass with detected hardware details
    """
    logical_cores, physical_cores = detect_cpu_cores()
    gpu_available, gpu_info = detect_gpu()

    # Calculate optimal Numba threads (cap at 16)
    numba_threads = min(physical_cores, 16)

    return HardwareInfo(
        cpu_count=logical_cores,
        cpu_count_physical=physical_cores,
        numba_threads=numba_threads,
        gpu_available=gpu_available,
        gpu_name=gpu_info.get('name') if gpu_info else None,
        gpu_memory_mb=gpu_info.get('memory_mb') if gpu_info else None,
        cuda_version=gpu_info.get('cuda_version') if gpu_info else None
    )


def configure_numba_threads(thread_count: Optional[int] = None) -> int:
    """
    Configure Numba thread count dynamically.

    Args:
        thread_count: Optional override for thread count

    Returns:
        Configured thread count
    """
    # Suppress OpenMP deprecation warning before importing Numba
    # KMP_WARNINGS=0 suppresses Intel OpenMP informational messages
    os.environ.setdefault('KMP_WARNINGS', '0')
    os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

    from numba import config, set_num_threads

    if thread_count is None:
        _, physical_cores = detect_cpu_cores()
        thread_count = min(physical_cores, 16)

    config.THREADING_LAYER = 'omp'
    set_num_threads(thread_count)

    return thread_count


def format_hardware_box(info: HardwareInfo, strategy: Optional[ParallelizationStrategy] = None) -> str:
    """
    Format hardware info as a rich box for CLI display.

    Args:
        info: Hardware information
        strategy: Optional parallelization strategy

    Returns:
        Formatted string for CLI output
    """
    lines = []

    # CPU info
    if strategy:
        cpu_line = f"  CPU: {info.cpu_count} cores ({info.cpu_count_physical} physical) → {strategy.scenario_workers} parallel workers"
    else:
        cpu_line = f"  CPU: {info.cpu_count} cores ({info.cpu_count_physical} physical)"
    lines.append(cpu_line)

    # GPU info
    if info.gpu_available and info.gpu_name:
        memory_str = f"{info.gpu_memory_mb // 1024}GB" if info.gpu_memory_mb and info.gpu_memory_mb >= 1024 else f"{info.gpu_memory_mb}MB"
        gpu_line = f"  GPU: {info.gpu_name} ({memory_str}) → Dijkstra acceleration"
    else:
        gpu_line = "  GPU: Not available → CPU mode"
    lines.append(gpu_line)

    # Numba threads
    if strategy:
        numba_line = f"  Numba: {strategy.numba_threads} threads (OpenMP)"
    else:
        numba_line = f"  Numba: {info.numba_threads} threads (OpenMP)"
    lines.append(numba_line)

    return "\n".join(lines)


def get_hardware_summary(info: HardwareInfo, strategy: ParallelizationStrategy) -> str:
    """
    Get a one-line hardware summary for progress display.

    Args:
        info: Hardware information
        strategy: Parallelization strategy

    Returns:
        Short summary string
    """
    parts = [f"{strategy.scenario_workers}x parallel"]

    if strategy.use_gpu:
        parts.append("GPU")

    return " | ".join(parts)
