"""
Hardware detection and parallelization configuration for SPKMC.

This module provides automatic hardware detection (CPU cores, GPU availability)
and optimal parallelization strategy configuration for maximum performance.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


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

    scenario_workers: int  # Level 1: multiprocessing for scenarios
    simulation_workers: int  # Level 2: joblib for samples/runs
    numba_threads: int  # Level 3: Numba OpenMP threads
    use_gpu: bool  # GPU acceleration flag

    @classmethod
    def auto_configure(
        cls, hardware: HardwareInfo, num_scenarios: int = 1
    ) -> "ParallelizationStrategy":
        """
        Automatically configure parallelization based on hardware and workload.

        Strategy for two-level parallelism:
        - Level 1: ProcessPoolExecutor for scenarios (using spawn context)
        - Level 2: Numba parallel prange for inner loops (CPU) or GPU kernels

        For CPU mode: balance scenario_workers * numba_threads <= physical_cores
        For GPU mode: limit scenario_workers to avoid GPU memory contention

        Args:
            hardware: Detected hardware information
            num_scenarios: Number of scenarios to execute

        Returns:
            Optimally configured ParallelizationStrategy
        """
        available_cores = hardware.cpu_count_physical

        if hardware.gpu_available:
            # GPU mode: limit parallel workers to avoid GPU memory contention
            # Each process loads cupy/cudf/cugraph (~200-500MB GPU memory each)
            # GPU driver handles time-slicing, but too many processes cause thrashing
            max_gpu_workers = 4  # Conservative limit for GPU memory
            if num_scenarios >= 2:
                scenario_workers = min(num_scenarios, max_gpu_workers)
            else:
                scenario_workers = 1
            # In GPU mode, most computation happens on GPU, so CPU threads are less critical
            # Give more threads per worker since they won't be the bottleneck
            numba_threads = max(4, min(available_cores // 2, 8))
            simulation_workers = 1
        elif num_scenarios >= 4:
            # CPU mode, many scenarios: parallelize at scenario level
            # Limit scenario workers and reduce Numba threads per worker
            scenario_workers = min(num_scenarios, max(2, available_cores // 4))
            # Each worker gets fewer Numba threads to avoid oversubscription
            numba_threads = max(2, min(available_cores // scenario_workers, 8))
            simulation_workers = 1
        elif num_scenarios > 1:
            # CPU mode, few scenarios (2-3): balance between levels
            scenario_workers = min(num_scenarios, 2)
            numba_threads = max(2, min(available_cores // scenario_workers, 8))
            simulation_workers = 1
        else:
            # Single scenario: all parallelism at Numba level
            scenario_workers = 1
            # Use 75% of physical cores for Numba, max 16
            numba_threads = max(2, min((available_cores * 3) // 4, 16))
            simulation_workers = 1

        return cls(
            scenario_workers=scenario_workers,
            simulation_workers=simulation_workers,
            numba_threads=numba_threads,
            use_gpu=hardware.gpu_available,
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

    The gpu_info dict contains:
        - name: GPU device name
        - memory_mb: Total GPU memory in MB
        - cuda_version: CUDA runtime version
        - compute_capability: GPU compute capability
        - libs_available: List of available RAPIDS libraries
        - libs_missing: List of missing RAPIDS libraries (if any)
    """
    gpu_info = None
    libs_available = []
    libs_missing = []

    # First, check if CUDA is available via cupy
    try:
        import cupy as cp

        # Try to access GPU (Device() verifies GPU is accessible)
        _ = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)

        runtime_ver = cp.cuda.runtime.runtimeGetVersion()
        cuda_major = runtime_ver // 1000
        cuda_minor = (runtime_ver % 1000) // 10
        gpu_info = {
            "name": (
                props["name"].decode("utf-8") if isinstance(props["name"], bytes) else props["name"]
            ),
            "memory_mb": props["totalGlobalMem"] // (1024 * 1024),
            "cuda_version": f"{cuda_major}.{cuda_minor}",
            "compute_capability": f"{props['major']}.{props['minor']}",
        }
        libs_available.append("cupy")

    except ImportError:
        libs_missing.append("cupy")
        return False, {"libs_missing": libs_missing, "reason": "cupy not installed"}
    except Exception as e:
        return False, {"reason": f"CUDA initialization failed: {e}"}

    # Check for optional RAPIDS libraries
    try:
        import cudf  # noqa: F401

        libs_available.append("cudf")
    except ImportError:
        libs_missing.append("cudf")

    try:
        import cugraph  # noqa: F401

        libs_available.append("cugraph")
    except ImportError:
        libs_missing.append("cugraph")

    gpu_info["libs_available"] = libs_available
    gpu_info["libs_missing"] = libs_missing

    # GPU acceleration requires ALL libraries (cupy, cudf, cugraph) for SSSP calculation
    # If any are missing, GPU mode won't work - only report as available if all present
    gpu_fully_available = len(libs_missing) == 0
    return gpu_fully_available, gpu_info


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
        gpu_name=gpu_info.get("name") if gpu_info else None,
        gpu_memory_mb=gpu_info.get("memory_mb") if gpu_info else None,
        cuda_version=gpu_info.get("cuda_version") if gpu_info else None,
    )


def configure_numba_threads(thread_count: Optional[int] = None) -> int:
    """
    Configure Numba thread count dynamically.

    Once Numba threads have been launched, the thread count cannot be changed.
    In that case, this function returns the current thread count.

    Args:
        thread_count: Optional override for thread count

    Returns:
        Configured (or current) thread count
    """
    # Suppress OpenMP deprecation warning before importing Numba
    # KMP_WARNINGS=0 suppresses Intel OpenMP informational messages
    os.environ.setdefault("KMP_WARNINGS", "0")
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

    from numba import config, get_num_threads, set_num_threads

    if thread_count is None:
        _, physical_cores = detect_cpu_cores()
        thread_count = min(physical_cores, 16)

    current_threads = get_num_threads()

    # If the current thread count matches the desired count, no change needed
    if current_threads == thread_count:
        return thread_count

    # Try to set the thread count, but it may fail if threads are already launched
    try:
        config.THREADING_LAYER = "omp"
        set_num_threads(thread_count)
        return thread_count
    except RuntimeError:
        # Threads already launched, return current count
        return int(current_threads)


def format_hardware_box(
    info: HardwareInfo,
    strategy: Optional[ParallelizationStrategy] = None,
    gpu_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format hardware info as a rich box for CLI display.

    Args:
        info: Hardware information
        strategy: Optional parallelization strategy
        gpu_details: Optional detailed GPU info from detect_gpu()

    Returns:
        Formatted string for CLI output
    """
    lines = []

    # CPU info
    cpu_line = f"  CPU: {info.cpu_count} cores ({info.cpu_count_physical} physical)"
    lines.append(cpu_line)

    # GPU info
    if info.gpu_available and info.gpu_name:
        memory_str = (
            f"{info.gpu_memory_mb // 1024}GB"
            if info.gpu_memory_mb and info.gpu_memory_mb >= 1024
            else f"{info.gpu_memory_mb}MB"
        )
        gpu_line = f"  GPU: {info.gpu_name} ({memory_str}) → CUDA acceleration"

        # Show available/missing RAPIDS libraries if provided
        if gpu_details:
            libs_missing = gpu_details.get("libs_missing", [])
            if libs_missing:
                gpu_line += f" (missing: {', '.join(libs_missing)})"
    else:
        # Show reason for GPU unavailability if known
        reason = ""
        if gpu_details:
            if "reason" in gpu_details:
                reason = f" ({gpu_details['reason']})"
            elif gpu_details.get("libs_missing"):
                reason = f" (install: {', '.join(gpu_details['libs_missing'])})"
        gpu_line = f"  GPU: Not available{reason} → CPU mode"
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
    if strategy.use_gpu:
        return "GPU"
    return "CPU"
