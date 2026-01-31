"""
SPKMC utilities module.

Provides hardware detection, parallelization, and GPU acceleration utilities.

Usage:
    # Import specific modules directly:
    from spkmc.utils.hardware import get_hardware_info
    from spkmc.utils.parallel import ParallelBatchExecutor
"""

# Lazy imports - modules are only loaded when accessed
__all__ = [
    # Hardware detection
    "HardwareInfo",
    "ParallelizationStrategy",
    "get_hardware_info",
    "configure_numba_threads",
    "format_hardware_box",
    "detect_cpu_cores",
    "detect_gpu",
    # Parallel execution
    "ScenarioResult",
    "ParallelBatchExecutor",
    "run_scenarios_parallel",
    # GPU utilities
    "is_gpu_available",
    "get_gpu_check_error",
]


def __getattr__(name: str) -> object:
    """Lazy import for utilities to avoid slow startup."""
    # Hardware utilities
    if name in (
        "HardwareInfo",
        "ParallelizationStrategy",
        "get_hardware_info",
        "configure_numba_threads",
        "format_hardware_box",
        "detect_cpu_cores",
        "detect_gpu",
    ):
        from spkmc.utils import hardware

        globals().update(
            {
                "HardwareInfo": hardware.HardwareInfo,
                "ParallelizationStrategy": hardware.ParallelizationStrategy,
                "get_hardware_info": hardware.get_hardware_info,
                "configure_numba_threads": hardware.configure_numba_threads,
                "format_hardware_box": hardware.format_hardware_box,
                "detect_cpu_cores": hardware.detect_cpu_cores,
                "detect_gpu": hardware.detect_gpu,
            }
        )
        return globals()[name]
    # Parallel utilities
    elif name in ("ScenarioResult", "ParallelBatchExecutor", "run_scenarios_parallel"):
        from spkmc.utils import parallel

        globals().update(
            {
                "ScenarioResult": parallel.ScenarioResult,
                "ParallelBatchExecutor": parallel.ParallelBatchExecutor,
                "run_scenarios_parallel": parallel.run_scenarios_parallel,
            }
        )
        return globals()[name]
    # GPU utilities
    elif name in ("is_gpu_available", "get_gpu_check_error"):
        from spkmc.utils import gpu_utils

        globals().update(
            {
                "is_gpu_available": gpu_utils.is_gpu_available,
                "get_gpu_check_error": gpu_utils.get_gpu_check_error,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'spkmc.utils' has no attribute '{name}'")
