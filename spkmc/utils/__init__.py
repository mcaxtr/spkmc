"""
SPKMC utilities module.

Provides hardware detection, parallelization, and GPU acceleration utilities.
"""

from spkmc.utils.hardware import (
    HardwareInfo,
    ParallelizationStrategy,
    get_hardware_info,
    configure_numba_threads,
    format_hardware_box,
    detect_cpu_cores,
    detect_gpu,
)

from spkmc.utils.parallel import (
    ScenarioResult,
    ParallelBatchExecutor,
    run_scenarios_parallel,
)

from spkmc.utils.gpu_utils import (
    is_gpu_available,
    get_gpu_check_error,
)

__all__ = [
    # Hardware detection
    'HardwareInfo',
    'ParallelizationStrategy',
    'get_hardware_info',
    'configure_numba_threads',
    'format_hardware_box',
    'detect_cpu_cores',
    'detect_gpu',
    # Parallel execution
    'ScenarioResult',
    'ParallelBatchExecutor',
    'run_scenarios_parallel',
    # GPU utilities
    'is_gpu_available',
    'get_gpu_check_error',
]
