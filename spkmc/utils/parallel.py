"""
Parallel execution utilities for SPKMC.

This module provides parallel execution infrastructure for running
multiple scenarios concurrently using multiprocessing.

Uses 'spawn' context on Linux to avoid OpenMP fork issues with Numba.
"""

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from spkmc.utils.hardware import ParallelizationStrategy


def _get_mp_context() -> mp.context.BaseContext:
    """
    Get appropriate multiprocessing context for the current platform.

    On Linux, 'fork' is the default but causes issues with OpenMP/Numba.
    We use 'spawn' or 'forkserver' to avoid:
    "fork() called from a process already using GNU OpenMP, this is unsafe"

    Returns:
        Multiprocessing context object
    """
    if sys.platform == "linux":
        # Use 'spawn' on Linux to avoid OpenMP fork issues
        # 'spawn' starts a fresh Python interpreter for each worker
        try:
            return mp.get_context("spawn")
        except ValueError:
            # Fallback to forkserver if spawn unavailable
            try:
                return mp.get_context("forkserver")
            except ValueError:
                # Last resort: use default (will likely fail with OpenMP)
                return mp.get_context()
    else:
        # On macOS/Windows, 'spawn' is already the default
        return mp.get_context()


# Global reference to progress queue (set by _init_worker in child processes)
_worker_progress_queue = None


def _init_worker(numba_threads: int, progress_queue: Optional[Any] = None) -> None:
    """
    Initialize worker process with proper Numba configuration.

    Called at worker process start to configure threading.
    Sets NUMBA_NUM_THREADS env var which Numba reads on first import.

    Args:
        numba_threads: Number of threads for Numba to use
        progress_queue: Optional Queue for progress updates (inherited from parent)
    """
    global _worker_progress_queue
    # Set environment variable BEFORE any Numba import
    # Numba reads NUMBA_NUM_THREADS when first imported
    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)
    os.environ["OMP_NUM_THREADS"] = str(numba_threads)
    # Store the progress queue for use by worker functions
    _worker_progress_queue = progress_queue


def worker_progress_callback(advance: int) -> None:
    """
    Send progress update from worker process to main process.

    This function is safe to call from any worker - it will do nothing
    if no progress queue was configured.

    Args:
        advance: Number of units to advance the progress bar
    """
    if _worker_progress_queue is not None:
        try:
            _worker_progress_queue.put(advance)
        except Exception:
            pass  # Ignore queue errors


def get_worker_progress_callback() -> Optional[Callable[[int], None]]:
    """
    Get a progress callback function for use in worker processes.

    Returns:
        A callback function if progress queue is available, None otherwise
    """
    if _worker_progress_queue is not None:
        return worker_progress_callback
    return None


@dataclass
class ScenarioResult:
    """Result container for a single scenario execution."""

    scenario_index: int
    label: str
    success: bool
    result_path: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


def run_scenarios_parallel(
    scenarios: List[Dict[str, Any]],
    execute_fn: Callable[[Dict[str, Any], int], ScenarioResult],
    strategy: ParallelizationStrategy,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[ScenarioResult]:
    """
    Execute scenarios in parallel using multiprocessing.

    Args:
        scenarios: List of scenario configurations
        execute_fn: Function to execute single scenario
            (scenario_dict, scenario_index) -> ScenarioResult
        strategy: Parallelization strategy configuration
        progress_callback: Optional callback for progress updates (completed, total, label)

    Returns:
        List of ScenarioResult in scenario order
    """
    num_scenarios = len(scenarios)

    if strategy.scenario_workers <= 1 or num_scenarios <= 1:
        # Sequential execution
        results: List[ScenarioResult] = []
        for i, scenario in enumerate(scenarios):
            result = execute_fn(scenario, i)
            results.append(result)
            if progress_callback:
                label = scenario.get("label", f"scenario_{i+1}")
                progress_callback(i + 1, num_scenarios, label)
        return results

    # Parallel execution using ProcessPoolExecutor with spawn context
    parallel_results: List[Optional[ScenarioResult]] = [None] * num_scenarios
    completed = 0
    mp_context = _get_mp_context()

    with ProcessPoolExecutor(
        max_workers=strategy.scenario_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(strategy.numba_threads,),
    ) as executor:
        # Submit all scenarios
        future_to_index = {
            executor.submit(execute_fn, scenario, i): i for i, scenario in enumerate(scenarios)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            scenario = scenarios[index]
            label = scenario.get("label", f"scenario_{index+1}")

            try:
                parallel_results[index] = future.result()
            except Exception as e:
                parallel_results[index] = ScenarioResult(
                    scenario_index=index, label=label, success=False, error=str(e)
                )

            completed += 1
            if progress_callback:
                progress_callback(completed, num_scenarios, label)

    return [r for r in parallel_results if r is not None]


class ParallelBatchExecutor:
    """
    Executor for parallel batch scenario execution with progress tracking.
    """

    def __init__(self, strategy: ParallelizationStrategy):
        """
        Initialize the executor.

        Args:
            strategy: Parallelization strategy configuration
        """
        self.strategy = strategy
        self._completed = 0
        self._total = 0
        self._results: List[ScenarioResult] = []

    @property
    def is_parallel(self) -> bool:
        """Check if parallel execution is enabled."""
        return self.strategy.scenario_workers > 1

    @property
    def worker_count(self) -> int:
        """Get the number of parallel workers."""
        return self.strategy.scenario_workers

    def execute(
        self,
        scenarios: List[Dict[str, Any]],
        scenario_executor: Callable[[Dict[str, Any], int, ParallelizationStrategy], ScenarioResult],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_error: Optional[Callable[[int, str, Exception], None]] = None,
    ) -> List[ScenarioResult]:
        """
        Execute all scenarios with the configured strategy.

        Args:
            scenarios: List of scenario configurations
            scenario_executor: Function to execute single scenario
                (scenario, index, strategy) -> ScenarioResult
            on_progress: Callback (completed, total, scenario_label)
            on_error: Callback (scenario_index, label, exception)

        Returns:
            List of ScenarioResult in order
        """
        self._total = len(scenarios)
        self._completed = 0
        self._results = [None] * self._total  # type: ignore

        if self.strategy.scenario_workers <= 1:
            # Sequential execution
            for i, scenario in enumerate(scenarios):
                label = scenario.get("label", f"scenario_{i+1}")
                try:
                    self._results[i] = scenario_executor(scenario, i, self.strategy)
                except Exception as e:
                    if on_error:
                        on_error(i, label, e)
                    self._results[i] = ScenarioResult(
                        scenario_index=i, label=label, success=False, error=str(e)
                    )

                self._completed += 1
                if on_progress:
                    on_progress(self._completed, self._total, label)
        else:
            # Parallel execution with spawn context to avoid OpenMP fork issues
            def wrapped_executor(scenario: Dict[str, Any], index: int) -> ScenarioResult:
                return scenario_executor(scenario, index, self.strategy)

            mp_context = _get_mp_context()
            with ProcessPoolExecutor(
                max_workers=self.strategy.scenario_workers,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(self.strategy.numba_threads,),
            ) as executor:
                future_to_index = {
                    executor.submit(wrapped_executor, scenario, i): i
                    for i, scenario in enumerate(scenarios)
                }

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    scenario = scenarios[index]
                    label = scenario.get("label", f"scenario_{index+1}")

                    try:
                        self._results[index] = future.result()
                    except Exception as e:
                        if on_error:
                            on_error(index, label, e)
                        self._results[index] = ScenarioResult(
                            scenario_index=index, label=label, success=False, error=str(e)
                        )

                    self._completed += 1
                    if on_progress:
                        on_progress(self._completed, self._total, label)

        return [r for r in self._results if r is not None]

    def get_summary(self) -> Tuple[int, int, int]:
        """
        Get execution summary.

        Returns:
            Tuple of (total, succeeded, failed)
        """
        succeeded = len([r for r in self._results if r and r.success])
        failed = len([r for r in self._results if r and not r.success])
        return self._total, succeeded, failed
