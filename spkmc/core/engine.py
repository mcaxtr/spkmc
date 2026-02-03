"""
Unified execution engine for SPKMC simulations.

This module provides the ExecutionEngine class that handles both single-run
and multi-scenario experiment execution with unified progress tracking,
hardware detection, and parallelization.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np

from spkmc.models import Scenario, SimulationResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spkmc.utils.hardware import HardwareInfo, ParallelizationStrategy


@dataclass
class ExecutionContext:
    """Context for simulation execution."""

    scenarios: List[Scenario]
    experiment_name: Optional[str] = None
    results_dir: Optional[Path] = None
    no_plot: bool = False
    export_format: str = "json"

    # Callbacks for progress reporting
    on_scenario_start: Optional[Callable[[int, str], None]] = None
    on_scenario_complete: Optional[Callable[[int, str, SimulationResult], None]] = None
    on_sample_progress: Optional[Callable[[int], None]] = None

    def total_samples(self) -> int:
        """Calculate total samples across all scenarios."""
        return sum(s.total_samples() for s in self.scenarios)


class ExecutionEngine:
    """
    Unified engine for executing SPKMC simulations.

    This engine handles both single-run and multi-scenario experiment execution
    with automatic hardware detection and parallelization.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the execution engine.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self._hardware: Optional["HardwareInfo"] = None
        self._strategy: Optional["ParallelizationStrategy"] = None
        self._failed_scenarios: List[Tuple[str, str]] = []

    @property
    def hardware(self) -> "HardwareInfo":
        """Get hardware information (lazy-loaded)."""
        if self._hardware is None:
            from spkmc.utils.hardware import get_hardware_info

            self._hardware = get_hardware_info()
        return self._hardware

    def configure_parallelization(self, num_scenarios: int) -> "ParallelizationStrategy":
        """
        Configure parallelization strategy based on hardware and workload.

        Args:
            num_scenarios: Number of scenarios to execute

        Returns:
            Configured ParallelizationStrategy
        """
        from spkmc.utils.hardware import ParallelizationStrategy, configure_numba_threads

        self._strategy = ParallelizationStrategy.auto_configure(self.hardware, num_scenarios)
        configure_numba_threads(self._strategy.numba_threads)
        return self._strategy

    @property
    def strategy(self) -> "ParallelizationStrategy":
        """Get the current parallelization strategy."""
        if self._strategy is None:
            self.configure_parallelization(1)
        # At this point _strategy is guaranteed to not be None
        assert self._strategy is not None
        return self._strategy

    def get_failed_scenarios(self) -> List[Tuple[str, str]]:
        """
        Get the list of failed scenarios from the last execution.

        Returns:
            List of (label, error_message) tuples for each failed scenario
        """
        return list(self._failed_scenarios)

    def execute(self, context: ExecutionContext) -> List[SimulationResult]:
        """
        Execute all scenarios in the context.

        Args:
            context: ExecutionContext with scenarios and configuration

        Returns:
            List of SimulationResult objects
        """
        # Clear failed scenarios from previous execution
        self._failed_scenarios = []

        # Configure parallelization
        num_scenarios = len(context.scenarios)
        strategy = self.configure_parallelization(num_scenarios)

        # Ensure results directory exists
        if context.results_dir:
            context.results_dir.mkdir(parents=True, exist_ok=True)

        # Determine execution mode
        use_parallel = strategy.scenario_workers > 1 and num_scenarios > 1

        if use_parallel:
            return self._execute_parallel(context, strategy)
        else:
            return self._execute_sequential(context, strategy)

    def _execute_sequential(
        self, context: ExecutionContext, strategy: "ParallelizationStrategy"
    ) -> List[SimulationResult]:
        """
        Execute scenarios sequentially.

        Args:
            context: Execution context
            strategy: Parallelization strategy

        Returns:
            List of SimulationResult objects
        """
        results: List[SimulationResult] = []

        for i, scenario in enumerate(context.scenarios):
            # Notify scenario start
            if context.on_scenario_start:
                context.on_scenario_start(i, scenario.label)

            # Execute scenario
            result = execute_scenario(
                scenario=scenario,
                index=i,
                results_dir=context.results_dir,
                experiment_name=context.experiment_name,
                use_gpu=strategy.use_gpu,
                export_format=context.export_format,
                progress_callback=context.on_sample_progress,
            )

            results.append(result)

            # Notify scenario complete
            if context.on_scenario_complete:
                context.on_scenario_complete(i, scenario.label, result)

        return results

    def _execute_parallel(
        self, context: ExecutionContext, strategy: "ParallelizationStrategy"
    ) -> List[SimulationResult]:
        """
        Execute scenarios in parallel using ProcessPoolExecutor.

        Args:
            context: Execution context
            strategy: Parallelization strategy

        Returns:
            List of SimulationResult objects in scenario order
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from spkmc.utils.parallel import _get_mp_context, _init_worker

        num_scenarios = len(context.scenarios)
        results: List[Optional[SimulationResult]] = [None] * num_scenarios

        mp_context = _get_mp_context()

        # Create a Queue for progress updates from workers
        progress_queue = mp_context.Queue()

        # Flag to signal the consumer thread to stop
        stop_consumer = threading.Event()

        # Consumer thread that reads progress updates from the queue
        def progress_consumer() -> None:
            while not stop_consumer.is_set():
                try:
                    advance = progress_queue.get(timeout=0.1)
                    if context.on_sample_progress:
                        context.on_sample_progress(advance)
                except Exception:
                    pass

        # Start the consumer thread
        consumer_thread = threading.Thread(target=progress_consumer, daemon=True)
        consumer_thread.start()

        try:
            with ProcessPoolExecutor(
                max_workers=strategy.scenario_workers,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(strategy.numba_threads, progress_queue),
            ) as executor:
                future_to_index = {}

                for i, scenario in enumerate(context.scenarios):
                    # Notify scenario start
                    if context.on_scenario_start:
                        context.on_scenario_start(i, scenario.label)

                    future = executor.submit(
                        execute_scenario,
                        scenario,
                        i,
                        context.results_dir,
                        context.experiment_name,
                        strategy.use_gpu,
                        context.export_format,
                        None,  # No direct callback - workers use queue
                    )
                    future_to_index[future] = i

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    scenario = context.scenarios[index]

                    try:
                        result = future.result()
                        results[index] = result

                        # Notify scenario complete
                        if context.on_scenario_complete:
                            context.on_scenario_complete(index, scenario.label, result)

                    except Exception as e:
                        # Track and log the failure
                        error_msg = str(e)
                        self._failed_scenarios.append((scenario.label, error_msg))
                        logger.warning(
                            "Scenario '%s' failed with error: %s",
                            scenario.label,
                            error_msg,
                        )

                        # Create error result
                        error_result = SimulationResult(
                            S_val=np.array([]),
                            I_val=np.array([]),
                            R_val=np.array([]),
                            time=np.array([]),
                            scenario_label=scenario.label,
                            metadata={"error": error_msg},
                        )
                        results[index] = error_result

        finally:
            # Stop the consumer thread and drain remaining queue items
            stop_consumer.set()
            while True:
                try:
                    advance = progress_queue.get_nowait()
                    if context.on_sample_progress:
                        context.on_sample_progress(advance)
                except Exception:
                    break
            consumer_thread.join(timeout=1.0)

            # Properly close the multiprocessing queue to avoid semaphore leak
            try:
                progress_queue.close()
                progress_queue.join_thread()
            except Exception:
                pass

        # Filter out None results (shouldn't happen, but be safe)
        return [r for r in results if r is not None]


def execute_scenario(
    scenario: Scenario,
    index: int,
    results_dir: Optional[Path],
    experiment_name: Optional[str],
    use_gpu: bool,
    export_format: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> SimulationResult:
    """
    Execute a single scenario and return the result.

    This function is designed to be called from both sequential and parallel modes.

    Args:
        scenario: Scenario to execute
        index: Index of the scenario
        results_dir: Path to results directory
        experiment_name: Name of the experiment
        use_gpu: Use GPU acceleration if available
        export_format: Format for saving results
        progress_callback: Optional callback called per sample

    Returns:
        SimulationResult with execution data
    """
    # Lazy imports for heavy modules
    from spkmc.core.distributions import create_distribution
    from spkmc.core.simulation import SPKMC
    from spkmc.io.data_manager import DataManager
    from spkmc.utils.parallel import get_worker_progress_callback

    # In parallel mode, get callback from the worker's global queue
    if progress_callback is None:
        progress_callback = get_worker_progress_callback()

    scenario_label = scenario.label
    scenario_num = index + 1
    start_time = time.time()

    # Determine output file path
    ext_map = {"json": ".json", "csv": ".csv", "excel": ".xlsx", "md": ".md", "html": ".html"}
    ext = ext_map.get(export_format, ".json")
    normalized_label = Scenario.normalize_label(scenario_label)

    if scenario.output_path:
        # Use explicit output path from scenario
        output_file = scenario.output_path
    elif results_dir:
        # Use results directory
        output_file = str(results_dir / f"{normalized_label}{ext}")
    elif experiment_name:
        # Use experiment path
        output_file = scenario.get_experiment_path(experiment_name, export_format)
    else:
        # Use run path
        output_file = scenario.get_run_path(export_format)

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Create distribution
    distribution_params = scenario.to_distribution_params()
    distribution = create_distribution(scenario.distribution, **distribution_params)

    # Create simulator
    simulator = SPKMC(distribution, use_gpu=use_gpu)

    # Create time steps
    time_steps = np.linspace(0, scenario.t_max, scenario.steps)

    # Simulation parameters
    simulation_params = scenario.to_simulation_params()
    simulation_params["show_progress"] = False

    # Execute simulation with progress callback
    simulation_callback: Optional[Callable[[int, int], None]] = None
    if progress_callback is not None:

        def _wrapped_callback(completed: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(completed)

        simulation_callback = _wrapped_callback

    result = simulator.run_simulation(
        scenario.network, time_steps, progress_callback=simulation_callback, **simulation_params
    )

    execution_time = time.time() - start_time

    # Build metadata
    metadata = scenario.to_metadata(experiment_name)
    metadata["scenario_number"] = scenario_num
    metadata["execution_time"] = execution_time

    # Create SimulationResult
    sim_result = SimulationResult.from_simulation_output(
        result=result,
        time_steps=time_steps,
        scenario_label=scenario_label,
        metadata=metadata,
        execution_time=execution_time,
        output_path=output_file,
    )

    # Save result
    output_data = sim_result.to_dict()
    DataManager.save(output_data, output_file)

    return sim_result
