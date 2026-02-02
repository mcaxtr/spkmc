"""
Unified display functions for SPKMC CLI.

This module provides consistent display formatting for both the `run`
and `experiments` commands, ensuring a unified user experience.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from rich.panel import Panel

from spkmc.cli.formatting import (
    console,
    create_progress_bar,
    format_info,
    format_param,
    format_title,
)

if TYPE_CHECKING:
    from spkmc.models import Scenario, SimulationResult
    from spkmc.utils.hardware import HardwareInfo, ParallelizationStrategy


def display_hardware_panel(
    hardware: "HardwareInfo",
    strategy: "ParallelizationStrategy",
) -> None:
    """
    Display hardware detection panel with GPU/CPU info.

    This is used by both run and experiments commands to show
    detected hardware and parallelization configuration.

    Args:
        hardware: Detected hardware information
        strategy: Configured parallelization strategy
    """
    from spkmc.utils.hardware import detect_gpu

    lines = []

    # CPU info
    cpu_info = f"CPU: {hardware.cpu_count} cores ({hardware.cpu_count_physical} physical)"
    lines.append(cpu_info)

    # GPU info - get detailed info for better messaging
    gpu_available, gpu_details = detect_gpu()
    if gpu_available and hardware.gpu_name:
        memory_str = (
            f"{hardware.gpu_memory_mb // 1024}GB"
            if hardware.gpu_memory_mb and hardware.gpu_memory_mb >= 1024
            else f"{hardware.gpu_memory_mb}MB"
        )
        gpu_info = f"GPU: {hardware.gpu_name} ({memory_str}) → CUDA acceleration"
        # Show missing optional libraries
        if gpu_details and gpu_details.get("libs_missing"):
            gpu_info += f" (optional: {', '.join(gpu_details['libs_missing'])})"
    else:
        # Show reason for GPU unavailability
        reason = ""
        if gpu_details:
            if "reason" in gpu_details:
                reason = f" ({gpu_details['reason']})"
            elif gpu_details.get("libs_missing"):
                reason = f" (install: {', '.join(gpu_details['libs_missing'])})"
        gpu_info = f"GPU: Not available{reason} → CPU mode"
    lines.append(gpu_info)

    # Numba info
    numba_info = f"Numba: {strategy.numba_threads} threads (OpenMP)"
    lines.append(numba_info)

    content = "\n".join(f"  {line}" for line in lines)
    panel = Panel(content, title="Hardware Detected", border_style="cyan")
    console.print(panel)
    console.print()


def display_scenario_config(
    scenario: "Scenario",
    title: str = "Configuration",
) -> None:
    """
    Display scenario configuration in a consistent format.

    Args:
        scenario: Scenario to display
        title: Title for the configuration section
    """
    console.print(format_title(f"SPKMC Simulation - {title}"))
    console.print(format_info("Configuration:"))

    # Network and distribution
    console.print(f"  {format_param('Network', scenario.network.upper())}")
    console.print(f"  {format_param('Distribution', scenario.distribution.capitalize())}")
    console.print(f"  {format_param('Nodes', scenario.nodes)}")

    # Network-specific parameters
    if scenario.network in ["er", "sf", "rrn"]:
        console.print(f"  {format_param('Average degree', scenario.k_avg)}")

    if scenario.network == "sf":
        console.print(f"  {format_param('Exponent', scenario.exponent)}")

    # Simulation parameters
    console.print(f"  {format_param('Samples', scenario.samples)}")
    console.print(f"  {format_param('Runs', scenario.num_runs)}")
    console.print(f"  {format_param('Initial infected', f'{scenario.initial_perc*100:.2f}%')}")
    console.print(f"  {format_param('Max time', scenario.t_max)}")
    console.print(f"  {format_param('Steps', scenario.steps)}")

    # Distribution parameters
    if scenario.distribution == "gamma":
        console.print(f"  {format_param('Shape', scenario.shape)}")
        console.print(f"  {format_param('Scale', scenario.scale)}")
    else:
        console.print(f"  {format_param('Mu', scenario.mu)}")

    console.print(f"  {format_param('Lambda', scenario.lambda_param)}")
    console.print()


def display_result_statistics(result: "SimulationResult") -> None:
    """
    Display simulation statistics for a result.

    Args:
        result: SimulationResult to display statistics for
    """
    console.print(format_title("Simulation Statistics"))
    stats = result.get_statistics()
    max_inf_str = f"{stats['peak_infected']:.4f} (at t={stats['peak_time']:.2f})"
    final_recovered_str = f"{stats['final_recovered']:.4f}"
    console.print(f"  {format_param('Peak infected', max_inf_str)}")
    console.print(f"  {format_param('Final recovered', final_recovered_str)}")
    console.print()


def display_execution_summary(
    execution_time: float,
    experiment_name: Optional[str] = None,
    total_scenarios: Optional[int] = None,
    scenarios_processed: Optional[int] = None,
    results_dir: Optional[Path] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Display execution summary after completing scenarios.

    Args:
        execution_time: Total execution time in seconds
        experiment_name: Optional experiment name
        total_scenarios: Total number of scenarios (expected)
        scenarios_processed: Number of scenarios actually processed
        results_dir: Optional results directory path
        output_path: Optional output file path (for single runs)
    """
    console.print(format_title("Execution Summary"))

    if experiment_name:
        console.print(f"  {format_param('Experiment', experiment_name)}")

    if total_scenarios is not None:
        console.print(f"  {format_param('Total scenarios', total_scenarios)}")

    if scenarios_processed is not None:
        console.print(f"  {format_param('Scenarios processed', scenarios_processed)}")

    console.print(f"  {format_param('Execution time', f'{execution_time:.2f} seconds')}")

    if results_dir:
        console.print(f"  {format_param('Results directory', str(results_dir))}")

    if output_path:
        console.print(f"  {format_param('Output file', output_path)}")

    console.print()


def display_experiment_header(
    experiment_name: str,
    num_scenarios: int,
    description: Optional[str] = None,
) -> None:
    """
    Display experiment header with name and scenario count.

    Args:
        experiment_name: Name of the experiment
        num_scenarios: Number of scenarios
        description: Optional experiment description
    """
    console.print(format_title(f"Experiment: {experiment_name}"))
    console.print(f"  {format_param('Scenarios', num_scenarios)}")
    if description:
        console.print(f"  {format_param('Description', description)}")
    console.print()


def display_scenarios_summary(scenarios: List["Scenario"]) -> None:
    """
    Display a brief summary of scenarios to be executed.

    Args:
        scenarios: List of Scenario objects
    """
    if not scenarios:
        return

    console.print(format_info("Scenarios:"))

    # Group scenarios by network/distribution for compact display
    for i, scenario in enumerate(scenarios, 1):
        # Build compact description
        net = scenario.network.upper()
        dist = scenario.distribution.capitalize()
        nodes = scenario.nodes
        samples = scenario.samples * scenario.num_runs

        label = scenario.label or f"Scenario {i}"
        desc = f"{net} network, {dist} dist, N={nodes}, {samples} samples"
        console.print(f"  {i}. {label}: {desc}")

    console.print()


def display_aggregate_statistics(results: List["SimulationResult"]) -> None:
    """
    Display aggregate statistics for multiple simulation results.

    Args:
        results: List of SimulationResult objects
    """
    if not results:
        return

    console.print(format_title("Simulation Statistics"))

    # For single result, show detailed stats
    if len(results) == 1:
        stats = results[0].get_statistics()
        max_inf_str = f"{stats['peak_infected']:.4f} (at t={stats['peak_time']:.2f})"
        final_recovered_str = f"{stats['final_recovered']:.4f}"
        console.print(f"  {format_param('Peak infected', max_inf_str)}")
        console.print(f"  {format_param('Final recovered', final_recovered_str)}")
    else:
        # For multiple results, show per-scenario summary
        for result in results:
            stats = result.get_statistics()
            label = result.scenario_label or "Unknown"
            peak = f"{stats['peak_infected']:.3f}"
            final = f"{stats['final_recovered']:.3f}"
            console.print(f"  {label}: peak={peak}, final={final}")

    console.print()


class ProgressTracker:
    """
    Context manager for unified progress tracking.

    Provides a consistent progress bar experience for both
    single runs and batch experiments.
    """

    def __init__(
        self,
        total_samples: int,
        verbose: bool = False,
        description: str = "Running simulation",
    ):
        """
        Initialize the progress tracker.

        Args:
            total_samples: Total number of samples to track
            verbose: Enable verbose output
            description: Description for the progress bar
        """
        self.total_samples = total_samples
        self.verbose = verbose
        self.description = description
        self._progress = None
        self._task = None

    def __enter__(self) -> "ProgressTracker":
        """Start the progress bar."""
        progress = create_progress_bar(self.description, self.total_samples, self.verbose)
        self._progress = progress
        if progress is not None:
            progress.__enter__()
            self._task = progress.add_task("Processing samples...", total=self.total_samples)
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the progress bar."""
        if self._progress is not None:
            self._progress.__exit__(*args)

    def advance(self, count: int = 1) -> None:
        """
        Advance the progress bar.

        Args:
            count: Number of units to advance
        """
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, advance=count)

    def get_callback(self) -> Any:
        """
        Get a callback function for advancing progress.

        Returns:
            Callback function that accepts an integer advance count
        """
        return self.advance
