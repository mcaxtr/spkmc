"""
SPKMC CLI commands.

This module contains command-line interface commands for the SPKMC algorithm,
including commands to run simulations, visualize results, and display information.
"""

from __future__ import annotations

import json
import os
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import click
import numpy as np
import questionary
from questionary import Style

from spkmc import __version__
from spkmc.cli.formatting import (
    console,
    create_progress_bar,
    format_info,
    format_param,
    format_title,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
    print_markdown,
    print_rich_table,
)
from spkmc.cli.validators import (
    validate_distribution_type,
    validate_exponent,
    validate_file_exists,
    validate_network_type,
    validate_output_file,
    validate_percentage,
    validate_positive,
    validate_positive_int,
)

if TYPE_CHECKING:
    from spkmc.utils.hardware import HardwareInfo
    from spkmc.utils.parallel import ParallelizationStrategy

# Lazy imports for heavy modules (to speed up CLI startup)
# These are imported inside functions that need them:
#   - spkmc.core.distributions (imports numba_utils - 60s JIT compilation)
#   - spkmc.core.simulation (imports numba_utils)
#   - spkmc.utils.hardware (GPU detection)
#   - spkmc.utils.parallel (imports simulation)
#
# Light imports that can stay at module level:
from spkmc.io.experiments import Experiment, ExperimentManager, PlotConfig
from spkmc.io.results import NumpyJSONEncoder

# Constants for default values
DEFAULT_N = 1000
DEFAULT_K_AVG = 10
DEFAULT_SAMPLES = 50
DEFAULT_INITIAL_PERC = 0.01
DEFAULT_T_MAX = 10.0
DEFAULT_STEPS = 100
DEFAULT_NUM_RUNS = 2
DEFAULT_SHAPE = 2.0
DEFAULT_SCALE = 1.0
DEFAULT_MU = 1.0
DEFAULT_LAMBDA = 1.0
DEFAULT_EXPONENT = 2.5


def display_experiments_menu(experiments: List[Experiment]) -> Optional[int]:
    """
    Display an interactive menu to select an experiment with arrow navigation.

    Args:
        experiments: List of available experiments

    Returns:
        Index of the selected experiment (1-based) or None to exit
    """
    # Custom style for the menu
    custom_style = Style(
        [
            ("qmark", "fg:cyan bold"),
            ("question", "fg:white bold"),
            ("answer", "fg:green bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan bold"),
            ("selected", "fg:green"),
            ("separator", "fg:gray"),
            ("instruction", "fg:gray italic"),
        ]
    )

    # Build choices with formatted display
    choices = []
    for i, exp in enumerate(experiments):
        scenarios_count = len(exp.scenarios)
        results_count = exp.result_count

        # Status indicator
        if results_count == 0:
            status = "○ Pending"
        elif results_count >= scenarios_count:
            status = "✓ Complete"
        else:
            status = f"◐ {results_count}/{scenarios_count}"

        # Format: "Name  |  scenarios  |  status"
        display = f"{exp.name:<35} │ {scenarios_count} scenarios │ {status}"
        choices.append(questionary.Choice(title=display, value=i + 1))

    # Add cancel option
    choices.append(questionary.Separator("─" * 70))
    choices.append(questionary.Choice(title="⮐ Cancel", value=-1))

    # Display header
    bc = "[bold cyan]"  # Style shorthand
    bce = "[/bold cyan]"
    bw = "[bold white]"
    bwe = "[/bold white]"
    dm = "[dim]"
    dme = "[/dim]"
    box_line = "═" * 70

    console.print()
    console.print(f"{bc}╔{box_line}╗{bce}")
    title_line = f"{bc}║{bce}                    {bw}SPKMC Experiment Runner{bwe}"
    console.print(f"{title_line}                         {bc}║{bce}")
    console.print(f"{bc}╠{box_line}╣{bce}")
    hint = f"{dm}Use ↑/↓ arrows to navigate, Enter to select, Ctrl+C to cancel{dme}"
    console.print(f"{bc}║{bce}  {hint}   {bc}║{bce}")
    console.print(f"{bc}╚{box_line}╝{bce}")
    console.print()

    try:
        result = questionary.select(
            "Select an experiment to run:",
            choices=choices,
            style=custom_style,
            instruction="",
            use_indicator=True,
            use_shortcuts=False,
        ).ask()

        # Handle cancel or Ctrl+C
        if result is None or result == -1:
            return None
        return int(result)
    except KeyboardInterrupt:
        return None


def _execute_single_scenario(
    scenario: Dict[str, Any],
    scenario_index: int,
    results_dir: Path,
    experiment_name: str,
    use_simple: bool,
    create_zip: bool,
    no_plot: bool,
    save_plot: bool,
    use_gpu: bool = False,
    force_rerun: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Execute a single scenario and return the result.

    This function is designed to be called from parallel workers.

    Args:
        scenario: Scenario configuration dictionary
        scenario_index: Index of the scenario in the experiment
        results_dir: Path to results directory
        experiment_name: Name of the experiment
        use_simple: Generate simplified CSV files
        create_zip: Create zip archives
        no_plot: Disable plot generation
        save_plot: Save plots to files
        use_gpu: Use GPU acceleration if available
        force_rerun: Force re-execution even if cached results exist
        progress_callback: Optional callback called per sample (called with 1 to advance by 1)

    Returns:
        Tuple of (result_dict, output_file_path, scenario_label)
    """
    # Lazy imports for heavy modules
    from spkmc.core.distributions import create_distribution
    from spkmc.core.simulation import SPKMC
    from spkmc.utils.parallel import get_worker_progress_callback
    from spkmc.visualization.plots import Visualizer

    # In parallel mode, get callback from the worker's global queue (set by initializer)
    # In sequential mode, use the passed callback
    if progress_callback is None:
        progress_callback = get_worker_progress_callback()
    scenario_num = scenario_index + 1

    # Extract parameters
    network_type = scenario.get("network_type", "er")
    dist_type = scenario.get("distribution", "exponential")
    nodes = scenario.get("network_size", DEFAULT_N)
    k_avg = scenario.get("k_avg", DEFAULT_K_AVG)
    shape = scenario.get("shape", DEFAULT_SHAPE)
    scale = scenario.get("scale", DEFAULT_SCALE)
    mu = scenario.get("mu", DEFAULT_MU)
    lambda_val = scenario.get("lambda", DEFAULT_LAMBDA)
    exponent = scenario.get("exponent", DEFAULT_EXPONENT)
    samples = scenario.get("samples", DEFAULT_SAMPLES)
    num_runs = scenario.get("num_runs", DEFAULT_NUM_RUNS)
    initial_perc = scenario.get("initial_perc", DEFAULT_INITIAL_PERC)
    t_max = scenario.get("t_max", DEFAULT_T_MAX)
    steps = scenario.get("steps", DEFAULT_STEPS)

    # Create filename
    scenario_label = f"scenario_{scenario_num:03d}"
    if "label" in scenario:
        scenario_label = scenario["label"]

    output_file = str(results_dir / f"{scenario_label}.json")

    # Check if already executed (skip if force_rerun is True)
    if not force_rerun and os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_result = json.load(f)
            existing_metadata = existing_result.get("metadata", {})

            # Check ALL parameters that affect simulation results
            existing_time = existing_result.get("time", [])
            existing_steps = len(existing_time) if existing_time else 0
            existing_t_max = max(existing_time) if existing_time else 0

            params_match = (
                existing_metadata.get("network_type", "").lower() == network_type.lower()
                and existing_metadata.get("distribution", "").lower() == dist_type.lower()
                and existing_metadata.get("N") == nodes
                and existing_metadata.get("samples") == samples
                and existing_metadata.get("num_runs", 1) == num_runs
                and existing_metadata.get("k_avg", DEFAULT_K_AVG) == k_avg
                and existing_steps == steps
                and abs(existing_t_max - t_max) < 1e-6
                and abs(existing_metadata.get("initial_perc", 0) - initial_perc) < 1e-6
                and abs(existing_metadata.get("shape", DEFAULT_SHAPE) - shape) < 1e-6
                and abs(existing_metadata.get("scale", DEFAULT_SCALE) - scale) < 1e-6
                and abs(existing_metadata.get("mu", DEFAULT_MU) - mu) < 1e-6
                and abs(existing_metadata.get("lambda", DEFAULT_LAMBDA) - lambda_val) < 1e-6
            )

            if params_match:
                # Advance progress bar for cached result
                if progress_callback is not None:
                    # Calculate expected samples for this scenario
                    if network_type == "cg":
                        cached_samples = samples
                    else:
                        cached_samples = num_runs * samples
                    progress_callback(cached_samples)
                return (existing_result, output_file, scenario_label)
        except Exception:
            pass

    # Debug output for worker process
    import sys

    if os.environ.get("SPKMC_DEBUG") == "1":
        print(
            f"[WORKER {scenario_index+1}] Starting: network={network_type}, nodes={nodes}, "
            f"samples={samples}, runs={num_runs}, use_gpu={use_gpu}",
            file=sys.stderr,
        )

    # Create distribution
    distribution_params = {"shape": shape, "scale": scale, "mu": mu, "lambda": lambda_val}
    distribution = create_distribution(dist_type, **distribution_params)

    # Create simulator
    simulator = SPKMC(distribution, use_gpu=use_gpu)

    # Debug: check if GPU is actually available in worker
    if os.environ.get("SPKMC_DEBUG") == "1":
        print(
            f"[WORKER {scenario_index+1}] Simulator created: use_gpu={simulator.use_gpu}, "
            f"gpu_available={simulator._gpu_available}",
            file=sys.stderr,
        )

    # Create time steps
    time_steps = np.linspace(0, t_max, steps)

    # Simulation parameters
    simulation_params = {
        "N": nodes,
        "samples": samples,
        "initial_perc": initial_perc,
        "overwrite": force_rerun,  # Force overwrite when re-running
    }

    if network_type in ["er", "cn", "rrn"]:
        simulation_params["k_avg"] = k_avg
        simulation_params["num_runs"] = num_runs

    if network_type == "cn":
        simulation_params["exponent"] = exponent

    # Disable inner progress bars during batch execution
    simulation_params["show_progress"] = False

    # Execute simulation with progress callback for per-sample updates
    # Wrap the callback to match expected signature (int, int) -> None
    simulation_callback: Optional[Callable[[int, int], None]] = None
    if progress_callback is not None:

        def _wrapped_callback(completed: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(completed)

        simulation_callback = _wrapped_callback

    result = simulator.run_simulation(
        network_type, time_steps, progress_callback=simulation_callback, **simulation_params
    )

    # Extract results (S=Susceptible, I=Infected, R=Recovered in SIR model)
    susceptible = result["S_val"]
    infected = result["I_val"]
    recovered = result["R_val"]
    has_error = result.get("has_error", False)

    if has_error:
        susceptible_err = result["S_err"]
        infected_err = result["I_err"]
        recovered_err = result["R_err"]

    # Prepare output
    metadata: Dict[str, Any] = {
        "network_type": network_type,
        "distribution": dist_type,
        "N": nodes,
        "samples": samples,
        "t_max": t_max,
        "steps": steps,
        "initial_perc": initial_perc,
        "scenario_number": scenario_num,
        "scenario_label": scenario_label,
        "experiment_name": experiment_name,
    }
    output_result: Dict[str, Any] = {
        "S_val": list(susceptible),
        "I_val": list(infected),
        "R_val": list(recovered),
        "time": list(time_steps),
        "metadata": metadata,
    }

    # Add distribution parameters
    if dist_type == "gamma":
        metadata["shape"] = shape
        metadata["scale"] = scale
    else:
        metadata["mu"] = mu

    metadata["lambda"] = lambda_val

    if network_type in ["er", "cn", "rrn"]:
        metadata["k_avg"] = k_avg
        metadata["num_runs"] = num_runs

    if network_type == "cn":
        metadata["exponent"] = exponent

    if has_error:
        output_result.update(
            {
                "S_err": list(susceptible_err),
                "I_err": list(infected_err),
                "R_err": list(recovered_err),
            }
        )

    # Save result
    with open(output_file, "w") as f:
        json.dump(output_result, f, indent=2, cls=NumpyJSONEncoder)

    # Generate CSV if needed
    if use_simple:
        csv_path = output_file.replace(".json", "_simple.csv")
        with open(csv_path, "w") as f:
            for j, t in enumerate(time_steps):
                erro = infected_err[j] if has_error else 0.0
                f.write(f"{t},{infected[j]},{erro}\n")

    # Generate individual plot if needed
    if save_plot and not no_plot:
        plot_path = output_file.replace(".json", ".png")
        title = f"{experiment_name} - {scenario_label}"
        if has_error:
            Visualizer.plot_result_with_error(
                susceptible,
                infected,
                recovered,
                susceptible_err,
                infected_err,
                recovered_err,
                time_steps,
                title,
                plot_path,
            )
        else:
            Visualizer.plot_result(susceptible, infected, recovered, time_steps, title, plot_path)

    # Create zip if needed
    if create_zip:
        zip_path = output_file.replace(".json", ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(output_file, os.path.basename(output_file))
            if use_simple:
                csv_p = output_file.replace(".json", "_simple.csv")
                if os.path.exists(csv_p):
                    zipf.write(csv_p, os.path.basename(csv_p))

    return (output_result, output_file, scenario_label)


def _display_hardware_panel(hardware: HardwareInfo, strategy: ParallelizationStrategy) -> None:
    """Display hardware detection panel."""
    from rich.panel import Panel

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


def run_experiment_scenarios(
    experiment: Experiment,
    verbose: bool,
    use_simple: bool,
    create_zip: bool,
    no_plot: bool,
    save_plot: bool,
    force_rerun: bool = False,
) -> List[str]:
    """
    Run all scenarios in an experiment with automatic parallelization.

    Args:
        experiment: Experiment to run
        verbose: Show detailed information
        use_simple: Generate simplified CSV files
        create_zip: Create zip archives with results
        no_plot: Disable plot generation
        save_plot: Save plots to files
        force_rerun: Force re-run even if cached results exist

    Returns:
        List of generated result file paths
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from spkmc.utils.hardware import ParallelizationStrategy, get_hardware_info
    from spkmc.utils.parallel import _get_mp_context, _init_worker
    from spkmc.visualization.plots import Visualizer

    # Detect hardware and configure parallelization
    hardware = get_hardware_info()
    strategy = ParallelizationStrategy.auto_configure(
        hardware, num_scenarios=len(experiment.scenarios)
    )

    # Configure Numba to use the calculated thread count
    from spkmc.utils.hardware import configure_numba_threads

    configure_numba_threads(strategy.numba_threads)

    # Display hardware panel
    _display_hardware_panel(hardware, strategy)

    # If force_rerun, clean ALL existing results first
    if force_rerun and experiment.results_dir.exists():
        import shutil

        shutil.rmtree(experiment.results_dir)

    # Ensure results directory exists
    results_dir = experiment.ensure_results_dir()

    result_files: List[str] = []
    all_results: List[Dict[str, Any]] = []
    all_labels: List[str] = []

    start_time = time.time()
    num_scenarios = len(experiment.scenarios)

    # Calculate total work units (samples) for granular progress
    total_samples = 0
    for scenario in experiment.scenarios:
        num_runs = scenario.get("num_runs", DEFAULT_NUM_RUNS)
        samples = scenario.get("samples", DEFAULT_SAMPLES)
        network_type = scenario.get("network_type", "er")
        # CG doesn't have num_runs, it runs samples directly
        if network_type == "cg":
            total_samples += samples
        else:
            total_samples += num_runs * samples

    # Determine execution mode
    # Use parallel execution when we have multiple scenarios AND multiple workers available
    # GPU mode: The GPU driver handles time-slicing between processes, so parallel
    # execution is safe. Each process submits work to the GPU queue.
    use_parallel = strategy.scenario_workers > 1 and num_scenarios > 1
    parallel_label = f"Running {num_scenarios} scenarios ({total_samples} samples)"

    with create_progress_bar(parallel_label, total_samples, verbose) as progress:
        task = progress.add_task("Processing samples...", total=total_samples)

        # Create a callback function to update the progress bar per sample
        def sample_progress_callback(advance: int) -> None:
            """Callback to update progress bar per sample completion."""
            progress.update(task, advance=advance)

        if use_parallel:
            # Parallel execution using ProcessPoolExecutor with spawn context
            # to avoid OpenMP fork issues on Linux
            if verbose:
                log_debug(
                    f"PARALLEL MODE: {strategy.scenario_workers} workers, {num_scenarios} scenarios"
                )
                log_debug(
                    f"GPU enabled: {strategy.use_gpu}, Numba threads: {strategy.numba_threads}"
                )
            ScenarioResult = Tuple[Optional[Dict[str, Any]], str, str]
            futures_results: List[Optional[ScenarioResult]] = [None] * num_scenarios
            mp_context = _get_mp_context()

            # Note: Child processes get thread config via _worker_initializer in parallel.py
            # We avoid setting NUMBA_NUM_THREADS in the parent process to prevent conflicts
            # when Numba is already initialized with a different thread count

            # Create a Queue for progress updates from workers
            # Queue must be created from the same mp_context as the workers
            # and passed via initializer (inherited, not pickled)
            progress_queue = mp_context.Queue()

            # Flag to signal the consumer thread to stop
            stop_consumer = threading.Event()

            # Consumer thread that reads progress updates from the queue
            def progress_consumer() -> None:
                while not stop_consumer.is_set():
                    try:
                        # Non-blocking get with short timeout
                        advance = progress_queue.get(timeout=0.1)
                        progress.update(task, advance=advance)
                    except Exception:
                        # Queue.Empty or other errors - just continue
                        pass

            # Start the consumer thread
            consumer_thread = threading.Thread(target=progress_consumer, daemon=True)
            consumer_thread.start()

            try:
                with ProcessPoolExecutor(
                    max_workers=strategy.scenario_workers,
                    mp_context=mp_context,
                    initializer=_init_worker,
                    initargs=(
                        strategy.numba_threads,
                        progress_queue,
                    ),  # Queue passed via initializer
                ) as executor:
                    future_to_index = {}

                    for i, scenario in enumerate(experiment.scenarios):
                        future = executor.submit(
                            _execute_single_scenario,
                            scenario,
                            i,
                            results_dir,
                            experiment.name,
                            use_simple,
                            create_zip,
                            no_plot,
                            save_plot,
                            strategy.use_gpu,
                            force_rerun,
                            # No progress_callback - workers get it from global queue
                        )
                        future_to_index[future] = i

                    parallel_start = time.time()

                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        scenario = experiment.scenarios[index]
                        scenario_label = scenario.get("label", f"scenario_{index+1:03d}")
                        scenario_time = time.time() - parallel_start

                        try:
                            result_tuple = future.result()
                            futures_results[index] = result_tuple
                            if verbose:
                                msg = f"Scenario {index+1} ({scenario_label}) "
                                msg += f"completed in {scenario_time:.1f}s"
                                log_debug(msg)
                        except Exception as e:
                            log_error(f"Error in scenario {index+1} ({scenario_label}): {e}")
                            futures_results[index] = None
            finally:
                # Stop the consumer thread and drain remaining queue items
                stop_consumer.set()
                # Drain remaining items from the queue
                while True:
                    try:
                        advance = progress_queue.get_nowait()
                        progress.update(task, advance=advance)
                    except Exception:
                        break
                consumer_thread.join(timeout=1.0)

            # Collect results in order
            for i, futures_result in enumerate(futures_results):
                if futures_result is not None:
                    output_result_item, output_file, scenario_label = futures_result
                    result_files.append(output_file)
                    if output_result_item is not None:
                        all_results.append(output_result_item)
                    all_labels.append(experiment.scenarios[i].get("label", scenario_label))

        else:
            # Sequential execution (original behavior)
            if verbose:
                log_debug(f"SEQUENTIAL MODE: 1 worker, {num_scenarios} scenarios")
                log_debug(
                    f"GPU enabled: {strategy.use_gpu}, Numba threads: {strategy.numba_threads}"
                )
            for i, scenario in enumerate(experiment.scenarios):
                scenario_label = scenario.get("label", f"scenario_{i+1:03d}")

                try:
                    result_tuple = _execute_single_scenario(
                        scenario,
                        i,
                        results_dir,
                        experiment.name,
                        use_simple,
                        create_zip,
                        no_plot,
                        save_plot,
                        strategy.use_gpu,
                        force_rerun,
                        progress_callback=sample_progress_callback,
                    )

                    if result_tuple is not None:
                        output_result, output_file, label = result_tuple
                        result_files.append(output_file)
                        if output_result is not None:
                            all_results.append(output_result)
                        all_labels.append(scenario.get("label", label))

                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    log_error(f"Error running scenario {i+1}: {e}")
                    # Show file and line number from traceback
                    tb_lines = tb.strip().split("\n")
                    for line in tb_lines:
                        if 'File "' in line:
                            console.print(f"[dim]{line.strip()}[/dim]")
                    if os.environ.get("SPKMC_DEBUG", "0") == "1":
                        console.print(f"[dim]{tb}[/dim]")

    # Calculate execution time
    execution_time = time.time() - start_time

    # Generate comparison plot with custom configuration
    if len(all_results) > 1:
        compare_path = str(results_dir / "comparison.png")
        try:
            Visualizer.compare_results_with_config(
                all_results, all_labels, experiment.plot_config, compare_path
            )
            log_success(f"Comparison plot saved to: {compare_path}")
        except Exception as e:
            log_error(f"Error generating comparison plot: {e}")

    # Generate AI analysis if available
    if experiment.description and all_results:
        from spkmc.analysis import try_generate_analysis

        analysis_path = try_generate_analysis(
            experiment_name=experiment.name,
            experiment_description=experiment.description,
            results=all_results,
            results_dir=results_dir,
            verbose=verbose,
        )
        if analysis_path:
            log_success(f"AI analysis generated at: {analysis_path}")

    # Show execution summary
    log_success(f"Completed in {execution_time:.1f}s")

    return result_files


def create_time_steps(t_max: float, steps: int) -> np.ndarray:
    """
    Create the time-step array.

    Args:
        t_max: Maximum simulation time
        steps: Number of time steps

    Returns:
        Array with time steps
    """
    result: np.ndarray = np.asarray(np.linspace(0, t_max, steps))
    return result


@click.group(help="CLI for the SPKMC algorithm (Shortest Path Kinetic Monte Carlo)")
@click.version_option(version=__version__, prog_name="spkmc")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode for debugging")
@click.option("--no-color", is_flag=True, help="Disable colors in output")
@click.option(
    "--simple",
    is_flag=True,
    help="Generate simplified CSV result file (time, infected, error)",
)
def cli(verbose: bool, no_color: bool, simple: bool) -> None:
    """Main command group for the SPKMC CLI."""
    # Configure verbose mode
    os.environ["SPKMC_VERBOSE"] = "1" if verbose else "0"

    if verbose:
        log_info("Verbose mode enabled")

    if no_color:
        log_info("Colors disabled in output")

    if simple:
        log_info("Simplified CSV output mode enabled")


@cli.command(help="Run an SPKMC simulation")
@click.option(
    "--simple",
    is_flag=True,
    help="Generate simplified CSV result file (time, infected, error)",
)
@click.option(
    "--network-type",
    "-n",
    type=str,
    default="er",
    show_default=True,
    callback=validate_network_type,
    help="Network type: er (Erdos-Renyi), cn (Complex), cg (Complete), rrn (Regular)",
)
@click.option(
    "--dist-type",
    "-d",
    type=str,
    default="gamma",
    show_default=True,
    callback=validate_distribution_type,
    help="Distribution type: Gamma or Exponential",
)
@click.option(
    "--shape",
    type=float,
    default=DEFAULT_SHAPE,
    show_default=True,
    callback=validate_positive,
    help="Shape parameter for the Gamma distribution",
)
@click.option(
    "--scale",
    type=float,
    default=DEFAULT_SCALE,
    show_default=True,
    callback=validate_positive,
    help="Scale parameter for the Gamma distribution",
)
@click.option(
    "--mu",
    type=float,
    default=DEFAULT_MU,
    show_default=True,
    callback=validate_positive,
    help="Mu parameter for the Exponential distribution (recovery)",
)
@click.option(
    "--lambda",
    "lambda_val",
    type=float,
    default=DEFAULT_LAMBDA,
    show_default=True,
    callback=validate_positive,
    help="Lambda parameter for infection times",
)
@click.option(
    "--exponent",
    type=float,
    default=DEFAULT_EXPONENT,
    show_default=True,
    callback=validate_exponent,
    help="Exponent for complex networks (CN)",
)
@click.option(
    "--nodes",
    "-N",
    type=int,
    default=DEFAULT_N,
    show_default=True,
    callback=validate_positive_int,
    help="Number of nodes in the network",
)
@click.option(
    "--k-avg",
    type=float,
    default=DEFAULT_K_AVG,
    show_default=True,
    callback=validate_positive,
    help="Average degree of the network",
)
@click.option(
    "--samples",
    "-s",
    type=int,
    default=DEFAULT_SAMPLES,
    show_default=True,
    callback=validate_positive_int,
    help="Number of samples per run",
)
@click.option(
    "--num-runs",
    "-r",
    type=int,
    default=DEFAULT_NUM_RUNS,
    show_default=True,
    callback=validate_positive_int,
    help="Number of runs (for averages)",
)
@click.option(
    "--initial-perc",
    "-i",
    type=float,
    default=DEFAULT_INITIAL_PERC,
    show_default=True,
    callback=validate_percentage,
    help="Initial percentage of infected",
)
@click.option(
    "--t-max",
    type=float,
    default=DEFAULT_T_MAX,
    show_default=True,
    callback=validate_positive,
    help="Maximum simulation time",
)
@click.option(
    "--steps",
    type=int,
    default=DEFAULT_STEPS,
    show_default=True,
    callback=validate_positive_int,
    help="Number of time steps",
)
@click.option(
    "--output",
    "-o",
    type=str,
    callback=validate_output_file,
    help="Path to save results (optional)",
)
@click.option(
    "--export-format",
    "-e",
    type=click.Choice(["json", "csv", "excel", "md", "html"]),
    help="Format for exporting results (optional)",
)
@click.option("--no-plot", is_flag=True, default=False, help="Do not display the results plot")
@click.option("--save-plot", type=str, help="Save the plot to a file (format: png, pdf, svg)")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing results")
@click.option("--zip", is_flag=True, default=False, help="Create a zip file with results")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed information during the simulation",
)
def run(
    simple: bool,
    network_type: str,
    dist_type: str,
    shape: float,
    scale: float,
    mu: float,
    lambda_val: float,
    exponent: float,
    nodes: int,
    k_avg: int,
    samples: int,
    num_runs: int,
    initial_perc: float,
    t_max: float,
    steps: int,
    output: Optional[str],
    export_format: Optional[str],
    no_plot: bool,
    save_plot: Optional[str],
    overwrite: bool,
    zip: bool,
    verbose: bool,
) -> None:
    """Run an SPKMC simulation with the specified parameters."""
    # Lazy imports for heavy modules
    from spkmc.core.distributions import create_distribution
    from spkmc.core.simulation import SPKMC
    from spkmc.io.export import ExportManager
    from spkmc.visualization.plots import Visualizer

    # Configure verbose mode
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"

    # Record simulation start
    start_time = time.time()
    log_debug(
        f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        verbose_only=False,
    )

    # Create the distribution
    distribution_params = {"shape": shape, "scale": scale, "mu": mu, "lambda": lambda_val}
    distribution = create_distribution(dist_type, **distribution_params)
    log_debug(
        f"{dist_type.capitalize()} distribution created with parameters: {distribution_params}",
        verbose_only=True,
    )

    # Create the simulator
    simulator = SPKMC(distribution)

    # Create time steps
    time_steps = create_time_steps(t_max, steps)
    log_debug(
        f"Time steps created: {time_steps[0]} to {time_steps[-1]} ({len(time_steps)} points)",
        verbose_only=True,
    )

    # Display simulation info
    console.print(format_title("SPKMC Simulation"))
    console.print(format_info("Configuration:"))

    # Use direct values without nesting formatting functions
    network_type_upper = network_type.upper()
    dist_type_cap = dist_type.capitalize()

    console.print(f"  {format_param('Network', network_type_upper)}")
    console.print(f"  {format_param('Distribution', dist_type_cap)}")
    console.print(f"  {format_param('Nodes', nodes)}")

    if network_type in ["er", "cn", "rrn"]:
        console.print(f"  {format_param('Average degree', k_avg)}")

    if network_type == "cn":
        console.print(f"  {format_param('Exponent', exponent)}")

    console.print(f"  {format_param('Samples', samples)}")
    console.print(f"  {format_param('Runs', num_runs)}")
    console.print(f"  {format_param('Initial infected', f'{initial_perc*100:.2f}%')}")
    console.print(f"  {format_param('Max time', t_max)}")
    console.print(f"  {format_param('Steps', steps)}")

    # Start progress bar
    with create_progress_bar("SPKMC Simulation", 1, verbose) as progress:
        task = progress.add_task("Running simulation...", total=1)

    # Network-specific parameters
    simulation_params = {
        "N": nodes,
        "samples": samples,
        "initial_perc": initial_perc,
        "overwrite": overwrite,
    }

    if network_type in ["er", "cn", "rrn"]:
        simulation_params["k_avg"] = k_avg
        simulation_params["num_runs"] = num_runs

    if network_type == "cn":
        simulation_params["exponent"] = exponent

    # Run the simulation
    try:
        log_debug("Starting simulation run", verbose_only=True)
        result = simulator.run_simulation(
            network_type, time_steps, progress_callback=None, **simulation_params
        )
        progress.update(task, advance=1)
        log_success("Simulation completed successfully!")
    except Exception as e:
        log_error(f"Error during simulation: {e}")
        return

    # Extract results (S=Susceptible, I=Infected, R=Recovered in SIR model)
    susceptible = result["S_val"]
    infected = result["I_val"]
    recovered = result["R_val"]
    has_error = result.get("has_error", False)

    if has_error:
        susceptible_err = result["S_err"]
        infected_err = result["I_err"]
        recovered_err = result["R_err"]

    # Calculate statistics
    max_infected: float = float(np.max(infected))
    max_infected_time = time_steps[np.argmax(infected)]
    final_recovered = recovered[-1]

    # Display statistics
    console.print(format_title("Simulation Statistics"))
    max_inf_str = f"{max_infected:.4f} (at t={max_infected_time:.2f})"
    console.print(f"  {format_param('Peak infected', max_inf_str)}")
    console.print(f"  {format_param('Final recovered', f'{final_recovered:.4f}')}")

    # Record execution time
    end_time = time.time()
    execution_time = end_time - start_time
    log_debug(f"Execution time: {execution_time:.2f} seconds", verbose_only=False)

    # Save results to a custom file if specified
    if output:
        # Build metadata dict separately to avoid mypy indexed assignment issues
        metadata_dict: Dict[str, Any] = {
            "network_type": network_type,
            "distribution": dist_type,
            "N": nodes,
            "initial_perc": initial_perc,
            "execution_time": execution_time,
        }

        # Add distribution-specific parameters
        if dist_type == "gamma":
            metadata_dict["shape"] = shape
            metadata_dict["scale"] = scale
        else:
            metadata_dict["mu"] = mu

        metadata_dict["lambda"] = lambda_val

        if network_type in ["er", "cn", "rrn"]:
            metadata_dict["k_avg"] = k_avg
            metadata_dict["num_runs"] = num_runs

        if network_type == "cn":
            metadata_dict["exponent"] = exponent

        output_result: Dict[str, Any] = {
            "S_val": list(susceptible),
            "I_val": list(infected),
            "R_val": list(recovered),
            "time": list(time_steps),
            "metadata": metadata_dict,
        }

        if has_error:
            output_result.update(
                {
                    "S_err": list(susceptible_err),
                    "I_err": list(infected_err),
                    "R_err": list(recovered_err),
                }
            )

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

            with open(output, "w") as f:
                json.dump(output_result, f, indent=2, cls=NumpyJSONEncoder)
            log_success(f"Results saved to: {output}")

            # Check whether --simple was passed globally or locally
            ctx = click.get_current_context()
            parent_simple = ctx.parent.params.get("simple", False) if ctx.parent else False
            use_simple = simple or parent_simple

            # Generate simplified CSV if --simple is enabled
            if use_simple:
                csv_path = output.replace(".json", "_simple.csv")
                with open(csv_path, "w") as f:
                    # Data without header (time, infected, error)
                    for idx, t in enumerate(time_steps):
                        erro = infected_err[idx] if has_error and infected_err is not None else 0.0
                        f.write(f"{t},{infected[idx]},{erro}\n")

                log_success(f"Simplified results saved to CSV: {csv_path}")

            # Export to additional format if specified
            if export_format:
                export_path = output.replace(".json", f".{export_format}")
                exported_file = ExportManager.export_results(
                    output_result, export_path, export_format
                )
                log_success(f"Results exported in {export_format.upper()} format: {exported_file}")

            # Create zip file with results if requested
            if zip:
                zip_path = output.replace(".json", ".zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    # Add main JSON file
                    zipf.write(output, os.path.basename(output))

                    # Add simplified CSV file if present
                    if use_simple and os.path.exists(csv_path):
                        zipf.write(csv_path, os.path.basename(csv_path))

                    # Add exported file if present
                    if export_format and os.path.exists(export_path):
                        zipf.write(export_path, os.path.basename(export_path))

                    # Add plot file if present
                    if save_plot and os.path.exists(save_plot):
                        zipf.write(save_plot, os.path.basename(save_plot))

                log_success(f"Results zipped at: {zip_path}")
        except Exception as e:
            log_error(f"Error saving results: {e}")

    # Plot results if requested
    if not no_plot or save_plot:
        log_info("Generating visualization...")

        title = (
            f"SPKMC Simulation - Network {network_type.upper()}, "
            f"Distribution {dist_type.capitalize()}"
        )

        try:
            if save_plot:
                # Save plot to a file
                if has_error:
                    Visualizer.plot_result_with_error(
                        susceptible,
                        infected,
                        recovered,
                        susceptible_err,
                        infected_err,
                        recovered_err,
                        time_steps,
                        title,
                        save_plot,
                    )
                else:
                    Visualizer.plot_result(
                        susceptible, infected, recovered, time_steps, title, save_plot
                    )
                log_success(f"Plot saved to: {save_plot}")
            elif not no_plot:
                # Show plot on screen
                if has_error:
                    Visualizer.plot_result_with_error(
                        susceptible,
                        infected,
                        recovered,
                        susceptible_err,
                        infected_err,
                        recovered_err,
                        time_steps,
                        title,
                    )
                else:
                    Visualizer.plot_result(susceptible, infected, recovered, time_steps, title)
        except Exception as e:
            log_error(f"Error generating visualization: {e}")


@cli.command(help="Visualize results from previous simulations")
@click.option(
    "--simple",
    is_flag=True,
    help="Generate simplified CSV result file (time, infected, error)",
)
@click.argument("path", type=str)
@click.option(
    "--with-error",
    "-e",
    is_flag=True,
    default=False,
    help="Show error bars (if available)",
)
@click.option(
    "--output",
    "-o",
    type=str,
    callback=validate_output_file,
    help="Save the plot to a file (optional)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "pdf", "svg", "jpg"]),
    default="png",
    help="Plot format (when used with --output)",
)
@click.option(
    "--dpi", type=int, default=300, help="Plot resolution in DPI (when used with --output)"
)
@click.option(
    "--export",
    "-x",
    type=click.Choice(["json", "csv", "excel", "md", "html"]),
    help="Export results in an additional format",
)
@click.option(
    "--states",
    "-s",
    type=str,
    multiple=True,
    help="Specific states to plot (e.g., --states infected --states recovered)",
)
@click.option(
    "--separate",
    is_flag=True,
    default=False,
    help="Plot multiple scenarios in separate charts (when a directory is provided)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed information")
def plot(
    simple: bool,
    path: str,
    with_error: bool,
    output: Optional[str],
    format: str,
    dpi: int,
    export: Optional[str],
    states: Tuple[str, ...],
    separate: bool,
    verbose: bool,
) -> None:
    """Visualize results from a previous simulation."""
    # Lazy imports for heavy modules
    from spkmc.io.export import ExportManager
    from spkmc.io.results import ResultManager
    from spkmc.visualization.plots import Visualizer

    # Check whether --simple was passed globally or locally
    ctx = click.get_current_context()
    parent_simple = ctx.parent.params.get("simple", False) if ctx.parent else False
    use_simple = simple or parent_simple
    # Configure verbose mode
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"

    # Check whether the path is a file or directory
    path_obj = Path(path)

    if not path_obj.exists():
        log_error(f"Path not found: {path}")
        ctx.exit(1)

    # Process states to plot
    valid_states = {"susceptible", "infected", "recovered", "s", "i", "r"}
    states_to_plot = set()

    if states:
        for state in states:
            state_lower = state.lower()
            if state_lower in valid_states:
                # Normalize to S, I, R
                if state_lower in ["susceptible", "s"]:
                    states_to_plot.add("S")
                elif state_lower in ["infected", "i"]:
                    states_to_plot.add("I")
                elif state_lower in ["recovered", "r"]:
                    states_to_plot.add("R")
            else:
                log_warning(f"Invalid state ignored: {state}")
    else:
        # If no state was specified, plot all
        states_to_plot = {"S", "I", "R"}

    if not states_to_plot:
        log_warning("No valid state specified. Plotting all states.")
        states_to_plot = {"S", "I", "R"}

    log_info(f"States to plot: {', '.join(sorted(states_to_plot))}")

    # Collect result files
    result_files = []

    if path_obj.is_file():
        # Single file
        if path_obj.suffix == ".json":
            result_files.append(path_obj)
        else:
            log_error(f"File must be JSON: {path}")
            ctx.exit(1)
    elif path_obj.is_dir():
        # Directory - find all JSON files
        json_files = list(path_obj.glob("*.json"))
        if not json_files:
            log_error(f"No JSON files found in directory: {path}")
            ctx.exit(1)
        result_files.extend(json_files)
        log_info(f"Found {len(result_files)} JSON files in directory")
    else:
        log_error(f"Invalid path: {path}")
        ctx.exit(1)

    # Process files
    if len(result_files) == 1:
        # Single file - original behavior
        result_file = result_files[0]
        try:
            log_info(f"Loading results from: {result_file}")
            result = ResultManager.load_result(str(result_file))
        except Exception as e:
            log_error(f"Error loading results file: {e}")
            ctx.exit(1)

        # Extract data (S=Susceptible, I=Infected, R=Recovered in SIR model)
        susceptible = np.array(result.get("S_val", []))
        infected = np.array(result.get("I_val", []))
        recovered = np.array(result.get("R_val", []))
        time_steps = np.array(result.get("time", []))

        if not len(susceptible) or not len(infected) or not len(recovered) or not len(time_steps):
            log_error("Incomplete data in results file.")
            ctx.exit(1)

        # Calculate basic statistics
        max_infected: float = float(np.max(infected))
        max_infected_time = time_steps[np.argmax(infected)]
        final_recovered = recovered[-1]

        console.print(format_title("Simulation Statistics"))
        max_inf_str = f"{max_infected:.4f} (at t={max_infected_time:.2f})"
        console.print(f"  {format_param('Peak infected', max_inf_str)}")
        console.print(f"  {format_param('Final recovered', f'{final_recovered:.4f}')}")

        # Check whether error data is available
        has_error = "S_err" in result and "I_err" in result and "R_err" in result

        # Extract metadata for the title
        metadata = result.get("metadata", {})
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")

        # Display metadata
        console.print(format_title("Simulation Metadata"))
        for key, value in metadata.items():
            console.print(f"  {format_param(key, value)}")

        title = f"SPKMC Simulation - Network {network_type}, Distribution {dist_type}, N={N}"

        try:
            log_info("Generating visualization...")

            if with_error and has_error:
                susceptible_err = np.array(result.get("S_err", []))
                infected_err = np.array(result.get("I_err", []))
                recovered_err = np.array(result.get("R_err", []))
                Visualizer.plot_result_with_error(
                    susceptible,
                    infected,
                    recovered,
                    susceptible_err,
                    infected_err,
                    recovered_err,
                    time_steps,
                    title,
                    output,
                    states_to_plot,
                )
            else:
                if with_error and not has_error:
                    log_warning("Error data not available. Showing plot without error bars.")
                Visualizer.plot_result(
                    susceptible, infected, recovered, time_steps, title, output, states_to_plot
                )

            if output:
                log_success(f"Plot saved to: {output}")

            # Export to additional format if specified
            if export:
                export_path = str(result_file).replace(".json", f".{export}")
                exported_file = ExportManager.export_results(result, export_path, export)
                log_success(f"Results exported in {export.upper()} format: {exported_file}")

            # Generate simplified CSV if --simple is enabled
            if use_simple:
                csv_path = str(result_file).replace(".json", "_simple.csv")
                with open(csv_path, "w") as f:
                    # Data without header (time, infected, error)
                    for idx, t in enumerate(time_steps):
                        erro = infected_err[idx] if has_error and infected_err is not None else 0.0
                        f.write(f"{t},{infected[idx]},{erro}\n")

                log_success(f"Simplified results saved to CSV: {csv_path}")

        except Exception as e:
            log_error(f"Error generating visualization: {e}")

    else:
        # Multiple files
        log_info(f"Processing {len(result_files)} result files...")

        if separate:
            # Plot each file separately
            for i, result_file in enumerate(result_files):
                log_info(f"Processing file {i+1}/{len(result_files)}: {result_file.name}")

                try:
                    result = ResultManager.load_result(str(result_file))

                    # Extract data (S=Susceptible, I=Infected, R=Recovered in SIR model)
                    susceptible = np.array(result.get("S_val", []))
                    infected = np.array(result.get("I_val", []))
                    recovered = np.array(result.get("R_val", []))
                    time_steps = np.array(result.get("time", []))

                    if (
                        not len(susceptible)
                        or not len(infected)
                        or not len(recovered)
                        or not len(time_steps)
                    ):
                        log_warning(f"Incomplete data in {result_file.name}, skipping...")
                        continue

                    # Check whether error data is available
                    has_error = "S_err" in result and "I_err" in result and "R_err" in result

                    # Extract metadata
                    metadata = result.get("metadata", {})

                    title = f"SPKMC Simulation - {result_file.stem}"

                    # Determine output file name
                    if output:
                        base_output = Path(output)
                        output_file = (
                            base_output.parent
                            / f"{base_output.stem}_{result_file.stem}{base_output.suffix}"
                        )
                    else:
                        output_file = None

                    if with_error and has_error:
                        susceptible_err = np.array(result.get("S_err", []))
                        infected_err = np.array(result.get("I_err", []))
                        recovered_err = np.array(result.get("R_err", []))
                        Visualizer.plot_result_with_error(
                            susceptible,
                            infected,
                            recovered,
                            susceptible_err,
                            infected_err,
                            recovered_err,
                            time_steps,
                            title,
                            str(output_file) if output_file else None,
                            states_to_plot,
                        )
                    else:
                        Visualizer.plot_result(
                            susceptible,
                            infected,
                            recovered,
                            time_steps,
                            title,
                            str(output_file) if output_file else None,
                            states_to_plot,
                        )

                    if output_file:
                        log_success(f"Plot saved to: {output_file}")

                except Exception as e:
                    log_error(f"Error processing {result_file.name}: {e}")
                    continue

        else:
            # Plot all files in a single comparison chart
            results_data = []
            labels = []

            for result_file in result_files:
                try:
                    result = ResultManager.load_result(str(result_file))

                    # Check required data
                    if all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                        results_data.append(result)
                        labels.append(result_file.stem)
                    else:
                        log_warning(f"Incomplete data in {result_file.name}, skipping...")

                except Exception as e:
                    log_error(f"Error loading {result_file.name}: {e}")
                    continue

            if not results_data:
                log_error("No valid file found to plot.")
                ctx.exit(1)

            try:
                log_info(
                    f"Generating comparison visualization for {len(results_data)} scenarios..."
                )

                title = f"SPKMC Simulation Comparison - {path_obj.name}"
                Visualizer.compare_results(results_data, labels, title, output, states_to_plot)

                if output:
                    log_success(f"Comparison plot saved to: {output}")

            except Exception as e:
                log_error(f"Error generating comparison visualization: {e}")


@cli.command(help="Show information about saved simulations")
@click.option(
    "--simple",
    is_flag=True,
    help="Generate simplified CSV result file (time, infected, error)",
)
@click.option(
    "--result-file",
    "-f",
    type=str,
    callback=validate_file_exists,
    help="Specific results file (optional)",
)
@click.option(
    "--list",
    "-l",
    "list_files",
    is_flag=True,
    default=False,
    help="List all available results files",
)
@click.option(
    "--export",
    "-e",
    type=click.Choice(["json", "csv", "excel", "md", "html"]),
    help="Export information in a specific format",
)
@click.option(
    "--output",
    "-o",
    type=str,
    callback=validate_output_file,
    help="Path to save the export (used with --export)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed information")
def info(
    simple: bool,
    result_file: Optional[str],
    list_files: bool,
    export: Optional[str],
    output: Optional[str],
    verbose: bool,
) -> None:
    """Show information about saved simulations."""
    # Lazy imports for heavy modules
    from spkmc.io.export import ExportManager
    from spkmc.io.results import ResultManager

    # Check whether --simple was passed globally or locally
    ctx = click.get_current_context()
    parent_simple = ctx.parent.params.get("simple", False) if ctx.parent else False
    use_simple = simple or parent_simple
    # Configure verbose mode
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"

    if list_files:
        files = ResultManager.list_results()
        if not files:
            log_info("No results files found.")
            return

        console.print(format_title("Available Results Files"))

        # Create a formatted table
        data = []
        for i, file in enumerate(files, 1):
            # Extrair metadados do caminho
            metadata = ResultManager.get_metadata_from_path(file)
            network = metadata.get("network_type", "").upper() if metadata else ""
            dist = metadata.get("distribution", "").capitalize() if metadata else ""
            nodes = metadata.get("N", "") if metadata else ""

            data.append(
                {
                    "Index": i,
                    "File": os.path.basename(file),
                    "Network": network,
                    "Distribution": dist,
                    "Nodes": nodes,
                }
            )

        print_rich_table(data, "Available Results")
        return

    if not result_file:
        log_error("Specify a results file or use --list to see available files.")
        return

    # Load results
    try:
        log_info(f"Loading results from: {result_file}")
        result = ResultManager.load_result(result_file)
    except Exception as e:
        log_error(f"Error loading results file: {e}")
        return

    # Generate simplified CSV if --simple is enabled
    if use_simple and result_file:
        try:
            # Extract required data
            time_steps = np.array(result.get("time", []))
            infected = np.array(result.get("I_val", []))
            has_error = "I_err" in result
            infected_err = np.array(result.get("I_err", [])) if has_error else None

            csv_path = result_file.replace(".json", "_simple.csv")
            with open(csv_path, "w") as f:
                # Data without header (time, infected, error)
                for idx, t in enumerate(time_steps):
                    erro = infected_err[idx] if has_error and infected_err is not None else 0.0
                    f.write(f"{t},{infected[idx]},{erro}\n")

            log_success(f"Simplified results saved to CSV: {csv_path}")
        except Exception as e:
            log_error(f"Error generating simplified CSV file: {e}")

    # Extract and show information
    metadata = result.get("metadata", {})
    if metadata:
        console.print(format_title("Simulation Parameters"))
        for key, value in metadata.items():
            if isinstance(value, dict):
                console.print(f"  {format_param(key, '')}")
                for subkey, subvalue in value.items():
                    console.print(f"    {format_param(subkey, subvalue)}")
            else:
                console.print(f"  {format_param(key, value)}")
    else:
        # Try to infer file info
        console.print(format_title("File Information"))
        console.print(f"  {format_param('File', os.path.basename(result_file))}")
        console.print(f"  {format_param('Size', f'{os.path.getsize(result_file)/1024:.2f} KB')}")

        # Format modification date
        mod_time = os.path.getmtime(result_file)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"  {format_param('Last modified', mod_time_str)}")

        # Extract metadata from path
        path_metadata = ResultManager.get_metadata_from_path(result_file)
        if path_metadata:
            console.print(format_title("Inferred Path Metadata"))
            for key, value in path_metadata.items():
                console.print(f"  {format_param(key, value)}")

        # Show basic statistics
        if "S_val" in result and "I_val" in result and "R_val" in result:
            susceptible = np.array(result.get("S_val", []))
            infected = np.array(result.get("I_val", []))
            recovered = np.array(result.get("R_val", []))
            time_steps = np.array(result.get("time", []))

            console.print(format_title("Statistics"))
            console.print(f"  {format_param('Time points', len(susceptible))}")

            max_infected: float = float(np.max(infected))
            max_infected_time = time_steps[np.argmax(infected)]
            max_info = f"{max_infected:.4f} (at t={max_infected_time:.2f})"
            console.print(f"  {format_param('Peak infected', max_info)}")
            console.print(f"  {format_param('Final recovered', f'{recovered[-1]:.4f}')}")

    # Check whether error data is available
    has_error = "S_err" in result and "I_err" in result and "R_err" in result
    if has_error:
        console.print(
            format_info("This file contains error data (use --with-error when plotting).")
        )

    # Exportar em formato adicional, se especificado
    if export:
        if not output:
            output = result_file.replace(".json", f".{export}")

        exported_file = ExportManager.export_results(result, output, export)
        log_success(f"Information exported in {export.upper()} format: {exported_file}")


@cli.command(help="Compare results from multiple simulations")
@click.option(
    "--simple",
    is_flag=True,
    help="Generate simplified CSV result file (time, infected, error)",
)
@click.argument("result_files", nargs=-1, type=str, required=True)
@click.option("--labels", "-l", multiple=True, help="Labels for each file (optional)")
@click.option(
    "--output",
    "-o",
    type=str,
    callback=validate_output_file,
    help="Save the plot to a file (optional)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "pdf", "svg", "jpg"]),
    default="png",
    help="Plot format (when used with --output)",
)
@click.option(
    "--dpi", type=int, default=300, help="Plot resolution in DPI (when used with --output)"
)
@click.option(
    "--export",
    "-e",
    type=click.Choice(["json", "csv", "excel", "md", "html"]),
    help="Export comparative results in an additional format",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed information")
def compare(
    simple: bool,
    result_files: Tuple[str, ...],
    labels: Tuple[str, ...],
    output: Optional[str],
    format: str,
    dpi: int,
    export: Optional[str],
    verbose: bool,
) -> None:
    """Compare results from multiple simulations."""
    # Lazy imports for heavy modules
    from spkmc.io.export import ExportManager
    from spkmc.io.results import ResultManager
    from spkmc.visualization.plots import Visualizer

    # Check whether --simple was passed globally or locally
    ctx = click.get_current_context()
    parent_simple = ctx.parent.params.get("simple", False) if ctx.parent else False
    use_simple = simple or parent_simple
    # Configure verbose mode
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"

    if not result_files:
        log_error("Specify at least one results file or directory.")
        return

    # Process arguments - expand directories to JSON files
    expanded_files = []
    for arg in result_files:
        if not os.path.exists(arg):
            log_error(f"The path '{arg}' does not exist.")
            return

        if os.path.isdir(arg):
            # If it's a directory, find all JSON files
            json_files = sorted([str(f) for f in Path(arg).glob("*.json")])
            if not json_files:
                log_error(f"No JSON files found in directory: {arg}")
                return
            log_info(f"Found {len(json_files)} JSON files in {arg}")
            expanded_files.extend(json_files)
        elif os.path.isfile(arg):
            # If it's a file, check it's JSON
            if not arg.endswith(".json"):
                log_warning(f"The file '{arg}' is not a JSON file, skipping...")
                continue
            expanded_files.append(arg)
        else:
            log_error(f"The path '{arg}' is not a valid file or directory.")
            return

    if not expanded_files:
        log_error("No valid JSON files found to compare.")
        return

    # Update files_to_compare with the expanded list
    files_to_compare: List[str] = expanded_files

    log_info(f"Comparing {len(files_to_compare)} results file(s)...")

    # Use filenames as default labels if none are provided
    labels_list: List[str]
    if not labels:
        labels_list = [os.path.basename(f) for f in files_to_compare]
    elif len(labels) < len(files_to_compare):
        # Fill with filenames if there are not enough labels
        labels_list = list(labels) + [os.path.basename(f) for f in files_to_compare[len(labels) :]]
    else:
        labels_list = list(labels)

    # Load results
    results = []
    metadata_list = []

    with create_progress_bar("Loading results", len(files_to_compare), verbose) as progress:
        task = progress.add_task("Loading files...", total=len(files_to_compare))

        for i, file in enumerate(files_to_compare):
            try:
                result = ResultManager.load_result(file)
                results.append(result)

                # Extract metadata for display
                metadata = result.get("metadata", {})
                metadata_list.append(
                    {
                        "File": os.path.basename(file),
                        "Network": metadata.get("network_type", "").upper(),
                        "Distribution": metadata.get("distribution", "").capitalize(),
                        "Nodes": metadata.get("N", ""),
                        "Label": labels_list[i],
                    }
                )

                progress.update(task, advance=1)
            except Exception as e:
                log_error(f"Error loading {file}: {e}")
                return

    # Display info about loaded files
    console.print(format_title("Files for Comparison"))
    print_rich_table(metadata_list, "Results to Compare")

    # Compare results
    try:
        log_info("Generating comparison visualization...")

        # Configure title with more information
        title = "SPKMC Simulation Comparison"
        if len(files_to_compare) <= 3:  # Add details only for a small set
            title += f" ({', '.join(labels_list)})"

        Visualizer.compare_results(results, labels_list, title, output)

        if output:
            log_success(f"Comparison plot saved to: {output}")

        # Export to additional format if specified
        if export:
            if output is None:
                log_error("--output is required when using --export")
                return
            export_path = output.replace(f".{format}", f".{export}")

            # Create a combined result for export
            combined_result = {
                "results": results,
                "labels": labels_list,
                "metadata": {
                    "comparison_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "files": files_to_compare,
                },
            }
            exported_file = ExportManager.export_results(combined_result, export_path, export)
            log_success(f"Comparative results exported in {export.upper()} format: {exported_file}")

        # Generate simplified CSV for each result
        if use_simple:
            for res_idx, result_file in enumerate(files_to_compare):
                try:
                    result = results[res_idx]
                    time_steps = np.array(result.get("time", []))
                    infected = np.array(result.get("I_val", []))
                    has_error = "I_err" in result
                    infected_err = np.array(result.get("I_err", [])) if has_error else None

                    csv_path = result_file.replace(".json", "_simple.csv")
                    with open(csv_path, "w") as f:
                        # Data without header (time, infected, error)
                        for idx, t in enumerate(time_steps):
                            erro = (
                                infected_err[idx] if has_error and infected_err is not None else 0.0
                            )
                            f.write(f"{t},{infected[idx]},{erro}\n")

                    label = labels_list[res_idx]
                    log_success(f"Results for '{label}' saved to: {csv_path}")
                except Exception as e:
                    log_error(f"Error generating CSV for '{labels_list[res_idx]}': {e}")

    except Exception as e:
        log_error(f"Error generating comparison visualization: {e}")


@cli.command(help="Run multiple simulation scenarios from a JSON file or experiment")
@click.option(
    "--simple",
    is_flag=True,
    help="Generate simplified CSV result file (time, infected, error)",
)
@click.argument("scenarios_file", type=str, default=None, required=False)
@click.option(
    "--all",
    "-a",
    "run_all",
    is_flag=True,
    default=False,
    help="Run all experiments in order (no interactive menu)",
)
@click.option(
    "--override",
    is_flag=True,
    default=False,
    help="Clear existing results and force a full re-run",
)
@click.option(
    "--experiments-dir",
    "-e",
    type=str,
    default=None,
    help="Base directory for experiments (default: experiments)",
)
@click.option(
    "--output-dir",
    "-o",
    type=str,
    default="./results",
    help="Directory to save results (default: ./results)",
)
@click.option("--prefix", "-p", type=str, default="", help="Prefix for output filenames")
@click.option(
    "--compare",
    "-c",
    is_flag=True,
    default=False,
    help="Generate a comparative visualization of results",
)
@click.option("--no-plot", is_flag=True, default=False, help="Disable individual plot generation")
@click.option("--save-plot", is_flag=True, default=False, help="Save plots to files")
@click.option(
    "--zip",
    is_flag=True,
    default=False,
    help="Create a zip file with results for each scenario",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed information during execution",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode with detailed Numba logs",
)
@click.option("--clear-cache", is_flag=True, default=False, help="Clear Numba cache before running")
def batch(
    simple: bool,
    scenarios_file: Optional[str],
    run_all: bool,
    override: bool,
    experiments_dir: str,
    output_dir: Optional[str],
    prefix: str,
    compare: bool,
    no_plot: bool,
    save_plot: bool,
    zip: bool,
    verbose: bool,
    debug: bool,
    clear_cache: bool,
) -> None:
    """
    Run multiple simulation scenarios from a JSON file or experiment.

    If no file is specified, show an interactive menu with experiments available in
    the 'experiments/' directory (or the directory specified with --experiments-dir).

    The JSON file must contain a list of objects, each representing a scenario with
    parameters for the simulation. Each scenario will run sequentially and results
    will be saved to separate files in the specified directory.
    """
    # Check whether --simple was passed globally or locally
    ctx = click.get_current_context()
    parent_simple = ctx.parent.params.get("simple", False) if ctx.parent else False
    use_simple = simple or parent_simple

    # Expand ~ to absolute path in output_dir
    output_dir_str: str = output_dir if output_dir else "./results"
    output_dir_str = os.path.expanduser(output_dir_str)

    # Configure verbose mode
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"

    # Configure debug mode
    if debug:
        os.environ["SPKMC_DEBUG"] = "1"
        log_info("Debug mode enabled - detailed Numba logs active")

    # Clear Numba cache if requested
    if clear_cache:
        from spkmc.utils.numba_utils import clear_numba_cache

        clear_numba_cache()

    # ============================================================
    # EXPERIMENT MODE: No file specified
    # ============================================================
    if scenarios_file is None:
        # Create experiment manager
        exp_manager = ExperimentManager(experiments_dir)
        experiments = exp_manager.list_experiments()

        if not experiments:
            log_error("No experiment found in the 'experiments/' directory.")
            log_info("Create experiments/<name>/data.json or specify a file.")
            return

        # Determine which experiments to run
        if run_all:
            # Run all experiments in order
            experiments_to_run = experiments
            log_info(f"Running all {len(experiments)} experiments in order...")
        else:
            # Show experiment menu
            selected = display_experiments_menu(experiments)

            if selected is None:
                log_info("Operation canceled by user.")
                return

            experiments_to_run = [experiments[selected - 1]]
            log_info(f"Selected experiment: {experiments_to_run[0].name}")

        # Run each experiment
        total_start_time = time.time()
        total_scenarios_processed = 0
        experiments_completed = 0

        for exp_index, experiment in enumerate(experiments_to_run):
            if run_all:
                console.print()
                console.print(
                    format_title(
                        f"Experiment {exp_index + 1}/{len(experiments_to_run)}: {experiment.name}"
                    )
                )

            # Check for existing results
            force_rerun = override  # --override always forces a re-run
            exp_name = experiment.name
            res_count = experiment.result_count
            if experiment.has_results:
                if override:
                    # With --override, clear results
                    experiment.clean_results()
                    log_info(f"Results for '{exp_name}' removed. Forced re-run.")
                elif run_all:
                    # In --all mode without --override, skip already run scenarios
                    msg = f"'{exp_name}' has {res_count} result(s). "
                    msg += "Existing scenarios will be skipped."
                    log_info(msg)
                else:
                    # Interactive mode: ask the user
                    log_warning(f"'{exp_name}' already has {res_count} result(s).")
                    if click.confirm(
                        "Do you want to clear existing results and re-run?",
                        default=False,
                    ):
                        experiment.clean_results()
                        force_rerun = True
                        log_success("Results removed. Full re-run.")
                    else:
                        log_info("Keeping results. Existing scenarios will be ignored.")

            # Run the experiment
            start_time = time.time()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_debug(f"Starting execution at {now_str}", verbose_only=False)

            result_files = run_experiment_scenarios(
                experiment=experiment,
                verbose=verbose,
                use_simple=use_simple,
                create_zip=zip,
                no_plot=no_plot,
                save_plot=save_plot,
                force_rerun=force_rerun,
            )

            # Experiment summary
            end_time = time.time()
            execution_time = end_time - start_time
            total_scenarios_processed += len(result_files)
            experiments_completed += 1

            if not run_all:
                console.print(format_title("Experiment Execution Summary"))
                console.print(f"  {format_param('Experiment', exp_name)}")
                num_scenarios = len(experiment.scenarios)
                console.print(f"  {format_param('Total scenarios', num_scenarios)}")
                console.print(f"  {format_param('Scenarios processed', len(result_files))}")
                exec_time_str = f"{execution_time:.2f} seconds"
                console.print(f"  {format_param('Execution time', exec_time_str)}")
                results_dir = str(experiment.results_dir)
                console.print(f"  {format_param('Results directory', results_dir)}")

            n_files = len(result_files)
            msg = f"'{exp_name}' completed. {n_files} scenario(s) in {execution_time:.1f}s."
            log_success(msg)

        # Final summary for --all
        if run_all:
            total_execution_time = time.time() - total_start_time
            console.print()
            console.print(format_title("Final Summary - All Experiments"))
            console.print(f"  {format_param('Experiments run', experiments_completed)}")
            console.print(
                f"  {format_param('Total scenarios processed', total_scenarios_processed)}"
            )
            total_time_str = f"{total_execution_time:.2f} seconds"
            console.print(f"  {format_param('Total time', total_time_str)}")
            log_success(f"All {experiments_completed} experiments completed.")

            # Generate AI collection summary for --all mode
            from spkmc.analysis import AIAnalyzer, extract_experiment_metrics

            if AIAnalyzer.is_available():
                all_exp_metrics = []
                for exp in experiments_to_run:
                    if exp.has_results and exp.description:
                        # Load results and extract metrics
                        results = []
                        for json_file in exp.results_dir.glob("*.json"):
                            if not json_file.name.startswith("comparison"):
                                try:
                                    with open(json_file) as f:
                                        results.append(json.load(f))
                                except Exception:
                                    continue

                        if results:
                            metrics = extract_experiment_metrics(exp.name, exp.description, results)
                            all_exp_metrics.append(metrics)

                # Generate and display collection summary
                if all_exp_metrics:
                    try:
                        analyzer = AIAnalyzer()
                        summary = analyzer.generate_collection_summary(all_exp_metrics)

                        if summary:
                            console.print()
                            console.print(format_title("AI Summary - All Experiments"))
                            print_markdown(summary)
                            console.print()
                    except Exception:
                        # AI analysis is optional - fail silently
                        pass

        return

    # ============================================================
    # FILE MODE: Traditional execution with a scenarios file
    # ============================================================

    # Record batch execution start
    start_time = time.time()
    log_debug(
        f"Starting batch execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        verbose_only=False,
    )

    # Check whether the scenarios file exists
    if not os.path.exists(scenarios_file):
        log_error(f"The scenarios file '{scenarios_file}' does not exist.")
        return

    # Load the scenarios file
    from pathlib import Path

    try:
        log_info(f"Loading scenarios from: {scenarios_file}")
        with open(scenarios_file, "r") as f:
            data = json.load(f)

        # Support both formats:
        # 1. Direct list of scenarios: [{...}, {...}]
        # 2. Experiment format: {"name": "...", "scenarios": [...], "plot": {...}}
        if isinstance(data, list):
            scenarios = data
            experiment_name = os.path.basename(scenarios_file).replace(".json", "")
            plot_config = None
            description = None
        elif isinstance(data, dict) and "scenarios" in data:
            scenarios = data["scenarios"]
            experiment_name = data.get(
                "name", os.path.basename(scenarios_file).replace(".json", "")
            )
            plot_config = data.get("plot")
            description = data.get("description")
        else:
            log_error(
                "The file must contain a list of scenarios or an object with a 'scenarios' key."
            )
            return

        if not isinstance(scenarios, list) or len(scenarios) == 0:
            log_error("No scenarios found in the file.")
            return

        log_info(f"Experiment: {experiment_name}")
        log_success(f"Loaded {len(scenarios)} scenarios for execution.")

        # Auto-detect experiment directory and use its results/ folder
        experiment_dir = Path(os.path.dirname(os.path.abspath(scenarios_file)))
        if output_dir_str == "./results":
            output_dir_str = str(experiment_dir / "results")
            log_info(f"Using experiment results directory: {output_dir_str}")

    except Exception as e:
        log_error(f"Error loading scenarios file: {e}")
        return

    # Create PlotConfig from dict
    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()

    # Create Experiment object
    experiment = Experiment(
        name=experiment_name,
        path=experiment_dir,
        description=description,
        plot_config=config,
        scenarios=scenarios,
    )

    # Handle --override for file mode
    force_rerun = override  # --override always forces a re-run
    if override and experiment.has_results:
        experiment.clean_results()
        log_info("Previous results removed. Forcing re-run.")

    # Use parallelized execution
    result_files = run_experiment_scenarios(
        experiment=experiment,
        verbose=verbose,
        use_simple=use_simple,
        create_zip=zip,
        no_plot=no_plot,
        save_plot=save_plot,
        force_rerun=force_rerun,
    )

    # Summary
    end_time = time.time()
    total_execution_time = end_time - start_time

    console.print(format_title("Execution Summary"))
    console.print(f"  {format_param('Experiment', experiment_name)}")
    console.print(f"  {format_param('Total scenarios', len(scenarios))}")
    console.print(f"  {format_param('Scenarios processed', len(result_files))}")
    console.print(f"  {format_param('Total time', f'{total_execution_time:.2f} seconds')}")
    console.print(f"  {format_param('Results directory', output_dir_str)}")

    log_success(f"Execution completed. {len(result_files)} scenario(s) processed.")


@cli.command(help="Clean results from all experiments")
@click.option(
    "--experiments-dir",
    "-e",
    type=str,
    default=None,
    help="Base directory for experiments (default: experiments)",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Auto-confirm without prompting")
@click.option(
    "--numba-cache",
    is_flag=True,
    default=False,
    help="Also clear the Numba compilation cache",
)
def clean(experiments_dir: Optional[str], yes: bool, numba_cache: bool) -> None:
    """
    Remove all results from all experiments.

    This command cleans the 'results/' directory of each experiment,
    allowing all simulations to be re-run from scratch.
    """
    import shutil
    from pathlib import Path

    # Create experiment manager
    exp_manager = ExperimentManager(experiments_dir)
    experiments = exp_manager.list_experiments()

    # Check for root results/ directory too
    root_results = Path("results")
    has_root_results = root_results.exists() and any(root_results.glob("*.json"))

    if not experiments and not has_root_results:
        log_error("No experiments or results found.")
        return

    # Count existing results
    total_results = sum(exp.result_count for exp in experiments)
    experiments_with_results = [exp for exp in experiments if exp.has_results]

    if not experiments_with_results and not has_root_results:
        log_info("No results to clean. All experiments are empty.")
        return

    # Show what will be removed
    console.print(format_title("Results Found"))
    for exp in experiments_with_results:
        console.print(f"  • {exp.name}: {exp.result_count} result(s)")
    if has_root_results:
        root_count = len(list(root_results.glob("*.json")))
        console.print(f"  • [root]/results/: {root_count} file(s)")
        total_results += root_count
    console.print()
    console.print(f"Total: {total_results} result(s)")
    console.print()

    # Confirm
    if not yes:
        if not click.confirm("Do you want to remove all results?", default=False):
            log_info("Operation canceled.")
            return

    # Clean results
    cleaned_count = 0
    for exp in experiments_with_results:
        try:
            exp.clean_results()
            cleaned_count += 1
            log_success(f"Results for '{exp.name}' removed.")
        except Exception as e:
            log_error(f"Error cleaning '{exp.name}': {e}")

    # Clean root results/ directory
    if has_root_results:
        try:
            shutil.rmtree(root_results)
            log_success("Results from '[root]/results/' removed.")
            cleaned_count += 1
        except Exception as e:
            log_error(f"Error cleaning '[root]/results/': {e}")

    # Clean Numba cache if requested
    if numba_cache:
        from spkmc.utils.numba_utils import clear_numba_cache

        clear_numba_cache()
        log_success("Numba cache cleared.")

    console.print()
    log_success(f"Cleanup completed. {cleaned_count} location(s) cleaned.")
