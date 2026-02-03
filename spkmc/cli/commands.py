"""
SPKMC CLI commands.

This module contains command-line interface commands for the SPKMC algorithm,
including commands to run simulations, visualize results, and display information.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from spkmc.analysis.ai_analyzer import AIAnalyzer

import click
import numpy as np
import questionary
from prompt_toolkit.keys import Keys
from questionary import Style

from spkmc import __version__
from spkmc.cli.formatting import (
    console,
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

# Lazy imports for heavy modules (to speed up CLI startup)
from spkmc.io.data_manager import DataManager

# These are imported inside functions that need them:
#   - spkmc.core.distributions (imports numba_utils - 60s JIT compilation)
#   - spkmc.core.simulation (imports numba_utils)
#   - spkmc.utils.hardware (GPU detection)
#   - spkmc.utils.parallel (imports simulation)
#
# Light imports that can stay at module level:
from spkmc.io.experiments import Experiment, ExperimentManager
from spkmc.visualization.plots import DEFAULT_PLOT_DPI

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


class MissingParameterError(Exception):
    """Raised when a required parameter is missing from a scenario."""

    def __init__(self, param_name: str, scenario_label: str):  # noqa: B042
        self.param_name = param_name
        self.scenario_label = scenario_label
        message = (
            f"Missing required parameter '{param_name}' in scenario '{scenario_label}'. "
            f"Please add this parameter to your experiment's data.json file."
        )
        super().__init__(message, param_name, scenario_label)


def _validate_scenario(scenario: Any, scenario_label: str) -> Dict[str, Any]:
    """
    Validate all required parameters in a scenario and return extracted values.

    Args:
        scenario: The scenario (dict or Scenario object)
        scenario_label: Label for error messages

    Returns:
        Dict with all validated parameters

    Raises:
        MissingParameterError: If any required parameter is missing
    """
    # Convert Scenario object to dict if needed
    if hasattr(scenario, "model_dump"):
        scenario = scenario.model_dump(by_alias=True)
    elif hasattr(scenario, "dict"):
        scenario = scenario.dict(by_alias=True)

    # Core required parameters for all scenarios
    required_params = [
        "network",
        "distribution",
        "nodes",
        "samples",
        "num_runs",
        "initial_perc",
        "t_max",
        "steps",
        "lambda",
    ]

    # Check all required params first
    for param in required_params:
        if scenario.get(param) is None:
            raise MissingParameterError(param, scenario_label)

    network = scenario["network"]
    distribution = scenario["distribution"]

    # Distribution-specific required parameters
    if distribution == "gamma":
        for param in ["shape", "scale"]:
            if scenario.get(param) is None:
                raise MissingParameterError(param, scenario_label)
    elif distribution == "exponential":
        if scenario.get("mu") is None:
            raise MissingParameterError("mu", scenario_label)

    # Network-specific required parameters
    # k_avg is required for er, sf, rrn but NOT for cg (complete graphs)
    if network in ["er", "sf", "rrn"]:
        if scenario.get("k_avg") is None:
            raise MissingParameterError("k_avg", scenario_label)
    if network == "sf":
        if scenario.get("exponent") is None:
            raise MissingParameterError("exponent", scenario_label)

    # Return all extracted values
    return {
        "network": network,
        "distribution": distribution,
        "nodes": scenario["nodes"],
        "k_avg": scenario.get("k_avg"),
        "samples": scenario["samples"],
        "num_runs": scenario["num_runs"],
        "initial_perc": scenario["initial_perc"],
        "t_max": scenario["t_max"],
        "steps": scenario["steps"],
        "lambda": scenario["lambda"],
        "shape": scenario.get("shape"),
        "scale": scenario.get("scale"),
        "mu": scenario.get("mu"),
        "exponent": scenario.get("exponent"),
    }


# Special return value for "Create New Experiment" menu option
CREATE_EXPERIMENT_SENTINEL = -999


def display_experiments_menu(experiments: List[Experiment]) -> Optional[int]:
    """
    Display an interactive menu to select an experiment with arrow navigation.

    Args:
        experiments: List of available experiments

    Returns:
        Index of the selected experiment (1-based), CREATE_EXPERIMENT_SENTINEL to create
        a new experiment, or None to exit
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
        # Use fixed-width columns for alignment
        scenarios_text = f"{scenarios_count:>3} scenarios"
        display = f"{exp.name:<35} │ {scenarios_text} │ {status:<12}"
        choices.append(questionary.Choice(title=display, value=i + 1))

    # Add separator and options
    choices.append(questionary.Separator("─" * 70))
    choices.append(
        questionary.Choice(title="[+] Create New Experiment", value=CREATE_EXPERIMENT_SENTINEL)
    )
    choices.append(questionary.Choice(title="[x] Cancel", value=-1))

    # Display header
    bc = "[bold cyan]"  # Style shorthand
    bce = "[/bold cyan]"
    bw = "[bold white]"
    bwe = "[/bold white]"
    dm = "[dim]"
    dme = "[/dim]"
    box_width = 70
    box_line = "═" * box_width

    # Center the title and hint text
    title = "SPKMC Experiment Runner"
    hint_text = "↑/↓ to navigate, Enter to select, Esc to cancel"
    title_padded = title.center(box_width)
    hint_padded = hint_text.center(box_width)

    console.print()
    console.print(f"{bc}╔{box_line}╗{bce}")
    console.print(f"{bc}║{bce}{bw}{title_padded}{bwe}{bc}║{bce}")
    console.print(f"{bc}╠{box_line}╣{bce}")
    console.print(f"{bc}║{bce}{dm}{hint_padded}{dme}{bc}║{bce}")
    console.print(f"{bc}╚{box_line}╝{bce}")
    console.print()

    try:
        # Create the select prompt
        prompt = questionary.select(
            "Select an experiment to run:",
            choices=choices,
            style=custom_style,
            instruction="",
            use_indicator=True,
            use_shortcuts=False,
        )

        # Add ESC key binding to cancel the prompt - use getattr for type safety
        key_bindings = prompt.application.key_bindings
        if key_bindings is not None:
            add_binding = getattr(key_bindings, "add", None)
            if add_binding is not None:

                @add_binding(Keys.Escape)
                def handle_escape(event: Any) -> None:
                    event.app.exit(result=None)

        result = prompt.ask()

        # Handle cancel or Ctrl+C
        if result is None or result == -1:
            return None
        return int(result)
    except KeyboardInterrupt:
        return None


def create_experiment_interactive() -> Optional[Experiment]:
    """
    Interactively create a new experiment configuration.

    Prompts the user for experiment name, description, and all simulation parameters,
    then generates a data.json file in experiments/<name>/.

    Returns:
        The created Experiment object, or None if creation was cancelled
    """
    console.print()
    console.print(format_title("Create New Experiment"))
    console.print()

    # Custom style for prompts
    custom_style = Style(
        [
            ("qmark", "fg:cyan bold"),
            ("question", "fg:white bold"),
            ("answer", "fg:green"),
            ("pointer", "fg:cyan bold"),
        ]
    )

    # Get experiment name
    name = questionary.text(
        "Experiment name (snake_case):",
        style=custom_style,
        validate=lambda x: len(x) > 0 and x.replace("_", "").replace("-", "").isalnum(),
    ).ask()

    if not name:
        return None

    # Check if experiment already exists
    exp_path = Path("experiments") / name
    if exp_path.exists():
        log_error(f"Experiment '{name}' already exists at {exp_path}")
        return None

    # Get description (hypothesis)
    description = questionary.text(
        "Description/Hypothesis:",
        style=custom_style,
    ).ask()

    if description is None:
        return None

    console.print()
    console.print(format_info("Simulation Parameters"))
    console.print(format_info("(Press Enter for default values)"))
    console.print()

    # Network type
    network_type = questionary.select(
        "Network type:",
        choices=[
            questionary.Choice("er - Erdos-Renyi (random)", value="er"),
            questionary.Choice("sf - Scale-Free Network (power-law)", value="sf"),
            questionary.Choice("cg - Complete Graph (fully connected)", value="cg"),
            questionary.Choice("rrn - Random Regular Network", value="rrn"),
        ],
        style=custom_style,
    ).ask()

    if network_type is None:
        return None

    # Distribution type
    dist_type = questionary.select(
        "Distribution type:",
        choices=[
            questionary.Choice("gamma - Gamma distribution", value="gamma"),
            questionary.Choice("exponential - Exponential distribution", value="exponential"),
        ],
        style=custom_style,
    ).ask()

    if dist_type is None:
        return None

    # Numeric parameters with defaults
    def get_float(prompt: str, default: float) -> Optional[float]:
        val = questionary.text(
            f"{prompt} [{default}]:",
            default=str(default),
            style=custom_style,
        ).ask()
        if val is None:
            return None
        try:
            return float(val) if val else default
        except ValueError:
            return default

    def get_int(prompt: str, default: int) -> Optional[int]:
        val = questionary.text(
            f"{prompt} [{default}]:",
            default=str(default),
            style=custom_style,
        ).ask()
        if val is None:
            return None
        try:
            return int(val) if val else default
        except ValueError:
            return default

    # Get all parameters
    nodes = get_int("Number of nodes", DEFAULT_N)
    if nodes is None:
        return None

    k_avg = get_float("Average degree (k_avg)", DEFAULT_K_AVG)
    if k_avg is None:
        return None

    # Distribution-specific parameters
    if dist_type == "gamma":
        shape = get_float("Shape (gamma)", DEFAULT_SHAPE)
        if shape is None:
            return None
        scale = get_float("Scale (gamma)", DEFAULT_SCALE)
        if scale is None:
            return None
        mu = DEFAULT_MU  # Not used for gamma
    else:
        shape = DEFAULT_SHAPE  # Not used for exponential
        scale = DEFAULT_SCALE
        mu_input = get_float("Mu (exponential)", DEFAULT_MU)
        if mu_input is None:
            return None
        mu = mu_input

    lambda_val = get_float("Lambda (infection rate)", DEFAULT_LAMBDA)
    if lambda_val is None:
        return None

    # Network-specific parameters
    if network_type == "sf":
        exponent = get_float("Exponent (scale-free networks)", DEFAULT_EXPONENT)
        if exponent is None:
            return None
    else:
        exponent = DEFAULT_EXPONENT

    initial_perc = get_float("Initial infected %", DEFAULT_INITIAL_PERC)
    if initial_perc is None:
        return None

    samples = get_int("Samples per run", DEFAULT_SAMPLES)
    if samples is None:
        return None

    num_runs = get_int("Number of runs", DEFAULT_NUM_RUNS)
    if num_runs is None:
        return None

    t_max = get_float("Maximum time (t_max)", DEFAULT_T_MAX)
    if t_max is None:
        return None

    steps = get_int("Time steps", DEFAULT_STEPS)
    if steps is None:
        return None

    console.print()
    console.print(format_info("Scenario Variations"))
    console.print(format_info("Create scenarios by varying a parameter"))
    console.print()

    # Choose parameter to vary
    vary_choices = [
        questionary.Choice("lambda - Infection rate", value="lambda"),
        questionary.Choice("k_avg - Average degree", value="k_avg"),
        questionary.Choice("nodes - Network size", value="nodes"),
        questionary.Choice("initial_perc - Initial infected", value="initial_perc"),
    ]

    if dist_type == "gamma":
        vary_choices.extend(
            [
                questionary.Choice("shape - Gamma shape", value="shape"),
                questionary.Choice("scale - Gamma scale", value="scale"),
            ]
        )
    else:
        vary_choices.append(questionary.Choice("mu - Exponential mu", value="mu"))

    if network_type == "sf":
        vary_choices.append(questionary.Choice("exponent - Power-law exponent", value="exponent"))

    vary_choices.append(questionary.Choice("(none) - Single scenario", value=None))

    vary_param = questionary.select(
        "Parameter to vary:",
        choices=vary_choices,
        style=custom_style,
    ).ask()

    # Build base parameters (global defaults)
    base_parameters: Dict[str, Any] = {
        "network": network_type,
        "distribution": dist_type,
        "nodes": nodes,
        "k_avg": k_avg,
        "lambda": lambda_val,
        "initial_perc": initial_perc,
        "samples": samples,
        "num_runs": num_runs,
        "t_max": t_max,
        "steps": steps,
    }

    if dist_type == "gamma":
        base_parameters["shape"] = shape
        base_parameters["scale"] = scale
    else:
        base_parameters["mu"] = mu

    if network_type == "sf":
        base_parameters["exponent"] = exponent

    # Build scenarios (only overrides, inheriting from parameters)
    scenarios: List[Dict[str, Any]] = []

    if vary_param is None:
        # Single scenario - only needs a label
        scenarios.append({"label": "baseline"})
    else:
        # Get variation values
        values_str = questionary.text(
            f"Values for {vary_param} (comma-separated):",
            style=custom_style,
        ).ask()

        if values_str is None:
            return None

        try:
            values = [float(v.strip()) for v in values_str.split(",")]
        except ValueError:
            log_error("Invalid values. Please enter comma-separated numbers.")
            return None

        for val in values:
            # Scenario only contains the override and label
            label_val = str(val).replace(".", "_")
            scenarios.append(
                {
                    vary_param: val,
                    "label": f"{vary_param}_{label_val}",
                }
            )

    # Create experiment data structure with parameters inheritance
    experiment_data = {
        "name": name.replace("_", " ").title(),
        "description": description,
        "plot": {
            "title": f"Effect of {vary_param or 'Parameters'} on Epidemic Dynamics",
            "xlabel": "Time",
            "ylabel": "Proportion of Individuals",
            "legend_position": "upper right",
            "figsize": [12, 8],
            "states_to_plot": ["I"],
            "dpi": 300,
            "grid": True,
        },
        "parameters": base_parameters,
        "scenarios": scenarios,
    }

    # Create directory and save
    exp_path.mkdir(parents=True, exist_ok=True)
    data_file = exp_path / "data.json"

    with open(data_file, "w") as f:
        json.dump(experiment_data, f, indent=2)

    log_success(f"Experiment created at: {data_file}")
    console.print()

    # Ask if user wants to run it
    run_now = questionary.confirm(
        "Run this experiment now?",
        default=True,
        style=custom_style,
    ).ask()

    if run_now:
        # Load and return the experiment
        exp_manager = ExperimentManager("experiments")
        experiment = exp_manager.load_experiment(name)
        return experiment

    return None


def _execute_single_scenario(
    scenario: Dict[str, Any],
    scenario_index: int,
    results_dir: Path,
    experiment_name: str,
    no_plot: bool,
    use_gpu: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    export_format: str = "json",
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Execute a single scenario and return the result.

    This function is designed to be called from parallel workers.

    Args:
        scenario: Scenario configuration dictionary
        scenario_index: Index of the scenario in the experiment
        results_dir: Path to results directory
        experiment_name: Name of the experiment
        no_plot: Disable plot generation
        use_gpu: Use GPU acceleration if available
        progress_callback: Optional callback called per sample (called with 1 to advance by 1)
        export_format: Format for saving results (json, csv, excel, md, html)

    Returns:
        Tuple of (result_dict, output_file_path, scenario_label)
    """
    # Lazy imports for heavy modules
    from spkmc.core.distributions import create_distribution
    from spkmc.core.simulation import SPKMC
    from spkmc.io.data_manager import DataManager as WorkerDataManager
    from spkmc.utils.parallel import get_worker_progress_callback

    # In parallel mode, get callback from the worker's global queue (set by initializer)
    # In sequential mode, use the passed callback
    if progress_callback is None:
        progress_callback = get_worker_progress_callback()
    scenario_num = scenario_index + 1

    # Create scenario label first (needed for error messages)
    scenario_label = f"scenario_{scenario_num:03d}"
    # Handle both dict and Scenario object
    if isinstance(scenario, dict):
        if "label" in scenario:
            scenario_label = scenario["label"]
    elif hasattr(scenario, "label") and scenario.label:
        scenario_label = scenario.label

    # Validate and extract all parameters at once
    params = _validate_scenario(scenario, scenario_label)
    network_type = params["network"]
    dist_type = params["distribution"]
    nodes = params["nodes"]
    k_avg = params["k_avg"]
    samples = params["samples"]
    num_runs = params["num_runs"]
    initial_perc = params["initial_perc"]
    t_max = params["t_max"]
    steps = params["steps"]
    lambda_val = params["lambda"]
    shape = params["shape"]
    scale = params["scale"]
    mu = params["mu"]
    exponent = params["exponent"]

    # Determine file extension based on export format
    ext_map = {"json": ".json", "csv": ".csv", "excel": ".xlsx", "md": ".md", "html": ".html"}
    ext = ext_map.get(export_format, ".json")
    # Normalize label for filename (lowercase, no spaces)
    from spkmc.io.experiments import Scenario

    normalized_label = Scenario.normalize_label(scenario_label)
    output_file = str(results_dir / f"{normalized_label}{ext}")

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
        "num_runs": num_runs,  # All network types use num_runs (including CG)
    }

    if network_type in ["er", "sf", "rrn"]:
        simulation_params["k_avg"] = k_avg

    if network_type == "sf":
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
        "network": network_type,
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

    if network_type in ["er", "sf", "rrn"]:
        metadata["k_avg"] = k_avg
        metadata["num_runs"] = num_runs

    if network_type == "sf":
        metadata["exponent"] = exponent

    if has_error:
        output_result.update(
            {
                "S_err": list(susceptible_err),
                "I_err": list(infected_err),
                "R_err": list(recovered_err),
            }
        )

    # Save result in the specified format
    WorkerDataManager.save(output_result, output_file)

    return (output_result, output_file, scenario_label)


def _run_experiment_with_engine(
    experiment: Experiment,
    verbose: bool,
    no_plot: bool,
    force_rerun: bool = False,
    run_analysis: bool = False,
    export_format: str = "json",
) -> List[str]:
    """
    Run all scenarios in an experiment using the unified ExecutionEngine.

    Args:
        experiment: Experiment to run
        verbose: Show detailed information
        no_plot: Disable plot generation
        force_rerun: Force re-run even if cached results exist
        run_analysis: Run AI analysis on results (requires OPENAI_API_KEY)
        export_format: Format for saving results (json, csv, excel, md, html)

    Returns:
        List of generated result file paths
    """
    from spkmc.cli.display import (
        ProgressTracker,
        display_aggregate_statistics,
        display_hardware_panel,
    )
    from spkmc.core.engine import ExecutionContext, ExecutionEngine
    from spkmc.models import Scenario
    from spkmc.visualization.plots import Visualizer

    # If force_rerun, clean ALL existing results first
    if force_rerun and experiment.results_dir.exists():
        import shutil

        shutil.rmtree(experiment.results_dir)

    # Ensure results directory exists
    results_dir = experiment.ensure_results_dir()

    # Convert scenarios to Scenario objects if needed
    scenarios: List[Scenario] = []
    try:
        for s in experiment.scenarios:
            if isinstance(s, Scenario):
                scenarios.append(s)
            elif isinstance(s, dict):
                # Convert dict to Scenario
                try:
                    scenarios.append(Scenario(**s))
                except Exception as e:
                    # Convert Pydantic ValidationError to MissingParameterError for better messages
                    from pydantic import ValidationError

                    if isinstance(e, ValidationError):
                        scenario_label = s.get("label", "unknown")
                        # Find first missing required field
                        for error in e.errors():
                            if error.get("type") == "missing":
                                field_name = error.get("loc", ["unknown"])[0]
                                # Map alias back to user-facing name
                                if field_name == "lambda_param":
                                    field_name = "lambda"
                                raise MissingParameterError(field_name, scenario_label) from e
                        # For other validation errors, provide context
                        log_error(f"Validation error in scenario '{scenario_label}': {e}")
                        raise
                    log_error(f"Error converting scenario: {e}")
                    raise
    except MissingParameterError as e:
        log_error(str(e))
        console.print(
            "\n[yellow]Hint: Add missing parameters to 'parameters' section "
            "in data.json for global defaults,\nor to individual scenarios "
            "for per-scenario values.[/yellow]"
        )
        raise click.Abort()

    # Filter out scenarios with existing results if not force_rerun
    ext_map = {"json": ".json", "csv": ".csv", "excel": ".xlsx", "md": ".md", "html": ".html"}
    ext = ext_map.get(export_format, ".json")
    cached_results: List[Tuple[Scenario, str]] = []  # (scenario, existing_file_path)
    scenarios_to_run: List[Scenario] = []

    if not force_rerun:
        for scenario in scenarios:
            output_file = results_dir / f"{scenario.normalized_label}{ext}"
            if output_file.exists():
                cached_results.append((scenario, str(output_file)))
            else:
                scenarios_to_run.append(scenario)

        if cached_results:
            log_info(
                f"Skipping {len(cached_results)} scenario(s) with existing results. "
                f"Use --override to re-run."
            )
    else:
        scenarios_to_run = scenarios

    # If all scenarios are cached, just return the cached file paths
    if not scenarios_to_run:
        log_info("All scenarios already have results. Nothing to execute.")
        return [path for _, path in cached_results]

    # Create execution engine
    engine = ExecutionEngine(verbose=verbose)
    strategy = engine.configure_parallelization(len(scenarios_to_run))

    # Display hardware panel first
    display_hardware_panel(engine.hardware, strategy)

    # Calculate total samples for progress (only for scenarios to run)
    total_samples = sum(s.total_samples() for s in scenarios_to_run)

    # Create execution context (only for scenarios that need to run)
    context = ExecutionContext(
        scenarios=scenarios_to_run,
        experiment_name=experiment.name,
        results_dir=results_dir,
        no_plot=no_plot,
        export_format=export_format,
    )

    # Execute with progress tracking
    all_results = []
    with ProgressTracker(
        total_samples, verbose, f"Running {len(scenarios_to_run)} scenario(s)"
    ) as tracker:
        context.on_sample_progress = tracker.get_callback()
        try:
            results = engine.execute(context)
            all_results = results
        except MissingParameterError as e:
            log_error(str(e))
            console.print(
                "\n[yellow]Hint: Add missing parameters to 'parameters' section "
                "in data.json for global defaults,\nor to individual scenarios "
                "for per-scenario values.[/yellow]"
            )
            raise click.Abort()
        except Exception as e:
            log_error(f"Error during execution: {e}")
            raise

    # Collect result file paths and data for comparison plot
    result_files: List[str] = []
    all_result_dicts: List[Dict[str, Any]] = []
    all_labels: List[str] = []

    # Include cached results first
    for scenario, cached_path in cached_results:
        result_files.append(cached_path)
        # Load cached result for comparison plot
        try:
            cached_data = DataManager.load(cached_path)
            all_result_dicts.append(cached_data)
            all_labels.append(scenario.label)
        except Exception:
            pass  # Skip if can't load cached result

    # Add newly executed results
    for result in all_results:
        if result.output_path:
            result_files.append(result.output_path)
        all_result_dicts.append(result.to_dict())
        all_labels.append(result.scenario_label)

    # Display aggregate statistics for all results
    display_aggregate_statistics(all_results)

    # Display any failed scenarios
    failed_scenarios = engine.get_failed_scenarios()
    if failed_scenarios:
        console.print()
        log_warning(f"{len(failed_scenarios)} scenario(s) failed during execution:")
        for label, error in failed_scenarios:
            console.print(f"  [red]✗[/red] {label}: {error}")
        console.print()

    # Generate comparison plot with custom configuration
    if len(all_result_dicts) > 1 and not no_plot:
        compare_path = str(results_dir / "comparison.png")
        try:
            Visualizer.compare_results_with_config(
                all_result_dicts, all_labels, experiment.plot, compare_path
            )
            log_success(f"Comparison plot saved to: {compare_path}")
        except Exception as e:
            log_error(f"Error generating comparison plot: {e}")

    # Generate AI analysis if available and requested
    if run_analysis and experiment.description and all_result_dicts:
        from spkmc.analysis import try_generate_analysis

        analysis_path = try_generate_analysis(
            experiment_name=experiment.name,
            experiment_description=experiment.description,
            results=all_result_dicts,
            results_dir=results_dir,
            verbose=verbose,
        )
        if analysis_path:
            log_success(f"AI analysis generated at: {analysis_path}")

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
@click.option(
    "--analyze",
    is_flag=True,
    help="Run AI analysis on results (requires OPENAI_API_KEY environment variable)",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, analyze: bool) -> None:
    """Main command group for the SPKMC CLI."""
    from spkmc.cli.utils import CLIContext, require_analyze_api_key

    # Store CLI context for subcommands
    ctx.ensure_object(dict)
    cli_context = CLIContext(verbose=verbose, analyze=analyze)
    cli_context.configure_environment()
    ctx.obj["cli_context"] = cli_context

    if verbose:
        log_info("Verbose mode enabled")

    # Validate --analyze flag has required API key
    if analyze:
        require_analyze_api_key()
        log_info("AI analysis enabled")


@cli.command(help="Run an SPKMC simulation")
@click.option(
    "--network-type",
    "-n",
    type=str,
    default="er",
    show_default=True,
    callback=validate_network_type,
    help="Network type: er (Erdos-Renyi), sf (Scale-Free), cg (Complete), rrn (Regular)",
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
    help="Exponent for scale-free networks (SF)",
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
    "--export",
    "-e",
    type=click.Choice(["json", "csv", "excel", "md", "html"]),
    default="json",
    show_default=True,
    help="Format for saving results",
)
@click.option("--no-plot", is_flag=True, default=False, help="Do not display the results plot")
@click.option("--override", is_flag=True, default=False, help="Overwrite existing results")
@click.pass_context
def run(
    ctx: click.Context,
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
    export: str,
    no_plot: bool,
    override: bool,
) -> None:
    """Run an SPKMC simulation with the specified parameters."""
    # Lazy imports
    from spkmc.cli.display import ProgressTracker, display_hardware_panel, display_result_statistics
    from spkmc.cli.utils import get_cli_context
    from spkmc.core.engine import ExecutionContext, ExecutionEngine
    from spkmc.models import Scenario
    from spkmc.utils.gpu_utils import check_gpu_suggestion
    from spkmc.visualization.plots import Visualizer

    # Check if user has GPU hardware but no GPU packages installed
    check_gpu_suggestion()

    cli_context = get_cli_context()
    verbose = cli_context.verbose
    run_analysis = cli_context.analyze

    # Record simulation start
    start_time = time.time()
    log_debug(
        f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        verbose_only=False,
    )

    # Determine output path with correct extension
    output_path: Optional[str] = None
    if output:
        ext_map = {
            "json": ".json",
            "csv": ".csv",
            "excel": ".xlsx",
            "md": ".md",
            "html": ".html",
        }
        ext = ext_map.get(export, ".json")
        # Use Path to properly handle extensions without affecting directory names
        output_p = Path(output)
        output_path = str(output_p.with_suffix(ext))

        # Check if file exists and --override not specified
        if Path(output_path).exists() and not override:
            log_error(
                f"Output file already exists: {output_path}\n"
                f"       Use --override to overwrite existing results."
            )
            ctx.exit(1)

    # Validate that N * initial_perc >= 1 to ensure at least 1 infected node
    min_infected = int(nodes * initial_perc)
    if min_infected < 1:
        log_error(
            f"Initial infected nodes would be 0 (N={nodes} × initial_perc={initial_perc} "
            f"= {nodes * initial_perc:.2f}).\n"
            f"       Increase --nodes or --initial-perc so that N × initial_perc ≥ 1."
        )
        ctx.exit(1)

    # Create Scenario from CLI arguments
    try:
        scenario = Scenario.from_cli_args(
            network_type=network_type,
            distribution=dist_type,
            nodes=nodes,
            samples=samples,
            num_runs=num_runs,
            k_avg=k_avg if network_type in ["er", "sf", "rrn"] else None,
            exponent=exponent if network_type == "sf" else None,
            shape=shape if dist_type == "gamma" else None,
            scale=scale if dist_type == "gamma" else None,
            mu=mu if dist_type == "exponential" else None,
            lambda_param=lambda_val,
            t_max=t_max,
            steps=steps,
            initial_perc=initial_perc,
            output_path=output_path,
        )
    except Exception as e:
        log_error(f"Error creating scenario: {e}")
        ctx.exit(1)

    # Create execution engine and configure parallelization
    engine = ExecutionEngine(verbose=verbose)
    strategy = engine.configure_parallelization(1)

    # Display hardware panel first
    display_hardware_panel(engine.hardware, strategy)

    # Determine results directory (for run storage or explicit output)
    results_dir: Optional[Path] = None
    if output_path:
        results_dir = Path(output_path).parent
        results_dir.mkdir(parents=True, exist_ok=True)

    # Create execution context
    context = ExecutionContext(
        scenarios=[scenario],
        results_dir=results_dir,
        no_plot=no_plot,
        export_format=export,
    )

    # Execute with progress tracking
    total_samples = scenario.total_samples()
    with ProgressTracker(total_samples, verbose, "Running simulation") as tracker:
        context.on_sample_progress = tracker.get_callback()
        try:
            results = engine.execute(context)
        except Exception as e:
            log_error(f"Error during simulation: {e}")
            ctx.exit(1)

    if not results:
        log_error("No results generated")
        ctx.exit(1)

    result = results[0]
    log_success("Simulation completed successfully!")

    # Display statistics
    display_result_statistics(result)

    # Record execution time
    execution_time = time.time() - start_time
    log_debug(f"Execution time: {execution_time:.2f} seconds", verbose_only=False)

    # Log output path if saved
    if result.output_path:
        log_success(f"Results saved to: {result.output_path}")

    # Plot results if requested
    if not no_plot:
        log_info("Generating visualization...")

        title = (
            f"SPKMC Simulation - Network {network_type.upper()}, "
            f"Distribution {dist_type.capitalize()}"
        )

        try:
            if result.has_error:
                Visualizer.plot_result_with_error(
                    result.S_val,
                    result.I_val,
                    result.R_val,
                    result.S_err,
                    result.I_err,
                    result.R_err,
                    result.time,
                    title,
                )
            else:
                Visualizer.plot_result(result.S_val, result.I_val, result.R_val, result.time, title)
        except Exception as e:
            log_error(f"Error generating visualization: {e}")

    # Run AI analysis if requested (requires output file)
    if run_analysis and result.output_path:
        log_info("Generating AI analysis...")
        try:
            from spkmc.analysis.ai_analyzer import AIAnalyzer

            analyzer = AIAnalyzer()
            # Determine output directory for analysis
            analysis_dir = (
                Path(result.output_path).parent
                if Path(result.output_path).parent.exists()
                else Path(".")
            )
            exp_name = f"{network_type}_{dist_type}_simulation"
            exp_desc = (
                f"SPKMC simulation with {network_type.upper()} network "
                f"and {dist_type.capitalize()} distribution"
            )
            analysis_path = analyzer.analyze_experiment(
                experiment_name=exp_name,
                experiment_description=exp_desc,
                results=[result.to_dict()],
                results_dir=analysis_dir,
            )
            if analysis_path:
                log_success(f"Analysis generated: {analysis_path}")
        except Exception as e:
            log_error(f"Error generating analysis: {e}")
    elif run_analysis and not result.output_path:
        log_warning("--analyze requires --output to save results for analysis")


@cli.command()
@click.argument("paths", nargs=-1, type=str, required=True)
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
    "--dpi",
    type=int,
    default=DEFAULT_PLOT_DPI,
    help="Plot resolution in DPI (when used with --output)",
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
    help="Plot each result in separate charts instead of comparison",
)
@click.option(
    "--labels",
    "-l",
    multiple=True,
    help="Custom labels for comparison (use multiple times)",
)
@click.pass_context
def plot(
    ctx: click.Context,
    paths: Tuple[str, ...],
    with_error: bool,
    output: Optional[str],
    format: str,
    dpi: int,
    export: Optional[str],
    states: Tuple[str, ...],
    separate: bool,
    labels: Tuple[str, ...],
) -> None:
    """
    Visualize and compare simulation results.

    Accepts one or more result files or directories. When multiple results are
    provided, creates a comparison plot. Use --separate to generate individual
    plots instead.

    \b
    Supported formats: JSON (.json), CSV (.csv), Excel (.xlsx, .xls)

    \b
    Examples:
        spkmc plot result.json
        spkmc plot data/experiment/
        spkmc plot result1.json result2.json -l "Scenario A" -l "Scenario B"
        spkmc plot data/exp1/ data/exp2/ -o comparison.png
        spkmc plot data/experiment/ --separate -e
    """
    # Lazy imports for heavy modules
    # Get CLI context for global options
    from spkmc.cli.utils import get_cli_context
    from spkmc.visualization.plots import Visualizer

    cli_context = get_cli_context()
    _ = cli_context.verbose  # Available for future verbose output

    if not paths:
        log_error("Specify at least one result file or directory.")
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

    # Helper to construct output path with correct format extension
    def get_output_path_with_format(base_path: Optional[str], fmt: str) -> Optional[str]:
        """Add format extension to output path if not already present."""
        if not base_path:
            return None
        p = Path(base_path)
        # Map format to extension
        ext_map = {"png": ".png", "pdf": ".pdf", "svg": ".svg", "jpg": ".jpg"}
        expected_ext = ext_map.get(fmt, ".png")
        # If path has no extension or different extension, add/replace it
        if p.suffix.lower() not in ext_map.values():
            return str(p.with_suffix(expected_ext))
        return str(p)

    # Apply format to output path
    output_with_format = get_output_path_with_format(output, format)

    # Collect result files from all paths
    result_files: List[Path] = []
    supported_extensions = {".json", ".csv", ".xlsx", ".xls"}

    for path in paths:
        path_obj = Path(path)

        if not path_obj.exists():
            log_error(f"Path not found: {path}")
            ctx.exit(1)

        if path_obj.is_file():
            # Single file
            if path_obj.suffix.lower() in supported_extensions:
                result_files.append(path_obj)
            else:
                log_warning(f"Unsupported format '{path_obj.suffix}', skipping: {path}")
        elif path_obj.is_dir():
            # Directory - find all supported files
            dir_files: List[Path] = []
            for ext in supported_extensions:
                dir_files.extend(path_obj.glob(f"*{ext}"))
            if not dir_files:
                log_warning(f"No result files found in directory: {path}")
            else:
                log_info(f"Found {len(dir_files)} result files in {path}")
                result_files.extend(sorted(dir_files))
        else:
            log_error(f"Invalid path: {path}")
            ctx.exit(1)

    if not result_files:
        log_error("No valid result files found.")
        ctx.exit(1)

    # Process files
    if len(result_files) == 1:
        # Single file - original behavior
        result_file = result_files[0]
        try:
            log_info(f"Loading results from: {result_file}")
            result = DataManager.load(str(result_file))
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
        network_type = metadata.get("network", "").upper()
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
                    output_with_format,
                    states_to_plot,
                    dpi,
                )
            else:
                if with_error and not has_error:
                    log_warning("Error data not available. Showing plot without error bars.")
                Visualizer.plot_result(
                    susceptible,
                    infected,
                    recovered,
                    time_steps,
                    title,
                    output_with_format,
                    states_to_plot,
                    dpi,
                )

            if output_with_format:
                log_success(f"Plot saved to: {output_with_format}")

            # Export to additional format if specified
            if export:
                ext_map = {"csv": ".csv", "excel": ".xlsx", "md": ".md", "html": ".html"}
                ext = ext_map.get(export, f".{export}")
                export_path = str(result_file).replace(".json", ext)
                exported_file = DataManager.save(result, export_path)
                log_success(f"Results exported in {export.upper()} format: {exported_file}")

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
                    result = DataManager.load(str(result_file))

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

                    # Determine output file name with correct format extension
                    if output_with_format:
                        base_output = Path(output_with_format)
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
                            dpi,
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
                            dpi,
                        )

                    if output_file:
                        log_success(f"Plot saved to: {output_file}")

                except Exception as e:
                    log_error(f"Error processing {result_file.name}: {e}")
                    continue

        else:
            # Plot all files in a single comparison chart
            results_data = []
            file_labels = []

            for result_file in result_files:
                try:
                    result = DataManager.load(str(result_file))

                    # Check required data
                    if not all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                        log_warning(f"Incomplete data in {result_file.name}, skipping...")
                        continue

                    # Check internal consistency - all arrays must have same length
                    time_len = len(result["time"])
                    s_len = len(result["S_val"])
                    i_len = len(result["I_val"])
                    r_len = len(result["R_val"])

                    if not (time_len == s_len == i_len == r_len):
                        log_warning(
                            f"Inconsistent array lengths in {result_file.name} "
                            f"(time={time_len}, S={s_len}, I={i_len}, R={r_len}), skipping..."
                        )
                        continue

                    results_data.append(result)
                    file_labels.append(result_file.stem)

                except Exception as e:
                    log_error(f"Error loading {result_file.name}: {e}")
                    continue

            if not results_data:
                log_error("No valid file found to plot.")
                ctx.exit(1)

            # Filter out files with incompatible step counts
            # Use the step count of the first file as reference
            reference_steps = len(results_data[0]["time"])
            compatible_results = []
            compatible_labels = []
            compatible_indices = []  # Track which original indices are kept
            skipped_count = 0

            for idx, (result, label) in enumerate(zip(results_data, file_labels)):
                steps = len(result["time"])
                if steps == reference_steps:
                    compatible_results.append(result)
                    compatible_labels.append(label)
                    compatible_indices.append(idx)
                else:
                    log_warning(
                        f"Skipping '{label}': incompatible step count "
                        f"({steps} vs {reference_steps} expected)"
                    )
                    skipped_count += 1

            if skipped_count > 0:
                log_info(
                    f"Skipped {skipped_count} file(s) with incompatible step counts. "
                    f"Use --separate to plot all files individually."
                )

            if not compatible_results:
                log_error("No compatible files found to plot.")
                ctx.exit(1)

            results_data = compatible_results
            file_labels = compatible_labels

            # Use custom labels if provided, otherwise use file names
            if labels:
                # Filter custom labels to only include compatible files
                labels_list = list(labels)
                final_labels = []
                for idx in compatible_indices:
                    if idx < len(labels_list):
                        final_labels.append(labels_list[idx])
                    else:
                        # Fall back to file label if custom label wasn't provided for this index
                        final_labels.append(file_labels[compatible_indices.index(idx)])
            else:
                final_labels = file_labels

            try:
                log_info(
                    f"Generating comparison visualization for {len(results_data)} scenarios..."
                )

                # Generate title from paths
                if len(paths) == 1:
                    title = f"SPKMC Simulation Comparison - {Path(paths[0]).name}"
                else:
                    title = "SPKMC Simulation Comparison"

                Visualizer.compare_results(
                    results_data, final_labels, title, output_with_format, states_to_plot, dpi
                )

                if output_with_format:
                    log_success(f"Comparison plot saved to: {output_with_format}")

            except Exception as e:
                log_error(f"Error generating comparison visualization: {e}")


@cli.command(help="Show information about saved simulations")
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
@click.pass_context
def info(
    ctx: click.Context,
    result_file: Optional[str],
    list_files: bool,
    export: Optional[str],
    output: Optional[str],
) -> None:
    """Show information about saved simulations."""
    # Lazy imports for heavy modules
    # Get CLI context for global options
    from spkmc.cli.utils import get_cli_context

    cli_context = get_cli_context()
    _ = cli_context.verbose  # Available for future verbose output

    if list_files:
        # List files from both legacy and new paths
        files = DataManager.list_all_results("data/runs")
        files.extend(DataManager.list_all_results("data/spkmc"))  # Legacy path
        if not files:
            log_info("No results files found.")
            return

        console.print(format_title("Available Results Files"))

        # Create a formatted table
        data = []
        for i, file in enumerate(files, 1):
            # Extract metadata from path
            metadata = DataManager.get_metadata_from_path(file)
            network = metadata.get("network", "").upper() if metadata else ""
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
        ctx.exit(1)
        return  # Unreachable but helps mypy understand control flow

    # Load results
    try:
        log_info(f"Loading results from: {result_file}")
        result = DataManager.load(result_file)
    except Exception as e:
        log_error(f"Error loading results file: {e}")
        ctx.exit(1)

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
        path_metadata = DataManager.get_metadata_from_path(result_file)
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

    # Export in additional format if specified
    if export:
        export_path: str
        if output:
            export_path = output
        else:
            ext_map = {"csv": ".csv", "excel": ".xlsx", "md": ".md", "html": ".html"}
            ext = ext_map.get(export, f".{export}")
            export_path = result_file.replace(".json", ext)

        exported_file = DataManager.save(result, export_path)
        log_success(f"Information exported in {export.upper()} format: {exported_file}")


@cli.command()
@click.argument("paths", nargs=-1, type=str)
@click.option(
    "--all",
    "-a",
    "analyze_all",
    is_flag=True,
    default=False,
    help="Analyze all experiments in the experiments directory",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="gpt-4o-mini",
    show_default=True,
    help="OpenAI model to use for analysis",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Regenerate analysis even if it already exists",
)
@click.option(
    "--output",
    "-o",
    type=str,
    help="Custom output path for analysis file (only for single path)",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    paths: tuple,
    analyze_all: bool,
    model: str,
    force: bool,
    output: Optional[str],
) -> None:
    """
    Generate AI-powered analysis for simulation results.

    Requires OPENAI_API_KEY environment variable to be set. Uses OpenAI's API
    to generate academic-style analysis of epidemic simulation results.

    You can also use the global --analyze flag with 'run' or 'experiments'
    commands to automatically generate analysis after execution.

    \b
    Examples:
        spkmc analyze result.json                    # Single file
        spkmc analyze result1.json result2.json     # Multiple files
        spkmc analyze data/my_experiment/           # Directory
        spkmc analyze --all                         # All experiments
        spkmc analyze data/results/ -o analysis.md  # Custom output
        spkmc analyze data/exp/ --model gpt-4o      # Different model

    \b
    Alternative (automatic analysis after execution):
        spkmc --analyze run -n er -d gamma -o result.json
        spkmc --analyze experiments
    """
    from spkmc.analysis.ai_analyzer import AIAnalyzer

    # Check if AI analysis is available
    if not AIAnalyzer.is_available():
        log_error("OPENAI_API_KEY environment variable not set.")
        log_info("Set it with: export OPENAI_API_KEY='your-api-key'")
        ctx.exit(1)

    analyzer = AIAnalyzer(model=model)
    log_info(f"Using model: {model}")

    # Determine which mode we're in
    if analyze_all:
        # Analyze all experiments
        _analyze_all_experiments(ctx, analyzer, force)
    elif paths:
        # Warn if --output is used with multiple paths
        if output and len(paths) > 1:
            log_warning("--output is ignored when analyzing multiple paths")
            output = None

        # Analyze each path
        for i, path in enumerate(paths):
            if len(paths) > 1:
                log_info(f"Analyzing [{i+1}/{len(paths)}]: {path}")
            _analyze_path(ctx, analyzer, path, force, output)
    else:
        # No arguments - show help
        log_error("Specify one or more paths or use --all to analyze all experiments.")
        log_info("Run 'spkmc analyze --help' for usage information.")
        ctx.exit(1)


def _analyze_path(
    ctx: click.Context,
    analyzer: "AIAnalyzer",
    path: str,
    force: bool,
    output: Optional[str],
) -> None:
    """Analyze a single file or directory of results."""

    path_obj = Path(path)

    if not path_obj.exists():
        log_error(f"Path not found: {path}")
        ctx.exit(1)

    # Determine output path
    if path_obj.is_file():
        results_dir = path_obj.parent
        analysis_path = Path(output) if output else results_dir / "analysis.md"
        result_files = [path_obj]
    else:
        results_dir = path_obj
        analysis_path = Path(output) if output else results_dir / "analysis.md"
        # Find all result files
        supported_extensions = {".json", ".csv", ".xlsx", ".xls"}
        result_files = []
        for ext in supported_extensions:
            result_files.extend(results_dir.glob(f"*{ext}"))
        result_files = sorted(result_files)

    if not result_files:
        log_error(f"No result files found in: {path}")
        ctx.exit(1)

    # Check if analysis already exists
    if analysis_path.exists() and not force:
        log_warning(f"Analysis already exists: {analysis_path}")
        log_info("Use --force to regenerate.")
        return

    # Load results
    log_info(f"Loading {len(result_files)} result file(s)...")
    results = []
    for f in result_files:
        try:
            result = DataManager.load(str(f))
            results.append(result)
        except Exception as e:
            log_warning(f"Failed to load {f.name}: {e}")

    if not results:
        log_error("No valid results to analyze.")
        ctx.exit(1)

    # Extract experiment name from path
    experiment_name = results_dir.name if results_dir.is_dir() else path_obj.stem

    # Try to get description from data.json if it exists
    data_json = results_dir / "data.json" if results_dir.is_dir() else None
    experiment_description = "Analysis of simulation results"
    if data_json is not None and data_json.exists():
        try:
            import json

            json_path: Path = data_json
            with open(json_path) as json_file:
                data_content = json.load(json_file)
                experiment_description = data_content.get("description", experiment_description)
        except Exception:
            pass

    # Generate analysis
    log_info("Generating AI analysis...")
    try:
        # Remove existing file if forcing
        if analysis_path.exists() and force:
            analysis_path.unlink()

        result_path = analyzer.analyze_experiment(
            experiment_name, experiment_description, results, results_dir
        )

        if result_path:
            log_success(f"Analysis generated: {result_path}")
        else:
            log_warning("Analysis generation returned no result.")
    except Exception as e:
        log_error(f"Failed to generate analysis: {e}")
        ctx.exit(1)


def _analyze_all_experiments(
    ctx: click.Context,
    analyzer: "AIAnalyzer",
    force: bool,
) -> None:
    """Analyze all experiments."""
    from spkmc.io.experiments import ExperimentManager

    exp_manager = ExperimentManager("experiments")
    experiments = exp_manager.list_experiments()

    if not experiments:
        log_error("No experiments found in 'experiments/'")
        ctx.exit(1)

    # Filter to experiments with results
    experiments_with_results = [e for e in experiments if e.has_results]

    if not experiments_with_results:
        log_error("No experiments with results found.")
        log_info("Run experiments first with: spkmc experiments")
        ctx.exit(1)

    log_info(f"Found {len(experiments_with_results)} experiment(s) with results")

    generated = 0
    skipped = 0
    failed = 0

    for experiment in experiments_with_results:
        results_dir = experiment.results_dir
        analysis_path = results_dir / "analysis.md"

        # Check if analysis already exists
        if analysis_path.exists() and not force:
            log_info(f"Skipping '{experiment.name}' (analysis exists)")
            skipped += 1
            continue

        # Load results
        result_files = sorted(results_dir.glob("*.json"))
        results = []
        for f in result_files:
            try:
                result = DataManager.load(str(f))
                results.append(result)
            except Exception:
                pass

        if not results:
            log_warning(f"No valid results for '{experiment.name}'")
            failed += 1
            continue

        # Generate analysis
        log_info(f"Analyzing '{experiment.name}'...")
        try:
            # Remove existing file if forcing
            if analysis_path.exists() and force:
                analysis_path.unlink()

            result_path = analyzer.analyze_experiment(
                experiment.name,
                experiment.description or "Analysis of simulation results",
                results,
                results_dir,
            )

            if result_path:
                log_success(f"  → {result_path}")
                generated += 1
            else:
                skipped += 1
        except Exception as e:
            log_error(f"  Failed: {e}")
            failed += 1

    # Summary
    console.print()
    console.print(format_title("Analysis Summary"))
    console.print(f"  {format_param('Generated', generated)}")
    console.print(f"  {format_param('Skipped', skipped)}")
    console.print(f"  {format_param('Failed', failed)}")


@cli.command(name="experiments", help="Run or create experiments from the experiments directory")
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
@click.option("--no-plot", is_flag=True, default=False, help="Disable individual plot generation")
@click.option(
    "--export",
    "-x",
    type=click.Choice(["json", "csv", "excel", "md", "html"]),
    default="json",
    show_default=True,
    help="Format for saving results",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode with detailed Numba logs",
)
@click.option("--clear-cache", is_flag=True, default=False, help="Clear Numba cache before running")
@click.option(
    "--analyze",
    is_flag=True,
    default=False,
    help="Run AI analysis after execution (requires OPENAI_API_KEY)",
)
@click.pass_context
def experiment(
    ctx: click.Context,
    scenarios_file: Optional[str],
    run_all: bool,
    override: bool,
    no_plot: bool,
    export: str,
    debug: bool,
    clear_cache: bool,
    analyze: bool,
) -> None:
    """
    Run or create experiments from the experiments directory.

    If no file is specified, show an interactive menu with experiments available in
    the 'experiments/' directory. The menu includes an option to create a new
    experiment interactively.

    The JSON file must contain a list of objects, each representing a scenario with
    parameters for the simulation. Each scenario will run sequentially and results
    will be saved to separate files in the specified directory.
    """
    # Get CLI context for global options
    from spkmc.cli.display import display_execution_summary
    from spkmc.cli.utils import get_cli_context
    from spkmc.utils.gpu_utils import check_gpu_suggestion

    # Check if user has GPU hardware but no GPU packages installed
    check_gpu_suggestion()

    cli_context = get_cli_context()
    verbose = cli_context.verbose
    # Merge global --analyze flag with command-specific --analyze flag
    analyze_enabled = cli_context.analyze or analyze
    if analyze_enabled:
        from spkmc.cli.utils import require_analyze_api_key

        require_analyze_api_key()

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
        exp_manager = ExperimentManager("experiments")
        experiments = exp_manager.list_experiments()

        # Determine which experiments to run
        if run_all:
            if not experiments:
                log_error("No experiments found in 'experiments/'.")
                ctx.exit(1)
            # Run all experiments in order
            experiments_to_run = experiments
            log_info(f"Running all {len(experiments)} experiments in order...")
        else:
            # Check for TTY before showing interactive menu
            import sys

            if not sys.stdin.isatty():
                log_error(
                    "Interactive mode requires a terminal. Use --all flag to run all experiments."
                )
                ctx.exit(1)

            # Show experiment menu (includes option to create new experiment)
            selected = display_experiments_menu(experiments)

            if selected is None:
                log_info("Operation canceled by user.")
                return

            if selected == CREATE_EXPERIMENT_SENTINEL:
                # Create a new experiment interactively
                created_exp = create_experiment_interactive()
                if created_exp is None:
                    log_info("Experiment creation cancelled.")
                    return
                experiments_to_run = [created_exp]
            else:
                if not experiments:
                    log_error("No experiments available to select.")
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

            # Run the experiment using ExecutionEngine
            start_time = time.time()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_debug(f"Starting execution at {now_str}", verbose_only=False)

            result_files = _run_experiment_with_engine(
                experiment=experiment,
                verbose=verbose,
                no_plot=no_plot,
                force_rerun=force_rerun,
                run_analysis=analyze_enabled,
                export_format=export,
            )

            # Experiment summary
            execution_time = time.time() - start_time
            total_scenarios_processed += len(result_files)
            experiments_completed += 1

            if not run_all:
                display_execution_summary(
                    execution_time=execution_time,
                    experiment_name=exp_name,
                    total_scenarios=len(experiment.scenarios),
                    scenarios_processed=len(result_files),
                    results_dir=experiment.results_dir,
                )

            log_success(
                f"'{exp_name}' completed. {len(result_files)} scenario(s) in {execution_time:.1f}s."
            )

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

            # Generate AI collection summary for --all mode (only when --analyze is enabled)
            if analyze_enabled:
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
                                metrics = extract_experiment_metrics(
                                    exp.name, exp.description, results
                                )
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

                    # Generate cross-experiment analysis from individual analysis.md files
                    experiment_analyses: List[Dict[str, str]] = []
                    for exp in experiments_to_run:
                        if exp.has_results:
                            analysis_path = exp.results_dir / "analysis.md"
                            if analysis_path.exists():
                                try:
                                    with open(analysis_path, "r", encoding="utf-8") as f:
                                        analysis_content = f.read()
                                    experiment_analyses.append(
                                        {
                                            "name": exp.name,
                                            "analysis": analysis_content,
                                        }
                                    )
                                except Exception:
                                    continue

                    if experiment_analyses:
                        try:
                            analyzer = AIAnalyzer()
                            cross_analysis_path = Path("cross_experiment_analysis.md")
                            result = analyzer.generate_cross_experiment_analysis(
                                experiment_analyses,
                                cross_analysis_path,
                                force=override,
                            )
                            if result:
                                log_success(f"Cross-experiment analysis saved to: {result}")
                            elif cross_analysis_path.exists():
                                log_info(
                                    "Cross-experiment analysis already exists. "
                                    "Use --override to regenerate."
                                )
                        except Exception as e:
                            log_warning(f"Failed to generate cross-experiment analysis: {e}")

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
        log_error(f"Scenarios file does not exist: {scenarios_file}")
        ctx.exit(1)

    # Load the scenarios file
    try:
        log_info(f"Loading scenarios from: {scenarios_file}")
        with open(scenarios_file, "r") as f:
            data = json.load(f)

        # Support both formats:
        # 1. Direct list of scenarios: [{...}, {...}]
        # 2. Experiment format: {"name": "...", "scenarios": [...], "parameters": {...}}
        experiment_dir = Path(os.path.dirname(os.path.abspath(scenarios_file)))

        if isinstance(data, list):
            # Direct list of complete scenarios (legacy format)
            scenarios = data
            experiment_name = os.path.basename(scenarios_file).replace(".json", "")
            log_info(f"Experiment: {experiment_name}")
            log_success(f"Loaded {len(scenarios)} scenarios for execution.")

            experiment = Experiment(
                name=experiment_name,
                path=experiment_dir,
                scenarios=scenarios,
            )
        elif isinstance(data, dict) and "scenarios" in data:
            # Experiment format with parameters - use ExperimentConfig for proper merging
            from spkmc.io.experiments import ExperimentConfig

            config = ExperimentConfig.from_dict(data)
            experiment = Experiment.from_config(config, path=experiment_dir)

            log_info(f"Experiment: {experiment.name}")
            log_success(f"Loaded {len(experiment.scenarios)} scenarios for execution.")
        else:
            log_error(
                "The file must contain a list of scenarios or an object with a 'scenarios' key."
            )
            ctx.exit(1)

    except Exception as e:
        log_error(f"Error loading scenarios file: {e}")
        ctx.exit(1)

    # Handle --override for file mode
    force_rerun = override  # --override always forces a re-run
    if override and experiment.has_results:
        experiment.clean_results()
        log_info("Previous results removed. Forcing re-run.")

    # Use ExecutionEngine for execution
    result_files = _run_experiment_with_engine(
        experiment=experiment,
        verbose=verbose,
        no_plot=no_plot,
        force_rerun=force_rerun,
        run_analysis=analyze_enabled,
        export_format=export,
    )

    # Summary
    total_execution_time = time.time() - start_time

    display_execution_summary(
        execution_time=total_execution_time,
        experiment_name=experiment.name,
        total_scenarios=len(experiment.scenarios),
        scenarios_processed=len(result_files),
        results_dir=experiment.results_dir,
    )

    log_success(f"Execution completed. {len(result_files)} scenario(s) processed.")


@cli.command(help="Clean results from experiments")
@click.argument("experiment_name", type=str, default=None, required=False)
@click.option("--yes", "-y", is_flag=True, default=False, help="Auto-confirm without prompting")
@click.option(
    "--numba-cache",
    is_flag=True,
    default=False,
    help="Also clear the Numba compilation cache",
)
@click.pass_context
def clean(
    ctx: click.Context,
    experiment_name: Optional[str],
    yes: bool,
    numba_cache: bool,
) -> None:
    """
    Remove results from experiments.

    If EXPERIMENT_NAME is provided, only that experiment's results are removed.
    If no argument is given, prompts to remove all results from all experiments.

    Examples:

        spkmc clean                     # Clean all experiments (with confirmation)

        spkmc clean network_comparison  # Clean only 'network_comparison' results

        spkmc clean -y                  # Clean all without confirmation
    """
    import shutil
    from pathlib import Path

    # Create experiment manager
    exp_manager = ExperimentManager("experiments")

    # ============================================================
    # SINGLE EXPERIMENT MODE: Clean specific experiment
    # ============================================================
    if experiment_name:
        try:
            exp = exp_manager.load_experiment(experiment_name)
        except FileNotFoundError:
            log_error(f"Experiment '{experiment_name}' not found.")
            ctx.exit(1)

        if not exp.has_results:
            log_info(f"Experiment '{experiment_name}' has no results to clean.")
            return

        # Show what will be removed
        console.print(format_title("Results to Remove"))
        console.print(f"  • {exp.name}: {exp.result_count} result(s)")
        console.print()

        # Confirm
        if not yes:
            if not click.confirm(f"Remove all results for '{experiment_name}'?", default=False):
                log_info("Operation canceled.")
                return

        # Clean results
        try:
            exp.clean_results()
            log_success(f"Results for '{exp.name}' removed.")
        except Exception as e:
            log_error(f"Error cleaning '{exp.name}': {e}")
            return

        # Clean Numba cache if requested
        if numba_cache:
            from spkmc.utils.numba_utils import clear_numba_cache

            clear_numba_cache()
            log_success("Numba cache cleared.")

        return

    # ============================================================
    # ALL EXPERIMENTS MODE: Clean all experiments
    # ============================================================
    experiments = exp_manager.list_experiments()

    # Check for root results/ directory too
    root_results = Path("results")
    has_root_results = root_results.exists() and any(root_results.glob("*.json"))

    if not experiments and not has_root_results:
        log_error("No experiments or results found.")
        ctx.exit(1)

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
