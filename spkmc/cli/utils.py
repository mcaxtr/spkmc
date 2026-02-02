"""
Shared utilities for the SPKMC CLI.

This module contains shared functions and dataclasses used across CLI commands,
including context management and consolidated plotting logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import click

if TYPE_CHECKING:
    from spkmc.io.experiments import PlotConfig


@dataclass
class CLIContext:
    """Holds global CLI options that are inherited by subcommands."""

    verbose: bool = False
    analyze: bool = False

    def configure_environment(self) -> None:
        """Configure environment variables based on CLI options."""
        os.environ["SPKMC_VERBOSE"] = "1" if self.verbose else "0"


def get_cli_context() -> CLIContext:
    """
    Extract CLI context from the current Click context.

    Returns:
        CLIContext object with global options

    Note:
        This function should be called from within a Click command context.
        It traverses the context hierarchy to find the root CLI group's parameters.
    """
    ctx = click.get_current_context()

    # Traverse to the root context to get global options
    root_ctx = ctx
    while root_ctx.parent is not None:
        root_ctx = root_ctx.parent

    # Extract global options from root context
    params = root_ctx.params or {}
    return CLIContext(
        verbose=params.get("verbose", False),
        analyze=params.get("analyze", False),
    )


def check_analyze_available() -> bool:
    """
    Check if AI analysis is available (OPENAI_API_KEY is set).

    Returns:
        True if OPENAI_API_KEY environment variable is set
    """
    return bool(os.environ.get("OPENAI_API_KEY"))


def require_analyze_api_key() -> None:
    """
    Raise an error if --analyze was requested but OPENAI_API_KEY is not set.

    Raises:
        click.UsageError: If OPENAI_API_KEY is not configured
    """
    if not check_analyze_available():
        raise click.UsageError(
            "The --analyze flag requires the OPENAI_API_KEY environment variable to be set.\n"
            "Set it with: export OPENAI_API_KEY=sk-..."
        )


def plot_results(
    results: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    plot_config: Optional[PlotConfig] = None,
    show: bool = True,
    with_error: bool = False,
    states: Optional[Set[str]] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Unified plotting function used by run, experiment, plot, and compare commands.

    This function consolidates plotting logic that was previously duplicated
    across multiple CLI commands.

    Args:
        results: List of result dictionaries containing S_val, I_val, R_val, time
        labels: Optional labels for each result (used in comparison plots)
        output_path: Path to save the plot (if None, displays interactively)
        plot_config: Optional PlotConfig object for customizing the plot
        show: Whether to display the plot (if False, only saves to file)
        with_error: Whether to show error bars (if available in results)
        states: Set of states to plot (e.g., {"S", "I", "R"}). If None, plots all.
        title: Optional title for the plot

    Returns:
        Path to saved plot if output_path provided, None otherwise
    """
    from spkmc.visualization.plots import Visualizer

    # Default to all states if not specified
    if states is None:
        states = {"S", "I", "R"}

    # Single result case
    if len(results) == 1:
        result = results[0]
        import numpy as np

        susceptible = np.array(result.get("S_val", []))
        infected = np.array(result.get("I_val", []))
        recovered = np.array(result.get("R_val", []))
        time_steps = np.array(result.get("time", []))

        # Check for error data
        has_error = "S_err" in result and "I_err" in result and "R_err" in result

        # Build title from metadata if not provided
        if title is None:
            metadata = result.get("metadata", {})
            network_type = metadata.get("network", "").upper()
            dist_type = metadata.get("distribution", "").capitalize()
            n_nodes = metadata.get("N", "")
            title = f"SPKMC Simulation - Network {network_type}, Distribution {dist_type}"
            if n_nodes:
                title += f", N={n_nodes}"

        # Determine output path for saving (None means display)
        save_path = output_path if output_path and not show else output_path

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
                save_path,
                states,
            )
        else:
            Visualizer.plot_result(
                susceptible, infected, recovered, time_steps, title, save_path, states
            )

        return output_path

    # Multiple results case - comparison plot
    else:
        # Use provided labels or generate default ones
        if labels is None:
            labels = [f"Result {i+1}" for i in range(len(results))]

        # Build title if not provided
        if title is None:
            title = "SPKMC Simulation Comparison"
            if len(labels) <= 3:
                title += f" ({', '.join(labels)})"

        # Use plot config if provided
        if plot_config is not None:
            Visualizer.compare_results_with_config(results, labels, plot_config, output_path)
        else:
            Visualizer.compare_results(results, labels, title, output_path, states)

        return output_path
