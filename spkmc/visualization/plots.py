"""
Result visualization for the SPKMC algorithm.

This module contains functions to visualize SPKMC simulation results,
including time-evolution plots of SIR states and comparisons between simulations.

Uses seaborn and matplotlib with publication-quality styling suitable for
academic papers and presentations.
"""

import contextlib
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Publication-quality color palettes (colorblind-friendly)
# Based on Paul Tol's colorblind-safe palette
COLORBLIND_PALETTE = [
    "#4477AA",  # blue
    "#EE6677",  # red/pink
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

# SIR-specific colors (semantically meaningful and colorblind-friendly)
SIR_COLORS = {
    "S": "#4477AA",  # blue for susceptible
    "I": "#EE6677",  # red/pink for infected
    "R": "#228833",  # green for recovered
}

# Line styles for distinguishing curves
LINE_STYLES = {
    "S": "-",  # solid for susceptible
    "I": "-",  # solid for infected
    "R": "--",  # dashed for recovered
}

# Default DPI for saved figures (single source of truth)
DEFAULT_PLOT_DPI = 300


def _setup_publication_style() -> None:
    """Configure matplotlib and seaborn for publication-quality figures."""
    # Use seaborn's whitegrid style as base
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    sns.set_palette(COLORBLIND_PALETTE)

    # Additional matplotlib customizations
    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.dpi": 150,
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
            "font.size": 11,
            # Axes
            "axes.linewidth": 1.2,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            # Grid
            "grid.alpha": 0.4,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            # Ticks
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            # Lines
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            # Saving
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def _get_scenario_colors(n_scenarios: int) -> List[str]:
    """Get a list of colorblind-friendly colors for scenarios."""
    if n_scenarios <= len(COLORBLIND_PALETTE):
        return COLORBLIND_PALETTE[:n_scenarios]

    # If we need more colors, cycle through the palette
    colors = []
    for i in range(n_scenarios):
        colors.append(COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)])
    return colors


@contextlib.contextmanager
def _suppress_macos_warning() -> Generator[None, None, None]:
    """
    Context manager to suppress macOS ApplePersistenceIgnoreState warning.

    This warning is printed by macOS Cocoa layer, not Python, so we redirect
    the file descriptor directly rather than using Python's sys.stderr.
    """
    if sys.platform != "darwin":
        yield
        return

    # On macOS, redirect stderr at the file descriptor level
    # to suppress Cocoa framework warnings
    stderr_fd = sys.stderr.fileno()
    try:
        # Save the original stderr
        saved_stderr = os.dup(stderr_fd)
        # Open /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Replace stderr with /dev/null
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)


def _show_plot() -> None:
    """Show plot with suppressed macOS warnings."""
    with _suppress_macos_warning():
        plt.show()


def _create_figure(
    figsize: Tuple[float, float] = (8, 5), **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a publication-quality figure with proper styling.

    Args:
        figsize: Figure size in inches (width, height)
        **kwargs: Additional arguments passed to plt.subplots

    Returns:
        Tuple of (figure, axes)
    """
    _setup_publication_style()

    with _suppress_macos_warning():
        fig, ax = plt.subplots(figsize=figsize, **kwargs)

    return fig, ax


if TYPE_CHECKING:
    from spkmc.io.experiments import PlotConfig


class Visualizer:
    """Class for visualizing simulation results with publication-quality plots."""

    @staticmethod
    def plot_result_with_error(
        S: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        S_err: np.ndarray,
        I_err: np.ndarray,
        R_err: np.ndarray,
        time: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        states_to_plot: Optional[set] = None,
        dpi: int = DEFAULT_PLOT_DPI,
    ) -> None:
        """
        Plot results with shaded error bands (publication-quality).

        Args:
            S: Proportion of susceptible
            I: Proportion of infected
            R: Proportion of recovered
            S_err: Standard error for susceptible
            I_err: Standard error for infected
            R_err: Standard error for recovered
            time: Time steps
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
            dpi: Resolution in dots per inch for saved figures (default: 300)
        """
        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        fig, ax = _create_figure(figsize=(8, 5))

        # Plot with shaded error bands (more elegant than error bars)
        if "S" in states_to_plot:
            ax.plot(
                time,
                S,
                color=SIR_COLORS["S"],
                linestyle=LINE_STYLES["S"],
                linewidth=2.0,
                label="Susceptible",
            )
            ax.fill_between(
                time,
                S - S_err,
                S + S_err,
                color=SIR_COLORS["S"],
                alpha=0.2,
            )

        if "I" in states_to_plot:
            ax.plot(
                time,
                I,
                color=SIR_COLORS["I"],
                linestyle=LINE_STYLES["I"],
                linewidth=2.0,
                label="Infected",
            )
            ax.fill_between(
                time,
                I - I_err,
                I + I_err,
                color=SIR_COLORS["I"],
                alpha=0.2,
            )

        if "R" in states_to_plot:
            ax.plot(
                time,
                R,
                color=SIR_COLORS["R"],
                linestyle=LINE_STYLES["R"],
                linewidth=2.0,
                label="Recovered",
            )
            ax.fill_between(
                time,
                R - R_err,
                R + R_err,
                color=SIR_COLORS["R"],
                alpha=0.2,
            )

        ax.set_xlabel("Time", fontweight="medium")
        ax.set_ylabel("Proportion of Population", fontweight="medium")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(time[0], time[-1])

        if title:
            ax.set_title(title, pad=15)
        else:
            ax.set_title("SIR Dynamics with Confidence Bands", pad=15)

        ax.legend(loc="best", framealpha=0.9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            _show_plot()

    @staticmethod
    def plot_result(
        S: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        time: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        states_to_plot: Optional[set] = None,
        dpi: int = DEFAULT_PLOT_DPI,
    ) -> None:
        """
        Plot results without error bands (publication-quality).

        Args:
            S: Proportion of susceptible
            I: Proportion of infected
            R: Proportion of recovered
            time: Time steps
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
            dpi: Resolution in dots per inch for saved figures (default: 300)
        """
        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        fig, ax = _create_figure(figsize=(8, 5))

        if "S" in states_to_plot:
            ax.plot(
                time,
                S,
                color=SIR_COLORS["S"],
                linestyle=LINE_STYLES["S"],
                linewidth=2.0,
                label="Susceptible",
            )

        if "I" in states_to_plot:
            ax.plot(
                time,
                I,
                color=SIR_COLORS["I"],
                linestyle=LINE_STYLES["I"],
                linewidth=2.0,
                label="Infected",
            )

        if "R" in states_to_plot:
            ax.plot(
                time,
                R,
                color=SIR_COLORS["R"],
                linestyle=LINE_STYLES["R"],
                linewidth=2.0,
                label="Recovered",
            )

        ax.set_xlabel("Time", fontweight="medium")
        ax.set_ylabel("Proportion of Population", fontweight="medium")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(time[0], time[-1])

        if title:
            ax.set_title(title, pad=15)
        else:
            ax.set_title("SIR Model Dynamics", pad=15)

        ax.legend(loc="best", framealpha=0.9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            _show_plot()

    @staticmethod
    def compare_results(
        results: List[Dict[str, Any]],
        labels: List[str],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        states_to_plot: Optional[set] = None,
        dpi: int = DEFAULT_PLOT_DPI,
    ) -> None:
        """
        Compare results from multiple simulations (publication-quality).

        Uses distinct colors for each scenario and different line styles
        for each SIR state. Colors are colorblind-friendly.

        Args:
            results: List of dictionaries with results
            labels: List of labels for each result
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
            dpi: Resolution in dots per inch for saved figures (default: 300)
        """
        if not results:
            raise ValueError("The results list is empty")

        if len(results) != len(labels):
            raise ValueError("The number of results and labels must match")

        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        # Adjust figure size based on number of scenarios (need room for legend)
        fig_width = 9 if len(results) <= 4 else 10
        fig, ax = _create_figure(figsize=(fig_width, 5.5))

        # Get colorblind-friendly colors for scenarios
        scenario_colors = _get_scenario_colors(len(results))

        # Line styles for states (to distinguish S, I, R within same scenario)
        state_styles = {"S": ":", "I": "-", "R": "--"}
        state_widths = {"S": 1.8, "I": 2.2, "R": 1.8}

        for idx, (result, label) in enumerate(zip(results, labels)):
            if not all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                raise ValueError(f"Result {idx} does not contain all required data")

            s_vals = np.array(result["S_val"])
            i_vals = np.array(result["I_val"])
            r_vals = np.array(result["R_val"])
            time = np.array(result["time"])

            color = scenario_colors[idx]

            if "S" in states_to_plot:
                ax.plot(
                    time,
                    s_vals,
                    color=color,
                    linestyle=state_styles["S"],
                    linewidth=state_widths["S"],
                    alpha=0.85,
                    label=f"S — {label}",
                )
            if "I" in states_to_plot:
                ax.plot(
                    time,
                    i_vals,
                    color=color,
                    linestyle=state_styles["I"],
                    linewidth=state_widths["I"],
                    alpha=0.95,
                    label=f"I — {label}",
                )
            if "R" in states_to_plot:
                ax.plot(
                    time,
                    r_vals,
                    color=color,
                    linestyle=state_styles["R"],
                    linewidth=state_widths["R"],
                    alpha=0.85,
                    label=f"R — {label}",
                )

        ax.set_xlabel("Time", fontweight="medium")
        ax.set_ylabel("Proportion of Population", fontweight="medium")
        ax.set_ylim(0, 1.05)

        if title:
            ax.set_title(title, pad=15)
        else:
            ax.set_title("Epidemic Dynamics Comparison", pad=15)

        # Position legend: outside for many scenarios, inside for few
        if len(results) > 3:
            ax.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=9,
                framealpha=0.9,
                title="State — Scenario",
                title_fontsize=10,
            )
        else:
            ax.legend(
                loc="best",
                fontsize=9,
                framealpha=0.9,
                title="State — Scenario",
                title_fontsize=10,
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            _show_plot()

    @staticmethod
    def compare_results_with_config(
        results: List[Dict[str, Any]],
        labels: List[str],
        plot_config: "PlotConfig",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare results from multiple simulations with custom configuration.

        Args:
            results: List of dictionaries with results
            labels: List of labels for each result
            plot_config: Custom plot configuration
            save_path: Path to save the plot (optional)
        """
        if not results:
            raise ValueError("The results list is empty")

        if len(results) != len(labels):
            raise ValueError("The number of results and labels must match")

        # Use config values
        states_to_plot = (
            set(plot_config.states_to_plot) if plot_config.states_to_plot else {"S", "I", "R"}
        )

        figsize_tuple: Tuple[float, float] = (plot_config.figsize[0], plot_config.figsize[1])
        fig, ax = _create_figure(figsize=figsize_tuple)

        # Get colorblind-friendly colors for scenarios
        scenario_colors = _get_scenario_colors(len(results))

        # Line styles for states
        state_styles = {"S": ":", "I": "-", "R": "--"}
        state_widths = {"S": 1.8, "I": 2.2, "R": 1.8}

        for idx, (result, label) in enumerate(zip(results, labels)):
            if not all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                raise ValueError(f"Result {idx} does not contain all required data")

            s_vals = np.array(result["S_val"])
            i_vals = np.array(result["I_val"])
            r_vals = np.array(result["R_val"])
            time = np.array(result["time"])

            color = scenario_colors[idx]

            if "S" in states_to_plot:
                ax.plot(
                    time,
                    s_vals,
                    color=color,
                    linestyle=state_styles["S"],
                    linewidth=state_widths["S"],
                    alpha=0.85,
                    label=f"S — {label}",
                )
            if "I" in states_to_plot:
                ax.plot(
                    time,
                    i_vals,
                    color=color,
                    linestyle=state_styles["I"],
                    linewidth=state_widths["I"],
                    alpha=0.95,
                    label=f"I — {label}",
                )
            if "R" in states_to_plot:
                ax.plot(
                    time,
                    r_vals,
                    color=color,
                    linestyle=state_styles["R"],
                    linewidth=state_widths["R"],
                    alpha=0.85,
                    label=f"R — {label}",
                )

        ax.set_xlabel(plot_config.xlabel, fontweight="medium")
        ax.set_ylabel(plot_config.ylabel, fontweight="medium")
        ax.set_ylim(0, 1.05)

        if plot_config.title:
            ax.set_title(plot_config.title, pad=15)
        else:
            ax.set_title("Epidemic Dynamics Comparison", pad=15)

        # Position legend based on number of scenarios
        if len(results) > 4:
            ax.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=9,
                framealpha=0.9,
                title="State — Scenario",
                title_fontsize=10,
            )
        else:
            ax.legend(
                loc=plot_config.legend_position,
                fontsize=9,
                framealpha=0.9,
            )

        if plot_config.grid:
            ax.grid(True, alpha=plot_config.grid_alpha, linestyle="-", linewidth=0.8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            _show_plot()

    @staticmethod
    def plot_network(
        G: nx.DiGraph, title: Optional[str] = None, save_path: Optional[str] = None
    ) -> None:
        """
        Plot the network used in the simulation (publication-quality).

        Args:
            G: Network graph
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
        """
        fig, ax = _create_figure(figsize=(8, 7))

        # Limit the number of nodes for visualization
        if G.number_of_nodes() > 100:
            import warnings

            warnings.warn(
                f"The network has {G.number_of_nodes()} nodes. "
                "Limiting visualization to 100 nodes.",
                stacklevel=2,
            )
            G = nx.DiGraph(G.subgraph(list(G.nodes())[:100]))

        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

        # Draw edges first (behind nodes)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#CCCCCC",
            arrows=True,
            arrowsize=8,
            alpha=0.6,
            width=0.8,
            connectionstyle="arc3,rad=0.1",
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=80,
            node_color=COLORBLIND_PALETTE[0],
            edgecolors="white",
            linewidths=1.0,
            alpha=0.9,
        )

        if title:
            ax.set_title(title, pad=15)
        else:
            ax.set_title(
                f"Network Structure ({G.number_of_nodes()} nodes, " f"{G.number_of_edges()} edges)",
                pad=15,
            )

        ax.axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            _show_plot()

    @staticmethod
    def create_summary_plot(result_path: str, output_dir: Optional[str] = None) -> str:
        """
        Create a publication-quality summary plot from a results file.

        Args:
            result_path: Path to the results file
            output_dir: Directory to save the plot (optional)

        Returns:
            Path to the generated plot
        """
        import json

        # Load results
        with open(result_path, "r") as f:
            result = json.load(f)

        # Extract data
        s_vals = np.array(result.get("S_val", []))
        i_vals = np.array(result.get("I_val", []))
        r_vals = np.array(result.get("R_val", []))
        time = np.array(result.get("time", []))

        # Check whether error data is available
        has_error = "S_err" in result and "I_err" in result and "R_err" in result

        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(result_path).replace(".json", ".png")
            save_path = os.path.join(output_dir, base_name)
        else:
            save_path = result_path.replace(".json", ".png")

        # Extract metadata for the title
        metadata = result.get("metadata", {})
        network_type = metadata.get("network", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")

        title = f"SIR Dynamics — {network_type} Network, {dist_type} Distribution (N={N})"

        # Plot results
        if has_error:
            s_err = np.array(result.get("S_err", []))
            i_err = np.array(result.get("I_err", []))
            r_err = np.array(result.get("R_err", []))
            Visualizer.plot_result_with_error(
                s_vals, i_vals, r_vals, s_err, i_err, r_err, time, title, save_path
            )
        else:
            Visualizer.plot_result(s_vals, i_vals, r_vals, time, title, save_path)

        return save_path
