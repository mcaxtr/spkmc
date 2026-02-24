"""
Result visualization for the SPKMC algorithm.

This module contains functions to visualize SPKMC simulation results,
including time-evolution plots of SIR states and comparisons between simulations.

Uses Plotly for interactive visualizations that work in both CLI (opens browser)
and web interface (embedded in Streamlit).
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Import from web plotting module to reuse code
from spkmc.web.plotting import (
    COLOR_I,
    COLOR_R,
    COLOR_S,
    STATE_COLORS,
    create_comparison_figure,
    create_sir_figure,
)

# Default DPI for saved figures (single source of truth)
DEFAULT_PLOT_DPI = 300

# Configure Plotly defaults for publication quality
pio.templates.default = "plotly_white"


def _save_or_show(
    fig: go.Figure,
    save_path: Optional[str] = None,
    format: str = "png",
    dpi: int = DEFAULT_PLOT_DPI,
    width: int = 800,
    height: int = 500,
) -> None:
    """
    Save figure to file or open in browser.

    Args:
        fig: Plotly figure
        save_path: Path to save the figure (if None, opens in browser)
        format: Output format ('png', 'jpg', 'svg', 'pdf', 'html')
        dpi: Resolution for raster formats
        width: Width in pixels
        height: Height in pixels
    """
    if save_path:
        # Determine format from extension if not specified
        if save_path.endswith(".html"):
            format = "html"
        elif save_path.endswith(".svg"):
            format = "svg"
        elif save_path.endswith(".pdf"):
            format = "pdf"
        elif save_path.endswith(".jpg") or save_path.endswith(".jpeg"):
            format = "jpg"
        else:
            format = "png"

        if format == "html":
            # Save as standalone HTML
            fig.write_html(save_path, include_plotlyjs="cdn")
        else:
            # Save as static image (requires kaleido)
            try:
                scale = dpi / 96  # Convert DPI to scale factor (96 is default)
                fig.write_image(
                    save_path,
                    format=format,
                    width=width,
                    height=height,
                    scale=scale,
                )
            except (ValueError, ImportError) as e:
                raise RuntimeError(
                    f"Failed to save plot as {format}: {e}. "
                    "Install kaleido for static image export: pip install kaleido"
                ) from e
    else:
        # Open in browser
        fig.show()


if TYPE_CHECKING:
    from spkmc.io.experiments import PlotConfig


class Visualizer:
    """Class for visualizing simulation results with interactive Plotly plots."""

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
        states_to_plot: Optional[Set[str]] = None,
        dpi: int = DEFAULT_PLOT_DPI,
    ) -> None:
        """
        Plot results with shaded error bands (interactive Plotly).

        Args:
            S: Proportion of susceptible
            I: Proportion of infected
            R: Proportion of recovered
            S_err: Standard error for susceptible
            I_err: Standard error for infected
            R_err: Standard error for recovered
            time: Time steps
            title: Plot title (optional)
            save_path: Path to save the plot (optional, opens in browser if None)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
            dpi: Resolution in dots per inch for saved figures (default: 300)
        """
        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        # Convert to list for Plotly
        states_list = list(states_to_plot)

        # Build result dict
        result_dict = {
            "time": time.tolist() if isinstance(time, np.ndarray) else time,
            "S_val": S.tolist() if isinstance(S, np.ndarray) else S,
            "I_val": I.tolist() if isinstance(I, np.ndarray) else I,
            "R_val": R.tolist() if isinstance(R, np.ndarray) else R,
            "S_err": S_err.tolist() if isinstance(S_err, np.ndarray) else S_err,
            "I_err": I_err.tolist() if isinstance(I_err, np.ndarray) else I_err,
            "R_err": R_err.tolist() if isinstance(R_err, np.ndarray) else R_err,
        }

        # Create figure using web plotting module
        fig = create_sir_figure(
            result_dict,
            title=title or "SIR Dynamics with Confidence Bands",
            states=states_list,
            show_error_bands=True,
            height=500,
        )

        _save_or_show(fig, save_path, dpi=dpi)

    @staticmethod
    def plot_result(
        S: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        time: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        states_to_plot: Optional[Set[str]] = None,
        dpi: int = DEFAULT_PLOT_DPI,
    ) -> None:
        """
        Plot results without error bands (interactive Plotly).

        Args:
            S: Proportion of susceptible
            I: Proportion of infected
            R: Proportion of recovered
            time: Time steps
            title: Plot title (optional)
            save_path: Path to save the plot (optional, opens in browser if None)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
            dpi: Resolution in dots per inch for saved figures (default: 300)
        """
        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        # Convert to list for Plotly
        states_list = list(states_to_plot)

        # Build result dict
        result_dict = {
            "time": time.tolist() if isinstance(time, np.ndarray) else time,
            "S_val": S.tolist() if isinstance(S, np.ndarray) else S,
            "I_val": I.tolist() if isinstance(I, np.ndarray) else I,
            "R_val": R.tolist() if isinstance(R, np.ndarray) else R,
        }

        # Create figure using web plotting module
        fig = create_sir_figure(
            result_dict,
            title=title or "SIR Model Dynamics",
            states=states_list,
            show_error_bands=False,
            height=500,
        )

        _save_or_show(fig, save_path, dpi=dpi)

    @staticmethod
    def compare_results(
        results: List[Dict[str, Any]],
        labels: List[str],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        states_to_plot: Optional[Set[str]] = None,
        dpi: int = DEFAULT_PLOT_DPI,
    ) -> None:
        """
        Compare results from multiple simulations (interactive Plotly).

        Args:
            results: List of dictionaries with results
            labels: List of labels for each result
            title: Plot title (optional)
            save_path: Path to save the plot (optional, opens in browser if None)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
            dpi: Resolution in dots per inch for saved figures (default: 300)
        """
        if not results:
            raise ValueError("The results list is empty")

        if len(results) != len(labels):
            raise ValueError("The number of results and labels must match")

        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        # Convert to list for Plotly
        states_list = list(states_to_plot)

        # Create figure using web plotting module
        fig = create_comparison_figure(
            results,
            labels,
            title=title or "Epidemic Dynamics Comparison",
            states=states_list,
            height=600,
        )

        _save_or_show(fig, save_path, dpi=dpi, height=600)

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
            save_path: Path to save the plot (optional, opens in browser if None)
        """
        if not results:
            raise ValueError("The results list is empty")

        if len(results) != len(labels):
            raise ValueError("The number of results and labels must match")

        # Use config values
        states_to_plot = (
            plot_config.states_to_plot if plot_config.states_to_plot else ["S", "I", "R"]
        )

        # Create figure using web plotting module
        fig = create_comparison_figure(
            results,
            labels,
            title=plot_config.title or "Epidemic Dynamics Comparison",
            states=states_to_plot,
            height=int(plot_config.figsize[1] * 100),  # Convert to pixels
        )

        # Apply labels, grid, and legend from config
        grid_color = f"rgba(0,0,0,{plot_config.grid_alpha})" if plot_config.grid else None
        fig.update_layout(
            xaxis_title=plot_config.xlabel,
            yaxis_title=plot_config.ylabel,
            xaxis_showgrid=plot_config.grid,
            yaxis_showgrid=plot_config.grid,
        )
        if plot_config.grid and grid_color:
            fig.update_layout(
                xaxis_gridcolor=grid_color,
                yaxis_gridcolor=grid_color,
            )
        # Map matplotlib-style legend_position to Plotly
        _LEGEND_MAP = {
            "best": dict(x=1.02, y=1, orientation="v"),
            "upper right": dict(x=1, y=1, xanchor="right"),
            "upper left": dict(x=0, y=1, xanchor="left"),
            "lower left": dict(x=0, y=0, xanchor="left", yanchor="bottom"),
            "lower right": dict(x=1, y=0, xanchor="right", yanchor="bottom"),
            "center": dict(x=0.5, y=0.5, xanchor="center", yanchor="middle"),
        }
        legend_kw = _LEGEND_MAP.get(plot_config.legend_position, {})
        if legend_kw:
            fig.update_layout(legend=legend_kw)

        _save_or_show(
            fig,
            save_path,
            dpi=plot_config.dpi,
            width=int(plot_config.figsize[0] * 100),
            height=int(plot_config.figsize[1] * 100),
        )

    @staticmethod
    def plot_network(G: Any, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plot the network used in the simulation (interactive Plotly).

        Args:
            G: Network graph (NetworkX)
            title: Plot title (optional)
            save_path: Path to save the plot (optional, opens in browser if None)
        """
        import networkx as nx

        # Limit the number of nodes for visualization
        if G.number_of_nodes() > 100:
            import warnings

            warnings.warn(
                f"The network has {G.number_of_nodes()} nodes. "
                "Limiting visualization to 100 nodes.",
                stacklevel=2,
            )
            G = nx.DiGraph(G.subgraph(list(G.nodes())[:100]))

        # Get layout
        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node trace
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=False,
                colorscale="YlGnBu",
                size=10,
                color=COLOR_S,
                line_width=2,
            ),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=title
                    or f"Network Structure ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
                    x=0.5,
                    xanchor="center",
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
            ),
        )

        _save_or_show(fig, save_path, width=800, height=700)

    @staticmethod
    def create_summary_plot(result_path: str, output_dir: Optional[str] = None) -> str:
        """
        Create an interactive summary plot from a results file.

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
            base_name = os.path.basename(result_path).replace(".json", ".html")
            save_path = os.path.join(output_dir, base_name)
        else:
            save_path = result_path.replace(".json", ".html")

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
