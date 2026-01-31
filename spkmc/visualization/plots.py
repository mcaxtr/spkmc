"""
Result visualization for the SPKMC algorithm.

This module contains functions to visualize SPKMC simulation results,
including time-evolution plots of SIR states and comparisons between simulations.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from spkmc.io.experiments import PlotConfig


class Visualizer:
    """Class for visualizing results."""

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
    ) -> None:
        """
        Plot results with error bars.

        Args:
            S: Proportion of susceptible
            I: Proportion of infected
            R: Proportion of recovered
            S_err: Error for susceptible
            I_err: Error for infected
            R_err: Error for recovered
            time: Time steps
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
        """
        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        plt.figure(figsize=(10, 6))

        if "R" in states_to_plot:
            plt.errorbar(time, R, yerr=R_err, label="Recovered", capsize=2, color="g")
        if "I" in states_to_plot:
            plt.errorbar(time, I, yerr=I_err, label="Infected", capsize=2, color="r")
        if "S" in states_to_plot:
            plt.errorbar(time, S, yerr=S_err, label="Susceptible", capsize=2, color="b")

        plt.xlabel("Time")
        plt.ylabel("Proportion of Individuals")

        if title:
            plt.title(title)
        else:
            plt.title("SIR Model Dynamics Over Time with Error Bars")

        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_result(
        S: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        time: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        states_to_plot: Optional[set] = None,
    ) -> None:
        """
        Plot results without error bars.

        Args:
            S: Proportion of susceptible
            I: Proportion of infected
            R: Proportion of recovered
            time: Time steps
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
        """
        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        plt.figure(figsize=(10, 6))

        if "R" in states_to_plot:
            plt.plot(time, R, "g-", label="Recovered")
        if "I" in states_to_plot:
            plt.plot(time, I, "r-", label="Infected")
        if "S" in states_to_plot:
            plt.plot(time, S, "b-", label="Susceptible")

        plt.xlabel("Time")
        plt.ylabel("Proportion of Individuals")

        if title:
            plt.title(title)
        else:
            plt.title("SIR Model Dynamics Over Time")

        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def compare_results(
        results: List[Dict[str, Any]],
        labels: List[str],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        states_to_plot: Optional[set] = None,
    ) -> None:
        """
        Compare results from multiple simulations.

        Args:
            results: List of dictionaries with results
            labels: List of labels for each result
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            states_to_plot: Set of states to plot ('S', 'I', 'R')
        """
        if not results:
            raise ValueError("The results list is empty")

        if len(results) != len(labels):
            raise ValueError("The number of results and labels must match")

        if states_to_plot is None:
            states_to_plot = {"S", "I", "R"}

        plt.figure(figsize=(12, 8))

        # Colors for scenarios (different color per scenario)
        scenario_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Line styles for states (different style per state type)
        state_styles = {"S": ":", "I": "-", "R": "--"}

        for idx, (result, label) in enumerate(zip(results, labels)):
            if not all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                raise ValueError(f"Result {idx} does not contain all required data")

            s_vals = np.array(result["S_val"])
            i_vals = np.array(result["I_val"])
            r_vals = np.array(result["R_val"])
            time = np.array(result["time"])

            color = scenario_colors[idx % len(scenario_colors)]

            if "S" in states_to_plot:
                plt.plot(
                    time,
                    s_vals,
                    color=color,
                    linestyle=state_styles["S"],
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"S - {label}",
                )
            if "I" in states_to_plot:
                plt.plot(
                    time,
                    i_vals,
                    color=color,
                    linestyle=state_styles["I"],
                    linewidth=2,
                    alpha=0.9,
                    label=f"I - {label}",
                )
            if "R" in states_to_plot:
                plt.plot(
                    time,
                    r_vals,
                    color=color,
                    linestyle=state_styles["R"],
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"R - {label}",
                )

        plt.xlabel("Time")
        plt.ylabel("Proportion of Individuals")

        if title:
            plt.title(title)
        else:
            plt.title("SPKMC Simulation Comparison")

        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

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

        plt.figure(figsize=plot_config.figsize)

        # Colors for scenarios (different color per scenario)
        scenario_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Line styles for states (different style per state type)
        state_styles = {"S": ":", "I": "-", "R": "--"}

        for idx, (result, label) in enumerate(zip(results, labels)):
            if not all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                raise ValueError(f"Result {idx} does not contain all required data")

            s_vals = np.array(result["S_val"])
            i_vals = np.array(result["I_val"])
            r_vals = np.array(result["R_val"])
            time = np.array(result["time"])

            color = scenario_colors[idx % len(scenario_colors)]

            if "S" in states_to_plot:
                plt.plot(
                    time,
                    s_vals,
                    color=color,
                    linestyle=state_styles["S"],
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"S - {label}",
                )
            if "I" in states_to_plot:
                plt.plot(
                    time,
                    i_vals,
                    color=color,
                    linestyle=state_styles["I"],
                    linewidth=2,
                    alpha=0.9,
                    label=f"I - {label}",
                )
            if "R" in states_to_plot:
                plt.plot(
                    time,
                    r_vals,
                    color=color,
                    linestyle=state_styles["R"],
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"R - {label}",
                )

        plt.xlabel(plot_config.xlabel)
        plt.ylabel(plot_config.ylabel)

        if plot_config.title:
            plt.title(plot_config.title)
        else:
            plt.title("SPKMC Simulation Comparison")

        # Position legend outside if many scenarios, otherwise use config position
        if len(results) > 4:
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
        else:
            plt.legend(loc=plot_config.legend_position, fontsize="small")

        if plot_config.grid:
            plt.grid(True, alpha=plot_config.grid_alpha)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=plot_config.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_network(
        G: nx.DiGraph, title: Optional[str] = None, save_path: Optional[str] = None
    ) -> None:
        """
        Plot the network used in the simulation.

        Args:
            G: Network graph
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 8))

        # Limit the number of nodes for visualization
        if G.number_of_nodes() > 100:
            import warnings

            warnings.warn(
                f"The network has {G.number_of_nodes()} nodes. "
                "Limiting visualization to 100 nodes.",
                stacklevel=2,
            )
            G = nx.DiGraph(G.subgraph(list(G.nodes())[:100]))

        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            node_size=30,
            node_color="skyblue",
            edge_color="gray",
            arrows=True,
            arrowsize=10,
            alpha=0.8,
        )

        if title:
            plt.title(title)
        else:
            plt.title(
                "Network Visualization "
                f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
            )

        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def create_summary_plot(result_path: str, output_dir: Optional[str] = None) -> str:
        """
        Create a summary plot from a results file.

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
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")

        title = f"SPKMC Simulation - Network {network_type}, Distribution {dist_type}, N={N}"

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
