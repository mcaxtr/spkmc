#!/usr/bin/env python3
"""
Example of programmatic use of the SPKMC CLI.

This script demonstrates how to call the SPKMC CLI programmatically from another
Python script to automate simulations and analyses.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_cli_command(command):
    """
    Run an SPKMC CLI command.

    Args:
        command: List with the command and its arguments

    Returns:
        Command output
    """
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None

    return result.stdout


def run_simulation(network_type, dist_type, output_file, **kwargs):
    """
    Run a simulation using the SPKMC CLI.

    Args:
        network_type: Network type ('er', 'cn', 'cg')
        dist_type: Distribution type ('gamma', 'exponential')
        output_file: Path to save results
        **kwargs: Additional simulation parameters

    Returns:
        Path to the results file
    """
    # Base command
    command = [
        "python",
        "spkmc_cli.py",
        "run",
        "--network-type",
        network_type,
        "--dist-type",
        dist_type,
        "--output",
        output_file,
        "--no-plot",
    ]  # Do not show the plot during execution

    # Add additional parameters
    for key, value in kwargs.items():
        # Convert underscores to hyphens in parameter names
        param_name = f"--{key.replace('_', '-')}"
        command.extend([param_name, str(value)])

    # Run the command
    run_cli_command(command)

    # Check whether the file was created
    if os.path.exists(output_file):
        return output_file
    else:
        print(f"Error: Results file was not created: {output_file}")
        return None


def plot_results(result_file, output_file=None, with_error=False):
    """
    Plot simulation results using the SPKMC CLI.

    Args:
        result_file: Path to the results file
        output_file: Path to save the plot (optional)
        with_error: If True, show error bars (if available)

    Returns:
        Path to the plot file (if output_file is provided)
    """
    # Base command
    command = ["python", "spkmc_cli.py", "plot", result_file]

    # Add options
    if with_error:
        command.append("--with-error")

    if output_file:
        command.extend(["--output", output_file])

    # Run the command
    run_cli_command(command)

    # Check whether the file was created
    if output_file and os.path.exists(output_file):
        return output_file
    else:
        return None


def compare_results(result_files, labels=None, output_file=None):
    """
    Compare results from multiple simulations using the SPKMC CLI.

    Args:
        result_files: List of paths to results files
        labels: List of labels for each file (optional)
        output_file: Path to save the plot (optional)

    Returns:
        Path to the plot file (if output_file is provided)
    """
    # Base command
    command = ["python", "spkmc_cli.py", "compare"] + result_files

    # Add labels
    if labels:
        for label in labels:
            command.extend(["-l", label])

    # Add output option
    if output_file:
        command.extend(["--output", output_file])

    # Run the command
    run_cli_command(command)

    # Check whether the file was created
    if output_file and os.path.exists(output_file):
        return output_file
    else:
        return None


def get_info(result_file):
    """
    Get information about a simulation using the SPKMC CLI.

    Args:
        result_file: Path to the results file

    Returns:
        Simulation information
    """
    # Command
    command = ["python", "spkmc_cli.py", "info", "--result-file", result_file]

    # Run the command
    return run_cli_command(command)


def main():
    """Main function for the example."""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Common parameters
    common_params = {
        "nodes": 1000,
        "samples": 50,
        "initial_perc": 0.01,
        "t_max": 10.0,
        "steps": 100,
        "num_runs": 2,
    }

    # Run simulation with Erdos-Renyi network and Gamma distribution
    er_gamma_file = str(results_dir / "er_gamma.json")
    er_gamma_params = {**common_params, "k_avg": 10, "shape": 2.0, "scale": 1.0, "lambda_val": 1.0}
    er_gamma_result = run_simulation("er", "gamma", er_gamma_file, **er_gamma_params)

    # Run simulation with Erdos-Renyi network and Exponential distribution
    er_exp_file = str(results_dir / "er_exponential.json")
    er_exp_params = {**common_params, "k_avg": 10, "mu": 1.0, "lambda_val": 1.0}
    er_exp_result = run_simulation("er", "exponential", er_exp_file, **er_exp_params)

    # Run simulation with Complex network and Gamma distribution
    cn_gamma_file = str(results_dir / "cn_gamma.json")
    cn_gamma_params = {
        **common_params,
        "k_avg": 10,
        "exponent": 2.5,
        "shape": 2.0,
        "scale": 1.0,
        "lambda_val": 1.0,
    }
    cn_gamma_result = run_simulation("cn", "gamma", cn_gamma_file, **cn_gamma_params)

    # Plot individual results
    if er_gamma_result:
        plot_results(er_gamma_result, str(plots_dir / "er_gamma.png"), with_error=True)
        print("Information about the ER-Gamma simulation:")
        print(get_info(er_gamma_result))

    if er_exp_result:
        plot_results(er_exp_result, str(plots_dir / "er_exponential.png"), with_error=True)

    if cn_gamma_result:
        plot_results(cn_gamma_result, str(plots_dir / "cn_gamma.png"), with_error=True)

    # Compare results
    if er_gamma_result and er_exp_result:
        compare_results(
            [er_gamma_result, er_exp_result],
            labels=["ER-Gamma", "ER-Exponential"],
            output_file=str(plots_dir / "comparison_er.png"),
        )

    if er_gamma_result and cn_gamma_result:
        compare_results(
            [er_gamma_result, cn_gamma_result],
            labels=["ER-Gamma", "CN-Gamma"],
            output_file=str(plots_dir / "comparison_networks.png"),
        )

    print("\nSimulations completed. Results saved to:")
    print(f"- Results files: {results_dir}")
    print(f"- Plots: {plots_dir}")


if __name__ == "__main__":
    main()
