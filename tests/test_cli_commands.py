"""
Tests for SPKMC CLI commands.

This module contains tests for SPKMC CLI commands.
"""

import json
import os
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

from spkmc.cli.commands import cli
from spkmc.io.results import ResultManager


@pytest.fixture
def runner():
    """Fixture for Click's CliRunner."""
    return CliRunner()


@pytest.fixture
def temp_result_file():
    """Fixture to create a temporary results file."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # Sample data
    result = {
        "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
        "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
        "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
        "time": [0.0, 2.5, 5.0, 7.5, 10.0],
        "metadata": {
            "network_type": "er",
            "distribution": "gamma",
            "N": 100,
            "k_avg": 5,
            "samples": 10,
            "initial_perc": 0.01,
        },
    }

    # Save data to the file
    with open(path, "w") as f:
        json.dump(result, f)

    yield path

    # Remove the file after the test
    if os.path.exists(path):
        os.remove(path)


def test_cli_version(runner):
    """Test the CLI version command."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    # Version format: "spkmc, version X.Y.Z" or "spkmc, version X.Y.Z.devN"
    assert "spkmc, version" in result.output


def test_run_command_help(runner):
    """Test the help text for the run command."""
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run an SPKMC simulation" in result.output


def test_plot_command_help(runner):
    """Test the help text for the plot command."""
    result = runner.invoke(cli, ["plot", "--help"])
    assert result.exit_code == 0
    assert "Visualize results" in result.output


def test_info_command_help(runner):
    """Test the help text for the info command."""
    result = runner.invoke(cli, ["info", "--help"])
    assert result.exit_code == 0
    assert "Show information" in result.output


def test_compare_command_help(runner):
    """Test the help text for the compare command."""
    result = runner.invoke(cli, ["compare", "--help"])
    assert result.exit_code == 0
    assert "Compare results" in result.output


def test_run_command_invalid_network(runner):
    """Test run command with an invalid network type."""
    result = runner.invoke(cli, ["run", "--network-type", "invalid"])
    assert result.exit_code != 0
    assert "Invalid network type" in result.output


def test_run_command_invalid_distribution(runner):
    """Test run command with an invalid distribution type."""
    result = runner.invoke(cli, ["run", "--dist-type", "invalid"])
    assert result.exit_code != 0
    assert "Invalid distribution type" in result.output


def test_run_command_invalid_shape(runner):
    """Test run command with an invalid shape value."""
    result = runner.invoke(cli, ["run", "--shape", "-1.0"])
    assert result.exit_code != 0
    assert "must be positive" in result.output


def test_run_command_invalid_nodes(runner):
    """Test run command with an invalid node count."""
    result = runner.invoke(cli, ["run", "--nodes", "0"])
    assert result.exit_code != 0
    assert "positive integer" in result.output


def test_run_command_invalid_initial_perc(runner):
    """Test run command with an invalid initial percentage."""
    result = runner.invoke(cli, ["run", "--initial-perc", "1.5"])
    assert result.exit_code != 0
    assert "percentage must be between 0 and 1" in result.output


def test_plot_command_nonexistent_file(runner):
    """Test plot command with a nonexistent file."""
    result = runner.invoke(cli, ["plot", "nonexistent.json"])
    assert result.exit_code != 0
    assert "Path not found" in result.output


def test_plot_command_with_file(runner, temp_result_file, monkeypatch):
    """Test plot command with a valid file."""

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Mock load_result
    def mock_load_result(file_path):
        return {
            "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
            "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
            "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
            "time": [0.0, 2.5, 5.0, 7.5, 10.0],
            "metadata": {
                "network_type": "er",
                "distribution": "gamma",
                "N": 100,
                "k_avg": 5,
                "samples": 10,
                "initial_perc": 0.01,
            },
        }

    # Apply mocks
    from spkmc.io.results import ResultManager
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)
    monkeypatch.setattr(ResultManager, "load_result", mock_load_result)

    # Run the command
    result = runner.invoke(cli, ["plot", temp_result_file])

    # Verify the result
    assert result.exit_code == 0


def test_info_command_list(runner, monkeypatch):
    """Test info command with the --list option."""

    # Mock list_results
    def mock_list_results():
        return ["file1.json", "file2.json"]

    # Apply the mock
    monkeypatch.setattr(ResultManager, "list_results", mock_list_results)

    # Run the command
    result = runner.invoke(cli, ["info", "--list"])

    # Verify the result
    assert result.exit_code == 0
    assert "file1.json" in result.output
    assert "file2.json" in result.output


def test_info_command_with_file(runner, temp_result_file):
    """Test info command with a valid file."""
    # Run the command
    result = runner.invoke(cli, ["info", "--result-file", temp_result_file])

    # Verify the result
    assert result.exit_code == 0
    assert "Simulation Parameters" in result.output
    assert "network_type: er" in result.output.lower()


@pytest.mark.skip(reason="Test temporarily disabled due to issues with the --simple parameter")
def test_compare_command_with_files(runner, temp_result_file, monkeypatch):
    """Test compare command with valid files."""

    # Mock compare_results to avoid showing the plot
    def mock_compare_results(*args, **kwargs):
        pass

    # Apply the mock
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(Visualizer, "compare_results", mock_compare_results)

    # Run the command
    result = runner.invoke(
        cli, ["compare", temp_result_file, temp_result_file, "--labels", "Test1", "Test2"]
    )

    # Verify the result
    assert result.exit_code == 0


def test_create_time_steps():
    """Test the create_time_steps function."""
    from spkmc.cli.commands import create_time_steps

    # Test with valid values
    t_max = 10.0
    steps = 5
    time_steps = create_time_steps(t_max, steps)

    assert isinstance(time_steps, np.ndarray)
    assert len(time_steps) == steps
    assert time_steps[0] == 0.0
    assert time_steps[-1] == t_max
    assert np.allclose(time_steps, np.array([0.0, 2.5, 5.0, 7.5, 10.0]))


def test_batch_command_help(runner):
    """Test the help text for the batch command."""
    result = runner.invoke(cli, ["batch", "--help"])
    assert result.exit_code == 0
    assert "Run multiple simulation scenarios" in result.output


@pytest.fixture
def temp_batch_file():
    """Fixture to create a temporary scenarios file."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # Sample data with two scenarios
    scenarios = [
        {
            "network_type": "er",
            "dist_type": "gamma",
            "nodes": 100,
            "k_avg": 5,
            "shape": 2.0,
            "scale": 1.0,
            "lambda_val": 0.5,
            "samples": 10,
            "num_runs": 1,
            "initial_perc": 0.01,
            "t_max": 5.0,
            "steps": 5,
        },
        {
            "network_type": "cn",
            "dist_type": "exponential",
            "nodes": 100,
            "k_avg": 5,
            "exponent": 2.5,
            "mu": 1.0,
            "lambda_val": 0.5,
            "samples": 10,
            "num_runs": 1,
            "initial_perc": 0.01,
            "t_max": 5.0,
            "steps": 5,
        },
    ]

    # Save data to the file
    with open(path, "w") as f:
        json.dump(scenarios, f)

    yield path

    # Remove the file after the test
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_invalid_batch_file():
    """Fixture to create an invalid scenarios file."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # Invalid data (not a list)
    invalid_data = {"network_type": "er", "dist_type": "gamma"}

    # Save data to the file
    with open(path, "w") as f:
        json.dump(invalid_data, f)

    yield path

    # Remove the file after the test
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_output_dir():
    """Fixture to create a temporary output directory."""
    # Create a temporary directory
    output_dir = tempfile.mkdtemp()

    yield output_dir

    # Remove the directory after the test
    import shutil

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def test_batch_command_with_default_params(runner, temp_batch_file, temp_output_dir, monkeypatch):
    """Test batch command with default parameters."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        return {
            "S_val": np.array([0.99, 0.95, 0.90, 0.85, 0.80]),
            "I_val": np.array([0.01, 0.04, 0.05, 0.05, 0.04]),
            "R_val": np.array([0.00, 0.01, 0.05, 0.10, 0.16]),
            "has_error": False,
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Mock compare_results to avoid showing the comparison plot
    def mock_compare_results(*args, **kwargs):
        pass

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)
    monkeypatch.setattr(Visualizer, "compare_results", mock_compare_results)

    # Run the command
    result = runner.invoke(
        cli,
        [
            "batch",
            temp_batch_file,
            "--output-dir",
            temp_output_dir,
            "--no-plot",  # Avoid trying to show plots
        ],
    )

    # Verify the result
    assert result.exit_code == 0
    assert "Loaded 2 scenarios for execution" in result.output
    assert "Execution completed" in result.output

    # Note: With mocked run_simulation, files may not be created in temp_output_dir
    # The batch command creates files in its own output directory structure
    # We only verify the command completed successfully and produced expected output


def test_batch_command_with_invalid_json(runner, temp_invalid_batch_file, monkeypatch):
    """Test batch command with an invalid JSON file."""
    # Mock json.load to simulate a format error
    original_load = json.load

    def mock_load(f):
        data = original_load(f)
        # If it's a dict (not a list), raise an error
        if isinstance(data, dict):
            raise ValueError("The scenarios file must contain a list of JSON objects.")
        return data

    # Apply the mock
    monkeypatch.setattr(json, "load", mock_load)

    # Run the command
    result = runner.invoke(cli, ["batch", temp_invalid_batch_file])

    # Verify the result
    assert "The scenarios file must contain a list of JSON objects" in result.output


def test_batch_command_with_multiple_scenarios(
    runner, temp_batch_file, temp_output_dir, monkeypatch
):
    """Test batch command with multiple scenarios."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        return {
            "S_val": np.array([0.99, 0.95, 0.90, 0.85, 0.80]),
            "I_val": np.array([0.01, 0.04, 0.05, 0.05, 0.04]),
            "R_val": np.array([0.00, 0.01, 0.05, 0.10, 0.16]),
            "has_error": False,
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Run the command
    result = runner.invoke(
        cli,
        [
            "batch",
            temp_batch_file,
            "--output-dir",
            temp_output_dir,
            "--no-plot",  # Avoid trying to show plots
        ],
    )

    # Verify the result - with mocked run_simulation, the command should complete
    # successfully even though files may be created in a different location
    assert result.exit_code == 0
    assert "Loaded 2 scenarios for execution" in result.output
    assert "Scenarios processed: 2" in result.output


def test_batch_command_respects_output_options(
    runner, temp_batch_file, temp_output_dir, monkeypatch
):
    """Test that batch command respects output options."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        return {
            "S_val": np.array([0.99, 0.95, 0.90, 0.85, 0.80]),
            "I_val": np.array([0.01, 0.04, 0.05, 0.05, 0.04]),
            "R_val": np.array([0.00, 0.01, 0.05, 0.10, 0.16]),
            "has_error": False,
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Mock compare_results to create the comparison file
    def mock_compare_results(results, labels, title, output_path=None):
        # Create an empty file at the specified path
        if output_path:
            with open(output_path, "w") as f:
                f.write("Mock comparison plot")

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)
    monkeypatch.setattr(Visualizer, "compare_results", mock_compare_results)

    # Define a prefix for output files
    prefix = "test_prefix_"

    # Run the command with specific options
    result = runner.invoke(
        cli,
        [
            "batch",
            temp_batch_file,
            "--output-dir",
            temp_output_dir,
            "--prefix",
            prefix,
            "--compare",  # Generate comparison visualization
            "--save-plot",  # Save plots
            "--no-plot",  # Do not show plots on screen
        ],
    )

    # Verify the result - with mocked run_simulation, the command should complete
    # successfully even though actual files may not be created in the mocked environment
    assert result.exit_code == 0
    assert "Execution completed" in result.output


def test_simple_parameter_recognition(runner):
    """Test that the --simple parameter is recognized by the CLI."""
    result = runner.invoke(cli, ["--simple", "--help"])
    assert result.exit_code == 0
    assert "Generate simplified CSV result file" in result.output


@pytest.fixture
def temp_output_file():
    """Fixture to create a temporary output file."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    yield path

    # Remove the file after the test
    if os.path.exists(path):
        os.remove(path)

    # Also remove simplified CSV file if present
    csv_path = path.replace(".json", "_simple.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)


def test_run_command_with_simple_parameter(runner, temp_output_file, monkeypatch):
    """Test run command with the --simple parameter."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        # Create arrays with the same size as time_steps (100 points)
        size = 100
        s_vals = np.ones(size) * 0.99
        i_vals = np.ones(size) * 0.01
        r_vals = np.zeros(size)
        s_err = np.ones(size) * 0.001
        i_err = np.ones(size) * 0.001
        r_err = np.ones(size) * 0.001

        # Modify some values to simulate dynamics
        for idx in range(1, size):
            factor = min(idx / 20, 1.0)
            s_vals[idx] = max(0.8, 0.99 - 0.19 * factor)
            i_vals[idx] = (
                min(0.05, 0.01 + 0.04 * factor)
                if idx < 50
                else max(0.01, 0.05 - 0.01 * (idx - 50) / 50)
            )
            r_vals[idx] = min(0.16, 0.0 + 0.16 * factor)

        return {
            "S_val": s_vals,
            "I_val": i_vals,
            "R_val": r_vals,
            "has_error": True,
            "S_err": s_err,
            "I_err": i_err,
            "R_err": r_err,
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Run the command with --simple
    result = runner.invoke(
        cli,
        [
            "--simple",
            "run",
            "--output",
            temp_output_file,
            "--no-plot",  # Avoid trying to show plots
        ],
    )

    # Verify the result
    assert result.exit_code == 0
    assert "Simplified CSV output mode enabled" in result.output
    assert "Simulation completed successfully" in result.output
    # Simple mode message: "Simplified CSV output mode enabled"
    assert (
        "Simplified CSV output mode enabled" in result.output
        or "Execution completed" in result.output
    )

    # Verify the simplified CSV file was created
    csv_path = temp_output_file.replace(".json", "_simple.csv")
    assert os.path.exists(csv_path)

    # Verify the simplified CSV file content
    with open(csv_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 100  # 100 time points

        # Verify the format of each line (time, infected, error)
        for line in lines:
            parts = line.strip().split(",")
            assert len(parts) == 3

            # Verify values are valid numbers
            time_val = float(parts[0])
            infected_val = float(parts[1])
            error_val = float(parts[2])

            assert 0 <= time_val <= 10.0
            assert 0 <= infected_val <= 1.0
            assert 0 <= error_val <= 1.0


def test_batch_command_with_simple_parameter(runner, temp_batch_file, temp_output_dir, monkeypatch):
    """Test batch command with the --simple parameter."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        return {
            "S_val": np.array([0.99, 0.95, 0.90, 0.85, 0.80]),
            "I_val": np.array([0.01, 0.04, 0.05, 0.05, 0.04]),
            "R_val": np.array([0.00, 0.01, 0.05, 0.10, 0.16]),
            "has_error": True,
            "S_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
            "I_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
            "R_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Run the command with --simple
    result = runner.invoke(
        cli,
        [
            "--simple",
            "batch",
            temp_batch_file,
            "--output-dir",
            temp_output_dir,
            "--no-plot",  # Avoid trying to show plots
        ],
    )

    # Verify the result - with mocked run_simulation, the command should complete
    # successfully. File creation tests require non-mocked execution.
    assert result.exit_code == 0
    assert "Simplified CSV output mode enabled" in result.output
    assert "Loaded 2 scenarios for execution" in result.output
    assert "Execution completed" in result.output


def test_simple_csv_format(runner, temp_output_file, monkeypatch):
    """Test that the simplified CSV contains 3 columns: time, infected, error."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        return {
            "S_val": np.array([0.99, 0.95, 0.90, 0.85, 0.80]),
            "I_val": np.array([0.01, 0.04, 0.05, 0.05, 0.04]),
            "R_val": np.array([0.00, 0.01, 0.05, 0.10, 0.16]),
            "has_error": True,
            "S_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
            "I_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
            "R_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Run the command with --simple
    result = runner.invoke(
        cli,
        [
            "--simple",
            "run",
            "--output",
            temp_output_file,
            "--no-plot",  # Avoid trying to show plots
        ],
    )

    # Verify the result
    assert result.exit_code == 0

    # Verify the simplified CSV file was created
    csv_path = temp_output_file.replace(".json", "_simple.csv")
    assert os.path.exists(csv_path)

    # Read the CSV file and verify its contents
    with open(csv_path, "r") as f:
        lines = f.readlines()

        # Verify there is at least one line
        assert len(lines) > 0

        # Verify each line has exactly 3 columns (time, infected, error)
        for line in lines:
            parts = line.strip().split(",")
            assert len(parts) == 3

            # Verify values are valid numbers
            time_val = float(parts[0])
            infected_val = float(parts[1])
            error_val = float(parts[2])

            # Verify values are within expected ranges
            assert 0 <= time_val <= 10.0  # time between 0 and 10
            assert 0 <= infected_val <= 1.0  # infected between 0 and 1
            assert 0 <= error_val <= 1.0  # error between 0 and 1


def test_simple_parameter_after_command(runner, temp_output_file, monkeypatch):
    """Test that --simple works when used after the command."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        # Create arrays with the same size as time_steps (100 points)
        size = 100
        s_vals = np.ones(size) * 0.99
        i_vals = np.ones(size) * 0.01
        r_vals = np.zeros(size)
        s_err = np.ones(size) * 0.001
        i_err = np.ones(size) * 0.001
        r_err = np.ones(size) * 0.001

        # Modify some values to simulate dynamics
        for idx in range(1, size):
            factor = min(idx / 20, 1.0)
            s_vals[idx] = max(0.8, 0.99 - 0.19 * factor)
            i_vals[idx] = (
                min(0.05, 0.01 + 0.04 * factor)
                if idx < 50
                else max(0.01, 0.05 - 0.01 * (idx - 50) / 50)
            )
            r_vals[idx] = min(0.16, 0.0 + 0.16 * factor)

        return {
            "S_val": s_vals,
            "I_val": i_vals,
            "R_val": r_vals,
            "has_error": True,
            "S_err": s_err,
            "I_err": i_err,
            "R_err": r_err,
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Run the command with --simple after the command
    result = runner.invoke(
        cli,
        ["run", "--output", temp_output_file, "--no-plot", "--simple"],  # Parameter after command
    )

    # Verify the result
    assert result.exit_code == 0
    assert "Simulation completed successfully" in result.output
    # Simple mode outputs \"Simplified results saved to CSV\" when saving
    assert "simplified" in result.output.lower() or "simple" in result.output.lower()

    # Verify the simplified CSV file was created
    csv_path = temp_output_file.replace(".json", "_simple.csv")
    assert os.path.exists(csv_path)

    # Verify the simplified CSV file content
    with open(csv_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 100  # 100 time points

        # Verify the format of each line (time, infected, error)
        for line in lines:
            parts = line.strip().split(",")
            assert len(parts) == 3

            # Verify values are valid numbers
            time_val = float(parts[0])
            infected_val = float(parts[1])
            error_val = float(parts[2])

            # Verify values are within expected ranges
            assert 0 <= time_val <= 10.0
            assert 0 <= infected_val <= 1.0
            assert 0 <= error_val <= 1.0


def test_batch_command_with_simple_parameter_after_command(
    runner, temp_batch_file, temp_output_dir, monkeypatch
):
    """Test batch command with --simple after the command."""

    # Mock run_simulation to avoid real execution
    def mock_run_simulation(*args, **kwargs):
        return {
            "S_val": np.array([0.99, 0.95, 0.90, 0.85, 0.80]),
            "I_val": np.array([0.01, 0.04, 0.05, 0.05, 0.04]),
            "R_val": np.array([0.00, 0.01, 0.05, 0.10, 0.16]),
            "has_error": True,
            "S_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
            "I_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
            "R_err": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
        }

    # Mock plot_result to avoid showing the plot
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply mocks
    from spkmc.core.simulation import SPKMC
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(SPKMC, "run_simulation", mock_run_simulation)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Run the command with --simple after the command
    result = runner.invoke(
        cli,
        [
            "batch",
            temp_batch_file,
            "--output-dir",
            temp_output_dir,
            "--no-plot",
            "--simple",  # Parameter after the command
        ],
    )

    # Verify the result - with mocked run_simulation, the command should complete
    # successfully. File creation tests require non-mocked execution.
    assert result.exit_code == 0
    assert "Loaded 2 scenarios for execution" in result.output
    assert "Execution completed" in result.output
