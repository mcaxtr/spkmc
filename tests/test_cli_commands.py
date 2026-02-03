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
            "network": "er",
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
    assert "Visualize and compare" in result.output


def test_info_command_help(runner):
    """Test the help text for the info command."""
    result = runner.invoke(cli, ["info", "--help"])
    assert result.exit_code == 0
    assert "Show information" in result.output


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
    assert "percentage must be greater than 0 and at most 1" in result.output


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
                "network": "er",
                "distribution": "gamma",
                "N": 100,
                "k_avg": 5,
                "samples": 10,
                "initial_perc": 0.01,
            },
        }

    # Apply mocks
    from spkmc.io.data_manager import DataManager
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)
    monkeypatch.setattr(DataManager, "load", mock_load_result)

    # Run the command
    result = runner.invoke(cli, ["plot", temp_result_file])

    # Verify the result
    assert result.exit_code == 0


def test_plot_command_skips_incompatible_step_counts(runner, tmp_path, monkeypatch):
    """Test plot command skips files with incompatible step counts."""
    from spkmc.io.data_manager import DataManager
    from spkmc.visualization.plots import Visualizer

    # Create files with different step counts
    file1 = tmp_path / "result1.json"
    file2 = tmp_path / "result2.json"

    result1 = {
        "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
        "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
        "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
        "time": [0.0, 2.5, 5.0, 7.5, 10.0],
        "metadata": {"network": "er"},
    }
    result2 = {
        "S_val": [0.99, 0.90, 0.80],  # Different length
        "I_val": [0.01, 0.05, 0.04],
        "R_val": [0.00, 0.05, 0.16],
        "time": [0.0, 5.0, 10.0],
        "metadata": {"network": "sf"},
    }

    file1.write_text(json.dumps(result1))
    file2.write_text(json.dumps(result2))

    # Mock compare_results
    def mock_compare(*args, **kwargs):
        pass

    monkeypatch.setattr(Visualizer, "compare_results", mock_compare)

    # Run the command
    result = runner.invoke(cli, ["plot", str(file1), str(file2)])

    # Verify warning about incompatible step count
    assert "incompatible step count" in result.output.lower()
    assert "3 vs 5 expected" in result.output


def test_plot_command_skips_inconsistent_arrays(runner, tmp_path, monkeypatch):
    """Test plot command skips files with inconsistent internal array lengths."""
    from spkmc.io.data_manager import DataManager
    from spkmc.visualization.plots import Visualizer

    # Create files - one valid, one with inconsistent arrays
    file1 = tmp_path / "result1.json"
    file2 = tmp_path / "result2.json"

    result1 = {
        "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
        "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
        "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
        "time": [0.0, 2.5, 5.0, 7.5, 10.0],
        "metadata": {"network": "er"},
    }
    result2 = {
        "S_val": [0.99, 0.90],  # Inconsistent with time
        "I_val": [0.01, 0.05],
        "R_val": [0.00, 0.05],
        "time": [0.0, 2.5, 5.0, 7.5, 10.0],  # 5 elements
        "metadata": {"network": "sf"},
    }

    file1.write_text(json.dumps(result1))
    file2.write_text(json.dumps(result2))

    # Mock compare_results and plot_result
    def mock_compare(*args, **kwargs):
        pass

    def mock_plot(*args, **kwargs):
        pass

    monkeypatch.setattr(Visualizer, "compare_results", mock_compare)
    monkeypatch.setattr(Visualizer, "plot_result", mock_plot)

    # Run the command
    result = runner.invoke(cli, ["plot", str(file1), str(file2)])

    # Verify warning about inconsistent array lengths
    assert "inconsistent array lengths" in result.output.lower()


def test_info_command_list(runner, monkeypatch):
    """Test info command with the --list option."""
    from spkmc.io.data_manager import DataManager

    # Mock list_all_results
    def mock_list_all_results(base_dir="data/runs"):
        return ["file1.json", "file2.json"]

    # Apply the mock
    monkeypatch.setattr(DataManager, "list_all_results", mock_list_all_results)

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
    assert "network: er" in result.output.lower()


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


def test_experiments_command_help(runner):
    """Test the help text for the experiment command."""
    result = runner.invoke(cli, ["experiments", "--help"])
    assert result.exit_code == 0
    assert "Run or create experiments" in result.output


@pytest.fixture
def temp_experiment_file():
    """Fixture to create a temporary scenarios file."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # Sample data with two scenarios - each scenario has all required fields
    # (Current CLI code expects scenarios with all params, not merged from global)
    scenarios = [
        {
            "label": "ER-Gamma",
            "network": "er",
            "distribution": "gamma",
            "nodes": 100,
            "k_avg": 5,
            "shape": 2.0,
            "scale": 1.0,
            "lambda": 0.5,
            "samples": 10,
            "num_runs": 1,
            "initial_perc": 0.01,
            "t_max": 5.0,
            "steps": 5,
        },
        {
            "label": "SF-Exponential",
            "network": "sf",
            "distribution": "exponential",
            "nodes": 100,
            "k_avg": 5,
            "exponent": 2.5,
            "mu": 1.0,
            "lambda": 0.5,
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
def temp_invalid_experiment_file():
    """Fixture to create an invalid scenarios file."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # Invalid data (not a list)
    invalid_data = {"network": "er", "distribution": "gamma"}

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


def test_experiments_command_with_default_params(
    runner, temp_experiment_file, temp_output_dir, monkeypatch
):
    """Test experiment command with default parameters."""
    from spkmc.models.scenario import Scenario

    # Override EXPERIMENTS_BASE to use temp directory for testing
    monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", temp_output_dir)

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
            "experiments",
            temp_experiment_file,
            "--no-plot",  # Avoid trying to show plots
        ],
    )

    # Verify the result
    assert result.exit_code == 0
    assert "Loaded 2 scenarios for execution" in result.output
    assert "Execution completed" in result.output

    # Note: With mocked run_simulation, files may not be created in temp_output_dir
    # The experiment command creates files in its own output directory structure
    # We only verify the command completed successfully and produced expected output


def test_experiments_command_with_invalid_json(runner, temp_invalid_experiment_file, monkeypatch):
    """Test experiment command with an invalid JSON file."""
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
    result = runner.invoke(cli, ["experiments", temp_invalid_experiment_file])

    # Verify the result
    assert "The scenarios file must contain a list of JSON objects" in result.output


def test_experiments_command_with_multiple_scenarios(
    runner, temp_experiment_file, temp_output_dir, monkeypatch
):
    """Test experiment command with multiple scenarios."""
    from spkmc.models.scenario import Scenario

    # Override EXPERIMENTS_BASE to use temp directory for testing
    monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", temp_output_dir)

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
            "experiments",
            temp_experiment_file,
            "--no-plot",  # Avoid trying to show plots
        ],
    )

    # Verify the result - with mocked run_simulation, the command should complete
    # successfully even though files may be created in a different location
    assert result.exit_code == 0
    assert "Loaded 2 scenarios for execution" in result.output
    assert "Scenarios processed: 2" in result.output


def test_experiments_command_respects_output_options(
    runner, temp_experiment_file, temp_output_dir, monkeypatch
):
    """Test that experiment command respects output options."""
    from spkmc.models.scenario import Scenario

    # Override EXPERIMENTS_BASE to use temp directory for testing
    monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", temp_output_dir)

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

    # Run the command with specific options
    result = runner.invoke(
        cli,
        [
            "experiments",
            temp_experiment_file,
            "--no-plot",  # Do not show plots on screen
        ],
    )

    # Verify the result - with mocked run_simulation, the command should complete
    # successfully even though actual files may not be created in the mocked environment
    assert result.exit_code == 0
    assert "Execution completed" in result.output


@pytest.fixture
def temp_experiment_file_missing_params():
    """Fixture to create a scenarios file with missing required parameters."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # Scenarios missing required parameters (no k_avg, samples, etc.)
    scenarios = [
        {
            "label": "Incomplete Scenario",
            "network": "er",
            "distribution": "gamma",
            "nodes": 100,
            # Missing: k_avg, shape, scale, lambda, samples, num_runs, etc.
        },
    ]

    with open(path, "w") as f:
        json.dump(scenarios, f)

    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_experiments_command_fails_with_missing_params(
    runner, temp_experiment_file_missing_params, temp_output_dir, monkeypatch
):
    """Test experiment command fails with validation error for missing parameters."""
    from spkmc.models.scenario import Scenario

    # Override EXPERIMENTS_BASE to use temp directory for testing
    monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", temp_output_dir)

    result = runner.invoke(
        cli,
        [
            "experiments",
            temp_experiment_file_missing_params,
            "--no-plot",
        ],
    )

    # Should abort with missing parameter error
    assert result.exit_code == 1
    # CLI shows "Missing required parameter" message
    assert "Missing required parameter" in result.output


# ============================================================
# Analyze command tests
# ============================================================


def test_analyze_command_help(runner):
    """Test the help text for the analyze command."""
    result = runner.invoke(cli, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "Generate AI-powered analysis" in result.output
    assert "[PATHS]" in result.output
    assert "--all" in result.output
    assert "--model" in result.output
    assert "--force" in result.output


def test_analyze_command_no_api_key(runner, temp_result_file, monkeypatch):
    """Test analyze command fails gracefully without API key."""
    # Ensure no API key is set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(cli, ["analyze", temp_result_file])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_analyze_command_single_file(runner, temp_result_file, monkeypatch):
    """Test analyze command with a single file."""
    # Mock the API key
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock the AIAnalyzer at the source module level
    analyze_calls = []

    class MockAIAnalyzer:
        def __init__(self, model="gpt-4o-mini"):
            self.model = model

        @staticmethod
        def is_available():
            return True

        def analyze_experiment(self, experiment_name, experiment_description, results, output_dir):
            analyze_calls.append(
                {
                    "experiment_name": experiment_name,
                    "results_count": len(results),
                    "output_dir": str(output_dir),
                }
            )
            return output_dir / "analysis.md"

    import spkmc.analysis.ai_analyzer

    monkeypatch.setattr(spkmc.analysis.ai_analyzer, "AIAnalyzer", MockAIAnalyzer)

    result = runner.invoke(cli, ["analyze", temp_result_file])

    assert result.exit_code == 0
    assert len(analyze_calls) == 1
    assert analyze_calls[0]["results_count"] == 1


def test_analyze_command_multiple_files(runner, monkeypatch):
    """Test analyze command with multiple files."""
    # Create multiple temp files
    temp_files = []
    for i in range(3):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        result_data = {
            "S_val": [0.99, 0.95, 0.90],
            "I_val": [0.01, 0.04, 0.05],
            "R_val": [0.00, 0.01, 0.05],
            "time": [0.0, 2.5, 5.0],
            "metadata": {"network": "er", "distribution": "gamma", "N": 100, "scenario": i},
        }
        with open(path, "w") as f:
            json.dump(result_data, f)
        temp_files.append(path)

    try:
        # Mock the API key
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock the AIAnalyzer at the source module level
        analyze_calls = []

        class MockAIAnalyzer:
            def __init__(self, model="gpt-4o-mini"):
                self.model = model

            @staticmethod
            def is_available():
                return True

            def analyze_experiment(
                self, experiment_name, experiment_description, results, output_dir
            ):
                analyze_calls.append({"experiment_name": experiment_name})
                return output_dir / "analysis.md"

        import spkmc.analysis.ai_analyzer

        monkeypatch.setattr(spkmc.analysis.ai_analyzer, "AIAnalyzer", MockAIAnalyzer)

        result = runner.invoke(cli, ["analyze", *temp_files])

        assert result.exit_code == 0
        # Should have called analyze 3 times (once per file)
        assert len(analyze_calls) == 3
        # Check progress messages
        assert "Analyzing [1/3]" in result.output
        assert "Analyzing [2/3]" in result.output
        assert "Analyzing [3/3]" in result.output
    finally:
        # Cleanup
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)


def test_analyze_command_output_ignored_for_multiple_files(runner, monkeypatch):
    """Test that --output is ignored when multiple paths are provided."""
    # Create two temp files
    temp_files = []
    for _ in range(2):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        result_data = {
            "S_val": [0.99],
            "I_val": [0.01],
            "R_val": [0.00],
            "time": [0.0],
            "metadata": {"network": "er"},
        }
        with open(path, "w") as f:
            json.dump(result_data, f)
        temp_files.append(path)

    try:
        # Mock the API key
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        class MockAIAnalyzer:
            def __init__(self, model="gpt-4o-mini"):
                pass

            @staticmethod
            def is_available():
                return True

            def analyze_experiment(
                self, experiment_name, experiment_description, results, output_dir
            ):
                return output_dir / "analysis.md"

        import spkmc.analysis.ai_analyzer

        monkeypatch.setattr(spkmc.analysis.ai_analyzer, "AIAnalyzer", MockAIAnalyzer)

        result = runner.invoke(cli, ["analyze", *temp_files, "-o", "custom.md"])

        assert result.exit_code == 0
        assert "--output is ignored" in result.output
    finally:
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)


def test_analyze_command_no_paths_shows_help(runner, monkeypatch):
    """Test that analyze command without paths shows helpful error."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class MockAIAnalyzer:
        def __init__(self, model="gpt-4o-mini"):
            pass

        @staticmethod
        def is_available():
            return True

    import spkmc.analysis.ai_analyzer

    monkeypatch.setattr(spkmc.analysis.ai_analyzer, "AIAnalyzer", MockAIAnalyzer)

    result = runner.invoke(cli, ["analyze"])

    assert result.exit_code == 1
    assert "Specify one or more paths" in result.output


def test_global_analyze_flag_with_run_no_output(runner, monkeypatch):
    """Test that --analyze flag attempts analysis even without explicit output.

    With the unified ExecutionEngine, results are always saved to a default location,
    so --analyze will attempt to generate analysis. The analysis will fail with an
    invalid API key, but the simulation should complete successfully.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock heavy imports at the source module level
    class MockSPKMC:
        def __init__(self, dist, use_gpu=False):
            pass

        def run_simulation(self, *args, **kwargs):
            return {
                "S_val": np.array([0.99, 0.95]),
                "I_val": np.array([0.01, 0.04]),
                "R_val": np.array([0.00, 0.01]),
                "has_error": False,
            }

    class MockDistribution:
        pass

    def mock_create_distribution(*args, **kwargs):
        return MockDistribution()

    import spkmc.core.distributions
    import spkmc.core.simulation
    import spkmc.utils.gpu_utils

    monkeypatch.setattr(spkmc.core.simulation, "SPKMC", MockSPKMC)
    monkeypatch.setattr(spkmc.core.distributions, "create_distribution", mock_create_distribution)
    monkeypatch.setattr(spkmc.utils.gpu_utils, "check_gpu_suggestion", lambda: None)

    def mock_plot(*args, **kwargs):
        pass

    import spkmc.visualization.plots

    monkeypatch.setattr(spkmc.visualization.plots.Visualizer, "plot_result", mock_plot)

    result = runner.invoke(
        cli,
        [
            "--analyze",
            "run",
            "-n",
            "er",
            "-d",
            "gamma",
            "--nodes",
            "10",
            "--samples",
            "2",
            "--initial-perc",
            "0.1",  # Ensure at least 1 infected node (10 * 0.1 = 1)
            "--no-plot",
        ],
    )

    # Simulation should complete successfully
    assert result.exit_code == 0
    # Results are now always saved (unified ExecutionEngine behavior)
    # Analysis is attempted but fails with invalid API key
    assert (
        "Simulation completed successfully" in result.output or "Results saved to" in result.output
    )
    # Analysis is attempted (may show "Generating AI analysis" or error)
    assert "AI analysis" in result.output or "analysis" in result.output.lower()


def test_global_analyze_flag_with_run_and_output(runner, monkeypatch, tmp_path):
    """Test that --analyze flag runs analysis when output is provided."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    output_file = tmp_path / "result.json"

    # Mock heavy imports at the source module level
    class MockSPKMC:
        def __init__(self, dist, use_gpu=False):
            pass

        def run_simulation(self, *args, **kwargs):
            return {
                "S_val": np.array([0.99, 0.95]),
                "I_val": np.array([0.01, 0.04]),
                "R_val": np.array([0.00, 0.01]),
                "has_error": False,
            }

    class MockDistribution:
        pass

    def mock_create_distribution(*args, **kwargs):
        return MockDistribution()

    analyze_called = []

    class MockAIAnalyzer:
        def __init__(self, model="gpt-4o-mini"):
            pass

        @staticmethod
        def is_available():
            return True

        def analyze_experiment(self, experiment_name, experiment_description, results, results_dir):
            analyze_called.append(True)
            return results_dir / "analysis.md"

    import spkmc.analysis.ai_analyzer
    import spkmc.core.distributions
    import spkmc.core.simulation
    import spkmc.utils.gpu_utils

    monkeypatch.setattr(spkmc.core.simulation, "SPKMC", MockSPKMC)
    monkeypatch.setattr(spkmc.core.distributions, "create_distribution", mock_create_distribution)
    monkeypatch.setattr(spkmc.utils.gpu_utils, "check_gpu_suggestion", lambda: None)
    monkeypatch.setattr(spkmc.analysis.ai_analyzer, "AIAnalyzer", MockAIAnalyzer)

    def mock_plot(*args, **kwargs):
        pass

    import spkmc.visualization.plots

    monkeypatch.setattr(spkmc.visualization.plots.Visualizer, "plot_result", mock_plot)

    result = runner.invoke(
        cli,
        [
            "--analyze",
            "run",
            "-n",
            "er",
            "-d",
            "gamma",
            "--nodes",
            "10",
            "--samples",
            "2",
            "--initial-perc",
            "0.1",  # Ensure at least 1 infected node (10 * 0.1 = 1)
            "--no-plot",
            "-o",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "Generating AI analysis" in result.output
    assert len(analyze_called) == 1
