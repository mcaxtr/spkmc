"""
Tests for plot command improvements in SPKMC.

This module contains specific tests for new plot command features,
including directory support, state filters, and multiple scenarios.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from spkmc.cli.commands import cli
from spkmc.io.results import ResultManager
from spkmc.visualization.plots import Visualizer


@pytest.fixture
def runner():
    """Fixture for Click's CliRunner."""
    return CliRunner()


@pytest.fixture
def sample_result_data():
    """Sample data for simulation results."""
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


@pytest.fixture
def temp_results_dir(sample_result_data):
    """Create a temporary directory with multiple result files."""
    temp_dir = tempfile.mkdtemp()

    # Create 3 result files with small variations
    for i in range(3):
        result = sample_result_data.copy()
        # Slightly modify data for each scenario
        result["I_val"] = [v * (1 + i * 0.1) for v in result["I_val"]]
        result["metadata"]["scenario"] = f"scenario_{i+1}"

        file_path = os.path.join(temp_dir, f"scenario_{i+1:03d}.json")
        with open(file_path, "w") as f:
            json.dump(result, f)

    yield temp_dir

    # Clean up the directory
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_single_result(sample_result_data):
    """Create a temporary file with a single result."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    with open(path, "w") as f:
        json.dump(sample_result_data, f)

    yield path

    if os.path.exists(path):
        os.remove(path)


def test_plot_single_file(runner, temp_single_result, monkeypatch):
    """Test plotting a single file (original behavior)."""

    # Mock to avoid showing plots
    def mock_plot_result(*args, **kwargs):
        pass

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    result = runner.invoke(cli, ["plot", temp_single_result])

    assert result.exit_code == 0
    assert "Simulation Statistics" in result.output
    assert "Peak infected" in result.output


def test_plot_directory(runner, temp_results_dir, monkeypatch):
    """Test plotting a directory with multiple files."""

    # Mock to avoid showing plots
    def mock_compare_results(*args, **kwargs):
        pass

    monkeypatch.setattr(Visualizer, "compare_results", mock_compare_results)

    result = runner.invoke(cli, ["plot", temp_results_dir])

    assert result.exit_code == 0
    assert "Found 3 JSON files in directory" in result.output
    assert "Generating comparison visualization for 3 scenarios" in result.output


def test_plot_with_states_filter(runner, temp_single_result, monkeypatch):
    """Test plotting with specific state filters."""
    # Mock to capture passed arguments
    plot_calls = []

    def mock_plot_result(s_vals, i_vals, r_vals, time, title, save_path, states_to_plot):
        plot_calls.append({"states_to_plot": states_to_plot})

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Test with only infected
    result = runner.invoke(cli, ["plot", temp_single_result, "--states", "infected"])
    assert result.exit_code == 0
    assert plot_calls[-1]["states_to_plot"] == {"I"}

    # Test with infected and recovered
    result = runner.invoke(
        cli, ["plot", temp_single_result, "--states", "infected", "--states", "recovered"]
    )
    assert result.exit_code == 0
    assert plot_calls[-1]["states_to_plot"] == {"I", "R"}

    # Test with abbreviations
    result = runner.invoke(cli, ["plot", temp_single_result, "--states", "s", "--states", "i"])
    assert result.exit_code == 0
    assert plot_calls[-1]["states_to_plot"] == {"S", "I"}


def test_plot_directory_separate(runner, temp_results_dir, monkeypatch):
    """Test plotting a directory with separate charts."""
    plot_calls = []

    def mock_plot_result(*args, **kwargs):
        plot_calls.append(kwargs.get("save_path") or args[-2] if len(args) > 2 else None)

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    result = runner.invoke(cli, ["plot", temp_results_dir, "--separate", "--output", "test.png"])

    assert result.exit_code == 0
    assert "Processing 3 result files" in result.output
    assert len(plot_calls) == 3
    # Verify output filenames are unique
    assert all("scenario_" in str(path) for path in plot_calls if path)


def test_plot_invalid_state(runner, temp_single_result, monkeypatch):
    """Test plotting with an invalid state."""

    def mock_plot_result(*args, **kwargs):
        pass

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    result = runner.invoke(cli, ["plot", temp_single_result, "--states", "invalid_state"])

    assert result.exit_code == 0
    assert "Invalid state ignored: invalid_state" in result.output


def test_plot_empty_directory(runner):
    """Test plotting an empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(cli, ["plot", temp_dir])

        assert result.exit_code != 0
        assert "No JSON files found in directory" in result.output


def test_plot_nonexistent_path(runner):
    """Test plotting with a nonexistent path."""
    result = runner.invoke(cli, ["plot", "/path/that/does/not/exist"])

    assert result.exit_code != 0
    assert "Path not found" in result.output


def test_plot_with_error_bars(runner, monkeypatch):
    """Test plotting with error bars."""
    # Create a result with error data
    result_with_error = {
        "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
        "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
        "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
        "S_err": [0.001, 0.002, 0.003, 0.004, 0.005],
        "I_err": [0.001, 0.002, 0.003, 0.002, 0.001],
        "R_err": [0.000, 0.001, 0.002, 0.003, 0.004],
        "time": [0.0, 2.5, 5.0, 7.5, 10.0],
        "metadata": {"network_type": "er", "distribution": "gamma", "N": 100},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(result_with_error, f)
        temp_file = f.name

    try:
        plot_calls = []

        def mock_plot_with_error(*args, **kwargs):
            plot_calls.append("with_error")

        monkeypatch.setattr(Visualizer, "plot_result_with_error", mock_plot_with_error)

        result = runner.invoke(cli, ["plot", temp_file, "--with-error"])

        assert result.exit_code == 0
        assert len(plot_calls) == 1
        assert plot_calls[0] == "with_error"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_load_results_from_directory():
    """Test the load_results_from_directory helper."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some JSON files
        for i in range(3):
            data = {
                "S_val": [0.99],
                "I_val": [0.01],
                "R_val": [0.00],
                "time": [0.0],
                "metadata": {"id": i},
            }

            file_path = os.path.join(temp_dir, f"result_{i}.json")
            with open(file_path, "w") as f:
                json.dump(data, f)

        # Create a non-JSON file that should be ignored
        with open(os.path.join(temp_dir, "readme.txt"), "w") as f:
            f.write("This is not a JSON file")

        # Test the function
        results = ResultManager.load_results_from_directory(temp_dir)

        assert len(results) == 3
        assert all(isinstance(r[0], Path) for r in results)
        assert all(isinstance(r[1], dict) for r in results)
        assert all("metadata" in r[1] for r in results)


def test_plot_directory_with_export(runner, temp_results_dir, monkeypatch):
    """Test plotting a directory with additional export."""

    def mock_compare_results(*args, **kwargs):
        pass

    def mock_export(*args, **kwargs):
        return "exported_file.csv"

    monkeypatch.setattr(Visualizer, "compare_results", mock_compare_results)

    # ExportManager is imported at module level; just verify the command runs
    result = runner.invoke(cli, ["plot", temp_results_dir, "--export", "csv"])

    # The command should run without errors; actual export may fail with multiple files
    assert result.exit_code == 0


def test_visualizer_states_filter():
    """Directly test Visualizer functions with state filters."""
    import matplotlib.pyplot as plt

    # Test data
    s_vals = np.array([0.99, 0.95, 0.90, 0.85, 0.80])
    i_vals = np.array([0.01, 0.04, 0.05, 0.05, 0.04])
    r_vals = np.array([0.00, 0.01, 0.05, 0.10, 0.16])
    time = np.array([0.0, 2.5, 5.0, 7.5, 10.0])

    # Test plot_result with different filters
    # We cannot verify visually; just ensure no errors

    # All states
    Visualizer.plot_result(
        s_vals,
        i_vals,
        r_vals,
        time,
        "Test",
        save_path="test_all.png",
        states_to_plot={"S", "I", "R"},
    )
    plt.close()

    # Only infected
    Visualizer.plot_result(
        s_vals, i_vals, r_vals, time, "Test", save_path="test_i.png", states_to_plot={"I"}
    )
    plt.close()

    # Infected and recovered
    Visualizer.plot_result(
        s_vals, i_vals, r_vals, time, "Test", save_path="test_ir.png", states_to_plot={"I", "R"}
    )
    plt.close()

    # Remove test files if created
    for file in ["test_all.png", "test_i.png", "test_ir.png"]:
        if os.path.exists(file):
            os.remove(file)


def test_plot_help(runner):
    """Test plot command help text."""
    result = runner.invoke(cli, ["plot", "--help"])

    assert result.exit_code == 0
    assert "--states" in result.output
    assert "--separate" in result.output
    # Check parts of the message
    assert "Specific states to plot" in result.output
    assert "separate charts" in result.output
