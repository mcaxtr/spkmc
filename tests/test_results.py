"""
Tests for the SPKMC results/data management module.

This module contains tests for the DataManager class and its functionality.
"""

import json
import os
import tempfile

import pytest

from spkmc.core.distributions import ExponentialDistribution, GammaDistribution
from spkmc.io.data_manager import DataManager
from spkmc.io.experiments import Scenario


@pytest.fixture
def gamma_distribution():
    """Fixture for a Gamma distribution."""
    return GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)


@pytest.fixture
def exponential_distribution():
    """Fixture for an Exponential distribution."""
    return ExponentialDistribution(mu=1.0, lmbd=1.0)


@pytest.fixture
def sample_result():
    """Fixture for a sample result."""
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


@pytest.fixture
def temp_result_file(sample_result):
    """Fixture to create a temporary results file."""
    # Create a temporary file in text mode for JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        # Save data to the file
        json.dump(sample_result, f)

        # Return the file path
        path = f.name

    yield path

    # Remove the file after the test
    if os.path.exists(path):
        os.remove(path)


def test_scenario_get_run_path():
    """Test run path generation using Scenario."""
    scenario = Scenario(
        label="test_scenario",
        network="er",
        distribution="gamma",
        nodes=1000,
        samples=50,
        k_avg=10.0,
        shape=2.0,
        scale=1.0,
        t_max=20.0,
        steps=100,
        initial_perc=0.01,
        **{"lambda": 1.0},
    )
    path = scenario.get_run_path()

    assert isinstance(path, str)
    assert "er" in path.lower()
    assert "gamma" in path.lower()
    assert "1000" in path
    assert "50" in path
    assert path.endswith(".json")


def test_scenario_get_run_path_sf():
    """Test run path generation for a scale-free network."""
    scenario = Scenario(
        label="test_sf_scenario",
        network="sf",
        distribution="gamma",
        nodes=1000,
        samples=50,
        k_avg=10.0,
        exponent=2.5,
        shape=2.0,
        scale=1.0,
        t_max=20.0,
        steps=100,
        initial_perc=0.01,
        **{"lambda": 1.0},
    )
    path = scenario.get_run_path()

    assert isinstance(path, str)
    assert "sf" in path.lower()
    assert "gamma" in path.lower()
    assert "1000" in path
    assert "50" in path
    assert "exp" in path  # Exponent in filename
    assert path.endswith(".json")


def test_load_result(temp_result_file, sample_result):
    """Test loading results."""
    result = DataManager.load(temp_result_file)

    assert isinstance(result, dict)
    assert "S_val" in result
    assert "I_val" in result
    assert "R_val" in result
    assert "time" in result
    assert "metadata" in result
    assert result["metadata"]["network"] == sample_result["metadata"]["network"]
    assert result["metadata"]["distribution"] == sample_result["metadata"]["distribution"]
    assert result["metadata"]["N"] == sample_result["metadata"]["N"]


def test_load_result_nonexistent():
    """Test loading a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        DataManager.load("nonexistent_file.json")


def test_save_result(sample_result):
    """Test saving results."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        # Save the results
        DataManager.save(sample_result, path)

        # Verify the file was created
        assert os.path.exists(path)

        # Load the results and verify
        with open(path, "r") as f:
            loaded_result = json.load(f)

        assert loaded_result["S_val"] == sample_result["S_val"]
        assert loaded_result["I_val"] == sample_result["I_val"]
        assert loaded_result["R_val"] == sample_result["R_val"]
        assert loaded_result["time"] == sample_result["time"]
        assert loaded_result["metadata"] == sample_result["metadata"]
    finally:
        # Remove the file
        if os.path.exists(path):
            os.remove(path)


def test_list_files():
    """Test listing files in a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        for i in range(3):
            path = os.path.join(temp_dir, f"result_{i}.json")
            with open(path, "w") as f:
                json.dump({"test": i}, f)

        # Test listing
        files = DataManager.list_files(temp_dir, "*.json")
        assert len(files) == 3


def test_exists():
    """Test file existence check."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        assert DataManager.exists(path) is True
        assert DataManager.exists("nonexistent_file.json") is False
    finally:
        os.remove(path)


def test_format_result_for_cli(sample_result):
    """Test formatting results for the CLI."""
    formatted = DataManager.format_result_for_cli(sample_result)

    assert isinstance(formatted, dict)
    assert "metadata" in formatted
    assert "max_infected" in formatted
    assert "final_recovered" in formatted
    assert "data_points" in formatted
    assert "has_error_data" in formatted
    assert formatted["max_infected"] == 0.05
    assert formatted["final_recovered"] == 0.16
    assert formatted["data_points"] == 5
    assert formatted["has_error_data"] is False


def test_load_result_from_csv(sample_result):
    """Test loading results from a CSV file."""
    import pandas as pd

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Create CSV with expected columns
        df = pd.DataFrame(
            {
                "Time": sample_result["time"],
                "Susceptible": sample_result["S_val"],
                "Infected": sample_result["I_val"],
                "Recovered": sample_result["R_val"],
            }
        )
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        loaded = DataManager.load(temp_path)

        assert "S_val" in loaded
        assert "I_val" in loaded
        assert "R_val" in loaded
        assert "time" in loaded
        assert loaded["S_val"] == sample_result["S_val"]
        assert loaded["I_val"] == sample_result["I_val"]
        assert loaded["R_val"] == sample_result["R_val"]
        assert loaded["time"] == sample_result["time"]
    finally:
        os.unlink(temp_path)


def test_load_result_from_csv_with_errors():
    """Test loading results from a CSV file with error columns."""
    import pandas as pd

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0],
                "Susceptible": [0.99, 0.95, 0.90],
                "Infected": [0.01, 0.04, 0.05],
                "Recovered": [0.00, 0.01, 0.05],
                "Susceptible_Error": [0.01, 0.02, 0.03],
                "Infected_Error": [0.005, 0.01, 0.015],
                "Recovered_Error": [0.0, 0.005, 0.01],
            }
        )
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        loaded = DataManager.load(temp_path)

        assert loaded.get("has_error", "S_err" in loaded) is True or "S_err" in loaded
        assert "S_err" in loaded
        assert "I_err" in loaded
        assert "R_err" in loaded
    finally:
        os.unlink(temp_path)


def test_load_result_from_excel(sample_result):
    """Test loading results from an Excel file."""
    import pandas as pd

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        temp_path = f.name

    try:
        # Create Excel with Data and Metadata sheets
        df_data = pd.DataFrame(
            {
                "Time": sample_result["time"],
                "Susceptible": sample_result["S_val"],
                "Infected": sample_result["I_val"],
                "Recovered": sample_result["R_val"],
            }
        )
        df_metadata = pd.DataFrame(
            {
                "Parameter": ["network", "distribution", "N"],
                "Value": ["er", "gamma", 100],
            }
        )

        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            df_data.to_excel(writer, sheet_name="Data", index=False)
            df_metadata.to_excel(writer, sheet_name="Metadata", index=False)

        loaded = DataManager.load(temp_path)

        assert "S_val" in loaded
        assert "I_val" in loaded
        assert "R_val" in loaded
        assert "time" in loaded
        assert loaded["S_val"] == sample_result["S_val"]
        assert loaded["metadata"]["network"] == "er"
        assert loaded["metadata"]["distribution"] == "gamma"
    finally:
        os.unlink(temp_path)


def test_load_auto_detects_format(sample_result, temp_result_file):
    """Test that load() auto-detects the file format."""
    # Test JSON
    loaded = DataManager.load(temp_result_file)
    assert "S_val" in loaded
    assert loaded["S_val"] == sample_result["S_val"]


def test_load_unsupported_format():
    """Test that load() raises an error for unsupported formats."""
    import tempfile

    # Create a temp file with unsupported extension
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"test content")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported format"):
            DataManager.load(temp_path)
    finally:
        os.unlink(temp_path)
