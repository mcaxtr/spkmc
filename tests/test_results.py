"""
Tests for the SPKMC results module.

This module contains tests for the ResultManager class and its functionality.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from spkmc.core.distributions import ExponentialDistribution, GammaDistribution
from spkmc.io.results import ResultManager


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
            "network_type": "er",
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


def test_get_result_path_er(gamma_distribution):
    """Test result path generation for an Erdos-Renyi network."""
    path = ResultManager.get_result_path("er", gamma_distribution, 1000, 50)

    assert isinstance(path, str)
    assert "er" in path.lower()
    assert "gamma" in path.lower()
    assert "1000" in path
    assert "50" in path
    assert path.endswith(".json")


def test_get_result_path_cn(gamma_distribution):
    """Test result path generation for a complex network."""
    path = ResultManager.get_result_path("cn", gamma_distribution, 1000, 50, 2.5)

    assert isinstance(path, str)
    assert "cn" in path.lower()
    assert "gamma" in path.lower()
    assert "1000" in path
    assert "50" in path
    assert "2.5" in path or "25" in path  # May be formatted as 2.5 or 25
    assert path.endswith(".json")


def test_load_result(temp_result_file, sample_result):
    """Test loading results."""
    result = ResultManager.load_result(temp_result_file)

    assert isinstance(result, dict)
    assert "S_val" in result
    assert "I_val" in result
    assert "R_val" in result
    assert "time" in result
    assert "metadata" in result
    assert result["metadata"]["network_type"] == sample_result["metadata"]["network_type"]
    assert result["metadata"]["distribution"] == sample_result["metadata"]["distribution"]
    assert result["metadata"]["N"] == sample_result["metadata"]["N"]


def test_load_result_nonexistent():
    """Test loading a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        ResultManager.load_result("nonexistent_file.json")


def test_save_result(sample_result):
    """Test saving results."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        # Save the results
        ResultManager.save_result(path, sample_result)

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


def test_list_results(monkeypatch):
    """Test listing results."""

    # Mock Path.exists
    def mock_exists(self):
        return True

    # Mock Path.iterdir
    def mock_iterdir(self):
        paths = [Path("data/spkmc/gamma"), Path("data/spkmc/exponential")]
        for path in paths:
            yield path

    # Mock the second iterdir level
    def mock_iterdir_level2(self):
        if str(self).endswith("gamma"):
            paths = [Path("data/spkmc/gamma/er"), Path("data/spkmc/gamma/cn")]
        else:
            paths = [Path("data/spkmc/exponential/er")]
        for path in paths:
            yield path

    # Mock the third iterdir level
    def mock_iterdir_level3(self):
        if str(self).endswith("er"):
            paths = [
                Path("data/spkmc/gamma/er/results_1000_50_2.0.json"),
                Path("data/spkmc/gamma/er/results_2000_100_2.0.json"),
            ]
        else:
            paths = [Path("data/spkmc/gamma/cn/results_25_1000_50_2.0.json")]
        for path in paths:
            yield path

    # Mock Path.glob
    def mock_glob(self, pattern):
        if str(self).endswith("er"):
            paths = [
                Path("data/spkmc/gamma/er/results_1000_50_2.0.json"),
                Path("data/spkmc/gamma/er/results_2000_100_2.0.json"),
            ]
        else:
            paths = [Path("data/spkmc/gamma/cn/results_25_1000_50_2.0.json")]
        for path in paths:
            yield path

    # Apply mocks
    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "iterdir", mock_iterdir)
    monkeypatch.setattr(Path, "glob", mock_glob)

    # Replace iterdir for different instances
    def patched_iterdir(self):
        if str(self).endswith("spkmc"):
            return mock_iterdir(self)
        elif str(self).endswith("gamma") or str(self).endswith("exponential"):
            return mock_iterdir_level2(self)
        else:
            return mock_iterdir_level3(self)

    monkeypatch.setattr(Path, "iterdir", patched_iterdir)

    # Test listing results
    results = ResultManager.list_results()

    assert isinstance(results, list)
    assert len(results) > 0
    # Check for path components (works on both Unix and Windows)
    assert any("gamma" in r and "er" in r for r in results)
    assert any("gamma" in r and "cn" in r for r in results)


def test_get_metadata_from_path():
    """Test extracting metadata from a file path."""
    path = "data/spkmc/gamma/er/results_1000_50_2.0.json"
    metadata = ResultManager.get_metadata_from_path(path)

    assert isinstance(metadata, dict)
    assert metadata["distribution"] == "gamma"
    assert metadata["network_type"] == "er"
    assert metadata["N"] == 1000
    assert metadata["samples"] == 50


def test_get_metadata_from_path_cn():
    """Test extracting metadata from a file path for a complex network."""
    path = "data/spkmc/gamma/cn/results_25_1000_50_2.0.json"
    metadata = ResultManager.get_metadata_from_path(path)

    assert isinstance(metadata, dict)
    assert metadata["distribution"] == "gamma"
    assert metadata["network_type"] == "cn"
    assert metadata["exponent"] == 2.5
    assert metadata["N"] == 1000
    assert metadata["samples"] == 50


def test_format_result_for_cli(sample_result):
    """Test formatting results for the CLI."""
    formatted = ResultManager.format_result_for_cli(sample_result)

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
