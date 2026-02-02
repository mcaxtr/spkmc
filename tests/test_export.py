"""
Tests for the SPKMC data export functionality.

This module contains tests for the DataManager class export functionality.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from spkmc.io.data_manager import DataManager


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
def sample_result_with_error():
    """Fixture for a sample result with error data."""
    return {
        "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
        "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
        "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
        "S_err": [0.001, 0.002, 0.003, 0.004, 0.005],
        "I_err": [0.001, 0.002, 0.003, 0.002, 0.001],
        "R_err": [0.000, 0.001, 0.002, 0.003, 0.004],
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


def test_export_to_csv(sample_result):
    """Test CSV export."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    try:
        # Export the results using DataManager
        exported_path = DataManager.save(sample_result, path)

        # Verify the file was created
        assert os.path.exists(exported_path)
        assert exported_path == path

        # Load the CSV and verify
        df = pd.read_csv(path)

        assert "Time" in df.columns
        assert "Susceptible" in df.columns
        assert "Infected" in df.columns
        assert "Recovered" in df.columns
        assert len(df) == len(sample_result["time"])
        assert df["Time"].tolist() == sample_result["time"]
        assert df["Susceptible"].tolist() == sample_result["S_val"]
        assert df["Infected"].tolist() == sample_result["I_val"]
        assert df["Recovered"].tolist() == sample_result["R_val"]
    finally:
        # Remove the file
        if os.path.exists(path):
            os.remove(path)


def test_export_to_csv_with_error(sample_result_with_error):
    """Test CSV export with error data."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    try:
        # Export the results using DataManager
        exported_path = DataManager.save(sample_result_with_error, path)

        # Verify the file was created
        assert os.path.exists(exported_path)

        # Load the CSV and verify
        df = pd.read_csv(path)

        assert "Time" in df.columns
        assert "Susceptible" in df.columns
        assert "Infected" in df.columns
        assert "Recovered" in df.columns
        assert "Susceptible_Error" in df.columns
        assert "Infected_Error" in df.columns
        assert "Recovered_Error" in df.columns
        assert len(df) == len(sample_result_with_error["time"])
        assert df["Susceptible_Error"].tolist() == sample_result_with_error["S_err"]
        assert df["Infected_Error"].tolist() == sample_result_with_error["I_err"]
        assert df["Recovered_Error"].tolist() == sample_result_with_error["R_err"]
    finally:
        # Remove the file
        if os.path.exists(path):
            os.remove(path)


def test_export_to_excel(sample_result):
    """Test Excel export."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name

    try:
        # Export the results using DataManager
        exported_path = DataManager.save(sample_result, path)

        # Verify the file was created
        assert os.path.exists(exported_path)

        # Load the Excel file and verify
        with pd.ExcelFile(path) as xls:
            # Verify sheets
            assert "Data" in xls.sheet_names
            assert "Metadata" in xls.sheet_names

            # Verify data sheet
            df_data = pd.read_excel(xls, "Data")
            assert "Time" in df_data.columns
            assert "Susceptible" in df_data.columns
            assert "Infected" in df_data.columns
            assert "Recovered" in df_data.columns
            assert len(df_data) == len(sample_result["time"])

            # Verify metadata sheet
            df_metadata = pd.read_excel(xls, "Metadata")
            assert "Parameter" in df_metadata.columns
            assert "Value" in df_metadata.columns
            assert len(df_metadata) == len(sample_result["metadata"])
    finally:
        # Remove the file
        if os.path.exists(path):
            os.remove(path)


def test_export_to_json(sample_result):
    """Test JSON export."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        # Export the results using DataManager
        exported_path = DataManager.save(sample_result, path)

        # Verify the file was created
        assert os.path.exists(exported_path)

        # Load the JSON and verify
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


def test_export_to_markdown(sample_result, monkeypatch):
    """Test Markdown export."""

    # Mock Visualizer.plot_result to avoid generating plots
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply the mock
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = f.name

    try:
        # Export the results using DataManager
        exported_path = DataManager.save(sample_result, path)

        # Verify the file was created
        assert os.path.exists(exported_path)

        # Read file content
        with open(path, "r") as f:
            content = f.read()

        # Verify content
        assert "# SPKMC Simulation Report" in content
        assert "## Simulation Parameters" in content
        assert "## Statistics" in content
        assert "## Simulation Data" in content
        assert "| Time | Susceptible | Infected | Recovered |" in content
        assert "Network Type | ER" in content
        assert "Distribution | Gamma" in content
        assert "Number of Nodes (N) | 100" in content
    finally:
        # Remove the file
        if os.path.exists(path):
            os.remove(path)


def test_export_to_html(sample_result, monkeypatch):
    """Test HTML export."""

    # Mock Visualizer.plot_result to avoid generating plots
    def mock_plot_result(*args, **kwargs):
        pass

    # Apply the mock
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        # Export the results using DataManager
        exported_path = DataManager.save(sample_result, path)

        # Verify the file was created
        assert os.path.exists(exported_path)

        # Read file content
        with open(path, "r") as f:
            content = f.read()

        # Verify content
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "SPKMC Simulation Report" in content
        assert "Simulation Parameters" in content
        assert "Network Type" in content
        assert "ER" in content
    finally:
        # Remove the file
        if os.path.exists(path):
            os.remove(path)


def test_export_invalid_format(sample_result):
    """Test export with an invalid format."""
    with pytest.raises(ValueError, match="Unsupported format"):
        DataManager.save(sample_result, "output.xyz")


def test_export_valid_formats(sample_result, monkeypatch):
    """Test export with valid formats."""
    import tempfile

    # Mock Visualizer.plot_result to avoid generating plots
    def mock_plot_result(*args, **kwargs):
        pass

    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot_result)

    # Test each valid format
    formats = [
        (".json", "json"),
        (".csv", "csv"),
        (".xlsx", "excel"),
        (".md", "markdown"),
        (".html", "html"),
    ]

    for ext, _fmt_name in formats:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            path = f.name

        try:
            result = DataManager.save(sample_result, path)
            assert result == path
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
