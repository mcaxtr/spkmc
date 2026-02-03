"""
Tests for experiment results directory handling.

Results should always be stored in the standard location:
- Experiment results: data/experiments/<experiment_name>/
- Individual run results: data/runs/

These paths are defined in Scenario.EXPERIMENTS_BASE and Scenario.RUNS_BASE.
"""

import json
from pathlib import Path

import pytest

from spkmc.io.experiments import ExperimentManager
from spkmc.models.experiment import Experiment
from spkmc.models.scenario import Scenario


class TestExperimentResultsDir:
    """Test that experiments correctly locate their results in the standard directory."""

    def test_results_dir_uses_scenario_experiments_base(self):
        """Results directory always uses Scenario.EXPERIMENTS_BASE."""
        experiment = Experiment(
            name="Test Experiment",
            scenarios=[{"label": "test"}],
        )
        # Results are always stored in Scenario.EXPERIMENTS_BASE
        expected = Path(Scenario.EXPERIMENTS_BASE) / "test_experiment"
        assert experiment.results_dir == expected

    def test_has_results_checks_standard_directory(self, tmp_path, monkeypatch):
        """has_results should check the standard directory (Scenario.EXPERIMENTS_BASE)."""
        # Temporarily override EXPERIMENTS_BASE for testing
        test_base = str(tmp_path / "experiments")
        monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", test_base)

        experiment = Experiment(
            name="Test Experiment",
            scenarios=[{"label": "test"}],
        )

        # Initially no results
        assert experiment.has_results is False

        # Create the results directory and add a file
        results_dir = Path(test_base) / "test_experiment"
        results_dir.mkdir(parents=True)
        (results_dir / "result.json").write_text("{}")

        # Now should have results
        assert experiment.has_results is True


class TestExperimentManagerResultsDir:
    """Test that ExperimentManager uses standard results directory."""

    @pytest.fixture
    def experiments_setup(self, tmp_path, monkeypatch):
        """Create a test experiment structure with standard directory layout."""
        # Override EXPERIMENTS_BASE for testing
        test_base = str(tmp_path / "data" / "experiments")
        monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", test_base)

        # Create experiments directory with a data.json
        exp_dir = tmp_path / "experiments" / "test_exp"
        exp_dir.mkdir(parents=True)

        data = {
            "name": "Test Experiment",
            "parameters": {
                "network": "er",
                "distribution": "gamma",
                "nodes": 100,
                "samples": 10,
                "k_avg": 4,
                "shape": 2,
                "scale": 0.5,
                "lambda": 0.5,
                "t_max": 5,
                "steps": 50,
                "initial_perc": 0.01,
            },
            "scenarios": [{"label": "baseline"}],
        }
        (exp_dir / "data.json").write_text(json.dumps(data))

        # Create results in the standard location (data/experiments/<name>/)
        results_dir = Path(test_base) / "test_experiment"
        results_dir.mkdir(parents=True)
        (results_dir / "result.json").write_text("{}")

        return tmp_path

    def test_loaded_experiment_uses_standard_results_dir(self, experiments_setup, monkeypatch):
        """
        Loaded experiments should use the standard results directory (Scenario.EXPERIMENTS_BASE).
        """
        tmp_path = experiments_setup
        exp_manager = ExperimentManager(str(tmp_path / "experiments"))
        experiments = exp_manager.list_experiments()

        assert len(experiments) == 1
        exp = experiments[0]

        # Results directory should always be Scenario.EXPERIMENTS_BASE / normalized_name
        expected_results_dir = Path(Scenario.EXPERIMENTS_BASE) / exp.normalized_name
        assert exp.results_dir == expected_results_dir

        # Since results are in the standard location, has_results should be True
        assert exp.has_results is True
