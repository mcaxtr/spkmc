"""
Tests for SPKMC models (Scenario, Experiment, etc.)

These tests ensure Pydantic models work correctly, including
class variable access patterns that can break in Pydantic v2.
"""

import tempfile
from pathlib import Path

import pytest

from spkmc.models.experiment import Experiment, ExperimentConfig
from spkmc.models.scenario import Scenario, ScenarioOverride


class TestScenarioClassVariables:
    """Test that class variables are accessible as class attributes."""

    def test_experiments_base_class_access(self):
        """EXPERIMENTS_BASE must be accessible as Scenario.EXPERIMENTS_BASE."""
        # This is the exact pattern that broke - accessing ClassVar as class attr
        assert Scenario.EXPERIMENTS_BASE == "data/experiments"

    def test_runs_base_class_access(self):
        """RUNS_BASE must be accessible as Scenario.RUNS_BASE."""
        assert Scenario.RUNS_BASE == "data/runs"

    def test_class_vars_not_in_model_fields(self):
        """Class variables should NOT appear in model fields."""
        assert "EXPERIMENTS_BASE" not in Scenario.model_fields
        assert "RUNS_BASE" not in Scenario.model_fields


class TestScenarioCreation:
    """Test Scenario model creation and validation."""

    @pytest.fixture
    def valid_gamma_scenario_data(self):
        return {
            "label": "test_scenario",
            "network": "er",
            "distribution": "gamma",
            "nodes": 1000,
            "samples": 50,
            "k_avg": 10.0,
            "shape": 2.0,
            "scale": 0.5,
            "lambda": 0.5,
            "t_max": 10.0,
            "steps": 100,
            "initial_perc": 0.01,
        }

    @pytest.fixture
    def valid_exp_scenario_data(self):
        return {
            "label": "test_exp_scenario",
            "network": "er",
            "distribution": "exponential",
            "nodes": 1000,
            "samples": 50,
            "k_avg": 10.0,
            "mu": 1.0,
            "lambda": 0.5,
            "t_max": 10.0,
            "steps": 100,
            "initial_perc": 0.01,
        }

    def test_create_gamma_scenario(self, valid_gamma_scenario_data):
        """Test creating a valid gamma distribution scenario."""
        scenario = Scenario(**valid_gamma_scenario_data)
        assert scenario.label == "test_scenario"
        assert scenario.network == "er"
        assert scenario.distribution == "gamma"

    def test_create_exponential_scenario(self, valid_exp_scenario_data):
        """Test creating a valid exponential distribution scenario."""
        scenario = Scenario(**valid_exp_scenario_data)
        assert scenario.distribution == "exponential"
        assert scenario.mu == 1.0

    def test_normalized_label(self, valid_gamma_scenario_data):
        """Test label normalization."""
        valid_gamma_scenario_data["label"] = "Test Scenario With Spaces"
        scenario = Scenario(**valid_gamma_scenario_data)
        assert scenario.normalized_label == "test_scenario_with_spaces"

    def test_missing_k_avg_for_er_network(self, valid_gamma_scenario_data):
        """ER network requires k_avg."""
        del valid_gamma_scenario_data["k_avg"]
        with pytest.raises(ValueError, match="k_avg required"):
            Scenario(**valid_gamma_scenario_data)

    def test_missing_shape_for_gamma(self, valid_gamma_scenario_data):
        """Gamma distribution requires shape and scale."""
        del valid_gamma_scenario_data["shape"]
        with pytest.raises(ValueError, match="shape and scale required"):
            Scenario(**valid_gamma_scenario_data)

    def test_total_samples_calculation(self, valid_gamma_scenario_data):
        """Test total_samples returns num_runs * samples."""
        valid_gamma_scenario_data["num_runs"] = 3
        valid_gamma_scenario_data["samples"] = 50
        scenario = Scenario(**valid_gamma_scenario_data)
        assert scenario.total_samples() == 150


class TestScenarioOverride:
    """Test ScenarioOverride for partial scenario definitions."""

    def test_minimal_override(self):
        """Override only needs a label."""
        override = ScenarioOverride(label="test")
        assert override.label == "test"
        assert override.network is None

    def test_override_with_params(self):
        """Override can specify any subset of parameters."""
        override = ScenarioOverride(label="test", nodes=2000, samples=100)
        assert override.nodes == 2000
        assert override.samples == 100
        assert override.network is None


class TestExperimentClassVariables:
    """Test Experiment model class variables."""

    def test_supported_extensions_class_access(self):
        """SUPPORTED_EXTENSIONS must be accessible as class attribute."""
        assert Experiment.SUPPORTED_EXTENSIONS == ["*.json", "*.csv", "*.xlsx", "*.md", "*.html"]

    def test_class_vars_not_in_model_fields(self):
        """Class variables should NOT appear in model fields."""
        assert "SUPPORTED_EXTENSIONS" not in Experiment.model_fields


class TestExperimentResultsDir:
    """Test Experiment.results_dir property - this is where the bug manifested."""

    @pytest.fixture
    def minimal_experiment(self):
        """Create minimal experiment for testing."""
        return Experiment(
            name="Test Experiment",
            scenarios=[{"label": "test", "network": "er", "distribution": "gamma"}],
        )

    def test_results_dir_uses_scenario_experiments_base(self, minimal_experiment):
        """results_dir should use Scenario.EXPERIMENTS_BASE when no custom base."""
        # This is the exact code path that broke
        results_dir = minimal_experiment.results_dir
        # Use Path for cross-platform comparison
        expected_base = Path(Scenario.EXPERIMENTS_BASE)
        assert results_dir.parent == expected_base

    def test_results_dir_always_uses_experiments_base(self):
        """results_dir should always use Scenario.EXPERIMENTS_BASE."""
        exp = Experiment(
            name="Test",
            scenarios=[{"label": "test"}],
        )
        # Results always go to Scenario.EXPERIMENTS_BASE, not a custom path
        assert exp.results_dir == Path(Scenario.EXPERIMENTS_BASE) / "test"

    def test_normalized_name(self, minimal_experiment):
        """Test experiment name normalization."""
        assert minimal_experiment.normalized_name == "test_experiment"

    def test_has_results_false_when_no_dir(self, minimal_experiment):
        """has_results returns False when directory doesn't exist."""
        # This property triggered the original bug
        assert minimal_experiment.has_results is False

    def test_has_results_true_when_files_exist(self, monkeypatch):
        """has_results returns True when result files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Override EXPERIMENTS_BASE for testing
            monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", str(base))

            exp = Experiment(
                name="test",
                scenarios=[{"label": "test"}],
            )
            # Create the results directory and a result file
            exp.ensure_results_dir()
            (exp.results_dir / "results.json").write_text("{}")

            assert exp.has_results is True

    def test_result_count(self, monkeypatch):
        """Test result_count returns correct count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Override EXPERIMENTS_BASE for testing
            monkeypatch.setattr(Scenario, "EXPERIMENTS_BASE", str(base))

            exp = Experiment(
                name="test",
                scenarios=[{"label": "test"}],
            )
            exp.ensure_results_dir()
            (exp.results_dir / "result1.json").write_text("{}")
            (exp.results_dir / "result2.json").write_text("{}")
            (exp.results_dir / "result3.csv").write_text("")

            assert exp.result_count == 3


class TestExperimentConfig:
    """Test ExperimentConfig loading from dict."""

    def test_from_dict_basic(self):
        """Test loading basic config from dict."""
        data = {
            "name": "Test Experiment",
            "parameters": {
                "network": "er",
                "distribution": "gamma",
                "nodes": 1000,
                "samples": 50,
                "k_avg": 10,
                "shape": 2,
                "scale": 0.5,
                "lambda": 0.5,
                "t_max": 10,
                "steps": 100,
                "initial_perc": 0.01,
            },
            "scenarios": [{"label": "baseline"}],
        }
        config = ExperimentConfig.from_dict(data)
        assert config.name == "Test Experiment"
        assert len(config.scenarios) == 1

    def test_from_dict_normalizes_keys(self):
        """Test that time_max -> t_max normalization works."""
        data = {
            "name": "Test",
            "parameters": {"time_max": 20, "time_points": 200},
            "scenarios": [{"label": "test"}],
        }
        config = ExperimentConfig.from_dict(data)
        assert config.parameters["t_max"] == 20
        assert config.parameters["steps"] == 200

    def test_from_dict_filters_comments(self):
        """Test that _comment scenarios are filtered out."""
        data = {
            "name": "Test",
            "parameters": {},
            "scenarios": [
                {"label": "real"},
                {"_comment": "This is a comment"},
                {"label": "also_real"},
            ],
        }
        config = ExperimentConfig.from_dict(data)
        assert len(config.scenarios) == 2
        assert all(s.label in ["real", "also_real"] for s in config.scenarios)


class TestExperimentFromConfig:
    """Test creating Experiment from ExperimentConfig."""

    def test_from_config_merges_parameters(self):
        """Test that global parameters are merged into scenarios."""
        data = {
            "name": "Merge Test",
            "parameters": {
                "network": "er",
                "distribution": "gamma",
                "nodes": 1000,
                "samples": 50,
                "k_avg": 10,
                "shape": 2,
                "scale": 0.5,
                "lambda": 0.5,
                "t_max": 10,
                "steps": 100,
                "initial_perc": 0.01,
            },
            "scenarios": [
                {"label": "small", "nodes": 500},
                {"label": "large", "nodes": 2000},
            ],
        }
        config = ExperimentConfig.from_dict(data)
        experiment = Experiment.from_config(config)

        assert len(experiment.scenarios) == 2
        # Scenarios should have merged parameters
        assert experiment.scenarios[0].nodes == 500  # Overridden
        assert experiment.scenarios[0].samples == 50  # From global
        assert experiment.scenarios[1].nodes == 2000  # Overridden
        assert experiment.scenarios[1].k_avg == 10  # From global
