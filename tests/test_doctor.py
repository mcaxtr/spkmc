"""
Tests for the --doctor flag and legacy key normalization.

Covers:
- ExperimentConfig.from_dict() normalizing legacy keys
- _run_doctor() rewriting data.json files
- Idempotency (running doctor twice produces no changes)
- No-op when data.json has no legacy keys
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from spkmc.cli.commands import LEGACY_KEY_MAPPING, _run_doctor
from spkmc.models.experiment import Experiment, ExperimentConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_params() -> Dict[str, Any]:
    """Return a complete set of valid experiment parameters (current schema)."""
    return {
        "network": "er",
        "distribution": "gamma",
        "nodes": 1000,
        "samples": 50,
        "k_avg": 10,
        "shape": 2.0,
        "scale": 0.5,
        "lambda": 0.5,
        "t_max": 10,
        "steps": 100,
        "initial_perc": 0.01,
    }


def _legacy_params() -> Dict[str, Any]:
    """Return parameters using ALL legacy key names."""
    return {
        "network_type": "er",
        "distribution": "gamma",
        "N": 1000,
        "samples": 50,
        "k_avg": 10,
        "shape": 2.0,
        "scale": 0.5,
        "lambda": 0.5,
        "time_max": 10,
        "time_points": 100,
        "initial_perc": 0.01,
    }


def _write_experiment(
    base_dir: Path,
    name: str,
    parameters: Dict[str, Any],
    scenarios: Any = None,
) -> Path:
    """Write a data.json file inside base_dir/name/."""
    exp_dir = base_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "parameters": parameters,
        "scenarios": scenarios or [{"label": "baseline"}],
    }
    data_file = exp_dir / "data.json"
    data_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data_file


# ---------------------------------------------------------------------------
# ExperimentConfig.from_dict() normalization tests
# ---------------------------------------------------------------------------


class TestKeyNormalization:
    """Test that from_dict() normalizes legacy keys correctly."""

    def test_network_type_to_network(self):
        params = _base_params()
        params["network_type"] = params.pop("network")
        data = {"name": "test", "parameters": params, "scenarios": [{"label": "s1"}]}
        config = ExperimentConfig.from_dict(data)
        assert "network" in config.parameters
        assert "network_type" not in config.parameters

    def test_network_size_to_nodes(self):
        params = _base_params()
        params["network_size"] = params.pop("nodes")
        data = {"name": "test", "parameters": params, "scenarios": [{"label": "s1"}]}
        config = ExperimentConfig.from_dict(data)
        assert config.parameters["nodes"] == 1000
        assert "network_size" not in config.parameters

    def test_N_to_nodes(self):
        params = _base_params()
        params["N"] = params.pop("nodes")
        data = {"name": "test", "parameters": params, "scenarios": [{"label": "s1"}]}
        config = ExperimentConfig.from_dict(data)
        assert config.parameters["nodes"] == 1000
        assert "N" not in config.parameters

    def test_time_max_to_t_max(self):
        params = _base_params()
        params["time_max"] = params.pop("t_max")
        data = {"name": "test", "parameters": params, "scenarios": [{"label": "s1"}]}
        config = ExperimentConfig.from_dict(data)
        assert config.parameters["t_max"] == 10
        assert "time_max" not in config.parameters

    def test_time_points_to_steps(self):
        params = _base_params()
        params["time_points"] = params.pop("steps")
        data = {"name": "test", "parameters": params, "scenarios": [{"label": "s1"}]}
        config = ExperimentConfig.from_dict(data)
        assert config.parameters["steps"] == 100
        assert "time_points" not in config.parameters

    def test_all_legacy_keys_at_once(self):
        data = {
            "name": "legacy_all",
            "parameters": _legacy_params(),
            "scenarios": [{"label": "s1"}],
        }
        config = ExperimentConfig.from_dict(data)
        experiment = Experiment.from_config(config)
        assert len(experiment.scenarios) == 1
        assert experiment.scenarios[0].network == "er"
        assert experiment.scenarios[0].nodes == 1000
        assert experiment.scenarios[0].t_max == 10
        assert experiment.scenarios[0].steps == 100

    def test_legacy_keys_in_scenario_override(self):
        params = _base_params()
        scenarios = [
            {"label": "override", "network_type": "sf", "N": 2000, "exponent": 2.5},
        ]
        data = {"name": "test", "parameters": params, "scenarios": scenarios}
        config = ExperimentConfig.from_dict(data)
        experiment = Experiment.from_config(config)
        assert experiment.scenarios[0].network == "sf"
        assert experiment.scenarios[0].nodes == 2000

    @pytest.mark.parametrize(
        "legacy_value,expected",
        [
            ("erdos_renyi", "er"),
            ("erdos-renyi", "er"),
            ("Erdos-Renyi", "er"),
            ("ER", "er"),
            ("cn", "sf"),
            ("CN", "sf"),
            ("complex_network", "sf"),
            ("complex-network", "sf"),
            ("scale_free", "sf"),
            ("scale-free", "sf"),
            ("Scale-Free", "sf"),
            ("SF", "sf"),
            ("complete_graph", "cg"),
            ("complete", "cg"),
            ("CG", "cg"),
            ("random_regular", "rrn"),
            ("random-regular", "rrn"),
            ("RRN", "rrn"),
        ],
    )
    def test_legacy_network_values_in_parameters(self, legacy_value, expected):
        """Legacy network type values (full names) are normalized to short codes."""
        params = _base_params()
        params["network"] = legacy_value
        data = {"name": "test", "parameters": params, "scenarios": [{"label": "s1"}]}
        config = ExperimentConfig.from_dict(data)
        assert config.parameters["network"] == expected

    def test_legacy_network_values_in_scenarios(self):
        """Legacy network type values in scenario overrides are normalized."""
        params = _base_params()
        scenarios = [
            {"label": "er_full", "network_type": "erdos_renyi"},
            {"label": "sf_full", "network_type": "scale_free", "exponent": 2.5},
            {"label": "cg_full", "network_type": "complete_graph"},
        ]
        data = {"name": "test", "parameters": params, "scenarios": scenarios}
        config = ExperimentConfig.from_dict(data)
        experiment = Experiment.from_config(config)
        assert experiment.scenarios[0].network == "er"
        assert experiment.scenarios[1].network == "sf"
        assert experiment.scenarios[2].network == "cg"


# ---------------------------------------------------------------------------
# _run_doctor() tests
# ---------------------------------------------------------------------------


class TestDoctor:
    """Test the _run_doctor() function."""

    def test_fixes_legacy_keys(self):
        """Doctor rewrites data.json with legacy keys to current schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = _write_experiment(Path(tmpdir), "legacy_exp", _legacy_params())

            _run_doctor(experiments_dir=tmpdir)

            fixed = json.loads(data_file.read_text(encoding="utf-8"))
            params = fixed["parameters"]
            assert "network" in params
            assert "nodes" in params
            assert "t_max" in params
            assert "steps" in params
            # Legacy keys must be gone
            assert "network_type" not in params
            assert "N" not in params
            assert "time_max" not in params
            assert "time_points" not in params

    def test_fixes_legacy_keys_in_scenarios(self):
        """Doctor fixes legacy keys inside individual scenarios too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _base_params()
            scenarios = [
                {"label": "s1", "network_type": "sf", "N": 2000, "exponent": 2.5},
                {"label": "s2", "network_type": "rrn"},
            ]
            data_file = _write_experiment(
                Path(tmpdir), "scenario_legacy", params, scenarios=scenarios
            )

            _run_doctor(experiments_dir=tmpdir)

            fixed = json.loads(data_file.read_text(encoding="utf-8"))
            for s in fixed["scenarios"]:
                assert "network_type" not in s
                assert "N" not in s

    def test_fixes_legacy_network_values(self):
        """Doctor normalizes full network type names to short codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _base_params()
            params["network_type"] = "erdos_renyi"
            scenarios = [
                {"label": "s1"},
                {"label": "sf_scenario", "network_type": "scale_free", "exponent": 2.5},
                {"label": "cg_scenario", "network_type": "complete_graph"},
            ]
            data_file = _write_experiment(Path(tmpdir), "value_legacy", params, scenarios=scenarios)

            _run_doctor(experiments_dir=tmpdir)

            fixed = json.loads(data_file.read_text(encoding="utf-8"))
            assert fixed["parameters"]["network"] == "er"
            assert fixed["scenarios"][1]["network"] == "sf"
            assert fixed["scenarios"][2]["network"] == "cg"

    def test_idempotent(self):
        """Running doctor twice should not change anything the second time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = _write_experiment(Path(tmpdir), "idem_exp", _legacy_params())

            # First run: should fix
            _run_doctor(experiments_dir=tmpdir)
            content_after_first = data_file.read_text(encoding="utf-8")

            # Second run: should be a no-op
            _run_doctor(experiments_dir=tmpdir)
            content_after_second = data_file.read_text(encoding="utf-8")

            assert content_after_first == content_after_second

    def test_no_op_for_current_schema(self):
        """Doctor leaves data.json untouched if it uses current schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = _write_experiment(Path(tmpdir), "current_exp", _base_params())
            original = data_file.read_text(encoding="utf-8")

            _run_doctor(experiments_dir=tmpdir)

            assert data_file.read_text(encoding="utf-8") == original

    def test_handles_missing_directory(self, capsys):
        """Doctor reports error gracefully for missing experiments dir."""
        _run_doctor(experiments_dir="/nonexistent/path")
        # Should not raise - just print an error message

    def test_handles_invalid_json(self):
        """Doctor skips files with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "broken_exp"
            exp_dir.mkdir()
            (exp_dir / "data.json").write_text("{ not valid json }")

            # Should not raise
            _run_doctor(experiments_dir=tmpdir)

    def test_preserves_extra_fields(self):
        """Doctor preserves fields it doesn't know about."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _legacy_params()
            params["custom_field"] = "keep_me"
            data_file = _write_experiment(Path(tmpdir), "extra_fields", params)

            _run_doctor(experiments_dir=tmpdir)

            fixed = json.loads(data_file.read_text(encoding="utf-8"))
            assert fixed["parameters"]["custom_field"] == "keep_me"

    def test_multiple_experiments(self):
        """Doctor processes multiple experiments in one run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _write_experiment(base, "exp_legacy", _legacy_params())
            _write_experiment(base, "exp_current", _base_params())

            _run_doctor(experiments_dir=tmpdir)

            # Legacy should be fixed
            legacy = json.loads((base / "exp_legacy" / "data.json").read_text(encoding="utf-8"))
            assert "network" in legacy["parameters"]
            assert "N" not in legacy["parameters"]

            # Current should be unchanged
            current = json.loads((base / "exp_current" / "data.json").read_text(encoding="utf-8"))
            assert "network" in current["parameters"]
