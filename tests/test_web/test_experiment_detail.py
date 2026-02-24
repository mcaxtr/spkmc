"""
Tests for experiment_detail page logic.

Covers update_scenario_in_experiment and related functions that manage
scenario editing, label collision detection, and result file lifecycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from spkmc.models.experiment import Experiment
from spkmc.models.scenario import Scenario

# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_data_json(exp_path: Path, data: Dict[str, Any]) -> None:
    """Write a data.json file for an experiment."""
    (exp_path / "data.json").write_text(json.dumps(data, indent=2))


def _read_data_json(exp_path: Path) -> Dict[str, Any]:
    """Read and parse an experiment's data.json."""
    return json.loads((exp_path / "data.json").read_text())


def _make_legacy_experiment(tmp_path: Path) -> Experiment:
    """Create a legacy experiment (no global ``parameters`` block).

    Returns an Experiment whose data.json stores full params in each scenario
    entry — the format used before the web interface introduced global params.
    """
    exp_path = tmp_path / "experiments" / "legacy_exp"
    exp_path.mkdir(parents=True)

    data = {
        "name": "Legacy Experiment",
        "description": "Pre-web-interface experiment",
        "scenarios": [
            {
                "label": "Baseline",
                "network": "er",
                "distribution": "gamma",
                "nodes": 500,
                "k_avg": 5.0,
                "lambda": 0.5,
                "shape": 2.0,
                "scale": 1.0,
                "samples": 10,
                "num_runs": 1,
                "initial_perc": 0.01,
                "t_max": 5.0,
                "steps": 50,
            },
            {
                "label": "High Lambda",
                "network": "er",
                "distribution": "gamma",
                "nodes": 500,
                "k_avg": 5.0,
                "lambda": 2.0,
                "shape": 2.0,
                "scale": 1.0,
                "samples": 10,
                "num_runs": 1,
                "initial_perc": 0.01,
                "t_max": 5.0,
                "steps": 50,
            },
        ],
    }
    _write_data_json(exp_path, data)

    scenarios = [
        Scenario(
            label="Baseline",
            network="er",
            distribution="gamma",
            nodes=500,
            k_avg=5.0,
            shape=2.0,
            scale=1.0,
            samples=10,
            initial_perc=0.01,
            t_max=5.0,
            steps=50,
            **{"lambda": 0.5},
        ),
        Scenario(
            label="High Lambda",
            network="er",
            distribution="gamma",
            nodes=500,
            k_avg=5.0,
            shape=2.0,
            scale=1.0,
            samples=10,
            initial_perc=0.01,
            t_max=5.0,
            steps=50,
            **{"lambda": 2.0},
        ),
    ]

    return Experiment(name="Legacy Experiment", scenarios=scenarios, path=exp_path)


def _make_modern_experiment(tmp_path: Path) -> Experiment:
    """Create a modern experiment with a global ``parameters`` block."""
    exp_path = tmp_path / "experiments" / "modern_exp"
    exp_path.mkdir(parents=True)

    data = {
        "name": "Modern Experiment",
        "description": "Experiment with global params",
        "parameters": {
            "network": "er",
            "distribution": "gamma",
            "nodes": 1000,
            "k_avg": 10.0,
            "lambda": 0.5,
            "shape": 2.0,
            "scale": 1.0,
            "samples": 50,
            "num_runs": 1,
            "initial_perc": 0.01,
            "t_max": 10.0,
            "steps": 100,
        },
        "scenarios": [
            {"label": "Baseline"},
            {"label": "High Lambda", "lambda": 2.0},
        ],
    }
    _write_data_json(exp_path, data)

    scenarios = [
        Scenario(
            label="Baseline",
            network="er",
            distribution="gamma",
            nodes=1000,
            k_avg=10.0,
            shape=2.0,
            scale=1.0,
            samples=50,
            initial_perc=0.01,
            t_max=10.0,
            steps=100,
            **{"lambda": 0.5},
        ),
        Scenario(
            label="High Lambda",
            network="er",
            distribution="gamma",
            nodes=1000,
            k_avg=10.0,
            shape=2.0,
            scale=1.0,
            samples=50,
            initial_perc=0.01,
            t_max=10.0,
            steps=100,
            **{"lambda": 2.0},
        ),
    ]

    return Experiment(
        name="Modern Experiment",
        scenarios=scenarios,
        path=exp_path,
        parameters=data["parameters"],
    )


# ── update_scenario_in_experiment ────────────────────────────────────────────


class TestUpdateScenarioInExperiment:
    """Tests for update_scenario_in_experiment()."""

    def test_noop_edit_on_legacy_experiment_preserves_results(self, tmp_path):
        """P1 regression: a no-op edit on a legacy experiment must NOT delete result files."""
        exp = _make_legacy_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create result and analysis files that should be preserved
        result_file = exp_path / "baseline.json"
        analysis_file = exp_path / "baseline_analysis.md"
        result_file.write_text('{"S_val": [1]}')
        analysis_file.write_text("# Analysis")

        from spkmc.web.pages.experiment_detail import update_scenario_in_experiment

        # Simulate a no-op edit: same label, empty overrides (matching hardcoded defaults).
        # For legacy experiments the form produces override_params containing only
        # values that differ from hardcoded defaults — NOT all stored params.
        # A no-op edit where some stored params happen to match defaults yields
        # a sparse override_params dict.
        update_scenario_in_experiment(
            experiment=exp,
            original_label="Baseline",
            new_label="Baseline",
            override_params={
                # Only include params that differ from hardcoded defaults.
                # For legacy scenarios these are the values the form would emit.
                "nodes": 500,  # differs from hardcoded default of 1000
                "k_avg": 5.0,  # differs from hardcoded default of 10.0
                "t_max": 5.0,  # differs from hardcoded default of 10.0
                "steps": 50,  # differs from hardcoded default of 100
                "samples": 10,  # differs from hardcoded default of 50
            },
        )

        # Result files must still exist (no-op edit should not delete them)
        assert result_file.exists(), "Result file was deleted by a no-op edit!"
        assert analysis_file.exists(), "Analysis file was deleted by a no-op edit!"

    def test_noop_edit_on_modern_experiment_preserves_results(self, tmp_path):
        """Modern experiments: no-op edit must NOT delete result files."""
        exp = _make_modern_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        result_file = exp_path / "high_lambda.json"
        analysis_file = exp_path / "high_lambda_analysis.md"
        result_file.write_text('{"S_val": [1]}')
        analysis_file.write_text("# Analysis")

        from spkmc.web.pages.experiment_detail import update_scenario_in_experiment

        # The override is the same as the existing one (lambda: 2.0)
        update_scenario_in_experiment(
            experiment=exp,
            original_label="High Lambda",
            new_label="High Lambda",
            override_params={"lambda": 2.0},
        )

        assert result_file.exists(), "Result file was deleted by a no-op edit!"
        assert analysis_file.exists(), "Analysis file was deleted by a no-op edit!"

    def test_actual_edit_deletes_stale_results(self, tmp_path):
        """When params actually change, stale result files must be deleted."""
        exp = _make_modern_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        result_file = exp_path / "high_lambda.json"
        analysis_file = exp_path / "high_lambda_analysis.md"
        result_file.write_text('{"S_val": [1]}')
        analysis_file.write_text("# Analysis")

        from spkmc.web.pages.experiment_detail import update_scenario_in_experiment

        # Change lambda from 2.0 to 3.0
        update_scenario_in_experiment(
            experiment=exp,
            original_label="High Lambda",
            new_label="High Lambda",
            override_params={"lambda": 3.0},
        )

        assert not result_file.exists(), "Result file was NOT deleted after param change!"
        assert not analysis_file.exists(), "Analysis file was NOT deleted after param change!"

    def test_label_rename_deletes_old_results(self, tmp_path):
        """Renaming a scenario must delete the old result files."""
        exp = _make_modern_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        old_result = exp_path / "high_lambda.json"
        old_analysis = exp_path / "high_lambda_analysis.md"
        old_result.write_text('{"S_val": [1]}')
        old_analysis.write_text("# Analysis")

        from spkmc.web.pages.experiment_detail import update_scenario_in_experiment

        update_scenario_in_experiment(
            experiment=exp,
            original_label="High Lambda",
            new_label="Very High Lambda",
            override_params={"lambda": 2.0},
        )

        assert not old_result.exists(), "Old result file was NOT deleted after rename!"
        assert not old_analysis.exists(), "Old analysis file was NOT deleted after rename!"

    def test_label_collision_raises_error(self, tmp_path):
        """Renaming to an existing scenario's normalized label must raise ValueError."""
        exp = _make_modern_experiment(tmp_path)

        from spkmc.web.pages.experiment_detail import update_scenario_in_experiment

        with pytest.raises(ValueError, match="conflicting name"):
            update_scenario_in_experiment(
                experiment=exp,
                original_label="High Lambda",
                new_label="Baseline",
                override_params={},
            )

    def test_empty_label_raises_error(self, tmp_path):
        """A label that normalizes to empty string must raise ValueError."""
        exp = _make_modern_experiment(tmp_path)

        from spkmc.web.pages.experiment_detail import update_scenario_in_experiment

        with pytest.raises(ValueError, match="normalizes to an empty"):
            update_scenario_in_experiment(
                experiment=exp,
                original_label="High Lambda",
                new_label="!!!",
                override_params={},
            )

    def test_legacy_actual_edit_deletes_stale_results(self, tmp_path):
        """Legacy experiment: actual param change must delete stale results."""
        exp = _make_legacy_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        result_file = exp_path / "baseline.json"
        analysis_file = exp_path / "baseline_analysis.md"
        result_file.write_text('{"S_val": [1]}')
        analysis_file.write_text("# Analysis")

        from spkmc.web.pages.experiment_detail import update_scenario_in_experiment

        # Change nodes from 500 to 600 (an actual parameter change)
        update_scenario_in_experiment(
            experiment=exp,
            original_label="Baseline",
            new_label="Baseline",
            override_params={
                "nodes": 600,  # CHANGED from 500
                "k_avg": 5.0,
                "t_max": 5.0,
                "steps": 50,
                "samples": 10,
            },
        )

        assert not result_file.exists(), "Result file was NOT deleted after param change!"
        assert not analysis_file.exists(), "Analysis file was NOT deleted after param change!"
