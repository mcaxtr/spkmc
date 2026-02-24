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


# ── delete_experiment ────────────────────────────────────────────────────────


class TestDeleteExperiment:
    """Tests for delete_experiment()."""

    def test_deletes_experiment_directory_and_contents(self, tmp_path):
        """delete_experiment removes the entire experiment directory tree."""
        exp = _make_modern_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None
        assert exp_path.is_dir()

        from spkmc.web.pages.experiment_detail import delete_experiment

        delete_experiment(exp)

        assert not exp_path.exists(), "Experiment directory was not deleted!"

    def test_deletes_result_and_analysis_files(self, tmp_path):
        """delete_experiment removes result, analysis, and experiment-level analysis files."""
        exp = _make_modern_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create scenario result files
        baseline_result = exp_path / "baseline.json"
        baseline_analysis = exp_path / "baseline_analysis.md"
        high_lambda_result = exp_path / "high_lambda.json"
        high_lambda_analysis = exp_path / "high_lambda_analysis.md"
        exp_analysis = exp_path / "analysis.md"

        baseline_result.write_text('{"S_val": [1]}')
        baseline_analysis.write_text("# Baseline Analysis")
        high_lambda_result.write_text('{"S_val": [2]}')
        high_lambda_analysis.write_text("# High Lambda Analysis")
        exp_analysis.write_text("# Experiment Analysis")

        from spkmc.web.pages.experiment_detail import delete_experiment

        delete_experiment(exp)

        assert not exp_path.exists(), "Experiment directory was not deleted!"
        assert not baseline_result.exists()
        assert not baseline_analysis.exists()
        assert not high_lambda_result.exists()
        assert not high_lambda_analysis.exists()
        assert not exp_analysis.exists()

    def test_raises_on_none_path(self):
        """delete_experiment raises ValueError when experiment has no path."""
        exp = Experiment(
            name="No Path",
            scenarios=[
                Scenario(
                    label="X",
                    network="er",
                    distribution="gamma",
                    nodes=100,
                    k_avg=5.0,
                    shape=2.0,
                    scale=1.0,
                    samples=10,
                    initial_perc=0.01,
                    t_max=5.0,
                    steps=50,
                    **{"lambda": 0.5},
                )
            ],
            path=None,
        )

        from spkmc.web.pages.experiment_detail import delete_experiment

        with pytest.raises(ValueError, match="has no path"):
            delete_experiment(exp)

    def test_raises_on_nonexistent_directory(self, tmp_path):
        """delete_experiment raises ValueError when directory does not exist."""
        ghost_path = tmp_path / "experiments" / "nonexistent_exp"
        exp = Experiment(
            name="Ghost",
            scenarios=[
                Scenario(
                    label="X",
                    network="er",
                    distribution="gamma",
                    nodes=100,
                    k_avg=5.0,
                    shape=2.0,
                    scale=1.0,
                    samples=10,
                    initial_perc=0.01,
                    t_max=5.0,
                    steps=50,
                    **{"lambda": 0.5},
                )
            ],
            path=ghost_path,
        )

        from spkmc.web.pages.experiment_detail import delete_experiment

        with pytest.raises(ValueError, match="does not exist"):
            delete_experiment(exp)


# ── update_experiment ────────────────────────────────────────────────────────


class TestUpdateExperiment:
    """Tests for update_experiment()."""

    # -- Shared constants for the modern experiment data.json shape --

    _GLOBAL_PARAMS = {
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
    }

    _ORIGINAL_SCENARIOS = [
        {"label": "Baseline"},
        {"label": "High Lambda", "lambda": 2.0},
    ]

    @staticmethod
    def _make_aligned_experiment(tmp_path: Path) -> Experiment:
        """Create a modern experiment whose directory name matches the normalized name.

        The shared ``_make_modern_experiment`` helper uses ``modern_exp`` as the
        directory name, which does NOT match the normalized form of
        "Modern Experiment" (``modern_experiment``).  Tests that assert no rename
        should use this helper instead so that ``update_experiment`` does not
        detect a spurious directory mismatch.
        """
        exp_path = tmp_path / "experiments" / "modern_experiment"
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

    def test_name_change_updates_data_json_and_renames_directory(self, tmp_path):
        """Changing the name rewrites data.json and renames the experiment directory."""
        exp = _make_modern_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        from spkmc.web.pages.experiment_detail import update_experiment

        new_path = update_experiment(
            experiment=exp,
            new_name="Renamed Experiment",
            new_description="Experiment with global params",
            new_parameters=dict(self._GLOBAL_PARAMS),
            new_scenarios=list(self._ORIGINAL_SCENARIOS),
        )

        assert new_path is not None, "Expected a new path after rename"
        assert new_path.is_dir(), "New directory does not exist"
        assert not exp_path.exists(), "Old directory still exists after rename"

        data = _read_data_json(new_path)
        assert data["name"] == "Renamed Experiment"

    def test_description_only_change_preserves_results(self, tmp_path):
        """Changing only the description preserves all result and analysis files."""
        exp = self._make_aligned_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create result files
        result_file = exp_path / "baseline.json"
        analysis_file = exp_path / "baseline_analysis.md"
        exp_analysis = exp_path / "analysis.md"
        result_file.write_text('{"S_val": [1]}')
        analysis_file.write_text("# Analysis")
        exp_analysis.write_text("# Experiment Analysis")

        from spkmc.web.pages.experiment_detail import update_experiment

        result = update_experiment(
            experiment=exp,
            new_name="Modern Experiment",
            new_description="Updated description",
            new_parameters=dict(self._GLOBAL_PARAMS),
            new_scenarios=list(self._ORIGINAL_SCENARIOS),
        )

        # No rename expected (name unchanged)
        assert result is None

        assert result_file.exists(), "Result file was deleted on description-only change!"
        assert analysis_file.exists(), "Analysis file was deleted on description-only change!"
        assert exp_analysis.exists(), "Experiment analysis was deleted on description-only change!"

        data = _read_data_json(exp_path)
        assert data["description"] == "Updated description"

    def test_global_param_change_invalidates_all_results(self, tmp_path):
        """Changing a global parameter invalidates ALL scenario results and experiment analysis."""
        exp = self._make_aligned_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create result files for both scenarios and experiment analysis
        baseline_result = exp_path / "baseline.json"
        baseline_analysis = exp_path / "baseline_analysis.md"
        hl_result = exp_path / "high_lambda.json"
        hl_analysis = exp_path / "high_lambda_analysis.md"
        exp_analysis = exp_path / "analysis.md"

        baseline_result.write_text('{"S_val": [1]}')
        baseline_analysis.write_text("# Baseline")
        hl_result.write_text('{"S_val": [2]}')
        hl_analysis.write_text("# High Lambda")
        exp_analysis.write_text("# Experiment Analysis")

        from spkmc.web.pages.experiment_detail import update_experiment

        changed_params = dict(self._GLOBAL_PARAMS)
        changed_params["nodes"] = 2000  # Change global param

        update_experiment(
            experiment=exp,
            new_name="Modern Experiment",
            new_description="Experiment with global params",
            new_parameters=changed_params,
            new_scenarios=list(self._ORIGINAL_SCENARIOS),
        )

        assert not baseline_result.exists(), "Baseline result was NOT invalidated!"
        assert not baseline_analysis.exists(), "Baseline analysis was NOT invalidated!"
        assert not hl_result.exists(), "High Lambda result was NOT invalidated!"
        assert not hl_analysis.exists(), "High Lambda analysis was NOT invalidated!"
        assert not exp_analysis.exists(), "Experiment analysis was NOT invalidated!"

    def test_noop_edit_returns_none_and_preserves_everything(self, tmp_path):
        """A no-op edit (nothing changed) returns None and preserves all files."""
        exp = self._make_aligned_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create result files
        baseline_result = exp_path / "baseline.json"
        hl_result = exp_path / "high_lambda.json"
        exp_analysis = exp_path / "analysis.md"

        baseline_result.write_text('{"S_val": [1]}')
        hl_result.write_text('{"S_val": [2]}')
        exp_analysis.write_text("# Experiment Analysis")

        from spkmc.web.pages.experiment_detail import update_experiment

        result = update_experiment(
            experiment=exp,
            new_name="Modern Experiment",
            new_description="Experiment with global params",
            new_parameters=dict(self._GLOBAL_PARAMS),
            new_scenarios=list(self._ORIGINAL_SCENARIOS),
        )

        assert result is None, "No-op edit should return None"
        assert baseline_result.exists(), "Baseline result was deleted on no-op edit!"
        assert hl_result.exists(), "High Lambda result was deleted on no-op edit!"
        assert exp_analysis.exists(), "Experiment analysis was deleted on no-op edit!"

    def test_name_collision_raises_value_error(self, tmp_path):
        """Renaming to a name that collides with an existing directory raises ValueError."""
        exp = self._make_aligned_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create a collision directory at the target path
        collision_path = exp_path.parent / "collision_name"
        collision_path.mkdir(parents=True)

        from spkmc.web.pages.experiment_detail import update_experiment

        with pytest.raises(ValueError, match="already exists"):
            update_experiment(
                experiment=exp,
                new_name="Collision Name",
                new_description="Experiment with global params",
                new_parameters=dict(self._GLOBAL_PARAMS),
                new_scenarios=list(self._ORIGINAL_SCENARIOS),
            )

    def test_empty_name_raises_value_error(self, tmp_path):
        """A name that normalizes to empty string raises ValueError."""
        exp = _make_modern_experiment(tmp_path)

        from spkmc.web.pages.experiment_detail import update_experiment

        with pytest.raises(ValueError, match="normalizes to an empty"):
            update_experiment(
                experiment=exp,
                new_name="!!!",
                new_description="Experiment with global params",
                new_parameters=dict(self._GLOBAL_PARAMS),
                new_scenarios=list(self._ORIGINAL_SCENARIOS),
            )

    def test_directory_rename_produces_correct_normalized_name(self, tmp_path):
        """Directory rename normalizes the new name correctly (lowercase, underscores)."""
        exp = _make_modern_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        from spkmc.web.pages.experiment_detail import update_experiment

        new_path = update_experiment(
            experiment=exp,
            new_name="My Cool Experiment",
            new_description="Experiment with global params",
            new_parameters=dict(self._GLOBAL_PARAMS),
            new_scenarios=list(self._ORIGINAL_SCENARIOS),
        )

        assert new_path is not None
        assert new_path.name == "my_cool_experiment"
        assert new_path.is_dir()

    def test_scenario_removal_deletes_result_files(self, tmp_path):
        """Removing a scenario invalidates only that scenario's result files."""
        exp = self._make_aligned_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create result files for both scenarios
        baseline_result = exp_path / "baseline.json"
        baseline_analysis = exp_path / "baseline_analysis.md"
        hl_result = exp_path / "high_lambda.json"
        hl_analysis = exp_path / "high_lambda_analysis.md"

        baseline_result.write_text('{"S_val": [1]}')
        baseline_analysis.write_text("# Baseline")
        hl_result.write_text('{"S_val": [2]}')
        hl_analysis.write_text("# High Lambda")

        from spkmc.web.pages.experiment_detail import update_experiment

        # Keep only Baseline, remove High Lambda
        update_experiment(
            experiment=exp,
            new_name="Modern Experiment",
            new_description="Experiment with global params",
            new_parameters=dict(self._GLOBAL_PARAMS),
            new_scenarios=[{"label": "Baseline"}],
        )

        # Baseline should be preserved; High Lambda should be deleted
        assert baseline_result.exists(), "Baseline result was deleted when only HL was removed!"
        assert baseline_analysis.exists(), "Baseline analysis was deleted when only HL was removed!"
        assert not hl_result.exists(), "High Lambda result was NOT deleted after removal!"
        assert not hl_analysis.exists(), "High Lambda analysis was NOT deleted after removal!"

    def test_scenario_override_change_invalidates_that_scenario(self, tmp_path):
        """Changing a scenario's overrides invalidates only that scenario's result files."""
        exp = self._make_aligned_experiment(tmp_path)
        exp_path = exp.path
        assert exp_path is not None

        # Create result files for both scenarios
        baseline_result = exp_path / "baseline.json"
        baseline_analysis = exp_path / "baseline_analysis.md"
        hl_result = exp_path / "high_lambda.json"
        hl_analysis = exp_path / "high_lambda_analysis.md"

        baseline_result.write_text('{"S_val": [1]}')
        baseline_analysis.write_text("# Baseline")
        hl_result.write_text('{"S_val": [2]}')
        hl_analysis.write_text("# High Lambda")

        from spkmc.web.pages.experiment_detail import update_experiment

        # Change High Lambda's override from 2.0 to 3.0
        update_experiment(
            experiment=exp,
            new_name="Modern Experiment",
            new_description="Experiment with global params",
            new_parameters=dict(self._GLOBAL_PARAMS),
            new_scenarios=[
                {"label": "Baseline"},
                {"label": "High Lambda", "lambda": 3.0},
            ],
        )

        # Baseline should be preserved; High Lambda should be invalidated
        assert baseline_result.exists(), "Baseline result was deleted on unrelated scenario change!"
        assert baseline_analysis.exists(), "Baseline analysis deleted on unrelated change!"
        assert not hl_result.exists(), "High Lambda result NOT invalidated after override change!"
        assert (
            not hl_analysis.exists()
        ), "High Lambda analysis NOT invalidated after override change!"

    def test_empty_scenario_list_raises_value_error(self, tmp_path):
        """An empty scenario list raises ValueError."""
        exp = _make_modern_experiment(tmp_path)

        from spkmc.web.pages.experiment_detail import update_experiment

        with pytest.raises(ValueError, match="at least one scenario"):
            update_experiment(
                experiment=exp,
                new_name="Modern Experiment",
                new_description="Experiment with global params",
                new_parameters=dict(self._GLOBAL_PARAMS),
                new_scenarios=[],
            )

    def test_duplicate_scenario_labels_raise_value_error(self, tmp_path):
        """Scenario entries whose labels normalize to the same string raise ValueError."""
        exp = _make_modern_experiment(tmp_path)

        from spkmc.web.pages.experiment_detail import update_experiment

        with pytest.raises(ValueError, match="conflict"):
            update_experiment(
                experiment=exp,
                new_name="Modern Experiment",
                new_description="Experiment with global params",
                new_parameters=dict(self._GLOBAL_PARAMS),
                new_scenarios=[
                    {"label": "My Scenario"},
                    {"label": "my-scenario"},  # normalizes to same "my_scenario"
                ],
            )
