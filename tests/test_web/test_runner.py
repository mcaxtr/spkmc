"""
Tests for SimulationRunner.

Tests cover file-based status management, completion detection, progress
reading, and cleanup. Subprocess execution is NOT tested — that belongs to
integration tests. The runner fixture bypasses __init__ so no real
.spkmc_web/ directory is created during the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def runner(tmp_path):
    """SimulationRunner with status_dir isolated to tmp_path."""
    from spkmc.web.runner import SimulationRunner

    r = SimulationRunner.__new__(SimulationRunner)
    r.status_dir = tmp_path / "status"
    r.status_dir.mkdir()
    r._processes = {}
    return r


@pytest.fixture()
def minimal_scenario():
    from spkmc.models.scenario import Scenario

    return Scenario(
        label="Baseline",
        network="er",
        distribution="gamma",
        nodes=1000,
        samples=50,
        k_avg=10.0,
        **{"lambda": 0.5},
        shape=2.0,
        scale=1.0,
        t_max=10.0,
        steps=100,
        initial_perc=0.01,
    )


@pytest.fixture()
def minimal_experiment(tmp_path, minimal_scenario):
    from spkmc.models.experiment import Experiment

    exp_path = tmp_path / "experiments" / "test_experiment"
    exp_path.mkdir(parents=True)
    return Experiment(
        name="Test Experiment",
        scenarios=[minimal_scenario],
        path=exp_path,
    )


# ── get_status ────────────────────────────────────────────────────────────────


class TestGetStatus:
    def test_returns_none_for_missing_run_id(self, runner):
        assert runner.get_status("nonexistent_run") is None

    def test_returns_parsed_dict_for_valid_status_file(self, runner):
        data = {"run_id": "run_1", "status": "running", "progress": 5, "total": 100}
        (runner.status_dir / "run_1.json").write_text(json.dumps(data))

        result = runner.get_status("run_1")
        assert result == data

    def test_returns_none_for_corrupted_json(self, runner):
        (runner.status_dir / "bad.json").write_text("{not: valid}")
        assert runner.get_status("bad") is None

    def test_returns_none_for_empty_file(self, runner):
        (runner.status_dir / "empty.json").write_text("")
        assert runner.get_status("empty") is None


# ── is_running ────────────────────────────────────────────────────────────────


class TestIsRunning:
    def test_returns_true_when_status_is_running(self, runner):
        (runner.status_dir / "r1.json").write_text(
            json.dumps({"run_id": "r1", "status": "running"})
        )
        assert runner.is_running("r1") is True

    def test_returns_false_when_status_is_completed(self, runner):
        (runner.status_dir / "r1.json").write_text(
            json.dumps({"run_id": "r1", "status": "completed"})
        )
        assert runner.is_running("r1") is False

    def test_returns_false_when_status_is_failed(self, runner):
        (runner.status_dir / "r1.json").write_text(json.dumps({"run_id": "r1", "status": "failed"}))
        assert runner.is_running("r1") is False

    def test_returns_false_for_nonexistent_run(self, runner):
        assert runner.is_running("ghost") is False


# ── check_completion ──────────────────────────────────────────────────────────


class TestCheckCompletion:
    def test_returns_true_when_result_file_exists(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "baseline.json").touch()

        assert runner.check_completion("my_exp", "Baseline") is True

    def test_returns_false_when_result_file_is_missing(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)

        assert runner.check_completion("my_exp", "Baseline") is False

    def test_label_is_normalized_before_checking(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)
        # "High Risk Scenario" normalizes to "high_risk_scenario"
        (exp_dir / "high_risk_scenario.json").touch()

        assert runner.check_completion("exp1", "High Risk Scenario") is True


# ── cleanup_status ────────────────────────────────────────────────────────────


class TestCleanupStatus:
    def test_removes_status_json_and_script_files(self, runner):
        run_id = "run_cleanup"
        (runner.status_dir / f"{run_id}.json").write_text("{}")
        (runner.status_dir / f"{run_id}_script.py").write_text("pass")

        runner.cleanup_status(run_id)

        assert not (runner.status_dir / f"{run_id}.json").exists()
        assert not (runner.status_dir / f"{run_id}_script.py").exists()

    def test_cleanup_is_idempotent_when_files_already_absent(self, runner):
        # Must not raise when called with a run_id that has no files
        runner.cleanup_status("nonexistent_run")
        runner.cleanup_status("nonexistent_run")

    def test_cleanup_handles_missing_script_file_gracefully(self, runner):
        run_id = "partial"
        (runner.status_dir / f"{run_id}.json").write_text("{}")
        # No script file

        runner.cleanup_status(run_id)
        assert not (runner.status_dir / f"{run_id}.json").exists()


# ── get_progress ──────────────────────────────────────────────────────────────


class TestGetProgress:
    def test_returns_progress_tuple_from_status_file(self, runner):
        data = {"run_id": "r1", "status": "running", "progress": 30, "total": 100}
        (runner.status_dir / "r1.json").write_text(json.dumps(data))

        assert runner.get_progress("r1") == (30, 100)

    def test_returns_none_for_missing_status_file(self, runner):
        assert runner.get_progress("ghost") is None

    def test_defaults_to_zero_when_progress_keys_absent(self, runner):
        (runner.status_dir / "r1.json").write_text(
            json.dumps({"run_id": "r1", "status": "running"})
        )
        assert runner.get_progress("r1") == (0, 0)

    def test_returns_full_progress_at_completion(self, runner):
        data = {"run_id": "r1", "status": "completed", "progress": 100, "total": 100}
        (runner.status_dir / "r1.json").write_text(json.dumps(data))

        progress, total = runner.get_progress("r1")
        assert progress == total == 100


# ── _build_execution_script ───────────────────────────────────────────────────


class TestBuildExecutionScript:
    def test_script_references_experiment_path(self, runner, minimal_experiment, minimal_scenario):
        script = runner._build_execution_script(minimal_experiment, minimal_scenario, "test_run_id")
        assert str(minimal_experiment.path) in script

    def test_script_contains_scenario_normalized_label(
        self, runner, minimal_experiment, minimal_scenario
    ):
        script = runner._build_execution_script(minimal_experiment, minimal_scenario, "test_run_id")
        assert minimal_scenario.normalized_label in script

    def test_script_contains_execution_engine_import(
        self, runner, minimal_experiment, minimal_scenario
    ):
        script = runner._build_execution_script(minimal_experiment, minimal_scenario, "test_run_id")
        assert "ExecutionEngine" in script

    def test_script_is_valid_python_syntax(self, runner, minimal_experiment, minimal_scenario):
        import ast

        script = runner._build_execution_script(minimal_experiment, minimal_scenario, "test_run_id")
        # Must not raise SyntaxError
        ast.parse(script)

    def test_script_contains_progress_callback(self, runner, minimal_experiment, minimal_scenario):
        script = runner._build_execution_script(minimal_experiment, minimal_scenario, "test_run_id")
        assert "_progress_callback" in script

    def test_script_uses_exact_status_file_path(self, runner, minimal_experiment, minimal_scenario):
        """P1 bugfix: script must use exact status file path, not prefix-glob discovery."""
        run_id = "sim--myexp--baseline--1700000000"
        script = runner._build_execution_script(minimal_experiment, minimal_scenario, run_id)
        assert f"{run_id}.json" in script
        assert "glob(" not in script

    def test_scenario_with_apostrophe_in_label_is_safe(self, runner, tmp_path):
        """P1 bugfix: scenario JSON with quotes must not break the generated script."""
        from spkmc.models.experiment import Experiment
        from spkmc.models.scenario import Scenario

        scenario = Scenario(
            label="O'Brien's Test",
            network="er",
            distribution="gamma",
            nodes=100,
            samples=10,
            k_avg=5.0,
            **{"lambda": 1.0},
            shape=2.0,
            scale=1.0,
            t_max=5.0,
            steps=50,
            initial_perc=0.01,
        )
        exp_path = tmp_path / "experiments" / "apos_exp"
        exp_path.mkdir(parents=True)
        experiment = Experiment(name="Apostrophe Exp", scenarios=[scenario], path=exp_path)

        import ast

        script = runner._build_execution_script(experiment, scenario, "test_apos_run_id")
        ast.parse(script)


# ── run_all_scenarios skips existing results ──────────────────────────────────


class TestRunAllScenariosSkipsExistingResults:
    def test_skips_scenario_with_existing_result(
        self, runner, minimal_experiment, tmp_path, monkeypatch
    ):
        """run_all_scenarios must not re-launch scenarios that already have results."""
        # Pre-create the result file for the baseline scenario
        result_file = minimal_experiment.path / "baseline.json"
        result_file.touch()

        launched = []

        def mock_run_scenario(exp, sc, show_progress=False):
            launched.append(sc.normalized_label)
            return "fake_run_id"

        runner.run_scenario = mock_run_scenario

        # Patch st.toast so it doesn't fail outside Streamlit
        with patch("spkmc.web.runner.st") as mock_st:
            run_ids = runner.run_all_scenarios(minimal_experiment, show_progress=False)

        assert "baseline" not in launched
        assert run_ids == []


# ── poll_running_simulations: dead process with output file ──────────────────


class _DictAttr(dict):
    """Dict that also supports attribute access (like Streamlit session_state)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class TestPollDeadProcessWithOutputFile:
    """Regression: a dead PID must be marked completed when the result file exists."""

    @staticmethod
    def _make_mock_st(runner, run_id):
        """Build a mock ``st`` module whose ``session_state`` behaves like a dict."""
        from unittest.mock import MagicMock

        mock_st = MagicMock()
        mock_st.session_state = _DictAttr(
            simulation_runner=runner,
            running_simulations={
                "sim--test_exp--baseline": {
                    "experiment_name": "test_exp",
                    "scenario_label": "Baseline",
                    "run_id": run_id,
                }
            },
        )
        return mock_st

    def test_dead_process_with_result_file_marks_completed(self, runner, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import spkmc.web.runner as runner_mod

        # Write a status file claiming "running" with a dead PID
        run_id = "sim--test_exp--baseline--1700000000"
        status_data = {
            "run_id": run_id,
            "status": "running",
            "pid": 99999999,  # PID that does not exist
        }
        (runner.status_dir / f"{run_id}.json").write_text(json.dumps(status_data))

        # Simulate result file existing (check_completion returns True)
        runner.check_completion = lambda *a, **kw: True

        mock_st = self._make_mock_st(runner, run_id)
        monkeypatch.setattr(runner_mod, "st", mock_st)

        # SessionState is imported locally inside poll_running_simulations
        mock_session = MagicMock()
        monkeypatch.setattr("spkmc.web.state.SessionState", mock_session)

        runner_mod.poll_running_simulations()

        # Must call mark_simulation_completed (not mark_simulation_failed)
        mock_session.mark_simulation_completed.assert_called_once_with("sim--test_exp--baseline")
        mock_session.mark_simulation_failed.assert_not_called()

    def test_dead_process_without_result_file_marks_failed(self, runner, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import spkmc.web.runner as runner_mod

        run_id = "sim--test_exp--baseline--1700000000"
        status_data = {
            "run_id": run_id,
            "status": "running",
            "pid": 99999999,
        }
        (runner.status_dir / f"{run_id}.json").write_text(json.dumps(status_data))

        # Simulate result file NOT existing (check_completion returns False)
        runner.check_completion = lambda *a, **kw: False

        mock_st = self._make_mock_st(runner, run_id)
        monkeypatch.setattr(runner_mod, "st", mock_st)

        mock_session = MagicMock()
        monkeypatch.setattr("spkmc.web.state.SessionState", mock_session)

        runner_mod.poll_running_simulations()

        # Must call mark_simulation_failed (no result file to rescue)
        mock_session.mark_simulation_failed.assert_called_once()
        mock_session.mark_simulation_completed.assert_not_called()
