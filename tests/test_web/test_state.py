"""
Tests for SessionState business logic.

Tests cover state machine transitions, scenario selection, progress tracking,
and disk-based restoration. Streamlit is patched at the module level so no
Streamlit runtime is required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeState(dict):
    """Minimal st.session_state substitute: dict with attribute access."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


class _FakeQueryParams(dict):
    """Minimal st.query_params substitute."""

    def pop(self, key, default=None):  # type: ignore[override]
        return super().pop(key, default)


@pytest.fixture()
def session(monkeypatch):
    """
    Patch st.session_state and st.query_params with plain dict substitutes.

    Returns the fake session_state dict so tests can pre-populate it.
    """
    state = _FakeState()
    params = _FakeQueryParams()

    import spkmc.web.state as state_module

    monkeypatch.setattr(state_module.st, "session_state", state)
    monkeypatch.setattr(state_module.st, "query_params", params)
    return state


# ── PID detection ─────────────────────────────────────────────────────────────


class TestIsPidAlive:
    def test_current_process_is_alive(self):
        from spkmc.web.state import _is_pid_alive

        assert _is_pid_alive(os.getpid()) is True

    def test_unreachable_pid_is_not_alive(self):
        from spkmc.web.state import _is_pid_alive

        # PID space on modern OSes is typically limited to ~4 million
        assert _is_pid_alive(999_999_999) is False

    def test_zero_pid_does_not_raise(self):
        from spkmc.web.state import _is_pid_alive

        # PID 0 means "same process group" on POSIX — we just care it doesn't raise
        result = _is_pid_alive(0)
        assert isinstance(result, bool)


# ── Scenario selection ────────────────────────────────────────────────────────


class TestScenarioSelection:
    @pytest.fixture(autouse=True)
    def _init(self, session):
        session["selected_scenarios"] = set()

    def test_toggle_adds_unselected_scenario(self, session):
        from spkmc.web.state import SessionState

        SessionState.toggle_scenario_selection("sc_1")
        assert "sc_1" in session["selected_scenarios"]

    def test_toggle_removes_already_selected_scenario(self, session):
        from spkmc.web.state import SessionState

        session["selected_scenarios"] = {"sc_1"}
        SessionState.toggle_scenario_selection("sc_1")
        assert "sc_1" not in session["selected_scenarios"]

    def test_double_toggle_restores_original_state(self, session):
        from spkmc.web.state import SessionState

        SessionState.toggle_scenario_selection("sc_1")
        SessionState.toggle_scenario_selection("sc_1")
        assert "sc_1" not in session["selected_scenarios"]

    def test_clear_empties_all_selections(self, session):
        from spkmc.web.state import SessionState

        session["selected_scenarios"] = {"sc_1", "sc_2", "sc_3"}
        SessionState.clear_scenario_selections()
        assert session["selected_scenarios"] == set()

    def test_get_returns_empty_set_when_key_absent(self, session):
        from spkmc.web.state import SessionState

        session.pop("selected_scenarios", None)
        assert SessionState.get_selected_scenarios() == set()

    def test_independent_scenarios_do_not_interfere(self, session):
        from spkmc.web.state import SessionState

        SessionState.toggle_scenario_selection("sc_a")
        SessionState.toggle_scenario_selection("sc_b")
        SessionState.toggle_scenario_selection("sc_a")  # remove sc_a
        assert "sc_a" not in session["selected_scenarios"]
        assert "sc_b" in session["selected_scenarios"]


# ── Simulation state machine ──────────────────────────────────────────────────


class TestSimulationStateMachine:
    @pytest.fixture(autouse=True)
    def _init(self, session):
        session["running_simulations"] = {}
        session["completed_simulations"] = set()
        session["failed_simulations"] = {}
        session["simulation_progress"] = {}

    def test_unknown_simulation_status_is_pending(self, session):
        from spkmc.web.state import SessionState

        assert SessionState.get_simulation_status("sim_1") == "pending"

    def test_added_simulation_is_running(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_simulation("sim_1", {"run_id": "r1"})
        assert SessionState.is_simulation_running("sim_1") is True
        assert SessionState.get_simulation_status("sim_1") == "running"

    def test_completed_simulation_is_removed_from_running(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_simulation("sim_1", {"run_id": "r1"})
        SessionState.mark_simulation_completed("sim_1")
        assert SessionState.is_simulation_running("sim_1") is False
        assert SessionState.get_simulation_status("sim_1") == "completed"

    def test_failed_simulation_is_removed_from_running(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_simulation("sim_1", {"run_id": "r1"})
        SessionState.mark_simulation_failed("sim_1", "Out of memory")
        assert SessionState.is_simulation_running("sim_1") is False
        assert SessionState.get_simulation_status("sim_1") == "failed"

    def test_completed_simulation_is_not_running(self, session):
        from spkmc.web.state import SessionState

        session["completed_simulations"].add("sim_1")
        assert SessionState.is_simulation_running("sim_1") is False

    def test_two_simulations_transition_independently(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_simulation("sim_a", {})
        SessionState.add_running_simulation("sim_b", {})
        SessionState.mark_simulation_completed("sim_a")
        assert SessionState.get_simulation_status("sim_a") == "completed"
        assert SessionState.get_simulation_status("sim_b") == "running"

    def test_remove_running_is_idempotent_when_absent(self, session):
        from spkmc.web.state import SessionState

        # Must not raise even if the simulation was never added
        SessionState.remove_running_simulation("sim_never_added")
        SessionState.remove_running_simulation("sim_never_added")


# ── Simulation progress ───────────────────────────────────────────────────────


class TestSimulationProgress:
    @pytest.fixture(autouse=True)
    def _init(self, session):
        session["simulation_progress"] = {}

    def test_set_and_get_progress(self, session):
        from spkmc.web.state import SessionState

        SessionState.set_simulation_progress("sim_1", 25, 100)
        result = SessionState.get_simulation_progress("sim_1")
        assert result == {"progress": 25, "total": 100}

    def test_get_progress_returns_none_for_unknown(self, session):
        from spkmc.web.state import SessionState

        assert SessionState.get_simulation_progress("unknown") is None

    def test_clear_progress_removes_entry(self, session):
        from spkmc.web.state import SessionState

        SessionState.set_simulation_progress("sim_1", 50, 100)
        SessionState.clear_simulation_progress("sim_1")
        assert SessionState.get_simulation_progress("sim_1") is None

    def test_clear_progress_is_idempotent_for_absent_key(self, session):
        from spkmc.web.state import SessionState

        # Must not raise
        SessionState.clear_simulation_progress("never_tracked")

    def test_updated_progress_overwrites_previous(self, session):
        from spkmc.web.state import SessionState

        SessionState.set_simulation_progress("sim_1", 10, 100)
        SessionState.set_simulation_progress("sim_1", 80, 100)
        result = SessionState.get_simulation_progress("sim_1")
        assert result["progress"] == 80


# ── Analysis state machine ────────────────────────────────────────────────────


class TestAnalysisStateMachine:
    @pytest.fixture(autouse=True)
    def _init(self, session):
        session["running_analyses"] = {}
        session["completed_analyses"] = set()
        session["failed_analyses"] = {}

    def test_unknown_analysis_status_is_pending(self, session):
        from spkmc.web.state import SessionState

        assert SessionState.get_analysis_status("analysis_1") == "pending"

    def test_added_analysis_is_running(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_analysis("analysis_1", {"run_id": "r1"})
        assert SessionState.is_analysis_running("analysis_1") is True
        assert SessionState.get_analysis_status("analysis_1") == "running"

    def test_completed_analysis_is_removed_from_running(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_analysis("analysis_1", {"run_id": "r1"})
        SessionState.mark_analysis_completed("analysis_1")
        assert SessionState.is_analysis_running("analysis_1") is False
        assert SessionState.get_analysis_status("analysis_1") == "completed"

    def test_failed_analysis_is_removed_from_running(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_analysis("analysis_1", {"run_id": "r1"})
        SessionState.mark_analysis_failed("analysis_1", "API key invalid")
        assert SessionState.is_analysis_running("analysis_1") is False
        assert SessionState.get_analysis_status("analysis_1") == "failed"

    def test_two_analyses_transition_independently(self, session):
        from spkmc.web.state import SessionState

        SessionState.add_running_analysis("analysis_a", {})
        SessionState.add_running_analysis("analysis_b", {})
        SessionState.mark_analysis_completed("analysis_a")
        assert SessionState.get_analysis_status("analysis_a") == "completed"
        assert SessionState.get_analysis_status("analysis_b") == "running"


# ── Disk restoration: simulations ─────────────────────────────────────────────


class TestRestoreRunningSimulations:
    """
    restore_running_simulations reads .spkmc_web/status/*.json files.
    monkeypatch.chdir ensures Path(".spkmc_web") resolves inside tmp_path.
    _is_pid_alive is patched to control alive/dead process scenarios.
    """

    @pytest.fixture(autouse=True)
    def _init(self, session):
        session["running_simulations"] = {}
        session["completed_simulations"] = set()
        session["failed_simulations"] = {}
        session["simulation_progress"] = {}

    def _write_status(self, status_dir: Path, data: dict) -> Path:
        f = status_dir / f"{data['run_id']}.json"
        f.write_text(json.dumps(data))
        return f

    def test_alive_process_is_restored_as_running(self, session, tmp_path, monkeypatch):
        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)

        self._write_status(
            status_dir,
            {
                "run_id": "sim--exp1--baseline--111",
                "experiment_name": "exp1",
                "scenario_label": "Baseline",
                "scenario_normalized": "baseline",
                "status": "running",
                "pid": 12345,
                "progress": 10,
                "total": 100,
            },
        )

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: True)

        from spkmc.web.state import SessionState

        SessionState.restore_running_simulations()

        assert SessionState.is_simulation_running("sim--exp1--baseline") is True

    def test_dead_process_with_result_file_is_marked_completed(
        self, session, tmp_path, monkeypatch
    ):
        from unittest.mock import patch

        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)

        self._write_status(
            status_dir,
            {
                "run_id": "sim--exp1--baseline--222",
                "experiment_name": "exp1",
                "scenario_label": "Baseline",
                "scenario_normalized": "baseline",
                "status": "running",
                "pid": 99999,
                "progress": 0,
                "total": 100,
            },
        )

        # Create the expected result file
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "baseline.json").touch()

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: False)

        # WebConfig is imported inside the function body: patch at its source module
        mock_config = MagicMock()
        mock_config.get_experiments_path.return_value = tmp_path / "experiments"
        with patch("spkmc.web.config.WebConfig", return_value=mock_config):
            from spkmc.web.state import SessionState

            SessionState.restore_running_simulations()

        assert SessionState.get_simulation_status("sim--exp1--baseline") == "completed"

    def test_dead_process_without_result_file_is_marked_failed(
        self, session, tmp_path, monkeypatch
    ):
        from unittest.mock import patch

        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)

        self._write_status(
            status_dir,
            {
                "run_id": "sim--exp1--scenario_a--333",
                "experiment_name": "exp1",
                "scenario_label": "Scenario A",
                "scenario_normalized": "scenario_a",
                "status": "running",
                "pid": 99999,
                "progress": 0,
                "total": 100,
            },
        )

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: False)

        mock_config = MagicMock()
        mock_config.get_experiments_path.return_value = tmp_path / "experiments"
        with patch("spkmc.web.config.WebConfig", return_value=mock_config):
            from spkmc.web.state import SessionState

            SessionState.restore_running_simulations()

        assert SessionState.get_simulation_status("sim--exp1--scenario_a") == "failed"

    def test_corrupted_status_file_is_silently_skipped(self, session, tmp_path, monkeypatch):
        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)
        (status_dir / "bad.json").write_text("{not: valid json}")

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: True)

        from spkmc.web.state import SessionState

        # Must not raise
        SessionState.restore_running_simulations()
        assert session["running_simulations"] == {}

    def test_completed_status_files_are_ignored(self, session, tmp_path, monkeypatch):
        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)

        self._write_status(
            status_dir,
            {
                "run_id": "sim--exp1--baseline--444",
                "experiment_name": "exp1",
                "scenario_normalized": "baseline",
                "status": "completed",  # already done
                "pid": 12345,
                "progress": 100,
                "total": 100,
            },
        )

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: True)

        from spkmc.web.state import SessionState

        SessionState.restore_running_simulations()
        assert session["running_simulations"] == {}

    def test_missing_status_dir_does_not_raise(self, session, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # .spkmc_web/status does NOT exist

        from spkmc.web.state import SessionState

        # Must return silently
        SessionState.restore_running_simulations()


# ── Disk restoration: analyses ────────────────────────────────────────────────


class TestRestoreRunningAnalyses:
    @pytest.fixture(autouse=True)
    def _init(self, session):
        session["running_analyses"] = {}
        session["completed_analyses"] = set()
        session["failed_analyses"] = {}

    def _write_status(self, status_dir: Path, data: dict) -> Path:
        f = status_dir / f"{data['run_id']}.json"
        f.write_text(json.dumps(data))
        return f

    def test_alive_experiment_analysis_is_restored(self, session, tmp_path, monkeypatch):
        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)

        self._write_status(
            status_dir,
            {
                "run_id": "exp_analysis--exp1--555",
                "type": "analysis",
                "analysis_type": "experiment",
                "experiment_name": "exp1",
                "scenario_normalized": "",
                "status": "running",
                "pid": 12345,
            },
        )

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: True)

        from spkmc.web.state import SessionState

        SessionState.restore_running_analyses()
        assert SessionState.is_analysis_running("exp_analysis--exp1") is True

    def test_non_analysis_type_status_files_are_ignored(self, session, tmp_path, monkeypatch):
        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)

        # Write a simulation status file (type is absent / not "analysis")
        self._write_status(
            status_dir,
            {
                "run_id": "sim--exp1--baseline--666",
                "experiment_name": "exp1",
                "scenario_normalized": "baseline",
                "status": "running",
                "pid": 12345,
            },
        )

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: True)

        from spkmc.web.state import SessionState

        SessionState.restore_running_analyses()
        assert session["running_analyses"] == {}

    def test_dead_analysis_with_result_file_is_marked_completed(
        self, session, tmp_path, monkeypatch
    ):
        from unittest.mock import patch

        import spkmc.web.state as state_module

        monkeypatch.chdir(tmp_path)
        status_dir = tmp_path / ".spkmc_web" / "status"
        status_dir.mkdir(parents=True)

        self._write_status(
            status_dir,
            {
                "run_id": "exp_analysis--exp1--777",
                "type": "analysis",
                "analysis_type": "experiment",
                "experiment_name": "exp1",
                "scenario_normalized": "",
                "status": "running",
                "pid": 99999,
            },
        )

        # Create the expected analysis output file
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "analysis.md").touch()

        monkeypatch.setattr(state_module, "_is_pid_alive", lambda pid: False)

        mock_config = MagicMock()
        mock_config.get_experiments_path.return_value = tmp_path / "experiments"
        with patch("spkmc.web.config.WebConfig", return_value=mock_config):
            from spkmc.web.state import SessionState

            SessionState.restore_running_analyses()

        assert SessionState.get_analysis_status("exp_analysis--exp1") == "completed"
