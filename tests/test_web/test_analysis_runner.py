"""
Tests for AnalysisRunner.

Tests cover file-based status management, completion detection (experiment
vs. scenario analysis types), script generation with correct content, and
cleanup. API keys are passed via subprocess env, not embedded in scripts.
Subprocess execution is not tested.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def runner(tmp_path):
    """AnalysisRunner with status_dir isolated to tmp_path."""
    from spkmc.web.analysis_runner import AnalysisRunner

    r = AnalysisRunner.__new__(AnalysisRunner)
    r.status_dir = tmp_path / "status"
    r.status_dir.mkdir()
    r._processes = {}
    return r


# ── get_status ────────────────────────────────────────────────────────────────


class TestGetStatus:
    def test_returns_none_for_missing_run_id(self, runner):
        assert runner.get_status("nonexistent") is None

    def test_returns_parsed_dict_for_valid_file(self, runner):
        data = {
            "run_id": "exp_analysis--exp1--1",
            "type": "analysis",
            "status": "running",
        }
        (runner.status_dir / "exp_analysis--exp1--1.json").write_text(json.dumps(data))
        assert runner.get_status("exp_analysis--exp1--1") == data

    def test_returns_none_for_corrupted_json(self, runner):
        (runner.status_dir / "bad.json").write_text("{not: valid}")
        assert runner.get_status("bad") is None


# ── cleanup_status ────────────────────────────────────────────────────────────


class TestCleanupStatus:
    def test_removes_both_status_and_script_files(self, runner):
        run_id = "exp_analysis--exp1--99"
        (runner.status_dir / f"{run_id}.json").write_text("{}")
        (runner.status_dir / f"{run_id}_script.py").write_text("pass")

        runner.cleanup_status(run_id)

        assert not (runner.status_dir / f"{run_id}.json").exists()
        assert not (runner.status_dir / f"{run_id}_script.py").exists()

    def test_cleanup_is_idempotent_when_files_absent(self, runner):
        runner.cleanup_status("never_existed")
        runner.cleanup_status("never_existed")


# ── check_completion ──────────────────────────────────────────────────────────


class TestCheckCompletion:
    def test_experiment_analysis_checks_analysis_md(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "analysis.md").touch()

        assert runner.check_completion("exp1", "experiment") is True

    def test_experiment_analysis_returns_false_when_file_missing(
        self, runner, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)

        assert runner.check_completion("exp1", "experiment") is False

    def test_scenario_analysis_checks_scenario_analysis_md(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "baseline_analysis.md").touch()

        assert runner.check_completion("exp1", "scenario", "baseline") is True

    def test_scenario_analysis_returns_false_when_file_missing(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)

        assert runner.check_completion("exp1", "scenario", "baseline") is False

    def test_experiment_and_scenario_paths_do_not_collide(self, runner, tmp_path, monkeypatch):
        """analysis.md and baseline_analysis.md are distinct files."""
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "experiments" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "analysis.md").touch()
        # Only experiment analysis exists; scenario analysis must report False

        assert runner.check_completion("exp1", "scenario", "baseline") is False


# ── _build_experiment_script ──────────────────────────────────────────────────


class TestBuildExperimentScript:
    @pytest.fixture()
    def script(self, runner, tmp_path):
        exp_path = tmp_path / "experiments" / "my_exp"
        exp_path.mkdir(parents=True)
        return runner._build_experiment_script(
            experiment_path=exp_path,
            experiment_name="My Experiment",
            experiment_description="Does X spread faster on SF networks?",
            model="gpt-4o-mini",
            run_id="test_exp_run_id",
        )

    def test_script_does_not_embed_api_key(self, script):
        """API key is passed via subprocess env, never written to script file."""
        assert "OPENAI_API_KEY" in script  # env var reference exists
        assert "sk-" not in script  # but no actual key value

    def test_script_references_experiment_path(self, script, tmp_path):
        assert "my_exp" in script

    def test_script_references_model_name(self, script):
        assert "gpt-4o-mini" in script

    def test_script_imports_ai_analyzer(self, script):
        assert "AIAnalyzer" in script

    def test_script_calls_analyze_experiment(self, script):
        assert "analyze_experiment" in script

    def test_script_is_valid_python_syntax(self, script):
        import ast

        ast.parse(script)

    def test_multiline_description_is_safely_embedded(self, runner, tmp_path):
        """P1 bugfix: multiline descriptions must not break the generated script."""
        exp_path = tmp_path / "experiments" / "exp1"
        exp_path.mkdir(parents=True)
        multiline_desc = "Line one\nLine two\nLine three with 'quotes'"
        script = runner._build_experiment_script(
            experiment_path=exp_path,
            experiment_name="Test\nNewline",
            experiment_description=multiline_desc,
            model="gpt-4o-mini",
            run_id="test_multiline_run_id",
        )
        import ast

        ast.parse(script)
        # repr() must be used for all user-provided strings
        assert repr(multiline_desc) in script
        assert repr("Test\nNewline") in script

    def test_script_writes_running_status(self, script):
        assert "_write_status" in script
        assert '"running"' in script or "'running'" in script

    def test_script_uses_exact_status_file_path(self, runner, tmp_path):
        """P1 bugfix: script must use exact status file path, not prefix-glob discovery."""
        exp_path = tmp_path / "experiments" / "my_exp"
        exp_path.mkdir(parents=True)
        run_id = "exp_analysis--my_exp--1700000000"
        script = runner._build_experiment_script(
            experiment_path=exp_path,
            experiment_name="My Experiment",
            experiment_description="Test",
            model="gpt-4o-mini",
            run_id=run_id,
        )
        assert f"{run_id}.json" in script
        assert "glob(" not in script


# ── _build_scenario_script ────────────────────────────────────────────────────


class TestBuildScenarioScript:
    @pytest.fixture()
    def script(self, runner, tmp_path):
        exp_path = tmp_path / "experiments" / "my_exp"
        exp_path.mkdir(parents=True)
        return runner._build_scenario_script(
            experiment_path=exp_path,
            scenario_label="High Transmission",
            scenario_normalized="high_transmission",
            model="gpt-4o",
            run_id="test_scenario_run_id",
        )

    def test_script_does_not_embed_api_key(self, script):
        """API key is passed via subprocess env, never written to script file."""
        assert "OPENAI_API_KEY" in script  # env var reference exists
        assert "sk-" not in script  # but no actual key value

    def test_script_references_scenario_normalized_label(self, script):
        assert "high_transmission" in script

    def test_script_calls_analyze_scenario(self, script):
        assert "analyze_scenario" in script

    def test_script_is_valid_python_syntax(self, script):
        import ast

        ast.parse(script)

    def test_script_loads_result_file(self, script):
        assert "result_file" in script

    def test_script_uses_exact_status_file_path(self, runner, tmp_path):
        """P1 bugfix: script must use exact status file path, not prefix-glob discovery."""
        exp_path = tmp_path / "experiments" / "my_exp"
        exp_path.mkdir(parents=True)
        run_id = "sc_analysis--my_exp--high_transmission--1700000000"
        script = runner._build_scenario_script(
            experiment_path=exp_path,
            scenario_label="High Transmission",
            scenario_normalized="high_transmission",
            model="gpt-4o",
            run_id=run_id,
        )
        assert f"{run_id}.json" in script
        assert "glob(" not in script


# ── poll_running_analyses: dead process with output file ─────────────────────


class _DictAttr(dict):
    """Dict that also supports attribute access (like Streamlit session_state)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class TestPollDeadAnalysisProcessWithOutputFile:
    """Regression: a dead analysis PID must be marked completed when .md exists."""

    @staticmethod
    def _make_mock_st(runner, run_id):
        """Build a mock ``st`` module whose ``session_state`` behaves like a dict."""
        from unittest.mock import MagicMock

        mock_st = MagicMock()
        mock_st.session_state = _DictAttr(
            analysis_runner=runner,
            running_analyses={
                "analysis--exp1": {
                    "experiment_name": "exp1",
                    "analysis_type": "experiment",
                    "scenario_normalized": "",
                    "run_id": run_id,
                }
            },
        )
        return mock_st

    def test_dead_process_with_output_file_marks_completed(self, runner, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import spkmc.web.analysis_runner as ar_mod

        run_id = "exp_analysis--exp1--1700000000"
        status_data = {
            "run_id": run_id,
            "status": "running",
            "pid": 99999999,
        }
        (runner.status_dir / f"{run_id}.json").write_text(json.dumps(status_data))

        # Simulate output file existing (check_completion returns True)
        runner.check_completion = lambda *a, **kw: True

        mock_st = self._make_mock_st(runner, run_id)
        monkeypatch.setattr(ar_mod, "st", mock_st)

        # SessionState is imported locally inside poll_running_analyses
        mock_session = MagicMock()
        monkeypatch.setattr("spkmc.web.state.SessionState", mock_session)

        ar_mod.poll_running_analyses()

        mock_session.mark_analysis_completed.assert_called_once_with("analysis--exp1")
        mock_session.mark_analysis_failed.assert_not_called()

    def test_dead_process_without_output_file_marks_failed(self, runner, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import spkmc.web.analysis_runner as ar_mod

        run_id = "exp_analysis--exp1--1700000000"
        status_data = {
            "run_id": run_id,
            "status": "running",
            "pid": 99999999,
        }
        (runner.status_dir / f"{run_id}.json").write_text(json.dumps(status_data))

        # Simulate output file NOT existing (check_completion returns False)
        runner.check_completion = lambda *a, **kw: False

        mock_st = self._make_mock_st(runner, run_id)
        monkeypatch.setattr(ar_mod, "st", mock_st)

        mock_session = MagicMock()
        monkeypatch.setattr("spkmc.web.state.SessionState", mock_session)

        ar_mod.poll_running_analyses()

        mock_session.mark_analysis_failed.assert_called_once()
        mock_session.mark_analysis_completed.assert_not_called()
