"""
Subprocess-based AI analysis runner for the web interface.

Runs AI analyses in background subprocesses so they survive browser refresh
and UI interactions. Follows the same pattern as SimulationRunner.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, cast

import streamlit as st


class AnalysisRunner:
    """Manages subprocess-based AI analysis execution."""

    def __init__(self) -> None:
        """Initialize the analysis runner."""
        self.status_dir = Path(".spkmc_web") / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        # Retain Popen handles so we can reap children and avoid zombies
        self._processes: Dict[str, subprocess.Popen] = {}  # type: ignore[type-arg]

    def run_experiment_analysis(
        self,
        experiment_path: Path,
        experiment_name: str,
        experiment_description: str,
        model: str,
        api_key: str,
    ) -> Optional[str]:
        """
        Launch a subprocess to run AI analysis on an entire experiment.

        Args:
            experiment_path: Path to the experiment directory
            experiment_name: Display name of the experiment
            experiment_description: Research question / description
            model: OpenAI model to use
            api_key: OpenAI API key

        Returns:
            Run ID if launched successfully, None otherwise
        """
        run_id = f"exp_analysis--{experiment_path.name}--{time.time_ns()}"

        status_file = self.status_dir / f"{run_id}.json"
        status_data = {
            "run_id": run_id,
            "type": "analysis",
            "analysis_type": "experiment",
            "experiment_name": experiment_path.name,
            "scenario_normalized": "",
            "status": "starting",
            "start_time": time.time(),
        }

        with open(status_file, "w") as f:
            json.dump(status_data, f)

        script_content = self._build_experiment_script(
            experiment_path, experiment_name, experiment_description, model, run_id
        )

        script_file = self.status_dir / f"{run_id}_script.py"
        with open(script_file, "w") as f:
            f.write(script_content)

        # Pass API key via environment so it never touches disk
        child_env = {**os.environ, "OPENAI_API_KEY": api_key}

        try:
            process = subprocess.Popen(
                [sys.executable, str(script_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env,
            )

            status_data["status"] = "running"
            status_data["pid"] = process.pid
            self._processes[run_id] = process

            with open(status_file, "w") as f:
                json.dump(status_data, f)

            return run_id

        except Exception as e:
            status_data["status"] = "failed"
            status_data["error"] = str(e)

            with open(status_file, "w") as f:
                json.dump(status_data, f)

            st.error(f"Failed to start analysis: {str(e)}")
            return None

    def run_scenario_analysis(
        self,
        experiment_path: Path,
        scenario_label: str,
        scenario_normalized: str,
        model: str,
        api_key: str,
    ) -> Optional[str]:
        """
        Launch a subprocess to run AI analysis on a single scenario.

        Args:
            experiment_path: Path to the experiment directory
            scenario_label: Display label of the scenario
            scenario_normalized: Normalized label for file naming
            model: OpenAI model to use
            api_key: OpenAI API key

        Returns:
            Run ID if launched successfully, None otherwise
        """
        run_id = f"sc_analysis--{experiment_path.name}--{scenario_normalized}--{time.time_ns()}"

        status_file = self.status_dir / f"{run_id}.json"
        status_data = {
            "run_id": run_id,
            "type": "analysis",
            "analysis_type": "scenario",
            "experiment_name": experiment_path.name,
            "scenario_normalized": scenario_normalized,
            "status": "starting",
            "start_time": time.time(),
        }

        with open(status_file, "w") as f:
            json.dump(status_data, f)

        script_content = self._build_scenario_script(
            experiment_path, scenario_label, scenario_normalized, model, run_id
        )

        script_file = self.status_dir / f"{run_id}_script.py"
        with open(script_file, "w") as f:
            f.write(script_content)

        # Pass API key via environment so it never touches disk
        child_env = {**os.environ, "OPENAI_API_KEY": api_key}

        try:
            process = subprocess.Popen(
                [sys.executable, str(script_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env,
            )

            status_data["status"] = "running"
            status_data["pid"] = process.pid
            self._processes[run_id] = process

            with open(status_file, "w") as f:
                json.dump(status_data, f)

            return run_id

        except Exception as e:
            status_data["status"] = "failed"
            status_data["error"] = str(e)

            with open(status_file, "w") as f:
                json.dump(status_data, f)

            st.error(f"Failed to start analysis: {str(e)}")
            return None

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a running or completed analysis."""
        status_file = self.status_dir / f"{run_id}.json"

        if not status_file.exists():
            return None

        try:
            with open(status_file, "r") as f:
                return cast(Dict[str, Any], json.load(f))
        except (json.JSONDecodeError, IOError):
            return None

    def cleanup_status(self, run_id: str) -> None:
        """Clean up status files and reap child process for a completed run."""
        # Reap child process to prevent zombies
        proc = self._processes.pop(run_id, None)
        if proc is not None:
            proc.poll()  # Non-blocking reap

        status_file = self.status_dir / f"{run_id}.json"
        script_file = self.status_dir / f"{run_id}_script.py"

        if status_file.exists():
            status_file.unlink()
        if script_file.exists():
            script_file.unlink()

    def check_completion(
        self,
        experiment_name: str,
        analysis_type: str,
        scenario_normalized: str = "",
    ) -> bool:
        """
        Check if an analysis has completed by looking for the .md file.

        Args:
            experiment_name: Name of the experiment
            analysis_type: "experiment" or "scenario"
            scenario_normalized: Normalized scenario label (for scenario type)

        Returns:
            True if analysis file exists
        """
        from spkmc.web.config import WebConfig

        config = WebConfig()
        exp_path = config.get_experiments_path() / experiment_name
        if analysis_type == "experiment":
            return (exp_path / "analysis.md").exists()
        return (exp_path / f"{scenario_normalized}_analysis.md").exists()

    def _build_experiment_script(
        self,
        experiment_path: Path,
        experiment_name: str,
        experiment_description: str,
        model: str,
        run_id: str,
    ) -> str:
        """Build a Python script to run experiment-level analysis."""
        # Use repr() for safe embedding — handles quotes, newlines, backslashes
        exp_path_repr = repr(str(experiment_path))
        exp_name_repr = repr(experiment_name)
        exp_desc_repr = repr(experiment_description)
        model_repr = repr(model)
        # Pass exact status file path so subprocess doesn't need prefix-glob discovery
        status_file_repr = repr(str(self.status_dir / f"{run_id}.json"))

        return f"""
import sys
import json
import os
import re
import time
from pathlib import Path

# Add package to path if needed
sys.path.insert(0, str(Path.cwd()))

# OPENAI_API_KEY is passed via subprocess environment (never written to disk)

# Exact status file path (set by runner before launching subprocess)
STATUS_FILE = {status_file_repr}

def _write_status(status, error=None):
    if STATUS_FILE is None:
        return
    try:
        with open(STATUS_FILE, "r") as fh:
            data = json.load(fh)
        data["status"] = status
        if error:
            data["error"] = error
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, STATUS_FILE)
    except Exception:
        pass

def _normalize_label(label):
    normalized = label.lower().strip()
    normalized = re.sub(r"[\\s\\-]+", "_", normalized)
    normalized = re.sub(r"[^\\w]", "", normalized)
    return normalized

_write_status("running")

experiment_path = Path({exp_path_repr})

# Load all completed scenario results
results = []
data_file = experiment_path / "data.json"
try:
    with open(data_file, "r") as fh:
        data = json.load(fh)
    for sc in data.get("scenarios", []):
        label = sc.get("label", "")
        normalized = _normalize_label(label)
        result_file = experiment_path / f"{{normalized}}.json"
        if result_file.exists():
            with open(result_file, "r") as rfh:
                results.append(json.load(rfh))
except Exception as e:
    _write_status("failed", error=f"Failed to load results: {{e}}")
    sys.exit(1)

if not results:
    _write_status("failed", error="No completed scenarios to analyze")
    sys.exit(1)

# Initialize backup paths before the try block so the except handler
# can always reference them without risking UnboundLocalError.
old_analysis = experiment_path / "analysis.md"
backup_analysis = experiment_path / "analysis.md.bak"

try:
    from spkmc.analysis.ai_analyzer import AIAnalyzer

    # Preserve existing analysis as backup so a failed re-analysis doesn't
    # permanently destroy the previous report.
    if old_analysis.exists():
        old_analysis.rename(backup_analysis)

    analyzer = AIAnalyzer(model={model_repr})
    analysis_path = analyzer.analyze_experiment(
        experiment_name={exp_name_repr},
        experiment_description={exp_desc_repr},
        results=results,
        results_dir=experiment_path,
    )

    if analysis_path:
        # Success — discard backup
        if backup_analysis.exists():
            backup_analysis.unlink()
        _write_status("completed")
        print("Analysis completed successfully")
    else:
        # Restore backup when analysis returns None
        if backup_analysis.exists():
            backup_analysis.rename(old_analysis)
        _write_status("failed", error="Analysis returned None (may already exist)")
    sys.exit(0)
except Exception as e:
    # Restore backup on any failure
    if backup_analysis.exists():
        backup_analysis.rename(old_analysis)
    _write_status("failed", error=str(e))
    print(f"Analysis failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    def _build_scenario_script(
        self,
        experiment_path: Path,
        scenario_label: str,
        scenario_normalized: str,
        model: str,
        run_id: str,
    ) -> str:
        """Build a Python script to run scenario-level analysis."""
        # Use repr() for safe embedding — handles quotes, newlines, backslashes
        exp_path_repr = repr(str(experiment_path))
        label_repr = repr(scenario_label)
        norm_repr = repr(scenario_normalized)
        model_repr = repr(model)
        # Pass exact status file path so subprocess doesn't need prefix-glob discovery
        status_file_repr = repr(str(self.status_dir / f"{run_id}.json"))

        return f"""
import sys
import json
import os
import time
from pathlib import Path

# Add package to path if needed
sys.path.insert(0, str(Path.cwd()))

# OPENAI_API_KEY is passed via subprocess environment (never written to disk)

# Exact status file path (set by runner before launching subprocess)
STATUS_FILE = {status_file_repr}

def _write_status(status, error=None):
    if STATUS_FILE is None:
        return
    try:
        with open(STATUS_FILE, "r") as fh:
            data = json.load(fh)
        data["status"] = status
        if error:
            data["error"] = error
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, STATUS_FILE)
    except Exception:
        pass

_write_status("running")

experiment_path = Path({exp_path_repr})
result_file = experiment_path / ({norm_repr} + ".json")

try:
    with open(result_file, "r") as fh:
        result_dict = json.load(fh)
except Exception as e:
    _write_status("failed", error=f"Failed to load result: {{e}}")
    sys.exit(1)

# Initialize backup paths before the try block so the except handler
# can always reference them without risking UnboundLocalError.
old_analysis = experiment_path / ({norm_repr} + "_analysis.md")
backup_analysis = experiment_path / ({norm_repr} + "_analysis.md.bak")

try:
    from spkmc.analysis.ai_analyzer import AIAnalyzer

    # Preserve existing analysis as backup so a failed re-analysis doesn't
    # permanently destroy the previous report.
    if old_analysis.exists():
        old_analysis.rename(backup_analysis)

    analyzer = AIAnalyzer(model={model_repr})
    analysis_path = analyzer.analyze_scenario(
        scenario_label={label_repr},
        result=result_dict,
        results_dir=experiment_path,
    )

    if analysis_path:
        # Success — discard backup
        if backup_analysis.exists():
            backup_analysis.unlink()
        _write_status("completed")
        print("Analysis completed successfully")
    else:
        # Restore backup when analysis returns None
        if backup_analysis.exists():
            backup_analysis.rename(old_analysis)
        _write_status("failed", error="Analysis returned None (may already exist)")
    sys.exit(0)
except Exception as e:
    # Restore backup on any failure
    if backup_analysis.exists():
        backup_analysis.rename(old_analysis)
    _write_status("failed", error=str(e))
    print(f"Analysis failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""


def poll_running_analyses() -> bool:
    """
    Poll all running analyses and update session state.

    Reads status from files and marks completed/failed analyses.
    Called by the scenario cards fragment every ~2 seconds.

    Returns:
        True if any analysis transitioned to completed or failed (caller
        should trigger a full page rerun so sections outside the fragment
        re-render with updated state).
    """
    if "analysis_runner" not in st.session_state:
        st.session_state.analysis_runner = AnalysisRunner()

    runner: AnalysisRunner = st.session_state.analysis_runner

    from spkmc.web.state import SessionState

    running = st.session_state.get("running_analyses", {})
    changed = False

    for analysis_id, info in list(running.items()):
        exp_name = info.get("experiment_name")
        analysis_type = info.get("analysis_type", "experiment")
        sc_normalized = info.get("scenario_normalized", "")
        run_id = info.get("run_id", analysis_id)

        if not exp_name:
            continue

        # Read status file
        status = runner.get_status(run_id)
        if status:
            file_status = status.get("status", "running")

            # Check if status file reports completion.
            # Do NOT fall back to check_completion() here — while status is
            # "running", a stale .md from the previous run may still exist on
            # disk (the subprocess deletes it shortly after starting).
            if file_status == "completed":
                SessionState.mark_analysis_completed(analysis_id)
                label = "experiment" if analysis_type == "experiment" else sc_normalized
                st.toast(f"Analysis complete: {label}")
                runner.cleanup_status(run_id)
                changed = True
                continue

            # Check if status file reports failure
            if file_status == "failed":
                error_msg = status.get("error", "Unknown error")
                SessionState.mark_analysis_failed(analysis_id, error_msg)
                st.toast(f"Analysis failed: {error_msg}")
                runner.cleanup_status(run_id)
                changed = True
                continue

            # Check if subprocess died without writing terminal status
            if file_status == "running":
                pid = status.get("pid")
                if pid is not None:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        # Process no longer exists — check if output was written
                        if runner.check_completion(exp_name, analysis_type, sc_normalized):
                            SessionState.mark_analysis_completed(analysis_id)
                            label = "experiment" if analysis_type == "experiment" else sc_normalized
                            st.toast(f"Analysis complete: {label}")
                        else:
                            SessionState.mark_analysis_failed(
                                analysis_id,
                                "Analysis process exited unexpectedly",
                            )
                            st.toast("Analysis failed: process exited unexpectedly")
                        runner.cleanup_status(run_id)
                        changed = True
                        continue
                    except OSError:
                        pass  # PermissionError etc — process may still exist

        # Fallback: check result file directly
        elif runner.check_completion(exp_name, analysis_type, sc_normalized):
            SessionState.mark_analysis_completed(analysis_id)
            runner.cleanup_status(run_id)
            changed = True

    return changed
