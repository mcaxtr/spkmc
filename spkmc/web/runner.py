"""
Subprocess-based simulation runner for the web interface.

Runs simulations in background subprocesses so they survive browser refresh
and UI interactions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import psutil
import streamlit as st

from spkmc.models import Experiment, Scenario


class SimulationRunner:
    """Manages subprocess-based simulation execution."""

    def __init__(self) -> None:
        """Initialize the simulation runner."""
        self.status_dir = Path(".spkmc_web") / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        # Retain Popen handles so we can reap children and avoid zombies
        self._processes: Dict[str, subprocess.Popen] = {}  # type: ignore[type-arg]

    def run_scenario(
        self, experiment: Experiment, scenario: Scenario, show_progress: bool = True
    ) -> Optional[str]:
        """
        Launch a subprocess to run a single scenario.

        Args:
            experiment: The parent experiment
            scenario: Scenario to execute
            show_progress: Whether to show progress indicators

        Returns:
            Subprocess ID if launched successfully, None otherwise
        """
        assert experiment.path is not None, "Experiment must have a path to run scenarios"

        # Generate unique ID for this run
        run_id = f"sim--{experiment.path.name}--{scenario.normalized_label}--{time.time_ns()}"

        # Create status file
        status_file = self.status_dir / f"{run_id}.json"
        status_data = {
            "run_id": run_id,
            "experiment_name": experiment.path.name,
            "scenario_label": scenario.label,
            "scenario_normalized": scenario.normalized_label,
            "status": "starting",
            "progress": 0,
            "total": scenario.total_samples(),
            "start_time": time.time(),
        }

        with open(status_file, "w") as f:
            json.dump(status_data, f)

        # Build command to execute scenario
        # We'll create a simple Python script that calls execute_scenario
        script_content = self._build_execution_script(experiment, scenario, run_id)

        # Write temporary script
        script_file = self.status_dir / f"{run_id}_script.py"
        with open(script_file, "w") as f:
            f.write(script_content)

        # Launch subprocess
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Store process info and retain handle for reaping
            status_data["status"] = "running"
            status_data["pid"] = process.pid
            self._processes[run_id] = process

            with open(status_file, "w") as f:
                json.dump(status_data, f)

            if show_progress:
                st.toast(f"Started: {scenario.label}")

            return run_id

        except Exception as e:
            # Mark as failed
            status_data["status"] = "failed"
            status_data["error"] = str(e)

            with open(status_file, "w") as f:
                json.dump(status_data, f)

            st.error(f"Failed to start simulation: {str(e)}")
            return None

    def run_all_scenarios(self, experiment: Experiment, show_progress: bool = True) -> List[str]:
        """
        Launch subprocesses to run all scenarios in an experiment.

        Args:
            experiment: The experiment to run
            show_progress: Whether to show progress indicators

        Returns:
            List of run IDs for all launched simulations
        """
        assert experiment.path is not None, "Experiment must have a path to run scenarios"

        run_ids = []

        for scenario in experiment.scenarios:
            # Skip if already has results
            result_file = experiment.path / f"{scenario.normalized_label}.json"
            if result_file.exists():
                continue

            run_id = self.run_scenario(experiment, scenario, show_progress=False)
            if run_id:
                run_ids.append(run_id)

        if show_progress and run_ids:
            st.toast(f"Started {len(run_ids)} simulations")

        return run_ids

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a running or completed simulation.

        Args:
            run_id: The run ID to check

        Returns:
            Status dictionary or None if not found
        """
        status_file = self.status_dir / f"{run_id}.json"

        if not status_file.exists():
            return None

        try:
            with open(status_file, "r") as f:
                return cast(Dict[str, Any], json.load(f))
        except (json.JSONDecodeError, IOError):
            return None

    def is_running(self, run_id: str) -> bool:
        """
        Check if a simulation is still running.

        Args:
            run_id: The run ID to check

        Returns:
            True if running, False otherwise
        """
        status = self.get_status(run_id)
        return status is not None and status.get("status") == "running"

    def check_completion(self, experiment_name: str, scenario_label: str) -> bool:
        """
        Check if a scenario has completed by looking for its result file.

        Args:
            experiment_name: Name of the experiment
            scenario_label: Label of the scenario

        Returns:
            True if result file exists, False otherwise
        """
        from spkmc.models.scenario import Scenario
        from spkmc.web.config import WebConfig

        normalized = Scenario.normalize_label(scenario_label)
        config = WebConfig()
        exp_path = config.get_experiments_path() / experiment_name
        result_file = exp_path / f"{normalized}.json"

        return result_file.exists()

    def cleanup_status(self, run_id: str) -> None:
        """
        Clean up status files and reap the child process for a completed run.

        Args:
            run_id: The run ID to clean up
        """
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

    def _build_execution_script(
        self, experiment: Experiment, scenario: Scenario, run_id: str
    ) -> str:
        """
        Build a Python script to execute a scenario.

        Args:
            experiment: The parent experiment
            scenario: Scenario to execute
            run_id: Unique run identifier (matches the status file name)

        Returns:
            Python script as a string
        """
        assert experiment.path is not None, "Experiment must have a path to build script"

        # Pass the exact status file path so the subprocess doesn't need
        # to discover it via glob (which is ambiguous for prefix-overlapping labels).
        status_file_repr = repr(str(self.status_dir / f"{run_id}.json"))

        script = f"""
import sys
import json
import os
import time
from pathlib import Path

# Add package to path if needed
sys.path.insert(0, str(Path.cwd()))

from spkmc.core.engine import ExecutionContext, ExecutionEngine
from spkmc.models import Scenario

# Exact status file path (set by runner before launching subprocess)
STATUS_FILE = {status_file_repr}

_progress_count = 0
_last_write = 0.0

def _progress_callback(completed):
    global _progress_count, _last_write
    _progress_count += completed
    now = time.time()
    if now - _last_write >= 0.5:
        _last_write = now
        _write_progress(_progress_count, "running")

def _write_progress(progress, status, error=None):
    if STATUS_FILE is None:
        return
    try:
        with open(STATUS_FILE, "r") as f:
            data = json.load(f)
        data["progress"] = progress
        data["status"] = status
        if error:
            data["error"] = error
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, STATUS_FILE)
    except Exception:
        pass

# Load scenario from experiment
experiment_path = Path({repr(str(experiment.path))})
scenario_data = {repr(scenario.model_dump_json())}
scenario = Scenario.model_validate_json(scenario_data)

# Set experiment context
scenario.experiment_name = {repr(experiment.path.name)}
scenario.output_path = str(experiment_path / {repr(scenario.normalized_label + '.json')})

# Create execution context with progress callback
context = ExecutionContext(
    scenarios=[scenario],
    experiment_name={repr(experiment.path.name)},
    results_dir=experiment_path,
    no_plot=True,
    export_format="json",
    on_sample_progress=_progress_callback,
)

# Execute
engine = ExecutionEngine(verbose=False)
try:
    results = engine.execute(context)
    _write_progress(scenario.total_samples(), "completed")
    print("Execution completed successfully")
    sys.exit(0)
except Exception as e:
    _write_progress(_progress_count, "failed", error=str(e))
    print(f"Execution failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        return script

    def get_progress(self, run_id: str) -> Optional[tuple]:
        """
        Get progress for a running simulation.

        Args:
            run_id: The run ID to check

        Returns:
            (progress, total) tuple or None if not available
        """
        status = self.get_status(run_id)
        if status is None:
            return None
        progress = status.get("progress", 0)
        total = status.get("total", 0)
        return (progress, total)


def _settle_scenario_backups(exp_name: str, scenario_label: str, succeeded: bool) -> None:
    """Clean up or restore ``.bak`` artifacts after a simulation terminates.

    On success the backups are stale and can be removed.  On failure the
    backups are restored so the user retains the previous successful result.
    """
    from spkmc.models.scenario import Scenario as ScenarioModel
    from spkmc.web.config import WebConfig

    config = WebConfig()
    exp_path = config.get_experiments_path() / exp_name
    normalized = ScenarioModel.normalize_label(scenario_label)

    result_bak = exp_path / f"{normalized}.json.bak"
    analysis_bak = exp_path / f"{normalized}_analysis.md.bak"

    if succeeded:
        result_bak.unlink(missing_ok=True)
        analysis_bak.unlink(missing_ok=True)
    else:
        result_file = exp_path / f"{normalized}.json"
        analysis_file = exp_path / f"{normalized}_analysis.md"
        if result_bak.exists() and not result_file.exists():
            result_bak.rename(result_file)
        else:
            result_bak.unlink(missing_ok=True)
        if analysis_bak.exists() and not analysis_file.exists():
            analysis_bak.rename(analysis_file)
        else:
            analysis_bak.unlink(missing_ok=True)


def poll_running_simulations() -> None:
    """
    Poll all running simulations and update session state.

    Reads progress from status files and marks completed/failed simulations.
    Called by the scenario cards fragment every ~2 seconds.
    """
    if "simulation_runner" not in st.session_state:
        st.session_state.simulation_runner = SimulationRunner()

    runner: SimulationRunner = st.session_state.simulation_runner

    from spkmc.web.state import SessionState

    running_sims = st.session_state.get("running_simulations", {})

    # Dict is keyed by scenario_id; run_id stored inside info
    for scenario_id, info in list(running_sims.items()):
        exp_name = info.get("experiment_name")
        scenario_label = info.get("scenario_label")
        run_id = info.get("run_id", scenario_id)

        if not (exp_name and scenario_label):
            continue

        # Read status file for progress
        status = runner.get_status(run_id)
        if status:
            progress = status.get("progress", 0)
            total = status.get("total", 0)
            file_status = status.get("status", "running")

            # Update progress in session state
            if total > 0:
                SessionState.set_simulation_progress(scenario_id, progress, total)

            # Check if status file reports completion
            if file_status == "completed" or runner.check_completion(exp_name, scenario_label):
                SessionState.mark_simulation_completed(scenario_id)
                SessionState.clear_simulation_progress(scenario_id)
                _settle_scenario_backups(exp_name, scenario_label, succeeded=True)
                st.toast(f"Completed: {scenario_label}")
                runner.cleanup_status(run_id)
                continue

            # Check if status file reports failure
            if file_status == "failed":
                error_msg = status.get("error", "Unknown error")
                SessionState.mark_simulation_failed(scenario_id, error_msg)
                SessionState.clear_simulation_progress(scenario_id)
                _settle_scenario_backups(exp_name, scenario_label, succeeded=False)
                st.toast(f"Failed: {scenario_label}")
                runner.cleanup_status(run_id)
                continue

            # Check if subprocess died without writing terminal status
            if file_status == "running":
                pid = status.get("pid")
                if pid is not None and not psutil.pid_exists(pid):
                    # Process no longer exists -- check if output was written
                    completed = runner.check_completion(exp_name, scenario_label)
                    if completed:
                        SessionState.mark_simulation_completed(scenario_id)
                        st.toast(f"Completed: {scenario_label}")
                    else:
                        SessionState.mark_simulation_failed(
                            scenario_id, "Process exited unexpectedly"
                        )
                        st.toast(f"Failed: {scenario_label}")
                    SessionState.clear_simulation_progress(scenario_id)
                    _settle_scenario_backups(exp_name, scenario_label, succeeded=completed)
                    runner.cleanup_status(run_id)
                    continue

        # Fallback: check result file directly
        elif runner.check_completion(exp_name, scenario_label):
            SessionState.mark_simulation_completed(scenario_id)
            SessionState.clear_simulation_progress(scenario_id)
            _settle_scenario_backups(exp_name, scenario_label, succeeded=True)
            st.toast(f"Completed: {scenario_label}")
            runner.cleanup_status(run_id)
