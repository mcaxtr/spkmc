"""
Session state management for the web interface.

Provides typed access to Streamlit's session state and initialization of
session-level variables.
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any, Dict, Optional, Set, cast

import streamlit as st


class SessionState:
    """Typed accessor for Streamlit session state."""

    @staticmethod
    def init() -> None:
        """Initialize session state with default values."""
        if "initialized" not in st.session_state:
            # Navigation state — restore from URL query params if present
            st.session_state.current_page = st.query_params.get("page", "dashboard")
            st.session_state.selected_experiment = None

            # UI state
            st.session_state.selected_scenarios = set()
            st.session_state.show_comparison_modal = False
            st.session_state.show_scenario_detail_modal = False
            st.session_state.selected_scenario_id = None

            # Simulation state
            st.session_state.running_simulations = {}  # Dict[str, subprocess_info]
            st.session_state.completed_simulations = set()  # Set[str]
            st.session_state.failed_simulations = {}  # Dict[str, error_message]
            st.session_state.simulation_progress = {}  # Dict[str, progress_info]

            # Analysis state (parallel to simulation state)
            st.session_state.running_analyses = {}  # Dict[str, subprocess_info]
            st.session_state.completed_analyses = set()  # Set[str]
            st.session_state.failed_analyses = {}  # Dict[str, error_message]

            # Form state
            st.session_state.creating_experiment = False
            st.session_state.creating_scenario = False

            # Mark as initialized
            st.session_state.initialized = True

    @staticmethod
    def get_current_page() -> str:
        """Get the current page name."""
        return cast(str, st.session_state.get("current_page", "dashboard"))

    @staticmethod
    def set_current_page(page: str) -> None:
        """Set the current page and sync to URL query params for refresh persistence."""
        st.session_state.current_page = page
        st.query_params["page"] = page

    @staticmethod
    def get_selected_experiment() -> Optional[str]:
        """Get the currently selected experiment name.

        Falls back to st.query_params to survive page refresh.
        """
        name: Optional[str] = cast(Optional[str], st.session_state.get("selected_experiment", None))
        if name is None:
            name = cast(Optional[str], st.query_params.get("experiment", None))
            if name:
                st.session_state.selected_experiment = name
        return name

    @staticmethod
    def set_selected_experiment(experiment_name: Optional[str]) -> None:
        """Set the currently selected experiment.

        Also syncs to st.query_params so the selection survives refresh.
        """
        st.session_state.selected_experiment = experiment_name
        # Clear scenario selections and stale UI flags when switching experiments
        st.session_state.selected_scenarios = set()
        st.session_state.show_comparison_modal = False
        st.session_state.show_scenario_detail_modal = False
        st.session_state.selected_scenario_id = None
        # Sync to query params for refresh persistence
        if experiment_name:
            st.query_params["experiment"] = experiment_name
        else:
            st.query_params.pop("experiment", None)

    @staticmethod
    def get_selected_scenarios() -> Set[str]:
        """Get the set of selected scenario IDs."""
        return cast(Set[str], st.session_state.get("selected_scenarios", set()))

    @staticmethod
    def toggle_scenario_selection(scenario_id: str) -> None:
        """Toggle a scenario's selection state."""
        selected = st.session_state.get("selected_scenarios", set())
        if scenario_id in selected:
            selected.remove(scenario_id)
        else:
            selected.add(scenario_id)
        st.session_state.selected_scenarios = selected

    @staticmethod
    def clear_scenario_selections() -> None:
        """Clear all scenario selections."""
        st.session_state.selected_scenarios = set()

    @staticmethod
    def is_simulation_running(simulation_id: str) -> bool:
        """Check if a simulation is currently running."""
        running = st.session_state.get("running_simulations", {})
        return simulation_id in running

    @staticmethod
    def add_running_simulation(simulation_id: str, info: Dict[str, Any]) -> None:
        """Add a simulation to the running set."""
        if "running_simulations" not in st.session_state:
            st.session_state.running_simulations = {}
        st.session_state.running_simulations[simulation_id] = info

    @staticmethod
    def remove_running_simulation(simulation_id: str) -> None:
        """Remove a simulation from the running set."""
        running = st.session_state.get("running_simulations", {})
        if simulation_id in running:
            del running[simulation_id]

    @staticmethod
    def mark_simulation_completed(simulation_id: str) -> None:
        """Mark a simulation as completed."""
        if "completed_simulations" not in st.session_state:
            st.session_state.completed_simulations = set()
        st.session_state.completed_simulations.add(simulation_id)
        # Clear opposite terminal state so reruns don't get stale status
        st.session_state.get("failed_simulations", {}).pop(simulation_id, None)
        SessionState.remove_running_simulation(simulation_id)

    @staticmethod
    def mark_simulation_failed(simulation_id: str, error_message: str) -> None:
        """Mark a simulation as failed."""
        if "failed_simulations" not in st.session_state:
            st.session_state.failed_simulations = {}
        st.session_state.failed_simulations[simulation_id] = error_message
        # Clear opposite terminal state so reruns don't get stale status
        st.session_state.get("completed_simulations", set()).discard(simulation_id)
        SessionState.remove_running_simulation(simulation_id)

    @staticmethod
    def get_simulation_status(simulation_id: str) -> str:
        """
        Get the status of a simulation.

        Returns:
            One of: 'pending', 'running', 'completed', 'failed'
        """
        if SessionState.is_simulation_running(simulation_id):
            return "running"
        # Check failed before completed — a rerun failure must not be masked
        # by a stale completion from a previous run.
        if simulation_id in st.session_state.get("failed_simulations", {}):
            return "failed"
        if simulation_id in st.session_state.get("completed_simulations", set()):
            return "completed"
        return "pending"

    @staticmethod
    def set_simulation_progress(sim_id: str, progress: int, total: int) -> None:
        """Store progress for a running simulation."""
        if "simulation_progress" not in st.session_state:
            st.session_state.simulation_progress = {}
        st.session_state.simulation_progress[sim_id] = {
            "progress": progress,
            "total": total,
        }

    @staticmethod
    def get_simulation_progress(sim_id: str) -> Optional[Dict[str, int]]:
        """Return progress dict or None if not tracked."""
        return cast(
            Optional[Dict[str, int]],
            st.session_state.get("simulation_progress", {}).get(sim_id),
        )

    @staticmethod
    def clear_simulation_progress(sim_id: str) -> None:
        """Remove progress tracking for a completed simulation."""
        progress = st.session_state.get("simulation_progress", {})
        progress.pop(sim_id, None)

    # ── Analysis tracking ──────────────────────────────────────

    @staticmethod
    def is_analysis_running(analysis_id: str) -> bool:
        """Check if an analysis is currently running."""
        running = st.session_state.get("running_analyses", {})
        return analysis_id in running

    @staticmethod
    def add_running_analysis(analysis_id: str, info: Dict[str, Any]) -> None:
        """Add an analysis to the running set."""
        if "running_analyses" not in st.session_state:
            st.session_state.running_analyses = {}
        st.session_state.running_analyses[analysis_id] = info

    @staticmethod
    def remove_running_analysis(analysis_id: str) -> None:
        """Remove an analysis from the running set."""
        running = st.session_state.get("running_analyses", {})
        if analysis_id in running:
            del running[analysis_id]

    @staticmethod
    def mark_analysis_completed(analysis_id: str) -> None:
        """Mark an analysis as completed."""
        if "completed_analyses" not in st.session_state:
            st.session_state.completed_analyses = set()
        st.session_state.completed_analyses.add(analysis_id)
        # Clear opposite terminal state so reruns don't get stale status
        st.session_state.get("failed_analyses", {}).pop(analysis_id, None)
        SessionState.remove_running_analysis(analysis_id)

    @staticmethod
    def mark_analysis_failed(analysis_id: str, error_message: str) -> None:
        """Mark an analysis as failed."""
        if "failed_analyses" not in st.session_state:
            st.session_state.failed_analyses = {}
        st.session_state.failed_analyses[analysis_id] = error_message
        # Clear opposite terminal state so reruns don't get stale status
        st.session_state.get("completed_analyses", set()).discard(analysis_id)
        SessionState.remove_running_analysis(analysis_id)

    @staticmethod
    def get_analysis_status(analysis_id: str) -> str:
        """
        Get the status of an analysis.

        Returns:
            One of: 'pending', 'running', 'completed', 'failed'
        """
        if SessionState.is_analysis_running(analysis_id):
            return "running"
        # Check failed before completed — a rerun failure must not be masked
        # by a stale completion from a previous run.
        if analysis_id in st.session_state.get("failed_analyses", {}):
            return "failed"
        if analysis_id in st.session_state.get("completed_analyses", set()):
            return "completed"
        return "pending"

    @staticmethod
    def restore_running_analyses() -> None:
        """Restore running analyses from status files on disk.

        Scans .spkmc_web/status/ for analysis status files, verifies the PID
        is still alive, and adds them back to session state. Called once on
        session init to survive page refresh.
        """
        status_dir = Path(".spkmc_web") / "status"
        if not status_dir.exists():
            return

        for status_file in sorted(
            list(status_dir.glob("exp_analysis--*.json"))
            + list(status_dir.glob("sc_analysis--*.json"))
        ):

            try:
                with open(status_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Only process analysis-type status files
            if data.get("type") != "analysis":
                continue

            file_status = data.get("status")
            if file_status not in ("running", "starting"):
                continue

            pid = data.get("pid")
            if not pid:
                continue

            # Check if process is still alive
            if not _is_pid_alive(pid):
                # Process died -- check if it completed by looking for result file
                exp_name = data.get("experiment_name", "")
                analysis_type = data.get("analysis_type", "")
                sc_normalized = data.get("scenario_normalized", "")

                if analysis_type == "experiment":
                    analysis_id = f"exp_analysis--{exp_name}"
                else:
                    analysis_id = f"sc_analysis--{exp_name}--{sc_normalized}"

                # Check if analysis file was actually written
                from spkmc.web.config import WebConfig

                config = WebConfig()
                exp_path = config.get_experiments_path() / exp_name

                if analysis_type == "experiment":
                    result_exists = (exp_path / "analysis.md").exists()
                else:
                    result_exists = (exp_path / f"{sc_normalized}_analysis.md").exists()

                if result_exists:
                    SessionState.mark_analysis_completed(analysis_id)
                else:
                    SessionState.mark_analysis_failed(analysis_id, "Process exited unexpectedly")

                # Clean up stale status file
                run_id = data.get("run_id", "")
                status_file.unlink(missing_ok=True)
                script_file = status_dir / f"{run_id}_script.py"
                if script_file.exists():
                    script_file.unlink(missing_ok=True)
                continue

            # Process is alive -- restore into session state
            exp_name = data.get("experiment_name", "")
            analysis_type = data.get("analysis_type", "")
            sc_normalized = data.get("scenario_normalized", "")

            if analysis_type == "experiment":
                analysis_id = f"exp_analysis--{exp_name}"
            else:
                analysis_id = f"sc_analysis--{exp_name}--{sc_normalized}"

            run_id = data.get("run_id", analysis_id)
            info = {
                "experiment_name": exp_name,
                "analysis_type": analysis_type,
                "scenario_normalized": sc_normalized,
                "run_id": run_id,
                "status": "running",
                "pid": pid,
            }
            SessionState.add_running_analysis(analysis_id, info)

    @staticmethod
    def restore_running_simulations() -> None:
        """Restore running simulations from status files on disk.

        Scans .spkmc_web/status/ for status files with running processes,
        verifies the PID is still alive, and adds them back to session state.
        Called once on session init to survive page refresh.
        """
        status_dir = Path(".spkmc_web") / "status"
        if not status_dir.exists():
            return

        for status_file in status_dir.glob("sim--*.json"):
            try:
                with open(status_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            file_status = data.get("status")
            if file_status not in ("running", "starting"):
                continue

            pid = data.get("pid")
            if not pid:
                continue

            # Check if process is still alive
            if not _is_pid_alive(pid):
                # Process died -- check if it completed by looking for result
                exp_name = data.get("experiment_name", "")
                sc_normalized = data.get("scenario_normalized", "")
                if exp_name and sc_normalized:
                    from spkmc.web.config import WebConfig

                    config = WebConfig()
                    result_path = config.get_experiments_path() / exp_name / f"{sc_normalized}.json"
                    scenario_id = f"sim--{exp_name}--{sc_normalized}"
                    if result_path.exists():
                        SessionState.mark_simulation_completed(scenario_id)
                    else:
                        SessionState.mark_simulation_failed(
                            scenario_id, "Process exited unexpectedly"
                        )
                # Clean up stale status file
                run_id = data.get("run_id", "")
                status_file.unlink(missing_ok=True)
                script_file = status_dir / f"{run_id}_script.py"
                if script_file.exists():
                    script_file.unlink(missing_ok=True)
                continue

            # Process is alive -- restore into session state
            exp_name = data.get("experiment_name", "")
            sc_normalized = data.get("scenario_normalized", "")
            scenario_id = f"sim--{exp_name}--{sc_normalized}"
            run_id = data.get("run_id", scenario_id)

            info = {
                "experiment_name": exp_name,
                "scenario_label": data.get("scenario_label", ""),
                "scenario_normalized": sc_normalized,
                "run_id": run_id,
                "status": "running",
                "pid": pid,
            }
            SessionState.add_running_simulation(scenario_id, info)

            # Restore progress
            progress = data.get("progress", 0)
            total = data.get("total", 0)
            if total > 0:
                SessionState.set_simulation_progress(scenario_id, progress, total)


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError etc — process likely exists but owned by another user
        return True
