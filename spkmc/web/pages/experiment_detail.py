"""
Experiment detail page - view and manage a single experiment.

Shows experiment overview, scenario cards, scenario detail modals,
and comparison functionality.
"""

from __future__ import annotations

import base64
import html as _html
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from spkmc.io.data_manager import DataManager
from spkmc.io.experiments import ExperimentManager
from spkmc.models import Experiment, Scenario
from spkmc.web.analysis_runner import AnalysisRunner, poll_running_analyses
from spkmc.web.components import result_metric_cards
from spkmc.web.config import WebConfig
from spkmc.web.plotting import create_comparison_figure, create_sir_figure
from spkmc.web.runner import SimulationRunner, poll_running_simulations
from spkmc.web.state import SessionState
from spkmc.web.styles import (
    COLORS,
    FONTS,
    _dedent,
    circular_progress_html,
    params_card,
    scenario_card,
    section_header,
)

# SVG icons used on this page
_ICON_NETWORK = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><circle cx="12" cy="5" r="3"/>'
    '<circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/>'
    '<line x1="12" y1="8" x2="5" y2="16"/>'
    '<line x1="12" y1="8" x2="19" y2="16"/></svg>'
)
_ICON_DIST = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M3 20h18"/>'
    '<path d="M3 20c2-6 5-16 9-16s7 10 9 16"/></svg>'
)
_ICON_SIM = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>'
)
_ICON_AI = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M12 2l2.09 6.26L20 10l-5.91 1.74L12 18l-2.09-6.26L4 10l5.91-1.74z"/>'
    '<path d="M18 14l1.18 3.54L22 19l-2.82.46L18 23l-1.18-3.54L14 19l2.82-.46z"/>'
    "</svg>"
)


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values with numeric type normalization.

    Handles the int/float mismatch that occurs when Pydantic coerces
    JSON integers but Python code uses floats (e.g. 10 vs 10.0).
    """
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return bool(a == b)


def _download_anchor(data: bytes, filename: str, mime: str, label: str = "Download") -> str:
    """
    Return an HTML anchor styled as a button using a base64 data URI.

    Avoids Streamlit's media file storage, which causes KeyError when a
    st.download_button is re-rendered inside a popover (stale file ID).
    """
    b64 = base64.b64encode(data).decode()
    safe_filename = _html.escape(filename, quote=True)
    safe_label = _html.escape(label)
    return (
        f'<a href="data:{mime};base64,{b64}" download="{safe_filename}" '
        f'style="display:flex;align-items:center;justify-content:center;gap:0.4rem;'
        f"width:100%;padding:0.4rem 0.75rem;margin-top:0.5rem;"
        f"background:white;border:1px solid #d1d5db;border-radius:0.5rem;"
        f"text-decoration:none;color:#374151;"
        f'font-size:0.875rem;font-weight:500;box-sizing:border-box;">'
        f"&#x2B07; {safe_label}</a>"
    )


def render() -> None:
    """Render the experiment detail page."""
    exp_name = SessionState.get_selected_experiment()
    if not exp_name:
        st.error("No experiment selected")
        if st.button("Back to Dashboard", key="detail_back_err"):
            SessionState.set_selected_experiment(None)
            st.rerun()
        return

    config = st.session_state.config
    exp_manager = ExperimentManager(str(config.get_experiments_path()))

    try:
        experiment = exp_manager.load_experiment(exp_name)
    except Exception as e:
        st.error(f"Failed to load experiment: {str(e)}")
        if st.button("Back to Dashboard", key="detail_back_err2"):
            SessionState.set_selected_experiment(None)
            st.rerun()
        return

    exp_path = experiment.path
    assert exp_path is not None

    # -- Header --
    with st.container(key="detail_back"):
        if st.button(
            "Back",
            key="detail_back_btn",
            icon=":material/arrow_back:",
        ):
            SessionState.set_selected_experiment(None)
            st.rerun()

    api_key = WebConfig.get_openai_api_key()
    exp_analysis_id = f"exp_analysis--{exp_path.name}"
    analysis_status = SessionState.get_analysis_status(exp_analysis_id)
    analysis_running = analysis_status == "running"
    analysis_file = exp_path / "analysis.md"
    has_analysis = analysis_file.exists()

    if analysis_running:
        ai_label = "Analyzing..."
        ai_icon = ":material/sync:"
        ai_disabled = True
        ai_help = "Analysis in progress..."
    elif has_analysis:
        ai_label = "Re-analyze"
        ai_icon = ":material/auto_awesome:"
        ai_disabled = not api_key
        ai_help = "Re-generate AI analysis" if api_key else "Set OpenAI API key in Preferences"
    else:
        ai_label = "Analyze experiment"
        ai_icon = ":material/auto_awesome:"
        ai_disabled = not api_key
        ai_help = "Generate AI analysis" if api_key else "Set OpenAI API key in Preferences"

    col_title, col_ai = st.columns([8, 2])
    with col_title:
        st.markdown(
            _dedent(f"""
<div style="margin-bottom:0.25rem;">
<h1 style="font-family:{FONTS['body']};font-size:1.875rem;font-weight:800;
color:{COLORS['gray_900']};margin:0;letter-spacing:-0.02em;">
{experiment.name}</h1>
</div>
"""),
            unsafe_allow_html=True,
        )
    with col_ai:
        with st.container(key="action_ai"):
            if st.button(
                ai_label,
                key="btn_ai",
                disabled=ai_disabled,
                help=ai_help,
                icon=ai_icon,
                width="stretch",
            ):
                if api_key:
                    run_ai_analysis(experiment)

    if experiment.description:
        st.caption(experiment.description)

    # -- Global Parameters --
    render_experiment_overview(experiment)

    # -- Spacer between sections --
    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    # -- Action bar + Scenarios --
    render_action_bar(experiment)
    _live_scenario_cards(experiment)

    # -- AI Analysis (always visible) --
    st.markdown(section_header("AI Analysis"), unsafe_allow_html=True)
    if has_analysis:
        with st.expander("View Analysis", expanded=True):
            try:
                with open(analysis_file, "r") as f:
                    st.markdown(f.read())
            except Exception as e:
                st.error(f"Failed to load analysis: {str(e)}")
    elif analysis_running:
        st.markdown(
            _dedent(f"""
<div style="display:flex;align-items:center;gap:0.75rem;padding:1.5rem;
background:{COLORS['white']};border-radius:12px;
border:1px solid {COLORS['gray_200']};box-shadow:0 1px 3px rgba(0,0,0,0.06);">
<div style="width:8px;height:8px;border-radius:50%;background:{COLORS['info']};
animation:pulse-dot 1.5s ease-in-out infinite;flex-shrink:0;"></div>
<span style="font-family:{FONTS['body']};font-size:0.875rem;
color:{COLORS['gray_600']};font-weight:500;">
Generating analysis... This may take a moment.</span>
</div>
"""),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _dedent(f"""
<div style="text-align:center;padding:2rem;background:{COLORS['white']};
border-radius:12px;border:1px solid {COLORS['gray_200']};
box-shadow:0 1px 3px rgba(0,0,0,0.06);">
<p style="font-family:{FONTS['body']};font-size:0.875rem;
color:{COLORS['gray_400']};margin:0;">
No analysis generated yet. Click "Analyze experiment" above to generate one.</p>
</div>
"""),
            unsafe_allow_html=True,
        )


def render_experiment_overview(experiment: Experiment) -> None:
    """Render global parameters as three refined cards."""
    st.markdown(section_header("Global Parameters"), unsafe_allow_html=True)

    params = experiment.parameters
    network_names = {
        "er": "Erdos-Renyi",
        "sf": "Scale-Free",
        "cg": "Complete Graph",
        "rrn": "Random Regular",
    }

    # Build rows for each card
    net_rows = [("Type", network_names.get(params.get("network", ""), "N/A"))]
    if "nodes" in params:
        net_rows.append(("Nodes", str(params["nodes"])))
    if "k_avg" in params:
        net_rows.append(("Avg Degree", str(params["k_avg"])))
    if "exponent" in params:
        net_rows.append(("Exponent", str(params["exponent"])))

    dist_rows = [("Type", params.get("distribution", "N/A").capitalize())]
    if "lambda" in params:
        dist_rows.append(("Infection Rate", str(params["lambda"])))
    if params.get("distribution") == "gamma":
        if "shape" in params:
            dist_rows.append(("Shape", str(params["shape"])))
        if "scale" in params:
            dist_rows.append(("Scale", str(params["scale"])))
    elif params.get("distribution") == "exponential":
        if "mu" in params:
            dist_rows.append(("Recovery Rate", str(params["mu"])))

    sim_rows = []
    if "samples" in params:
        sim_rows.append(("Samples", str(params["samples"])))
    if "num_runs" in params:
        sim_rows.append(("Runs", str(params["num_runs"])))
    if "t_max" in params:
        sim_rows.append(("Max Time", str(params["t_max"])))
    if "steps" in params:
        sim_rows.append(("Steps", str(params["steps"])))

    with st.container(key="params_section"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                params_card("Network", _ICON_NETWORK, net_rows),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                params_card("Distribution", _ICON_DIST, dist_rows),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                params_card("Simulation", _ICON_SIM, sim_rows),
                unsafe_allow_html=True,
            )


def render_action_bar(experiment: Experiment) -> None:
    """Render the scenarios section header with Add Scenario button."""
    col_title, col_add = st.columns([8, 2])
    with col_title:
        st.markdown(
            section_header("Scenarios"),
            unsafe_allow_html=True,
        )
    with col_add:
        with st.container(key="action_add_scenario"):
            if st.button(
                "Add Scenario",
                key="btn_add_scenario_bar",
                width="stretch",
                icon=":material/add:",
            ):
                show_add_scenario_modal(experiment)


@st.fragment(run_every=timedelta(seconds=2))
def _live_scenario_cards(experiment: Experiment) -> None:
    """Fragment wrapper that polls progress and re-renders scenario cards.

    Runs every 2 seconds to check subprocess status files and update
    progress bars without triggering a full page rerun.
    """
    poll_running_simulations()
    analysis_changed = poll_running_analyses()
    if analysis_changed:
        # Full page rerun so AI section (outside this fragment) re-renders
        st.rerun()
    render_scenario_cards(experiment)


def _get_scenario_entry(experiment: Experiment, label: str) -> dict[str, Any] | None:
    """Read a scenario's raw entry from data.json."""
    exp_path = experiment.path
    assert exp_path is not None
    data_file = exp_path / "data.json"
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("scenarios", []):
            if s.get("label") == label:
                result: dict[str, Any] = s
                return result
    except Exception:
        pass
    return None


def render_scenario_cards(experiment: Experiment) -> None:
    """Render scenarios as clickable cards with run and delete buttons."""
    if not experiment.scenarios:
        st.markdown(
            _dedent(f"""
<div style="text-align:center;padding:2rem;color:{COLORS['gray_500']};
font-family:{FONTS['body']};font-size:0.875rem;">
No scenarios defined yet. Add one above.
</div>
"""),
            unsafe_allow_html=True,
        )
        return

    exp_path = experiment.path
    assert exp_path is not None

    if "simulation_runner" not in st.session_state:
        st.session_state.simulation_runner = SimulationRunner()
    runner: SimulationRunner = st.session_state.simulation_runner

    for sc in experiment.scenarios:
        result_file = exp_path / f"{sc.normalized_label}.json"
        has_result = result_file.exists()
        scenario_id = f"sim--{exp_path.name}--{sc.normalized_label}"
        sc_status = "completed" if has_result else "created"
        if not has_result:
            sc_entry = _get_scenario_entry(experiment, sc.label)
            if sc_entry and sc_entry.get("status") == "edited":
                sc_status = "edited"
        sim_state = SessionState.get_simulation_status(scenario_id)
        if sim_state == "running":
            sc_status = "running"
        elif sim_state == "failed":
            sc_status = "failed"

        override_text = get_override_summary(sc, experiment.parameters)

        # Calculate progress fraction for running scenarios
        progress_frac = -1.0
        if sc_status == "running":
            prog_info = SessionState.get_simulation_progress(scenario_id)
            if prog_info and prog_info["total"] > 0:
                progress_frac = prog_info["progress"] / prog_info["total"]

        # Card container with overlay button
        with st.container(key=f"sc_card_{scenario_id}"):
            is_baseline = sc.label == "Baseline"
            col_body, col_run, col_edit, col_del = st.columns([8.5, 0.5, 0.5, 0.5])

            with col_body:
                st.markdown(
                    scenario_card(
                        label=sc.label,
                        override_text=override_text,
                        status=sc_status,
                        progress=progress_frac,
                    ),
                    unsafe_allow_html=True,
                )

            with col_run:
                with st.container(key=f"sc_run_{scenario_id}"):
                    is_running = sim_state == "running"
                    if st.button(
                        "",
                        key=f"run_sc_{scenario_id}",
                        width="stretch",
                        disabled=is_running,
                        icon=":material/play_arrow:",
                        help=(
                            "Running..."
                            if is_running
                            else "Re-run this scenario" if has_result else "Run this scenario"
                        ),
                    ):
                        if has_result:
                            show_rerun_scenario_dialog(experiment, sc, runner)
                        else:
                            _start_scenario_run(experiment, sc, runner)

            with col_edit:
                if not is_baseline:
                    with st.container(key=f"sc_edit_{scenario_id}"):
                        if st.button(
                            "",
                            key=f"edit_sc_{scenario_id}",
                            width="stretch",
                            disabled=is_running,
                            icon=":material/edit:",
                            help="Running..." if is_running else "Edit this scenario",
                        ):
                            show_edit_scenario_modal(experiment, sc)

            with col_del:
                if not is_baseline:
                    with st.container(key=f"sc_del_{scenario_id}"):
                        if st.button(
                            "",
                            key=f"del_sc_{scenario_id}",
                            width="stretch",
                            disabled=is_running,
                            icon=":material/delete:",
                            help="Running..." if is_running else "Delete this scenario",
                        ):
                            show_delete_scenario_dialog(experiment, sc)

            # Invisible overlay button for card click
            if st.button("open", key=f"sc_btn_{scenario_id}"):
                show_scenario_detail_modal(experiment, sc)


def get_override_summary(scenario: Scenario, global_params: Dict[str, Any]) -> str:
    """
    Get a summary of parameters that differ from global parameters.

    Args:
        scenario: The scenario to check
        global_params: Global experiment parameters

    Returns:
        String summary of overridden parameters
    """
    overrides = []
    skip_keys = {"label", "experiment_name", "output_path"}

    # Check each parameter
    scenario_dict = scenario.model_dump(by_alias=True)
    for key, value in scenario_dict.items():
        if key in skip_keys:
            continue
        if value is None:
            continue

        # Show if key doesn't exist in global (new param) or value differs
        if key not in global_params or not _values_equal(value, global_params[key]):
            overrides.append(f"{key}: {value}")

    return " | ".join(overrides) if overrides else ""


@st.dialog("Scenario Details", width="large")
def show_scenario_detail_modal(experiment: Experiment, scenario: Scenario) -> None:
    """Show detailed modal for a single scenario."""
    _modal_body_fragment(experiment, scenario)


@st.fragment(run_every=timedelta(seconds=2))
def _modal_body_fragment(experiment: Experiment, scenario: Scenario) -> None:
    """Fragment handling the full modal body.

    Renders title row (with AI/Export when results exist), parameters,
    Run button, and content area. Runs every 2 seconds to poll progress.
    Uses st.rerun(scope="fragment") for clean DOM transitions.
    """
    exp_path = experiment.path
    assert exp_path is not None

    scenario_id = f"sim--{exp_path.name}--{scenario.normalized_label}"
    modal_running_key = f"_modal_running_{scenario_id}"

    poll_running_simulations()
    analysis_changed = poll_running_analyses()
    if analysis_changed:
        st.rerun(scope="fragment")

    result_file = exp_path / f"{scenario.normalized_label}.json"
    analysis_file = exp_path / f"{scenario.normalized_label}_analysis.md"

    sim_status = SessionState.get_simulation_status(scenario_id)
    has_result = result_file.exists()

    # Clear modal running latch when simulation finishes (success or failure).
    # The latch bridges the gap between "user clicked Run" and "status file written".
    # Once sim_status reflects a terminal state, the latch is no longer needed.
    latch_on = st.session_state.get(modal_running_key, False)
    if latch_on and sim_status != "running":
        # Simulation finished (completed/failed/pending) — clear latch and refresh
        if has_result or sim_status in ("failed", "completed"):
            st.session_state.pop(modal_running_key, None)
            st.rerun(scope="fragment")

    is_running = sim_status == "running" or latch_on

    # Pre-read result data for title row actions + content
    result_json = None
    result_dict = None
    if has_result:
        try:
            with open(result_file, "r") as f:
                result_json = f.read()
            result_dict = json.loads(result_json)
        except Exception as e:
            st.error(f"Failed to load results: {str(e)}")
            return

    # -- Title row: title + action buttons (AI, Export, Run) --
    is_baseline = scenario.label == "Baseline"
    show_run = True  # All scenarios (including Baseline) can be run
    sc_key = scenario_id

    if has_result and result_json:
        api_key = WebConfig.get_openai_api_key()
        has_analysis = analysis_file.exists()
        sc_analysis_id = f"sc_analysis--{exp_path.name}--{scenario.normalized_label}"
        sc_analysis_status = SessionState.get_analysis_status(sc_analysis_id)
        sc_analysis_running = sc_analysis_status == "running"

        if sc_analysis_running:
            sc_ai_label = "Analyzing..."
            sc_ai_icon = ":material/sync:"
            sc_ai_disabled = True
            sc_ai_help = "Analysis in progress..."
        elif has_analysis:
            sc_ai_label = "Re-analyze"
            sc_ai_icon = ":material/auto_awesome:"
            sc_ai_disabled = not api_key
            sc_ai_help = (
                "Re-generate AI analysis" if api_key else "Set OpenAI API key in Preferences"
            )
        else:
            sc_ai_label = "Analyze scenario"
            sc_ai_icon = ":material/auto_awesome:"
            sc_ai_disabled = not api_key
            sc_ai_help = (
                "Analyze this scenario with AI" if api_key else "Set OpenAI API key in Preferences"
            )

        if show_run:
            cols = st.columns([4, 1.5, 1.5, 1.5])
            col_title, col_ai, col_export, col_run = cols
        else:
            cols = st.columns([6, 2, 2])
            col_title, col_ai, col_export = cols
        with col_title:
            st.title(scenario.label)
        with col_ai:
            with st.container(key=f"modal_action_ai_{sc_key}"):
                if st.button(
                    sc_ai_label,
                    key=f"modal_btn_ai_{sc_key}",
                    disabled=sc_ai_disabled,
                    help=sc_ai_help,
                    icon=sc_ai_icon,
                    width="stretch",
                ):
                    run_scenario_ai_analysis(experiment, scenario, result_file)
        with col_export:
            with st.container(key=f"modal_action_export_{sc_key}"):
                with st.popover(
                    "Export",
                    icon=":material/download:",
                    use_container_width=True,
                ):
                    _export_fmt = st.radio(
                        "Format",
                        options=["json", "csv", "excel", "md", "html"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"export_fmt_{sc_key}",
                    )
                    assert result_dict is not None
                    _export_data, _export_mime, _export_ext = DataManager.to_bytes(
                        result_dict, _export_fmt
                    )
                    st.markdown(
                        _download_anchor(
                            _export_data,
                            f"{scenario.normalized_label}{_export_ext}",
                            _export_mime,
                        ),
                        unsafe_allow_html=True,
                    )
    else:
        if show_run:
            col_title, col_run = st.columns([8, 2])
        else:
            col_title = st.columns([1])[0]
            col_run = None
        with col_title:
            st.title(scenario.label)

    # -- Run button in title row (for non-Baseline) --
    if show_run:
        if "simulation_runner" not in st.session_state:
            st.session_state.simulation_runner = SimulationRunner()
        runner: SimulationRunner = st.session_state.simulation_runner
        assert col_run is not None
        with col_run:
            with st.container(key=f"modal_action_run_{sc_key}"):
                run_label = "Re-run scenario" if has_result else "Run scenario"
                if st.button(
                    run_label,
                    type="primary",
                    key=f"modal_btn_run_{sc_key}",
                    width="stretch",
                    icon=":material/play_arrow:",
                    disabled=is_running,
                ):
                    # Move stale artifacts to .bak BEFORE spawning so
                    # fast scenarios don't race.  Restore if launch fails.
                    bak_r, bak_a = _backup_scenario_artifacts(
                        result_file, analysis_file, has_result
                    )
                    sid = _start_scenario_run_no_rerun(experiment, scenario, runner)
                    _finalize_artifact_backups(
                        result_file, analysis_file, bak_r, bak_a, sid is not None
                    )
                    if sid:
                        st.session_state[modal_running_key] = True
                    st.rerun(scope="fragment")

    # -- Parameters --
    render_scenario_parameters(scenario, experiment.parameters, experiment_name=exp_path.name)

    # -- Content area --
    if has_result and result_dict:
        _render_result_content(result_dict, experiment, scenario, analysis_file)
    elif is_running:
        prog_info = SessionState.get_simulation_progress(scenario_id)
        progress = 0.0
        if prog_info and prog_info["total"] > 0:
            progress = prog_info["progress"] / prog_info["total"]
        st.markdown(
            circular_progress_html(progress, "Running simulation..."),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _dedent(f"""
<div style="display:flex;flex-direction:column;align-items:center;
justify-content:center;text-align:center;padding:3rem 2rem;
color:{COLORS['gray_500']};min-height:200px;">
<p style="font-family:{FONTS['body']};font-size:1rem;
font-weight:600;color:{COLORS['gray_700']};margin:0 0 0.5rem 0;">
No results available</p>
<p style="font-family:{FONTS['body']};font-size:0.875rem;
color:{COLORS['gray_400']};margin:0;">
Run this scenario to generate simulation results.</p>
</div>
"""),
            unsafe_allow_html=True,
        )


def _render_result_content(
    result_dict: dict[str, Any],
    experiment: Experiment,
    scenario: Scenario,
    analysis_file: Path,
) -> None:
    """Render result metrics, chart, comparison, and AI analysis."""
    exp_path = experiment.path
    assert exp_path is not None

    has_analysis = analysis_file.exists()

    # -- Key Metrics --
    st.subheader("Key Metrics")
    result_metric_cards(result_dict)

    st.divider()

    # -- SIR Dynamics --
    st.subheader("SIR Dynamics")

    # -- Chart controls (single row) --
    (
        col_s,
        col_i,
        col_r,
        col_spacer,
        col_err,
        col_type,
    ) = st.columns([0.8, 0.8, 0.8, 1.875, 0.6, 0.8])
    sc_key = f"sim--{exp_path.name}--{scenario.normalized_label}"
    with col_s:
        show_s = st.checkbox("Susceptible", value=True, key=f"modal_show_s_{sc_key}")
    with col_i:
        show_i = st.checkbox("Infected", value=True, key=f"modal_show_i_{sc_key}")
    with col_r:
        show_r = st.checkbox("Recovered", value=True, key=f"modal_show_r_{sc_key}")
    with col_err:
        show_errors = st.checkbox(
            "Error bars",
            value=True,
            key=f"modal_show_errors_{sc_key}",
        )
    with col_type:
        chart_type_label = st.selectbox(
            "Chart Type",
            ["Lines", "Lines + Markers", "Area"],
            key=f"modal_chart_mode_{sc_key}",
            label_visibility="collapsed",
        )

    # Map selectbox label to chart_mode parameter
    chart_mode_map = {
        "Lines": "lines",
        "Lines + Markers": "lines+markers",
        "Area": "area",
    }
    chart_mode = chart_mode_map.get(chart_type_label, "lines")

    states_to_plot = []
    if show_s:
        states_to_plot.append("S")
    if show_i:
        states_to_plot.append("I")
    if show_r:
        states_to_plot.append("R")

    # -- Discover other scenarios with results --
    other_scenarios = []
    for other_sc in experiment.scenarios:
        if other_sc.label == scenario.label:
            continue
        other_file = exp_path / f"{other_sc.normalized_label}.json"
        if other_file.exists():
            other_scenarios.append(other_sc)

    # -- Reserve visual space for chart (rendered after controls) --
    chart_container = st.container()

    # -- Compare controls (execute first to set state) --
    comparing = False
    comp_results: List[Dict] = []
    comp_labels: List[str] = []

    if other_scenarios:
        st.subheader("Compare with Other Scenarios")
        compare_options = [sc.label for sc in other_scenarios]

        compare_key = f"modal_compare_{exp_path.name}_{scenario.normalized_label}"
        selected_labels = st.multiselect(
            "Select scenarios to compare",
            options=compare_options,
            key=compare_key,
            label_visibility="collapsed",
        )

        # Auto-trigger comparison when scenarios are selected
        if selected_labels:
            comp_results.append(result_dict)
            comp_labels.append(scenario.label)

            for sel_label in selected_labels:
                for other_sc in other_scenarios:
                    if other_sc.label == sel_label:
                        sel_file = exp_path / f"{other_sc.normalized_label}.json"
                        try:
                            with open(sel_file, "r") as f:
                                comp_results.append(json.load(f))
                            comp_labels.append(sel_label)
                        except Exception:
                            continue

            if len(comp_results) >= 2:
                comparing = True

    # -- Render chart into reserved container --
    with chart_container:
        if not states_to_plot:
            st.warning("Select at least one state to display")
        elif comparing:
            config = st.session_state.config
            fig = create_comparison_figure(
                comp_results,
                comp_labels,
                title=f"Comparison: {experiment.name}",
                states=states_to_plot,
                height=config.get("chart_height", 500),
                template=config.get("chart_template", "plotly_white"),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            config = st.session_state.config
            fig = create_sir_figure(
                result_dict,
                title=scenario.label,
                states=states_to_plot,
                show_error_bands=show_errors and "S_err" in result_dict,
                height=config.get("chart_height", 500),
                chart_mode=chart_mode,
                state_colors={
                    "S": config.get("chart_color_s", "#4477AA"),
                    "I": config.get("chart_color_i", "#EE6677"),
                    "R": config.get("chart_color_r", "#228833"),
                },
                template=config.get("chart_template", "plotly_white"),
            )
            st.plotly_chart(fig, width="stretch")

    # -- Comparison statistics --
    if comparing:
        st.subheader("Comparison Statistics")
        render_comparison_stats(comp_results, comp_labels)

    # -- AI Analysis (always visible) --
    st.divider()
    st.subheader("AI Analysis")

    sc_analysis_id = f"sc_analysis--{exp_path.name}--{scenario.normalized_label}"
    sc_analysis_running = SessionState.get_analysis_status(sc_analysis_id) == "running"

    if has_analysis:
        try:
            with open(analysis_file, "r") as f:
                st.markdown(f.read())
        except Exception as e:
            st.error(f"Failed to load analysis: {str(e)}")
    elif sc_analysis_running:
        st.markdown(
            _dedent(f"""
<div style="display:flex;align-items:center;gap:0.75rem;padding:1.25rem;
background:{COLORS['gray_50']};border-radius:8px;">
<div style="width:8px;height:8px;border-radius:50%;background:{COLORS['info']};
animation:pulse-dot 1.5s ease-in-out infinite;flex-shrink:0;"></div>
<span style="font-family:{FONTS['body']};font-size:0.875rem;
color:{COLORS['gray_600']};font-weight:500;">
Generating analysis...</span>
</div>
"""),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _dedent(f"""
<div style="text-align:center;padding:1.5rem;background:{COLORS['gray_50']};
border-radius:8px;">
<p style="font-family:{FONTS['body']};font-size:0.813rem;
color:{COLORS['gray_400']};margin:0;">
No analysis generated yet. Click "Analyze scenario" above to generate one.</p>
</div>
"""),
            unsafe_allow_html=True,
        )


def render_scenario_parameters(
    scenario: Scenario,
    global_params: Dict[str, Any],
    experiment_name: str = "",
) -> None:
    """Render scenario parameters with visual distinction for overrides."""
    scenario_dict = scenario.model_dump(by_alias=True)

    network_keys = ["network", "nodes", "k_avg", "exponent"]
    dist_keys = ["distribution", "shape", "scale", "mu", "lambda"]
    sim_keys = ["samples", "num_runs", "t_max", "steps", "initial_perc"]

    def _build_rows(keys: list) -> list:
        rows = []
        for key in keys:
            if key in scenario_dict and scenario_dict[key] is not None:
                val = scenario_dict[key]
                is_override = not _values_equal(global_params.get(key), val)
                rows.append((key, str(val), is_override))
        return rows

    sc_key = (
        f"{experiment_name}_{scenario.normalized_label}"
        if experiment_name
        else scenario.normalized_label
    )
    with st.container(key=f"modal_params_section_{sc_key}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                params_card("Network", _ICON_NETWORK, _build_rows(network_keys)),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                params_card("Distribution", _ICON_DIST, _build_rows(dist_keys)),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                params_card("Simulation", _ICON_SIM, _build_rows(sim_keys)),
                unsafe_allow_html=True,
            )


def render_comparison_stats(results: List[Dict], labels: List[str]) -> None:
    """Render a comparison table of key statistics."""
    from datetime import timedelta

    import humanize
    import numpy as np

    stats = []
    for result_dict, label in zip(results, labels):
        I_val = np.array(result_dict["I_val"])
        R_val = np.array(result_dict["R_val"])
        time = np.array(result_dict["time"])

        exec_time = result_dict.get("metadata", {}).get("execution_time")
        if exec_time is not None:
            duration = humanize.precisedelta(
                timedelta(seconds=exec_time), minimum_unit="seconds", format="%0.0f"
            )
        else:
            duration = "N/A"

        stats.append(
            {
                "Scenario": label,
                "Peak Infected": f"{np.max(I_val):.2%}",
                "Peak Time": f"{time[np.argmax(I_val)]:.2f}",
                "Final Size": f"{R_val[-1]:.2%}",
                "Duration": duration,
            }
        )

    df = pd.DataFrame(stats)
    st.dataframe(df, width="stretch", hide_index=True)


@st.dialog("Add Scenario", width="large")
def show_add_scenario_modal(experiment: Experiment) -> None:
    """Show modal to add a new scenario to the experiment."""
    exp_path = experiment.path
    assert exp_path is not None
    exp_key = exp_path.name
    st.title("Add New Scenario")

    label = st.text_input(
        "Scenario Label",
        placeholder="e.g., High Infection Rate",
        help="Descriptive name for this scenario",
    )

    st.subheader("Parameter Overrides")
    st.caption(
        "Values are pre-filled with experiment defaults. "
        "Only changed values will be saved as overrides."
    )

    global_params = experiment.parameters
    override_params: Dict[str, Any] = {}

    # -- Network Overrides --
    with st.expander("Network Overrides", expanded=False):
        network_options = ["er", "sf", "cg", "rrn"]
        network_names = {
            "er": "Erdos-Renyi",
            "sf": "Scale-Free",
            "cg": "Complete Graph",
            "rrn": "Random Regular",
        }
        global_network = global_params.get("network", "er")
        global_idx = (
            network_options.index(global_network) if global_network in network_options else 0
        )
        override_network = st.selectbox(
            "Network Type",
            options=network_options,
            format_func=lambda x: network_names.get(x, x),
            index=global_idx,
            key=f"add_sc_network_{exp_key}",
            help=f"Experiment default: {network_names.get(global_network, global_network)}",
        )
        network_changed = override_network != global_network
        if network_changed:
            override_params["network"] = override_network

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            global_nodes = int(global_params.get("nodes", 1000))
            override_nodes = st.number_input(
                "Nodes",
                min_value=1,
                value=global_nodes,
                step=100,
                key=f"add_sc_nodes_{exp_key}",
                help=f"Experiment default: {global_nodes}",
            )
            if override_nodes != global_nodes:
                override_params["nodes"] = override_nodes

        with col_n2:
            global_k_avg = float(global_params.get("k_avg", 10.0))
            override_k_avg = st.number_input(
                "Average Degree (k_avg)",
                min_value=0.1,
                value=global_k_avg,
                step=1.0,
                key=f"add_sc_k_avg_{exp_key}",
                help=f"Experiment default: {global_k_avg}",
            )
            # Always include k_avg when network type is overridden (required for er/sf/rrn)
            if override_k_avg != global_k_avg or network_changed:
                override_params["k_avg"] = override_k_avg

        # Exponent only relevant for scale-free networks
        effective_network = override_network or global_network
        if effective_network == "sf":
            global_exponent = float(global_params.get("exponent", 2.5))
            override_exponent = st.number_input(
                "Power-law Exponent",
                min_value=0.1,
                value=global_exponent,
                step=0.1,
                key=f"add_sc_exponent_{exp_key}",
                help=f"Experiment default: {global_exponent}",
            )
            # Always include exponent when network type is overridden to sf
            if override_exponent != global_exponent or network_changed:
                override_params["exponent"] = override_exponent

    # -- Distribution Overrides --
    with st.expander("Distribution Overrides", expanded=False):
        dist_options = ["gamma", "exponential"]
        global_dist = global_params.get("distribution", "gamma")
        global_dist_idx = dist_options.index(global_dist) if global_dist in dist_options else 0
        override_dist = st.selectbox(
            "Distribution Type",
            options=dist_options,
            format_func=lambda x: x.capitalize(),
            index=global_dist_idx,
            key=f"add_sc_distribution_{exp_key}",
            help=f"Experiment default: {global_dist.capitalize()}",
        )
        if override_dist != global_dist:
            override_params["distribution"] = override_dist

        global_lambda = float(global_params.get("lambda", 1.0))
        override_lambda = st.number_input(
            "Infection Rate (lambda)",
            min_value=0.01,
            value=global_lambda,
            step=0.1,
            key=f"add_sc_lambda_{exp_key}",
            help=f"Experiment default: {global_lambda}",
        )
        if override_lambda != global_lambda:
            override_params["lambda"] = override_lambda

        effective_dist = override_dist or global_dist
        # When distribution is overridden, always include the
        # distribution-specific required params so that
        # Scenario.from_merged() won't fail validation.
        dist_changed = override_dist != global_dist
        if effective_dist == "gamma":
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                global_shape = float(global_params.get("shape", 2.0))
                override_shape = st.number_input(
                    "Shape",
                    min_value=0.01,
                    value=global_shape,
                    step=0.1,
                    key=f"add_sc_shape_{exp_key}",
                    help=f"Experiment default: {global_shape}",
                )
                if override_shape != global_shape or dist_changed:
                    override_params["shape"] = override_shape
            with col_d2:
                global_scale = float(global_params.get("scale", 1.0))
                override_scale = st.number_input(
                    "Scale",
                    min_value=0.01,
                    value=global_scale,
                    step=0.1,
                    key=f"add_sc_scale_{exp_key}",
                    help=f"Experiment default: {global_scale}",
                )
                if override_scale != global_scale or dist_changed:
                    override_params["scale"] = override_scale
        elif effective_dist == "exponential":
            global_mu = float(global_params.get("mu", 1.0))
            override_mu = st.number_input(
                "Recovery Rate (mu)",
                min_value=0.01,
                value=global_mu,
                step=0.1,
                key=f"add_sc_mu_{exp_key}",
                help=f"Experiment default: {global_mu}",
            )
            if override_mu != global_mu or dist_changed:
                override_params["mu"] = override_mu

    # -- Simulation Overrides --
    with st.expander("Simulation Overrides", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            global_samples = int(global_params.get("samples", 50))
            override_samples = st.number_input(
                "Samples",
                min_value=1,
                value=global_samples,
                step=10,
                key=f"add_sc_samples_{exp_key}",
                help=f"Experiment default: {global_samples}",
            )
            if override_samples != global_samples:
                override_params["samples"] = override_samples
        with col_s2:
            global_num_runs = int(global_params.get("num_runs", 2))
            override_num_runs = st.number_input(
                "Number of Runs",
                min_value=1,
                value=global_num_runs,
                step=1,
                key=f"add_sc_num_runs_{exp_key}",
                help=f"Experiment default: {global_num_runs}",
            )
            if override_num_runs != global_num_runs:
                override_params["num_runs"] = override_num_runs

        col_s3, col_s4 = st.columns(2)
        with col_s3:
            global_t_max = float(global_params.get("t_max", 10.0))
            override_t_max = st.number_input(
                "Max Time (t_max)",
                min_value=0.01,
                value=global_t_max,
                step=1.0,
                key=f"add_sc_t_max_{exp_key}",
                help=f"Experiment default: {global_t_max}",
            )
            if override_t_max != global_t_max:
                override_params["t_max"] = override_t_max
        with col_s4:
            global_steps = int(global_params.get("steps", 100))
            override_steps = st.number_input(
                "Steps",
                min_value=1,
                value=global_steps,
                step=10,
                key=f"add_sc_steps_{exp_key}",
                help=f"Experiment default: {global_steps}",
            )
            if override_steps != global_steps:
                override_params["steps"] = override_steps

        global_initial_perc = float(global_params.get("initial_perc", 0.01))
        override_initial_perc = st.number_input(
            "Initial Infected Fraction",
            min_value=0.001,
            max_value=1.0,
            value=global_initial_perc,
            step=0.01,
            format="%.3f",
            key=f"add_sc_initial_perc_{exp_key}",
            help=f"Experiment default: {global_initial_perc}",
        )
        if override_initial_perc != global_initial_perc:
            override_params["initial_perc"] = override_initial_perc

    # Action buttons (pinned to bottom via CSS on modal_actions container)
    with st.container(key=f"modal_actions_add_{exp_key}"):
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Cancel", width="stretch"):
                st.rerun()

        with col2:
            if st.button(
                "Add Scenario",
                type="primary",
                width="stretch",
                icon=":material/add:",
            ):
                if not label:
                    st.error("Please provide a scenario label")
                    return

                try:
                    add_scenario_to_experiment(experiment, label, override_params)
                    st.success(f"Scenario '{label}' added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add scenario: {str(e)}")


def add_scenario_to_experiment(
    experiment: Experiment, label: str, override_params: Dict[str, Any]
) -> None:
    """
    Add a new scenario to an existing experiment.

    Args:
        experiment: The experiment to add to
        label: Scenario label
        override_params: Parameters that override global settings

    Raises:
        ValueError: If a scenario with the same normalized label already exists
    """
    from spkmc.models.scenario import Scenario as ScenarioModel

    exp_path = experiment.path
    assert exp_path is not None

    # Load current data.json
    data_file = exp_path / "data.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check that label normalizes to a non-empty filename
    new_norm = ScenarioModel.normalize_label(label)
    if not new_norm:
        raise ValueError(
            f"Scenario label '{label}' normalizes to an empty filename. "
            "Use a label with at least one alphanumeric character."
        )

    # Check for normalized label collision
    for sc in data.get("scenarios", []):
        existing_norm = ScenarioModel.normalize_label(sc.get("label", ""))
        if existing_norm == new_norm:
            raise ValueError(
                f"A scenario with a conflicting name already exists: '{sc.get('label')}' "
                f"(both normalize to '{new_norm}')"
            )

    # Add new scenario.
    # When a global `parameters` block exists, store only overrides (label + diffs).
    # For legacy experiments without globals, include the full effective parameter
    # set so the scenario entry remains valid on reload.
    global_params = data.get("parameters", {})
    if global_params:
        new_scenario: Dict[str, Any] = {"label": label, **override_params}
    else:
        # Derive defaults from the first existing scenario (minus meta keys)
        existing = data.get("scenarios", [])
        base_params: Dict[str, Any] = {}
        if existing:
            meta_keys = {"label", "status"}
            base_params = {k: v for k, v in existing[0].items() if k not in meta_keys}
        new_scenario = {"label": label, **base_params, **override_params}
    data.setdefault("scenarios", []).append(new_scenario)

    # Write back (atomic to prevent corruption on crash)
    from spkmc.web import atomic_json_write

    atomic_json_write(data_file, data)


def _start_scenario_run(
    experiment: Experiment, scenario: Scenario, runner: SimulationRunner
) -> None:
    """Start a scenario simulation run."""
    exp_path = experiment.path
    assert exp_path is not None

    run_id = runner.run_scenario(experiment, scenario, show_progress=True)
    if run_id:
        status_info = runner.get_status(run_id)
        if status_info:
            # Key by scenario_id (stable, matches render lookups).
            # Store run_id inside info for status file lookups.
            scenario_id = f"sim--{exp_path.name}--{scenario.normalized_label}"
            status_info["run_id"] = run_id
            SessionState.add_running_simulation(scenario_id, status_info)
        st.rerun()


def _backup_scenario_artifacts(
    result_file: Path, analysis_file: Path, has_result: bool
) -> Tuple[Optional[Path], Optional[Path]]:
    """Rename scenario result/analysis to ``.bak`` so the UI sees them as absent.

    Returns ``(backup_result, backup_analysis)`` paths (None when the
    original did not exist).  Call :func:`_finalize_artifact_backups`
    after the launch attempt to discard or restore backups.
    """
    backup_result: Optional[Path] = None
    backup_analysis: Optional[Path] = None
    if has_result and result_file.exists():
        backup_result = result_file.with_suffix(".json.bak")
        result_file.rename(backup_result)
    if analysis_file.exists():
        backup_analysis = analysis_file.with_suffix(".md.bak")
        analysis_file.rename(backup_analysis)
    return backup_result, backup_analysis


def _finalize_artifact_backups(
    result_file: Path,
    analysis_file: Path,
    backup_result: Optional[Path],
    backup_analysis: Optional[Path],
    launch_ok: bool,
) -> None:
    """Keep backups on successful launch, or restore originals on failure.

    On launch success the `.bak` files are intentionally kept: the subprocess
    has only *started* — it may still fail later.  The backups are harmless
    (the UI only checks canonical filenames, never `.bak`) and act as a
    safety net.  They are naturally superseded when the scenario is re-run
    successfully or when the experiment/scenario is deleted.
    """
    if not launch_ok:
        if backup_result is not None and backup_result.exists():
            backup_result.rename(result_file)
        if backup_analysis is not None and backup_analysis.exists():
            backup_analysis.rename(analysis_file)


def _start_scenario_run_no_rerun(
    experiment: Experiment, scenario: Scenario, runner: SimulationRunner
) -> str | None:
    """Start a scenario run without calling st.rerun().

    Returns the scenario_id if started successfully, None otherwise.
    Used by the modal to keep the dialog open while polling progress.
    """
    exp_path = experiment.path
    assert exp_path is not None

    run_id = runner.run_scenario(experiment, scenario, show_progress=True)
    if run_id:
        status_info = runner.get_status(run_id)
        if status_info:
            scenario_id = f"sim--{exp_path.name}--{scenario.normalized_label}"
            status_info["run_id"] = run_id
            SessionState.add_running_simulation(scenario_id, status_info)
            return scenario_id
    return None


@st.dialog("Re-run Scenario")
def show_rerun_scenario_dialog(
    experiment: Experiment,
    scenario: Scenario,
    runner: SimulationRunner,
) -> None:
    """Show confirmation dialog before re-running a completed scenario."""
    exp_path = experiment.path
    assert exp_path is not None

    scope = f"sim--{exp_path.name}--{scenario.normalized_label}"
    st.markdown(
        f"**{scenario.label}** already has results. "
        "Re-running will overwrite the existing data.",
    )

    col_cancel, col_rerun = st.columns(2)
    with col_cancel:
        if st.button("Cancel", key=f"rerun_cancel_{scope}", width="stretch"):
            st.rerun()
    with col_rerun:
        if st.button(
            "Re-run",
            key=f"rerun_confirm_{scope}",
            type="primary",
            width="stretch",
            icon=":material/play_arrow:",
        ):
            # Move stale artifacts to .bak BEFORE spawning so
            # fast scenarios don't race.  Restore if launch fails.
            result_file = exp_path / f"{scenario.normalized_label}.json"
            analysis_file = exp_path / f"{scenario.normalized_label}_analysis.md"
            bak_r, bak_a = _backup_scenario_artifacts(
                result_file, analysis_file, result_file.exists()
            )
            sid = _start_scenario_run_no_rerun(experiment, scenario, runner)
            _finalize_artifact_backups(result_file, analysis_file, bak_r, bak_a, sid is not None)
            st.rerun()


@st.dialog("Delete Scenario")
def show_delete_scenario_dialog(experiment: Experiment, scenario: Scenario) -> None:
    """Show confirmation dialog before deleting a scenario."""
    exp_path = experiment.path
    assert exp_path is not None

    scope = f"sim--{exp_path.name}--{scenario.normalized_label}"
    st.markdown(
        f"Are you sure you want to delete **{scenario.label}**?",
    )

    # Block deletion of the last scenario (would make experiment unloadable)
    if len(experiment.scenarios) <= 1:
        st.error("Cannot delete the only scenario in an experiment.")
        if st.button("Close", key=f"del_close_{scope}", width="stretch"):
            st.rerun()
        return

    result_file = exp_path / f"{scenario.normalized_label}.json"
    if result_file.exists():
        st.warning("This scenario has results that will also be deleted.")

    col_cancel, col_delete = st.columns(2)
    with col_cancel:
        if st.button("Cancel", key=f"del_cancel_{scope}", width="stretch"):
            st.rerun()
    with col_delete:
        if st.button(
            "Delete",
            key=f"del_confirm_{scope}",
            type="primary",
            width="stretch",
        ):
            delete_scenario_from_experiment(experiment, scenario)
            st.rerun()


def delete_scenario_from_experiment(experiment: Experiment, scenario: Scenario) -> None:
    """
    Remove a scenario from an experiment.

    Deletes the scenario entry from data.json and removes the result file
    if it exists.

    Args:
        experiment: The parent experiment
        scenario: The scenario to delete
    """
    exp_path = experiment.path
    assert exp_path is not None

    # Remove from data.json
    data_file = exp_path / "data.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios_list = data.get("scenarios", [])
    new_scenarios = [s for s in scenarios_list if s.get("label") != scenario.label]
    if not new_scenarios:
        raise ValueError("Cannot delete the last scenario in an experiment")
    data["scenarios"] = new_scenarios

    # Write back (atomic to prevent corruption on crash)
    from spkmc.web import atomic_json_write

    atomic_json_write(data_file, data)

    # Remove result and analysis files if they exist
    result_file = exp_path / f"{scenario.normalized_label}.json"
    if result_file.exists():
        result_file.unlink()
    analysis_file = exp_path / f"{scenario.normalized_label}_analysis.md"
    if analysis_file.exists():
        analysis_file.unlink()


@st.dialog("Edit Scenario", width="large")
def show_edit_scenario_modal(experiment: Experiment, scenario: Scenario) -> None:
    """Show modal to edit an existing scenario."""
    exp_path = experiment.path
    assert exp_path is not None

    edit_scope = f"sim--{exp_path.name}--{scenario.normalized_label}"
    st.title(f"Edit: {scenario.label}")

    original_label = scenario.label
    label = st.text_input(
        "Scenario Label",
        value=scenario.label,
        help="Descriptive name for this scenario",
        key=f"edit_sc_label_{edit_scope}",
    )

    st.subheader("Parameter Overrides")
    st.caption(
        "Values are pre-filled with this scenario's current settings. "
        "Only values that differ from experiment defaults will be saved as overrides."
    )

    global_params = experiment.parameters
    scenario_dict = scenario.model_dump(by_alias=True)
    override_params: Dict[str, Any] = {}

    # -- Network Overrides --
    with st.expander("Network Overrides", expanded=False):
        network_options = ["er", "sf", "cg", "rrn"]
        network_names = {
            "er": "Erdos-Renyi",
            "sf": "Scale-Free",
            "cg": "Complete Graph",
            "rrn": "Random Regular",
        }
        current_network = scenario_dict.get("network", global_params.get("network", "er"))
        global_network = global_params.get("network", "er")
        current_idx = (
            network_options.index(current_network) if current_network in network_options else 0
        )
        override_network = st.selectbox(
            "Network Type",
            options=network_options,
            format_func=lambda x: network_names.get(x, x),
            index=current_idx,
            key=f"edit_sc_network_{edit_scope}",
            help=f"Experiment default: {network_names.get(global_network, global_network)}",
        )
        network_changed = not _values_equal(override_network, global_network)
        if network_changed:
            override_params["network"] = override_network

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            global_nodes = int(global_params.get("nodes", 1000))
            current_nodes = int(scenario_dict.get("nodes", global_nodes))
            override_nodes = st.number_input(
                "Nodes",
                min_value=1,
                value=current_nodes,
                step=100,
                key=f"edit_sc_nodes_{edit_scope}",
                help=f"Experiment default: {global_nodes}",
            )
            if not _values_equal(override_nodes, global_nodes):
                override_params["nodes"] = override_nodes

        with col_n2:
            global_k_avg = float(global_params.get("k_avg", 10.0))
            current_k_avg = float(scenario_dict.get("k_avg", global_k_avg))
            override_k_avg = st.number_input(
                "Average Degree (k_avg)",
                min_value=0.1,
                value=current_k_avg,
                step=1.0,
                key=f"edit_sc_k_avg_{edit_scope}",
                help=f"Experiment default: {global_k_avg}",
            )
            # Always include k_avg when network type is overridden (required for er/sf/rrn)
            if not _values_equal(override_k_avg, global_k_avg) or network_changed:
                override_params["k_avg"] = override_k_avg

        effective_network = override_network or current_network
        if effective_network == "sf":
            global_exponent = float(global_params.get("exponent", 2.5))
            current_exponent = float(scenario_dict.get("exponent", global_exponent))
            override_exponent = st.number_input(
                "Power-law Exponent",
                min_value=0.1,
                value=current_exponent,
                step=0.1,
                key=f"edit_sc_exponent_{edit_scope}",
                help=f"Experiment default: {global_exponent}",
            )
            # Always include exponent when network type is overridden to sf
            if not _values_equal(override_exponent, global_exponent) or network_changed:
                override_params["exponent"] = override_exponent

    # -- Distribution Overrides --
    with st.expander("Distribution Overrides", expanded=False):
        dist_options = ["gamma", "exponential"]
        global_dist = global_params.get("distribution", "gamma")
        current_dist = scenario_dict.get("distribution", global_dist)
        current_dist_idx = dist_options.index(current_dist) if current_dist in dist_options else 0
        override_dist = st.selectbox(
            "Distribution Type",
            options=dist_options,
            format_func=lambda x: x.capitalize(),
            index=current_dist_idx,
            key=f"edit_sc_distribution_{edit_scope}",
            help=f"Experiment default: {global_dist.capitalize()}",
        )
        if not _values_equal(override_dist, global_dist):
            override_params["distribution"] = override_dist

        global_lambda = float(global_params.get("lambda", 1.0))
        current_lambda = float(scenario_dict.get("lambda", global_lambda))
        override_lambda = st.number_input(
            "Infection Rate (lambda)",
            min_value=0.01,
            value=current_lambda,
            step=0.1,
            key=f"edit_sc_lambda_{edit_scope}",
            help=f"Experiment default: {global_lambda}",
        )
        if not _values_equal(override_lambda, global_lambda):
            override_params["lambda"] = override_lambda

        effective_dist = override_dist or current_dist
        # When distribution is overridden, always include the
        # distribution-specific required params so that
        # Scenario.from_merged() won't fail validation.
        dist_changed = not _values_equal(override_dist, global_dist)
        if effective_dist == "gamma":
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                global_shape = float(global_params.get("shape", 2.0))
                current_shape = float(scenario_dict.get("shape", global_shape))
                override_shape = st.number_input(
                    "Shape",
                    min_value=0.01,
                    value=current_shape,
                    step=0.1,
                    key=f"edit_sc_shape_{edit_scope}",
                    help=f"Experiment default: {global_shape}",
                )
                if not _values_equal(override_shape, global_shape) or dist_changed:
                    override_params["shape"] = override_shape
            with col_d2:
                global_scale = float(global_params.get("scale", 1.0))
                current_scale = float(scenario_dict.get("scale", global_scale))
                override_scale = st.number_input(
                    "Scale",
                    min_value=0.01,
                    value=current_scale,
                    step=0.1,
                    key=f"edit_sc_scale_{edit_scope}",
                    help=f"Experiment default: {global_scale}",
                )
                if not _values_equal(override_scale, global_scale) or dist_changed:
                    override_params["scale"] = override_scale
        elif effective_dist == "exponential":
            global_mu = float(global_params.get("mu", 1.0))
            current_mu = float(scenario_dict.get("mu", global_mu))
            override_mu = st.number_input(
                "Recovery Rate (mu)",
                min_value=0.01,
                value=current_mu,
                step=0.1,
                key=f"edit_sc_mu_{edit_scope}",
                help=f"Experiment default: {global_mu}",
            )
            if not _values_equal(override_mu, global_mu) or dist_changed:
                override_params["mu"] = override_mu

    # -- Simulation Overrides --
    with st.expander("Simulation Overrides", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            global_samples = int(global_params.get("samples", 50))
            current_samples = int(scenario_dict.get("samples", global_samples))
            override_samples = st.number_input(
                "Samples",
                min_value=1,
                value=current_samples,
                step=10,
                key=f"edit_sc_samples_{edit_scope}",
                help=f"Experiment default: {global_samples}",
            )
            if not _values_equal(override_samples, global_samples):
                override_params["samples"] = override_samples
        with col_s2:
            global_num_runs = int(global_params.get("num_runs", 2))
            current_num_runs = int(scenario_dict.get("num_runs", global_num_runs))
            override_num_runs = st.number_input(
                "Number of Runs",
                min_value=1,
                value=current_num_runs,
                step=1,
                key=f"edit_sc_num_runs_{edit_scope}",
                help=f"Experiment default: {global_num_runs}",
            )
            if not _values_equal(override_num_runs, global_num_runs):
                override_params["num_runs"] = override_num_runs

        col_s3, col_s4 = st.columns(2)
        with col_s3:
            global_t_max = float(global_params.get("t_max", 10.0))
            current_t_max = float(scenario_dict.get("t_max", global_t_max))
            override_t_max = st.number_input(
                "Max Time (t_max)",
                min_value=0.01,
                value=current_t_max,
                step=1.0,
                key=f"edit_sc_t_max_{edit_scope}",
                help=f"Experiment default: {global_t_max}",
            )
            if not _values_equal(override_t_max, global_t_max):
                override_params["t_max"] = override_t_max
        with col_s4:
            global_steps = int(global_params.get("steps", 100))
            current_steps = int(scenario_dict.get("steps", global_steps))
            override_steps = st.number_input(
                "Steps",
                min_value=1,
                value=current_steps,
                step=10,
                key=f"edit_sc_steps_{edit_scope}",
                help=f"Experiment default: {global_steps}",
            )
            if not _values_equal(override_steps, global_steps):
                override_params["steps"] = override_steps

        global_initial_perc = float(global_params.get("initial_perc", 0.01))
        current_initial_perc = float(scenario_dict.get("initial_perc", global_initial_perc))
        override_initial_perc = st.number_input(
            "Initial Infected Fraction",
            min_value=0.001,
            max_value=1.0,
            value=current_initial_perc,
            step=0.01,
            format="%.3f",
            key=f"edit_sc_initial_perc_{edit_scope}",
            help=f"Experiment default: {global_initial_perc}",
        )
        if not _values_equal(override_initial_perc, global_initial_perc):
            override_params["initial_perc"] = override_initial_perc

    # Action buttons (pinned to bottom via CSS on modal_actions container)
    with st.container(key=f"modal_actions_edit_{edit_scope}"):
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Cancel", width="stretch", key=f"edit_sc_cancel_{edit_scope}"):
                st.rerun()

        with col2:
            if st.button(
                "Save Changes",
                type="primary",
                width="stretch",
                icon=":material/save:",
                key=f"edit_sc_save_{edit_scope}",
            ):
                if not label:
                    st.error("Please provide a scenario label")
                    return

                try:
                    update_scenario_in_experiment(
                        experiment, original_label, label, override_params
                    )
                    st.success(f"Scenario '{label}' updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update scenario: {str(e)}")


def update_scenario_in_experiment(
    experiment: Experiment,
    original_label: str,
    new_label: str,
    override_params: Dict[str, Any],
) -> None:
    """Update an existing scenario in the experiment's data.json.

    Args:
        experiment: The parent experiment
        original_label: The scenario's current label (for lookup)
        new_label: The new label (may be same as original)
        override_params: Parameters that override global settings

    Raises:
        ValueError: If new_label normalizes to the same value as another scenario
    """
    from spkmc.models.scenario import Scenario as ScenarioModel

    exp_path = experiment.path
    assert exp_path is not None

    data_file = exp_path / "data.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check that label normalizes to a non-empty filename
    new_norm = ScenarioModel.normalize_label(new_label)
    if not new_norm:
        raise ValueError(
            f"Scenario label '{new_label}' normalizes to an empty filename. "
            "Use a label with at least one alphanumeric character."
        )

    # Check for normalized label collision (excluding the scenario being edited)
    for sc in data.get("scenarios", []):
        if sc.get("label") == original_label:
            continue
        existing_norm = ScenarioModel.normalize_label(sc.get("label", ""))
        if existing_norm == new_norm:
            raise ValueError(
                f"A scenario with a conflicting name already exists: '{sc.get('label')}' "
                f"(both normalize to '{new_norm}')"
            )

    # Detect whether anything actually changed before writing.
    scenarios_list = data.get("scenarios", [])
    old_entry: Dict[str, Any] = {}
    old_index = -1
    for i, s in enumerate(scenarios_list):
        if s.get("label") == original_label:
            old_entry = dict(s)
            old_index = i
            break

    old_norm = ScenarioModel.normalize_label(original_label)
    label_changed = new_label != original_label

    # Compare *effective* parameters (globals + overrides) so legacy experiments
    # that store full params in scenario entries don't trigger false positives.
    global_params = data.get("parameters", {})
    meta_keys = {"label", "status"}
    old_overrides = {k: v for k, v in old_entry.items() if k not in meta_keys}
    effective_old = {**global_params, **old_overrides}
    if global_params:
        # Modern format: effective_new is globals + new overrides.
        effective_new = {**global_params, **override_params}
    else:
        # Legacy format: override_params only contains values that differ from
        # hardcoded form defaults — a SUBSET of the full param set. Start from
        # effective_old so that keys present in old_overrides but matching the
        # hardcoded defaults don't produce false-positive diffs.
        effective_new = {**effective_old, **override_params}
    params_changed = effective_old != effective_new

    if not label_changed and not params_changed:
        return  # No-op: nothing changed, preserve result files and data.json

    # Replace the scenario entry and mark as edited.
    # When a global `parameters` block exists, store only overrides.
    # For legacy experiments without globals, preserve the full parameter
    # set so the scenario entry remains valid on reload.
    if old_index >= 0:
        if global_params:
            scenarios_list[old_index] = {
                "label": new_label,
                "status": "edited",
                **override_params,
            }
        else:
            # Legacy format: keep all existing params, apply overrides on top
            saved_entry = {k: v for k, v in old_entry.items() if k not in meta_keys}
            scenarios_list[old_index] = {
                "label": new_label,
                "status": "edited",
                **saved_entry,
                **override_params,
            }

    # Write back (atomic to prevent corruption on crash)
    from spkmc.web import atomic_json_write

    atomic_json_write(data_file, data)

    # Delete stale result/analysis files since parameters or label changed.
    old_result = exp_path / f"{old_norm}.json"
    if old_result.exists():
        old_result.unlink()
    old_analysis = exp_path / f"{old_norm}_analysis.md"
    if old_analysis.exists():
        old_analysis.unlink()


def run_ai_analysis(experiment: Experiment) -> None:
    """
    Launch subprocess-based AI analysis on an experiment.

    Args:
        experiment: The experiment to analyze
    """
    exp_path = experiment.path
    assert exp_path is not None

    api_key = WebConfig.get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key not found. Please set it in Preferences.")
        return

    # Check there are completed scenarios
    has_results = any(
        (exp_path / f"{sc.normalized_label}.json").exists() for sc in experiment.scenarios
    )
    if not has_results:
        st.warning("No completed scenarios to analyze. Run some scenarios first.")
        return

    config = st.session_state.config
    model = config.get("ai_model", "gpt-4o-mini")

    if "analysis_runner" not in st.session_state:
        st.session_state.analysis_runner = AnalysisRunner()
    runner: AnalysisRunner = st.session_state.analysis_runner

    run_id = runner.run_experiment_analysis(
        experiment_path=exp_path,
        experiment_name=experiment.name,
        experiment_description=experiment.description or "No description provided",
        model=model,
        api_key=api_key,
    )

    if run_id:
        analysis_id = f"exp_analysis--{exp_path.name}"
        SessionState.add_running_analysis(
            analysis_id,
            {
                "experiment_name": exp_path.name,
                "analysis_type": "experiment",
                "scenario_normalized": "",
                "run_id": run_id,
                "status": "running",
            },
        )
        st.toast("Analysis started...")
        st.rerun()


def run_scenario_ai_analysis(
    experiment: Experiment,
    scenario: Scenario,
    result_file: Path,
) -> None:
    """
    Launch subprocess-based AI analysis on a single scenario.

    Args:
        experiment: The parent experiment
        scenario: The scenario to analyze
        result_file: Path to the scenario's result JSON file
    """
    exp_path = experiment.path
    assert exp_path is not None

    api_key = WebConfig.get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key not found. Please set it in Preferences.")
        return

    config = st.session_state.config
    model = config.get("ai_model", "gpt-4o-mini")

    if "analysis_runner" not in st.session_state:
        st.session_state.analysis_runner = AnalysisRunner()
    runner: AnalysisRunner = st.session_state.analysis_runner

    run_id = runner.run_scenario_analysis(
        experiment_path=exp_path,
        scenario_label=scenario.label,
        scenario_normalized=scenario.normalized_label,
        model=model,
        api_key=api_key,
    )

    if run_id:
        analysis_id = f"sc_analysis--{exp_path.name}--{scenario.normalized_label}"
        SessionState.add_running_analysis(
            analysis_id,
            {
                "experiment_name": exp_path.name,
                "analysis_type": "scenario",
                "scenario_normalized": scenario.normalized_label,
                "run_id": run_id,
                "status": "running",
            },
        )
        st.toast("Scenario analysis started...")
        st.rerun(scope="fragment")
