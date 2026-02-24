"""
Dashboard page - main experiments list view.

Shows all experiments, summary stats, and provides "Create Experiment" functionality.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from spkmc.io.experiments import ExperimentManager
from spkmc.models import Experiment, ExperimentConfig, ScenarioOverride
from spkmc.web.components import (
    distribution_config_form,
    experiment_status_badge,
    network_config_form,
    simulation_params_form,
)
from spkmc.web.runner import poll_running_simulations
from spkmc.web.state import SessionState
from spkmc.web.styles import (
    ICONS,
    empty_state,
    experiment_card,
    page_header,
    section_header,
    stat_card,
)


def render() -> None:
    """Render the dashboard page."""
    # Page header
    st.markdown(
        page_header("Experiments", subtitle="Manage and run SPKMC epidemic simulation experiments"),
        unsafe_allow_html=True,
    )

    # Load experiments
    config = st.session_state.config
    exp_manager = ExperimentManager(str(config.get_experiments_path()))
    experiments = exp_manager.list_experiments()

    # Summary stats row with beautiful cards
    render_summary_stats(experiments)

    # Spacer between stats and experiments
    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)

    # Experiments list or empty state
    if experiments:
        render_experiments_list(experiments)
    else:
        render_empty_state_ui()


def render_summary_stats(experiments: List[Experiment]) -> None:
    """Render beautiful summary statistics cards."""
    total_experiments = len(experiments)
    total_scenarios = sum(len(exp.scenarios) for exp in experiments)

    # Count completed scenarios
    completed_scenarios = 0
    for exp in experiments:
        if exp.path is None:
            continue
        for scenario in exp.scenarios:
            result_file = exp.path / f"{scenario.normalized_label}.json"
            if result_file.exists():
                completed_scenarios += 1

    # Recent activity (last modified experiment)
    last_activity = "Never"
    if experiments:
        most_recent = max(
            (exp.path for exp in experiments if exp.path is not None and exp.path.exists()),
            key=lambda p: p.stat().st_mtime,
            default=None,
        )
        if most_recent:
            last_modified = datetime.fromtimestamp(most_recent.stat().st_mtime)
            last_activity = last_modified.strftime("%Y-%m-%d %H:%M")

    # Use columns for grid layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            stat_card("Total Experiments", str(total_experiments), ICONS["flask"]),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            stat_card("Total Scenarios", str(total_scenarios), ICONS["file"]),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            stat_card("Completed Scenarios", str(completed_scenarios), ICONS["check"]),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            stat_card("Last Activity", last_activity, ICONS["clock"]), unsafe_allow_html=True
        )


def render_experiments_list(experiments: List[Experiment]) -> None:
    """Render header + create button, then delegate cards to polling fragment."""
    col_header, col_create = st.columns([8, 2], vertical_alignment="bottom")
    with col_header:
        st.markdown(
            section_header("All Experiments"),
            unsafe_allow_html=True,
        )
    with col_create:
        if st.button(
            "Create Experiment",
            type="primary",
            width="stretch",
            key="btn_create_exp",
        ):
            show_create_experiment_modal()

    _live_experiment_cards(experiments)


@st.fragment(run_every=timedelta(seconds=2))
def _live_experiment_cards(experiments: List[Experiment]) -> None:
    """Fragment that polls running simulations and re-renders experiment cards."""
    poll_running_simulations()

    for idx, exp in enumerate(experiments):
        if exp.path is None:
            continue

        exp_path = exp.path
        scenario_count = len(exp.scenarios)

        # Count completed scenarios
        completed = sum(
            1 for s in exp.scenarios if (exp_path / f"{s.normalized_label}.json").exists()
        )

        # Get last modified time
        if exp_path.exists():
            last_mod = datetime.fromtimestamp(exp_path.stat().st_mtime)
            last_modified = last_mod.strftime("%Y-%m-%d %H:%M")
        else:
            last_modified = "Unknown"

        # Determine status by checking actual running simulations
        scenario_statuses = [
            SessionState.get_simulation_status(f"sim--{exp_path.name}--{s.normalized_label}")
            for s in exp.scenarios
        ]
        any_running = "running" in scenario_statuses
        any_failed = "failed" in scenario_statuses

        if any_running:
            status = "running"
        elif scenario_count > 0 and completed == scenario_count:
            status = "complete"
        elif any_failed:
            status = "failed"
        else:
            status = "pending"

        # Clickable card container: invisible button overlays the card HTML
        with st.container(key=f"exp_card_{idx}"):
            st.markdown(
                experiment_card(
                    name=exp.name,
                    description=exp.description or "No description provided",
                    scenarios_complete=completed,
                    scenarios_total=scenario_count,
                    last_run=last_modified,
                    status=status,
                ),
                unsafe_allow_html=True,
            )
            if st.button("select", key=f"exp_btn_{idx}"):
                SessionState.set_selected_experiment(exp_path.name)
                st.rerun()


def render_empty_state_ui() -> None:
    """Render beautiful empty state when no experiments exist."""
    st.markdown(
        empty_state(
            title="No experiments yet",
            message="Create your first experiment to start running epidemic simulations on networks. "
            "Each experiment can contain multiple scenarios with different parameters.",
        ),
        unsafe_allow_html=True,
    )

    # Add some spacing and the CTA button
    st.markdown('<div style="margin-top: 2rem; text-align: center;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Create Your First Experiment", type="primary", width="stretch"):
            show_create_experiment_modal()
    st.markdown("</div>", unsafe_allow_html=True)


def _init_scenario_state() -> None:
    """Initialize session state for scenario list if not present."""
    if "create_exp_scenarios" not in st.session_state:
        st.session_state.create_exp_scenarios = []
        st.session_state.create_exp_sc_counter = 0


def _add_scenario() -> None:
    """Append a new scenario to the session state list."""
    counter = st.session_state.create_exp_sc_counter
    sc_id = f"sc_{counter}"
    st.session_state.create_exp_scenarios.append(
        {
            "id": sc_id,
            "label": "",
        }
    )
    st.session_state.create_exp_sc_counter = counter + 1
    st.session_state.create_exp_last_added = sc_id


def _remove_scenario(sc_id: str) -> None:
    """Remove a scenario from the session state list by its ID."""
    st.session_state.create_exp_scenarios = [
        s for s in st.session_state.create_exp_scenarios if s["id"] != sc_id
    ]


def _render_scenario(
    sc_id: str,
    default_label: str,
    index: int,
    can_remove: bool,
) -> None:
    """
    Render a single scenario expander with override toggles and forms.

    Args:
        sc_id: Unique scenario ID (e.g. "sc_0")
        default_label: Default label text for this scenario
        index: Display index (1-based)
        can_remove: Whether the remove button should be enabled
    """
    label_key = f"{sc_id}_label"
    current_label = st.session_state.get(label_key, default_label)
    display_label = current_label if current_label else "Untitled"
    header = f"Scenario {index}: {display_label}"

    last_added = st.session_state.get("create_exp_last_added")
    with st.expander(header, expanded=(sc_id == last_added)):
        st.text_input(
            "Label *",
            value=default_label,
            key=label_key,
            placeholder="e.g., High Infection Rate",
            help="Required. Name for this scenario",
        )

        # Override toggle checkboxes
        col_net, col_dist, col_sim = st.columns(3)
        with col_net:
            override_net = st.checkbox(
                "Override Network",
                key=f"{sc_id}_override_net",
            )
        with col_dist:
            override_dist = st.checkbox(
                "Override Distribution",
                key=f"{sc_id}_override_dist",
            )
        with col_sim:
            override_sim = st.checkbox(
                "Override Simulation",
                key=f"{sc_id}_override_sim",
            )

        # Render override forms when toggled
        if override_net:
            st.markdown("---")
            st.caption("Network Overrides")
            network_config_form(key_prefix=f"{sc_id}_net")

        if override_dist:
            st.markdown("---")
            st.caption("Distribution Overrides")
            distribution_config_form(key_prefix=f"{sc_id}_dist")

        if override_sim:
            st.markdown("---")
            st.caption("Simulation Overrides")
            simulation_params_form(key_prefix=f"{sc_id}_sim")

        if not (override_net or override_dist or override_sim):
            st.caption("Using all global defaults")

        # Remove button
        if can_remove:
            st.button(
                "Remove",
                key=f"{sc_id}_remove",
                on_click=_remove_scenario,
                args=(sc_id,),
            )


def _collect_scenario_overrides(
    sc_id: str,
    global_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Collect override dict for one scenario by reading widget state.

    Only includes keys whose values actually differ from global_params.

    Args:
        sc_id: Unique scenario ID
        global_params: The global parameter dict to diff against

    Returns:
        Dict with "label" and any differing override keys
    """
    result: Dict[str, Any] = {
        "label": st.session_state.get(f"{sc_id}_label", "Untitled"),
    }

    override_net = st.session_state.get(f"{sc_id}_override_net", False)
    override_dist = st.session_state.get(f"{sc_id}_override_dist", False)
    override_sim = st.session_state.get(f"{sc_id}_override_sim", False)

    if override_net:
        net_params = _read_form_values_network(f"{sc_id}_net")
        for key, value in net_params.items():
            if global_params.get(key) != value:
                result[key] = value

    if override_dist:
        dist_params = _read_form_values_distribution(f"{sc_id}_dist")
        for key, value in dist_params.items():
            if global_params.get(key) != value:
                result[key] = value

    if override_sim:
        sim_params = _read_form_values_simulation(f"{sc_id}_sim")
        for key, value in sim_params.items():
            if global_params.get(key) != value:
                result[key] = value

    return result


def _read_form_values_network(key_prefix: str) -> Dict[str, Any]:
    """Read network form widget values from session state.

    Only reads conditional parameters (k_avg, exponent) when the
    current network type actually uses them, avoiding stale session
    state from previously-rendered conditional widgets.
    """
    result: Dict[str, Any] = {}
    network_type = st.session_state.get(f"{key_prefix}_type")
    if network_type is not None:
        result["network"] = network_type
    nodes = st.session_state.get(f"{key_prefix}_nodes")
    if nodes is not None:
        result["nodes"] = nodes
    # k_avg only exists for er, sf, rrn
    if network_type in ("er", "sf", "rrn"):
        k_avg = st.session_state.get(f"{key_prefix}_k_avg")
        if k_avg is not None:
            result["k_avg"] = k_avg
    # exponent only exists for sf
    if network_type == "sf":
        exponent = st.session_state.get(f"{key_prefix}_exponent")
        if exponent is not None:
            result["exponent"] = exponent
    return result


def _read_form_values_distribution(key_prefix: str) -> Dict[str, Any]:
    """Read distribution form widget values from session state.

    Only reads conditional parameters (shape/scale for gamma, mu for
    exponential) when the current distribution type uses them.
    """
    result: Dict[str, Any] = {}
    dist_type = st.session_state.get(f"{key_prefix}_type")
    if dist_type is not None:
        result["distribution"] = dist_type
    lambda_val = st.session_state.get(f"{key_prefix}_lambda")
    if lambda_val is not None:
        result["lambda"] = lambda_val
    # shape and scale only exist for gamma
    if dist_type == "gamma":
        shape = st.session_state.get(f"{key_prefix}_shape")
        if shape is not None:
            result["shape"] = shape
        scale = st.session_state.get(f"{key_prefix}_scale")
        if scale is not None:
            result["scale"] = scale
    # mu only exists for exponential
    elif dist_type == "exponential":
        mu = st.session_state.get(f"{key_prefix}_mu")
        if mu is not None:
            result["mu"] = mu
    return result


def _read_form_values_simulation(key_prefix: str) -> Dict[str, Any]:
    """Read simulation form widget values from session state."""
    result: Dict[str, Any] = {}
    samples = st.session_state.get(f"{key_prefix}_samples")
    if samples is not None:
        result["samples"] = samples
    num_runs = st.session_state.get(f"{key_prefix}_num_runs")
    if num_runs is not None:
        result["num_runs"] = num_runs
    initial_perc = st.session_state.get(f"{key_prefix}_initial_perc")
    if initial_perc is not None:
        # The widget stores percentage (0-100), convert back to fraction
        result["initial_perc"] = initial_perc / 100.0
    t_max = st.session_state.get(f"{key_prefix}_t_max")
    if t_max is not None:
        result["t_max"] = t_max
    steps = st.session_state.get(f"{key_prefix}_steps")
    if steps is not None:
        result["steps"] = steps
    return result


def _cleanup_scenario_state() -> None:
    """Remove all dialog-related keys from session state after dialog closes."""
    prefixes = ("sc_", "create_network_", "create_dist_", "create_sim_")
    explicit_keys = (
        "create_exp_scenarios",
        "create_exp_sc_counter",
        "create_exp_baseline",
        "create_exp_last_added",
    )
    keys_to_remove = [k for k in st.session_state if k.startswith(prefixes) or k in explicit_keys]
    for key in keys_to_remove:
        del st.session_state[key]


@st.dialog("Create New Experiment", width="large")
def show_create_experiment_modal() -> None:
    """Show the create experiment modal dialog."""
    _init_scenario_state()

    st.markdown("### Experiment Configuration")

    # Basic info
    st.subheader("Basic Information")
    name = st.text_input(
        "Experiment Name",
        placeholder="e.g., Network Comparison Study",
        help="Descriptive name for your experiment",
    )

    description = st.text_area(
        "Description",
        placeholder="What are you testing?",
        help="Brief description of the experiment's purpose",
    )

    # Global parameters
    st.subheader("Global Parameters")
    st.caption("These parameters will be inherited by all scenarios (can be overridden)")

    with st.expander("Network Configuration", expanded=True):
        network_params = network_config_form(key_prefix="create_network")

    with st.expander("Distribution Configuration", expanded=True):
        dist_params = distribution_config_form(key_prefix="create_dist")

    with st.expander("Simulation Parameters", expanded=True):
        sim_params = simulation_params_form(key_prefix="create_sim")

    # Scenarios section
    st.subheader("Scenarios")
    st.caption(
        "Each scenario inherits the global parameters above. "
        "Override specific values to create different conditions."
    )

    include_baseline = st.checkbox(
        "Include Baseline scenario",
        value=True,
        key="create_exp_baseline",
        help="Adds a Baseline scenario using all global defaults",
    )

    # Show baseline preview when checkbox is checked
    if include_baseline:
        with st.expander("Scenario 1: Baseline", expanded=False):
            st.caption("Uses all global defaults (no overrides)")

    # Render each scenario
    scenario_list = st.session_state.create_exp_scenarios
    offset = 1 if include_baseline else 0

    for idx, sc in enumerate(scenario_list):
        _render_scenario(
            sc_id=sc["id"],
            default_label=sc["label"],
            index=idx + 1 + offset,
            can_remove=True,
        )

    # Add Scenario button (below all scenarios)
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col2:
        st.button(
            "+ Add Scenario",
            on_click=_add_scenario,
            width="stretch",
        )

    # Action buttons
    st.divider()
    spacer, col_cancel, col_create = st.columns([6, 2, 2])

    with col_cancel:
        if st.button("Cancel", width="stretch"):
            _cleanup_scenario_state()
            st.rerun()

    with col_create:
        if st.button("Create Experiment", type="primary", width="stretch"):
            if not name:
                st.error("Please provide an experiment name")
                return

            # Validate all scenario labels are non-empty
            for sc in scenario_list:
                sc_label = st.session_state.get(f"{sc['id']}_label", "").strip()
                if not sc_label:
                    st.error("All scenarios must have a label.")
                    return

            # Validate normalized label uniqueness and non-emptiness
            from spkmc.models.scenario import Scenario as ScenarioModel

            seen_normalized: Dict[str, str] = {}
            if include_baseline:
                seen_normalized["baseline"] = "Baseline"
            for sc in scenario_list:
                sc_label = st.session_state.get(f"{sc['id']}_label", "").strip()
                norm = ScenarioModel.normalize_label(sc_label)
                if not norm:
                    st.error(
                        f"Scenario label '{sc_label}' normalizes to an empty filename. "
                        "Use a label with at least one alphanumeric character."
                    )
                    return
                if norm in seen_normalized:
                    st.error(
                        f"Scenario labels '{seen_normalized[norm]}' and '{sc_label}' "
                        f"conflict (both normalize to '{norm}'). Use distinct names."
                    )
                    return
                seen_normalized[norm] = sc_label

            global_params = {**network_params, **dist_params, **sim_params}

            # Collect scenario overrides
            scenarios = [
                _collect_scenario_overrides(sc["id"], global_params) for sc in scenario_list
            ]

            # Prepend baseline if checked
            if include_baseline:
                scenarios.insert(0, {"label": "Baseline"})

            if not scenarios:
                st.error("Add at least one scenario or include baseline")
                return

            # Create the experiment
            try:
                exp_path = create_experiment(
                    name=name,
                    description=description,
                    global_params=global_params,
                    scenarios=scenarios,
                )

                # Auto-run baseline scenario
                if include_baseline:
                    _auto_run_baseline(exp_path)

                _cleanup_scenario_state()
                st.success(f"Experiment '{name}' created successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create experiment: {str(e)}")


def create_experiment(
    name: str,
    description: str,
    global_params: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
) -> Path:
    """
    Create a new experiment in the experiments directory.

    Args:
        name: Experiment name
        description: Experiment description
        global_params: Global parameters dictionary
        scenarios: List of scenario override dictionaries

    Returns:
        Path to the created experiment directory
    """
    config = st.session_state.config
    exp_dir = config.get_experiments_path()

    # Create normalized directory name
    from spkmc.models.scenario import Scenario

    dir_name = Scenario.normalize_label(name)
    if not dir_name:
        raise ValueError(
            f"Experiment name '{name}' normalizes to an empty directory name. "
            "Use a name with at least one alphanumeric character."
        )
    exp_path = Path(exp_dir) / dir_name

    # Check if already exists
    if exp_path.exists():
        raise ValueError(f"Experiment '{dir_name}' already exists")

    # Create directory
    exp_path.mkdir(parents=True, exist_ok=True)

    # Build experiment config
    config_dict = {
        "name": name,
        "description": description,
        "parameters": global_params,
        "scenarios": scenarios,
    }

    # Write data.json (atomic to prevent corruption on crash)
    from spkmc.web import atomic_json_write

    data_file = exp_path / "data.json"
    atomic_json_write(data_file, config_dict)

    return exp_path


def _auto_run_baseline(exp_path: Path) -> None:
    """Start the baseline scenario run for a freshly created experiment.

    Args:
        exp_path: Path to the experiment directory
    """
    from spkmc.web.runner import SimulationRunner

    config = st.session_state.config
    exp_manager = ExperimentManager(str(config.get_experiments_path()))

    try:
        experiment = exp_manager.load_experiment(exp_path.name)
    except Exception:
        return

    # Find the baseline scenario
    baseline = None
    for sc in experiment.scenarios:
        if sc.label == "Baseline":
            baseline = sc
            break

    if baseline is None:
        return

    if "simulation_runner" not in st.session_state:
        st.session_state.simulation_runner = SimulationRunner()
    runner: SimulationRunner = st.session_state.simulation_runner

    run_id = runner.run_scenario(experiment, baseline, show_progress=True)
    if run_id:
        status_info = runner.get_status(run_id)
        if status_info and experiment.path is not None:
            scenario_id = f"sim--{experiment.path.name}--{baseline.normalized_label}"
            status_info["run_id"] = run_id
            SessionState.add_running_simulation(scenario_id, status_info)
