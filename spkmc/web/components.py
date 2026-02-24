"""
Reusable UI components for the web interface.

Provides form builders, cards, and other reusable widgets used across pages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from spkmc.web.config import WebConfig


def status_badge(status: str) -> str:
    """
    Generate HTML for a status badge.

    Args:
        status: One of 'pending', 'running', 'completed', 'failed'

    Returns:
        HTML string for the status badge
    """
    status_map = {
        "pending": ("Pending", "status-badge status-pending"),
        "created": ("Created", "status-badge status-created"),
        "running": ("Running", "status-badge status-running"),
        "completed": ("Completed", "status-badge status-completed"),
        "failed": ("Failed", "status-badge status-failed"),
    }

    text, css_class = status_map.get(status, ("Unknown", "status-badge"))
    return f'<span class="{css_class}">{text}</span>'


def network_config_form(key_prefix: str = "network") -> Dict[str, Any]:
    """
    Render dynamic form fields for network configuration.

    Args:
        key_prefix: Unique prefix for form element keys

    Returns:
        Dictionary of network configuration values
    """
    config = st.session_state.config

    col1, col2 = st.columns(2)

    with col1:
        network_type = st.selectbox(
            "Network Type",
            options=["er", "sf", "cg", "rrn"],
            format_func=lambda x: {
                "er": "Erdős-Rényi (Random)",
                "sf": "Scale-Free",
                "cg": "Complete Graph",
                "rrn": "Random Regular",
            }[x],
            index=0,
            key=f"{key_prefix}_type",
            help="Network topology structure",
        )

    with col2:
        nodes = st.number_input(
            "Number of Nodes",
            min_value=10,
            max_value=100000,
            value=config.get("default_nodes", 1000),
            step=100,
            key=f"{key_prefix}_nodes",
            help="Size of the network (population)",
        )

    result = {"network": network_type, "nodes": nodes}

    # Network-specific parameters
    if network_type in ["er", "sf", "rrn"]:
        col1, col2 = st.columns(2)
        with col1:
            k_avg = st.number_input(
                "Average Degree (k_avg)",
                min_value=1.0,
                max_value=float(nodes),
                value=float(config.get("default_k_avg", 10.0)),
                step=1.0,
                key=f"{key_prefix}_k_avg",
                help="Average number of connections per node",
            )
        result["k_avg"] = k_avg

    if network_type == "sf":
        with col2:
            exponent = st.number_input(
                "Power-law Exponent",
                min_value=2.0,
                max_value=5.0,
                value=float(config.get("default_exponent", 2.5)),
                step=0.1,
                key=f"{key_prefix}_exponent",
                help="Controls hub concentration (lower = more hubs)",
            )
        result["exponent"] = exponent

    return result


def distribution_config_form(key_prefix: str = "distribution") -> Dict[str, Any]:
    """
    Render dynamic form fields for distribution configuration.

    Args:
        key_prefix: Unique prefix for form element keys

    Returns:
        Dictionary of distribution configuration values
    """
    config = st.session_state.config

    col1, col2 = st.columns(2)

    with col1:
        dist_type = st.selectbox(
            "Distribution Type",
            options=["gamma", "exponential"],
            format_func=lambda x: x.capitalize(),
            index=0,
            key=f"{key_prefix}_type",
            help="Recovery time distribution",
        )

    with col2:
        lambda_param = st.number_input(
            "Infection Rate (λ)",
            min_value=0.01,
            max_value=10.0,
            value=float(config.get("default_lambda", 1.0)),
            step=0.1,
            key=f"{key_prefix}_lambda",
            help="Transmission rate along edges",
        )

    result = {"distribution": dist_type, "lambda": lambda_param}

    # Distribution-specific parameters
    col1, col2 = st.columns(2)

    if dist_type == "gamma":
        with col1:
            shape = st.number_input(
                "Shape Parameter",
                min_value=0.1,
                max_value=10.0,
                value=float(config.get("default_shape", 2.0)),
                step=0.1,
                key=f"{key_prefix}_shape",
                help="Controls recovery time distribution shape",
            )
        with col2:
            scale = st.number_input(
                "Scale Parameter",
                min_value=0.1,
                max_value=10.0,
                value=float(config.get("default_scale", 1.0)),
                step=0.1,
                key=f"{key_prefix}_scale",
                help="Controls recovery time scale",
            )
        result["shape"] = shape
        result["scale"] = scale

    elif dist_type == "exponential":
        with col1:
            mu = st.number_input(
                "Recovery Rate (μ)",
                min_value=0.01,
                max_value=10.0,
                value=float(config.get("default_mu", 1.0)),
                step=0.1,
                key=f"{key_prefix}_mu",
                help="Exponential recovery rate",
            )
        result["mu"] = mu

    return result


def simulation_params_form(key_prefix: str = "simulation") -> Dict[str, Any]:
    """
    Render form fields for simulation parameters.

    Args:
        key_prefix: Unique prefix for form element keys

    Returns:
        Dictionary of simulation configuration values
    """
    config = st.session_state.config

    col1, col2, col3 = st.columns(3)

    with col1:
        samples = st.number_input(
            "Samples",
            min_value=1,
            max_value=10000,
            value=config.get("default_samples", 50),
            step=10,
            key=f"{key_prefix}_samples",
            help="Monte Carlo samples per run",
        )

    with col2:
        num_runs = st.number_input(
            "Number of Runs",
            min_value=1,
            max_value=100,
            value=config.get("default_num_runs", 2),
            step=1,
            key=f"{key_prefix}_num_runs",
            help="Independent runs for error estimation",
        )

    with col3:
        initial_perc = (
            st.number_input(
                "Initial Infected (%)",
                min_value=0.01,
                max_value=100.0,
                value=float(config.get("default_initial_perc", 0.01)) * 100,
                step=0.1,
                key=f"{key_prefix}_initial_perc",
                help="Percentage of initially infected nodes",
            )
            / 100.0
        )

    col1, col2 = st.columns(2)

    with col1:
        t_max = st.number_input(
            "Max Time",
            min_value=0.1,
            max_value=1000.0,
            value=float(config.get("default_t_max", 10.0)),
            step=1.0,
            key=f"{key_prefix}_t_max",
            help="Simulation duration",
        )

    with col2:
        steps = st.number_input(
            "Time Steps",
            min_value=10,
            max_value=10000,
            value=config.get("default_steps", 100),
            step=10,
            key=f"{key_prefix}_steps",
            help="Number of time points to record",
        )

    return {
        "samples": samples,
        "num_runs": num_runs,
        "initial_perc": initial_perc,
        "t_max": t_max,
        "steps": steps,
    }


def result_metric_cards(result_dict: Dict[str, Any]) -> None:
    """
    Display key metrics from a simulation result.

    Args:
        result_dict: Result dictionary with S_val, I_val, R_val arrays
    """
    import numpy as np

    if "I_val" not in result_dict or "time" not in result_dict:
        st.warning("No result data available")
        return

    I_val = np.array(result_dict["I_val"])
    R_val = np.array(result_dict["R_val"])
    time = np.array(result_dict["time"])

    peak_infected = float(np.max(I_val))
    peak_time = float(time[np.argmax(I_val)])
    final_recovered = float(R_val[-1])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Peak Infected",
            value=f"{peak_infected:.1%}",
            help="Maximum proportion of infected individuals",
        )

    with col2:
        st.metric(
            label="Peak Time",
            value=f"{peak_time:.2f}",
            help="Time at which peak infection occurred",
        )

    with col3:
        st.metric(
            label="Final Epidemic Size",
            value=f"{final_recovered:.1%}",
            help="Proportion of population that was infected",
        )


def experiment_status_badge(experiment: Any) -> str:
    """
    Generate status badge for an experiment based on its scenarios.

    Args:
        experiment: Experiment object with scenarios

    Returns:
        HTML string for the experiment status badge
    """
    # Guard against None path and empty scenario list
    if experiment.path is None or not experiment.scenarios:
        return status_badge("pending")

    # Check scenario statuses
    has_results = any(
        (experiment.path / f"{s.normalized_label}.json").exists() for s in experiment.scenarios
    )

    all_complete = all(
        (experiment.path / f"{s.normalized_label}.json").exists() for s in experiment.scenarios
    )

    if all_complete:
        return status_badge("completed")
    elif has_results:
        return status_badge("running")  # Partial completion
    else:
        return status_badge("pending")
