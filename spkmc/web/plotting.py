"""
Plotly figure builders for the web interface.

Provides functions to create interactive Plotly charts for SIR simulation results
and comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SIR state colors (matching the plan)
COLOR_S = "#4477AA"
COLOR_I = "#EE6677"
COLOR_R = "#228833"

STATE_COLORS = {
    "S": COLOR_S,
    "I": COLOR_I,
    "R": COLOR_R,
}


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a hex color string to an rgba() string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def create_sir_figure(
    result_dict: Dict[str, Any],
    title: str = "SIR Dynamics",
    states: Optional[List[str]] = None,
    show_error_bands: bool = True,
    height: int = 500,
    chart_mode: str = "lines",
    state_colors: Optional[Dict[str, str]] = None,
    template: str = "plotly_white",
) -> go.Figure:
    """
    Create an interactive Plotly figure for a single SIR simulation result.

    Args:
        result_dict: Result dictionary containing S_val, I_val, R_val, time, etc.
        title: Plot title
        states: List of states to plot (default: ['S', 'I', 'R'])
        show_error_bands: Whether to show error bands (if S_err, I_err, R_err exist)
        height: Figure height in pixels
        chart_mode: One of "lines", "lines+markers", or "area"
        state_colors: Override colors for S/I/R states (keys: "S", "I", "R")
        template: Plotly template name

    Returns:
        Plotly Figure object
    """
    if states is None:
        states = ["S", "I", "R"]

    effective_colors = {**STATE_COLORS, **(state_colors or {})}

    fig = go.Figure()

    time = result_dict.get("time", [])
    has_errors = show_error_bands and "S_err" in result_dict

    for state in states:
        state_upper = state.upper()
        val_key = f"{state_upper}_val"
        err_key = f"{state_upper}_err"

        if val_key not in result_dict:
            continue

        y_val = result_dict[val_key]
        color = effective_colors.get(state_upper, "#666666")

        # Determine trace mode and fill from chart_mode
        if chart_mode == "lines+markers":
            trace_mode = "lines+markers"
            trace_fill = None
            trace_fillcolor = None
        elif chart_mode == "area":
            trace_mode = "lines"
            trace_fill = "tozeroy"
            trace_fillcolor = _hex_to_rgba(color, 0.15)
        else:
            trace_mode = "lines"
            trace_fill = None
            trace_fillcolor = None

        # Build error_y config if applicable
        error_y_config = None
        if has_errors and err_key in result_dict:
            y_err = result_dict[err_key]
            error_y_config = dict(
                type="data",
                array=y_err,
                visible=True,
                color=color,
                thickness=1.5,
                width=4,
            )

        # Main line (with optional error bars attached)
        fig.add_trace(
            go.Scatter(
                x=time,
                y=y_val,
                mode=trace_mode,
                name=state_upper,
                line=dict(color=color, width=2),
                fill=trace_fill,
                fillcolor=trace_fillcolor,
                error_y=error_y_config,
                hovertemplate=f"{state_upper}: %{{y:.4f}}<br>Time: %{{x:.2f}}<extra></extra>",
            )
        )

    # Compute explicit x-axis range to prevent layout shift
    # when error bars or markers add visual padding
    x_max = float(max(time)) if len(time) > 0 else 1
    x_range = [0, x_max]

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            range=x_range,
        ),
        yaxis=dict(
            title="Proportion of Population",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            range=[0, 1],
        ),
        template=template,
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_comparison_figure(
    results: List[Dict[str, Any]],
    labels: List[str],
    title: str = "Scenario Comparison",
    states: Optional[List[str]] = None,
    height: int = 600,
    template: str = "plotly_white",
) -> go.Figure:
    """
    Create an interactive Plotly figure comparing multiple SIR simulation results.

    Args:
        results: List of result dictionaries
        labels: List of labels for each result
        title: Plot title
        states: List of states to plot (default: ['S', 'I', 'R'])
        height: Figure height in pixels
        template: Plotly template name

    Returns:
        Plotly Figure object
    """
    if states is None:
        states = ["S", "I", "R"]

    fig = go.Figure()

    # Color palette for different scenarios (cycling if needed)
    scenario_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, (result_dict, label) in enumerate(zip(results, labels)):
        time = result_dict.get("time", [])
        base_color = scenario_colors[idx % len(scenario_colors)]

        for state in states:
            state_upper = state.upper()
            val_key = f"{state_upper}_val"

            if val_key not in result_dict:
                continue

            y_val = result_dict[val_key]

            # Use different line styles for different states
            line_dash = "solid" if state_upper == "I" else "dot" if state_upper == "S" else "dash"

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=y_val,
                    mode="lines",
                    name=f"{label} - {state_upper}",
                    line=dict(color=base_color, width=2, dash=line_dash),
                    hovertemplate=f"{label} - {state_upper}: %{{y:.4f}}<br>Time: %{{x:.2f}}<extra></extra>",
                )
            )

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            title="Proportion of Population",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            range=[0, 1],
        ),
        template=template,
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    return fig


def create_metric_card_figure(
    value: float,
    title: str,
    subtitle: str = "",
    color: str = "#4477AA",
) -> go.Figure:
    """
    Create a simple metric card figure for displaying key statistics.

    Args:
        value: The metric value to display
        title: Metric title
        subtitle: Optional subtitle/description
        color: Color for the metric value

    Returns:
        Plotly Figure object (minimal chart used as a card)
    """
    fig = go.Figure()

    # Create a minimal figure that just displays the metric
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=value,
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em'>{subtitle}</span>"},
            number={"font": {"size": 48, "color": color}},
        )
    )

    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig
