"""
Settings page - configure web interface preferences and API keys.

Manages OpenAI API keys, AI model selection, directory paths,
chart preferences, default simulation parameters, and export format.

All changes auto-save on widget interaction (no Save button required).
"""

from __future__ import annotations

import textwrap

import streamlit as st

from spkmc.web.config import WebConfig
from spkmc.web.styles import COLORS, FONTS, page_header


def _dedent(html: str) -> str:
    """Strip leading whitespace from HTML to prevent Markdown code-block rendering."""
    return textwrap.dedent(html).strip()


# ── SVG icons for section headers (Feather/Lucide style, 16x16) ──

_ICON_AI = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 '
    '12 10 13 2"/></svg>'
)

_ICON_CHART = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/>'
    '<line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" '
    'y2="14"/></svg>'
)

_ICON_SLIDERS = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/>'
    '<line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" '
    'y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" '
    'x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
    '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" '
    'y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>'
)

_ICON_FOLDER = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 '
    '1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>'
)

_ICON_ALERT = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 '
    '3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/>'
    '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
)


# ── HTML helpers ───────────────────────────────────────────


def _section_icon(
    title: str,
    subtitle: str,
    icon_svg: str,
    icon_bg: str = "",
    icon_color: str = "",
) -> str:
    """Create a section header with icon for the preferences page."""
    bg = icon_bg or COLORS["teal_100"]
    color = icon_color or COLORS["teal_500"]
    return _dedent(
        f"""
<div style="display:flex;align-items:center;gap:0.625rem;margin:1.5rem 0 0.75rem 0;">
<div style="width:32px;height:32px;background:{bg};border-radius:8px;display:flex;align-items:center;justify-content:center;color:{color};flex-shrink:0;">{icon_svg}</div>
<div>
<div style="font-family:{FONTS['body']};font-size:1rem;font-weight:700;color:{COLORS['gray_800']};letter-spacing:-0.01em;">{title}</div>
<div style="font-family:{FONTS['body']};font-size:0.75rem;color:{COLORS['gray_500']};margin-top:0.063rem;">{subtitle}</div>
</div>
</div>
"""
    )


def _sublabel(title: str) -> str:
    """Create a small uppercase subsection label inside a card."""
    return _dedent(
        f"""
<div class="pref-sublabel">{title}</div>
"""
    )


def _status_badge(configured: bool) -> str:
    """Create an API key status badge."""
    if configured:
        bg = COLORS["success_bg"]
        color = COLORS["success"]
        icon = (
            '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="3" stroke-linecap="round" '
            'stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
        )
        text = "Configured"
    else:
        bg = COLORS["warning_bg"]
        color = COLORS["warning"]
        icon = (
            '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="3" stroke-linecap="round" '
            'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="8" x2="12" y2="12"/>'
            '<line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
        )
        text = "Not configured"

    return _dedent(
        f"""
<div style="display:inline-flex;align-items:center;gap:0.375rem;padding:0.188rem 0.625rem;border-radius:99px;background:{bg};color:{color};font-family:{FONTS['body']};font-size:0.688rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.5rem;">{icon} {text}</div>
"""
    )


# ── Main render ────────────────────────────────────────────


def render() -> None:
    """Render the settings page. All values auto-save on change."""
    st.markdown(
        page_header(
            "Preferences",
            subtitle="Configure web interface and simulation defaults",
        ),
        unsafe_allow_html=True,
    )

    config: WebConfig = st.session_state.config

    # Consume the post-reset flag so auto-save doesn't overwrite defaults.
    skip_autosave = st.session_state.pop("pref_skip_autosave", False)

    # ── AI & Intelligence ─────────────────────────────────
    st.markdown(
        _section_icon(
            "AI & Intelligence",
            "API key and model for AI-powered analysis",
            _ICON_AI,
        ),
        unsafe_allow_html=True,
    )

    with st.container(key="pref_card_ai"):
        current_key = WebConfig.get_openai_api_key()
        st.markdown(_status_badge(bool(current_key)), unsafe_allow_html=True)

        col_key, col_model = st.columns([3, 1])

        with col_key:
            new_key = st.text_input(
                "API Key",
                value=current_key or "",
                type="password",
                placeholder="sk-...",
                help="Your OpenAI API key for AI analysis features",
            )

        with col_model:
            model_options = [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
                "gpt-4.1",
                "o3-mini",
            ]
            current_model = config.get("ai_model", "gpt-4o-mini")
            model_index = (
                model_options.index(current_model) if current_model in model_options else 0
            )
            selected_model = st.selectbox(
                "AI Model",
                options=model_options,
                index=model_index,
                help="OpenAI model used for AI analysis",
            )

    # ── Visualization ─────────────────────────────────────
    st.markdown(
        _section_icon(
            "Visualization",
            "Chart appearance and SIR state colors",
            _ICON_CHART,
        ),
        unsafe_allow_html=True,
    )

    with st.container(key="pref_card_viz"):
        col_chart, col_sep, col_colors = st.columns([10, 1, 5])

        with col_chart:
            sub_h, sub_t = st.columns(2)

            with sub_h:
                chart_height = st.number_input(
                    "Default Height (px)",
                    min_value=300,
                    max_value=1000,
                    value=config.get("chart_height", 500),
                    step=50,
                )

            with sub_t:
                template_options = [
                    "plotly_white",
                    "plotly_dark",
                    "simple_white",
                    "ggplot2",
                ]
                current_template = config.get("chart_template", "plotly_white")
                template_index = (
                    template_options.index(current_template)
                    if current_template in template_options
                    else 0
                )
                chart_template = st.selectbox(
                    "Template",
                    options=template_options,
                    index=template_index,
                )

        with col_sep:
            st.markdown(
                _dedent(
                    """
<div style="height:100%;min-height:80px;display:flex;align-items:center;justify-content:center;padding-top:1.75rem;">
<div style="width:1px;height:60px;background:var(--gray-200,#e5e7eb);"></div>
</div>
"""
                ),
                unsafe_allow_html=True,
            )

        with col_colors:
            sub_s, sub_i, sub_r = st.columns([1, 1, 1], gap="small")
            with sub_s:
                color_s = st.color_picker(
                    "Susceptible",
                    value=config.get("chart_color_s", "#4477AA"),
                )
            with sub_i:
                color_i = st.color_picker(
                    "Infected",
                    value=config.get("chart_color_i", "#EE6677"),
                )
            with sub_r:
                color_r = st.color_picker(
                    "Recovered",
                    value=config.get("chart_color_r", "#228833"),
                )

    # ── Simulation Defaults ───────────────────────────────
    st.markdown(
        _section_icon(
            "Simulation Defaults",
            "Default parameters for new experiments",
            _ICON_SLIDERS,
        ),
        unsafe_allow_html=True,
    )

    with st.container(key="pref_card_sim"):
        col_net, col_dist, col_sim = st.columns(3)

        with col_net:
            st.markdown(_sublabel("Network"), unsafe_allow_html=True)

            default_nodes = st.number_input(
                "Nodes",
                min_value=10,
                max_value=100000,
                value=config.get("default_nodes", 1000),
                step=100,
            )
            default_k_avg = st.number_input(
                "k_avg",
                min_value=1.0,
                max_value=100.0,
                value=float(config.get("default_k_avg", 10.0)),
                step=1.0,
            )
            default_exponent = st.number_input(
                "Exponent (SF)",
                min_value=2.0,
                max_value=5.0,
                value=float(config.get("default_exponent", 2.5)),
                step=0.1,
            )

        with col_dist:
            st.markdown(_sublabel("Distribution"), unsafe_allow_html=True)

            default_shape = st.number_input(
                "Shape (Gamma)",
                min_value=0.1,
                max_value=10.0,
                value=float(config.get("default_shape", 2.0)),
                step=0.1,
            )
            default_scale = st.number_input(
                "Scale (Gamma)",
                min_value=0.1,
                max_value=10.0,
                value=float(config.get("default_scale", 1.0)),
                step=0.1,
            )
            default_mu = st.number_input(
                "\u03bc (Exponential)",
                min_value=0.01,
                max_value=10.0,
                value=float(config.get("default_mu", 1.0)),
                step=0.1,
            )
            default_lambda = st.number_input(
                "\u03bb",
                min_value=0.01,
                max_value=10.0,
                value=float(config.get("default_lambda", 1.0)),
                step=0.1,
            )

        with col_sim:
            st.markdown(_sublabel("Simulation"), unsafe_allow_html=True)

            default_samples = st.number_input(
                "Samples",
                min_value=1,
                max_value=10000,
                value=config.get("default_samples", 50),
                step=10,
            )
            default_num_runs = st.number_input(
                "Runs per scenario",
                min_value=1,
                max_value=100,
                value=config.get("default_num_runs", 2),
                step=1,
            )
            default_t_max = st.number_input(
                "t_max",
                min_value=0.1,
                max_value=1000.0,
                value=float(config.get("default_t_max", 10.0)),
                step=1.0,
            )
            default_steps = st.number_input(
                "Steps",
                min_value=10,
                max_value=10000,
                value=config.get("default_steps", 100),
                step=10,
            )
            default_initial_perc = (
                st.number_input(
                    "Initial % infected",
                    min_value=0.01,
                    max_value=100.0,
                    value=float(config.get("default_initial_perc", 0.01)) * 100,
                    step=0.1,
                )
                / 100.0
            )

    # ── Storage & Export ──────────────────────────────────
    st.markdown(
        _section_icon(
            "Storage & Export",
            "Directory paths and default export format",
            _ICON_FOLDER,
        ),
        unsafe_allow_html=True,
    )

    with st.container(key="pref_card_storage"):
        st.markdown(_sublabel("Directories"), unsafe_allow_html=True)
        col_data, col_exp = st.columns(2)

        with col_data:
            data_dir = st.text_input(
                "Data Directory",
                value=config.get("data_directory", "data"),
                help="Where simulation results are stored",
            )

        with col_exp:
            experiments_dir = st.text_input(
                "Experiments Directory",
                value=config.get("experiments_directory", "experiments"),
                help="Where experiment configurations are stored",
            )

    # ── Auto-save ─────────────────────────────────────────

    # Save API key when it changes (writes to secrets.toml).
    # Rerun immediately to refresh the status badge.
    old_key = current_key or ""
    new_key_str = new_key or ""
    if new_key_str != old_key:
        try:
            WebConfig.set_openai_api_key(new_key_str)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save API key: {str(e)}")

    # Auto-save all other config values on every render pass,
    # except the render immediately after a reset (to preserve defaults).
    if not skip_autosave:
        config.update(
            {
                "ai_model": selected_model,
                "chart_height": chart_height,
                "chart_template": chart_template,
                "chart_color_s": color_s,
                "chart_color_i": color_i,
                "chart_color_r": color_r,
                "default_nodes": default_nodes,
                "default_k_avg": default_k_avg,
                "default_exponent": default_exponent,
                "default_shape": default_shape,
                "default_scale": default_scale,
                "default_mu": default_mu,
                "default_lambda": default_lambda,
                "default_samples": default_samples,
                "default_num_runs": default_num_runs,
                "default_t_max": default_t_max,
                "default_steps": default_steps,
                "default_initial_perc": default_initial_perc,
                "data_directory": data_dir,
                "experiments_directory": experiments_dir,
            }
        )

    # ── Danger Zone ───────────────────────────────────────
    st.markdown(
        _section_icon(
            "Danger Zone",
            "Irreversible actions",
            _ICON_ALERT,
            icon_bg=COLORS["error_bg"],
            icon_color=COLORS["error"],
        ),
        unsafe_allow_html=True,
    )

    with st.container(key="pref_card_danger"):
        col_text, col_btn = st.columns([3, 1])

        with col_text:
            st.markdown(
                _dedent(
                    f"""
<div style="font-family:{FONTS['body']};font-size:0.875rem;color:{COLORS['gray_600']};line-height:1.6;padding-top:0.25rem;">
Reset all preferences to their default values. This action cannot be undone.
</div>
"""
                ),
                unsafe_allow_html=True,
            )

        with col_btn:
            with st.container(key="pref_reset"):
                if st.button("Reset all", type="secondary"):
                    config.config = WebConfig.DEFAULTS.copy()
                    config.save()
                    # Skip auto-save on the next render so defaults are preserved.
                    st.session_state["pref_skip_autosave"] = True
                    st.rerun()
