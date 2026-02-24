"""
SPKMC Web Interface - Main Application.

This is the entry point for the Streamlit web interface. It handles page routing,
sidebar navigation, and applies custom CSS styling.
"""

from __future__ import annotations

import streamlit as st

from spkmc import __version__
from spkmc.web.config import WebConfig
from spkmc.web.state import SessionState
from spkmc.web.styles import get_global_styles

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="SPKMC - Epidemic Simulation Manager",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    """Render the sidebar navigation with brand, nav items, and footer."""
    with st.sidebar:
        current_page = SessionState.get_current_page()

        # ── Brand ────────────────────────────────
        st.markdown(
            '<div style="margin-bottom:9rem;">'
            "<div style=\"font-family:'Plus Jakarta Sans',sans-serif;"
            "font-size:1.5rem;font-weight:800;color:white;"
            'letter-spacing:-0.03em;line-height:1;">SPKMC</div>'
            "<div style=\"font-family:'Plus Jakarta Sans',sans-serif;"
            "font-size:0.688rem;color:rgba(255,255,255,0.35);"
            'font-weight:500;margin-top:0.375rem;letter-spacing:0.02em;">'
            "Epidemic Simulation Manager</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Navigation ──────────────────────────
        if st.button(
            "Experiments",
            key="nav_experiments",
            width="stretch",
            type="primary" if current_page == "dashboard" else "secondary",
        ):
            SessionState.set_selected_experiment(None)
            SessionState.set_current_page("dashboard")
            st.rerun()

        if st.button(
            "Preferences",
            key="nav_settings",
            width="stretch",
            type="primary" if current_page == "settings" else "secondary",
        ):
            SessionState.set_selected_experiment(None)
            SessionState.set_current_page("settings")
            st.rerun()

        # ── Version footer (fixed to sidebar bottom) ──
        st.markdown(
            '<div class="sidebar-version-footer">'
            '<div style="padding-top:0.75rem;'
            "border-top:1px solid rgba(255,255,255,0.06);"
            "font-family:'JetBrains Mono',monospace;font-size:0.625rem;"
            "color:rgba(255,255,255,0.18);letter-spacing:0.02em;"
            'text-align:right;">'
            f"v{__version__}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Main application entry point."""
    # Apply global styles
    st.markdown(get_global_styles(), unsafe_allow_html=True)

    # Initialize session state
    SessionState.init()

    # Load configuration
    if "config" not in st.session_state:
        st.session_state.config = WebConfig()

    # Restore running simulations and analyses from disk (survives refresh)
    if not st.session_state.get("_sims_restored"):
        SessionState.restore_running_simulations()
        SessionState.restore_running_analyses()
        st.session_state._sims_restored = True

    # Render sidebar
    render_sidebar()

    # Page routing
    current_page = SessionState.get_current_page()

    if current_page == "dashboard":
        from spkmc.web.pages import dashboard

        if SessionState.get_selected_experiment():
            from spkmc.web.pages import experiment_detail

            experiment_detail.render()
        else:
            dashboard.render()

    elif current_page == "settings":
        from spkmc.web.pages import settings

        settings.render()

    else:
        from spkmc.web.pages import dashboard

        dashboard.render()


if __name__ == "__main__":
    main()
