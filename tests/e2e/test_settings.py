"""
E2E tests for the settings (Preferences) page: section cards,
inputs, defaults, and the reset button.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import navigate_to_settings

pytestmark = pytest.mark.e2e


@pytest.fixture
def settings_page(app_page):
    """Navigate to the settings page."""
    navigate_to_settings(app_page)
    # Use the unique subtitle as indicator that settings loaded
    expect(app_page.get_by_text("Configure web interface and simulation defaults")).to_be_visible(
        timeout=8000
    )
    return app_page


# ── Page structure ─────────────────────────────────────


def test_settings_page_renders(settings_page):
    """All section cards are visible on the settings page."""
    # 4 main section cards + danger zone = 5 containers
    expect(settings_page.locator(".st-key-pref_card_ai")).to_be_visible(timeout=8000)
    expect(settings_page.locator(".st-key-pref_card_viz")).to_be_visible()
    expect(settings_page.locator(".st-key-pref_card_sim")).to_be_visible()
    expect(settings_page.locator(".st-key-pref_card_storage")).to_be_visible()


# ── AI & Intelligence section ──────────────────────────


def test_ai_section_shows_not_configured(settings_page):
    """The AI section shows 'Not configured' when no API key is set."""
    ai_card = settings_page.locator(".st-key-pref_card_ai")
    expect(ai_card).to_contain_text("Not configured", ignore_case=True)


def test_model_selectbox_options(settings_page):
    """The AI model dropdown is present with model options."""
    ai_card = settings_page.locator(".st-key-pref_card_ai")
    # The selectbox should show the default model
    expect(ai_card).to_contain_text("gpt-4o-mini")


# ── Visualization section ──────────────────────────────


def test_chart_height_input_visible(settings_page):
    """The chart height number input is present."""
    viz_card = settings_page.locator(".st-key-pref_card_viz")
    expect(viz_card).to_contain_text("Default Height")


def test_template_selectbox_visible(settings_page):
    """The chart template dropdown is present."""
    viz_card = settings_page.locator(".st-key-pref_card_viz")
    expect(viz_card).to_contain_text("plotly_white")


def test_color_pickers_visible(settings_page):
    """Three color picker inputs are visible for S, I, R."""
    viz_card = settings_page.locator(".st-key-pref_card_viz")
    expect(viz_card.get_by_text("Susceptible")).to_be_visible()
    expect(viz_card.get_by_text("Infected")).to_be_visible()
    expect(viz_card.get_by_text("Recovered")).to_be_visible()


# ── Simulation Defaults section ────────────────────────


def test_simulation_defaults_section(settings_page):
    """The simulation defaults section shows Network, Distribution, Simulation subsections."""
    sim_card = settings_page.locator(".st-key-pref_card_sim")
    expect(sim_card.get_by_text("Network")).to_be_visible()
    expect(sim_card.get_by_text("Distribution")).to_be_visible()
    expect(sim_card.get_by_text("Simulation")).to_be_visible()


def test_default_nodes_reflects_config(settings_page):
    """The default nodes input reflects the value from the test config (100)."""
    sim_card = settings_page.locator(".st-key-pref_card_sim")
    # Find the Nodes input and check its value
    nodes_input = sim_card.locator("input[type='number']").first
    expect(nodes_input).to_have_value("100")


# ── Storage & Export section ───────────────────────────


def test_storage_inputs_visible(settings_page):
    """Data and Experiments directory inputs are present."""
    storage_card = settings_page.locator(".st-key-pref_card_storage")
    expect(storage_card.get_by_text("Data Directory")).to_be_visible()
    expect(storage_card.get_by_text("Experiments Directory")).to_be_visible()


# ── Danger Zone ────────────────────────────────────────


def test_danger_zone_reset_button(settings_page):
    """The Reset all button is present in the danger zone."""
    reset_container = settings_page.locator(".st-key-pref_reset")
    expect(reset_container).to_be_visible(timeout=8000)
    reset_btn = reset_container.locator("button")
    expect(reset_btn).to_be_visible()
    expect(reset_btn).to_contain_text("Reset all")
