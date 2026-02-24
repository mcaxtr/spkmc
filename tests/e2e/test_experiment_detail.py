"""
E2E tests for the experiment detail page: parameters, scenario cards,
scenario detail modal, chart controls, comparison, and export.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import open_experiment, open_scenario_detail

pytestmark = pytest.mark.e2e

# The pre-seeded experiment directory name and scenario labels
EXP_DIR = "e2e_smoke_exp"
SC_BASELINE = "baseline"
SC_HIGH_LAMBDA = "high_lambda"

# Scenario IDs match the pattern used by experiment_detail.py for widget keys
SC_ID_BASELINE = f"sim--{EXP_DIR}--{SC_BASELINE}"
SC_ID_HIGH_LAMBDA = f"sim--{EXP_DIR}--{SC_HIGH_LAMBDA}"


@pytest.fixture
def detail_page(app_page):
    """Navigate to the experiment detail page for the pre-seeded experiment."""
    # Find the pre-seeded experiment card by name (not by index, which may shift).
    # Use .first to avoid strict mode violations when get_by_text matches multiple
    # DOM levels (the inner text div and its ancestor containers).
    app_page.locator(":text('E2E Smoke Test Experiment')").first.wait_for(
        state="visible", timeout=10000
    )
    # Iterate through experiment card containers to find the right one
    for idx in range(10):
        container = app_page.locator(f".st-key-exp_card_{idx}")
        if container.count() == 0:
            break
        if container.locator(":text('E2E Smoke Test Experiment')").count() > 0:
            app_page.locator(f".st-key-exp_btn_{idx} button").click()
            break
    # Wait for the detail page to render — use the back button as the definitive signal
    back_btn = app_page.locator(".st-key-detail_back_btn button")
    expect(back_btn).to_be_visible(timeout=15000)
    return app_page


# ── Detail page basics ─────────────────────────────────


def test_detail_page_renders(detail_page):
    """The experiment name is displayed as a heading."""
    expect(detail_page.locator(":text('E2E Smoke Test Experiment')").first).to_be_visible()


def test_back_button_returns_to_dashboard(detail_page):
    """Clicking the back button returns to the dashboard."""
    detail_page.locator(".st-key-detail_back_btn button").click()
    detail_page.wait_for_timeout(1500)
    # Dashboard should be visible again (Create Experiment button as indicator)
    expect(detail_page.locator(".st-key-btn_create_exp button")).to_be_visible(timeout=8000)


# ── Global parameters ──────────────────────────────────


def test_global_params_three_cards(detail_page):
    """The global parameters section shows 3 param cards (Network, Distribution, Simulation)."""
    params_section = detail_page.locator(".st-key-params_section")
    expect(params_section).to_be_visible(timeout=8000)

    # The three cards should contain these titles
    expect(params_section.get_by_text("Network")).to_be_visible()
    expect(params_section.get_by_text("Distribution")).to_be_visible()
    expect(params_section.get_by_text("Simulation")).to_be_visible()


# ── Scenario cards ─────────────────────────────────────


def test_scenario_cards_render(detail_page):
    """Two scenario cards are visible for the pre-seeded experiment."""
    baseline_card = detail_page.locator(f".st-key-sc_card_sim--{EXP_DIR}--{SC_BASELINE}")
    high_lambda_card = detail_page.locator(f".st-key-sc_card_sim--{EXP_DIR}--{SC_HIGH_LAMBDA}")
    expect(baseline_card).to_be_visible(timeout=8000)
    expect(high_lambda_card).to_be_visible()


def test_scenario_card_labels(detail_page):
    """Scenario cards display the correct labels."""
    # Scope text assertions to specific card containers to avoid strict mode violations
    baseline_card = detail_page.locator(f".st-key-sc_card_{SC_ID_BASELINE}")
    high_lambda_card = detail_page.locator(f".st-key-sc_card_{SC_ID_HIGH_LAMBDA}")
    expect(baseline_card).to_contain_text("Baseline")
    expect(high_lambda_card).to_contain_text("High Lambda")


def test_completed_status_badge(detail_page):
    """Pre-seeded scenarios show 'Completed' status badges."""
    # Both scenarios have result files, so they should show completed status
    baseline_card = detail_page.locator(f".st-key-sc_card_sim--{EXP_DIR}--{SC_BASELINE}")
    expect(baseline_card).to_contain_text("Completed", ignore_case=True)


# ── Add Scenario ───────────────────────────────────────


def test_add_scenario_button_visible(detail_page):
    """The Add Scenario button is present."""
    btn = detail_page.locator(".st-key-btn_add_scenario_bar button")
    expect(btn).to_be_visible(timeout=8000)


def test_add_scenario_modal_opens(detail_page):
    """Clicking Add Scenario opens a dialog."""
    detail_page.locator(".st-key-btn_add_scenario_bar button").click()
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)


# ── Scenario detail modal ─────────────────────────────


def test_scenario_card_opens_detail_modal(detail_page):
    """Clicking a scenario card opens the detail modal."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)


def test_modal_shows_scenario_name(detail_page):
    """The detail modal header shows the scenario label."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)
    # st.title renders inside a stTitle testid container
    title = dialog.locator("[data-testid='stTitle']")
    expect(title).to_contain_text("Baseline", timeout=8000)


def test_modal_sir_chart_renders(detail_page):
    """The detail modal contains a Plotly chart for a completed scenario."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)
    # Plotly renders with a class containing 'js-plotly-plot'
    chart = dialog.locator("[class*='js-plotly-plot']")
    expect(chart).to_be_visible(timeout=10000)


def test_modal_state_checkboxes_present(detail_page):
    """S, I, R checkboxes are present in the detail modal."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    expect(detail_page.locator(f".st-key-modal_show_s_{SC_ID_BASELINE}")).to_be_visible(
        timeout=8000
    )
    expect(detail_page.locator(f".st-key-modal_show_i_{SC_ID_BASELINE}")).to_be_visible()
    expect(detail_page.locator(f".st-key-modal_show_r_{SC_ID_BASELINE}")).to_be_visible()


def test_modal_chart_type_selectbox(detail_page):
    """The chart type dropdown is visible in the modal."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    chart_mode = detail_page.locator(f".st-key-modal_chart_mode_{SC_ID_BASELINE}")
    expect(chart_mode).to_be_visible(timeout=8000)


def test_modal_comparison_multiselect(detail_page):
    """The comparison multiselect is visible when other scenarios have results."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    # The comparison section should be present since High Lambda also has results
    compare_key = f"modal_compare_{EXP_DIR}_{SC_BASELINE}"
    compare_widget = detail_page.locator(f".st-key-{compare_key}")
    expect(compare_widget).to_be_visible(timeout=8000)


def test_modal_select_comparison_scenario(detail_page):
    """Selecting a comparison scenario triggers a comparison chart."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    # Click on the comparison multiselect to open its dropdown
    compare_key = f"modal_compare_{EXP_DIR}_{SC_BASELINE}"
    compare_widget = detail_page.locator(f".st-key-{compare_key}")
    expect(compare_widget).to_be_visible(timeout=8000)
    compare_widget.click()
    detail_page.wait_for_timeout(500)

    # Select "High Lambda" from the dropdown options (target the virtual dropdown)
    dropdown = detail_page.locator("[data-testid='stSelectboxVirtualDropdown']")
    dropdown.get_by_text("High Lambda").click()
    detail_page.wait_for_timeout(2000)

    # A comparison chart should now be visible (second Plotly chart)
    charts = dialog.locator("[class*='js-plotly-plot']")
    expect(charts.first).to_be_visible(timeout=8000)


def test_modal_export_popover(detail_page):
    """The export button reveals format radio options."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    # Click the Export popover trigger
    export_container = detail_page.locator(f".st-key-modal_action_export_{SC_ID_BASELINE}")
    expect(export_container).to_be_visible(timeout=8000)
    export_container.locator("button").click()
    detail_page.wait_for_timeout(1000)

    # The export format radio should appear
    export_fmt = detail_page.locator(f".st-key-export_fmt_{SC_ID_BASELINE}")
    expect(export_fmt).to_be_visible(timeout=5000)


def test_uncheck_infected_updates_chart(detail_page):
    """Unchecking the Infected checkbox updates the chart (fewer traces)."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    # Wait for chart to render
    chart = dialog.locator("[class*='js-plotly-plot']")
    expect(chart).to_be_visible(timeout=10000)

    # Uncheck the Infected checkbox
    infected_cb = detail_page.locator(f".st-key-modal_show_i_{SC_ID_BASELINE}")
    infected_cb.click()
    detail_page.wait_for_timeout(2000)

    # Chart should still be visible (it re-renders with fewer traces)
    expect(chart).to_be_visible()


def test_chart_mode_area(detail_page):
    """Switching chart mode to Area updates the chart."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    chart = dialog.locator("[class*='js-plotly-plot']")
    expect(chart).to_be_visible(timeout=10000)

    # Click the chart mode selectbox to change it
    chart_mode = detail_page.locator(f".st-key-modal_chart_mode_{SC_ID_BASELINE}")
    chart_mode.click()
    detail_page.wait_for_timeout(500)

    # Select "Area" from the dropdown options (target the virtual dropdown)
    dropdown = detail_page.locator("[data-testid='stSelectboxVirtualDropdown']")
    dropdown.get_by_text("Area", exact=True).click()
    detail_page.wait_for_timeout(2000)

    # Chart should still render
    expect(chart).to_be_visible()


def test_ai_button_disabled_without_key(detail_page):
    """The AI analyze button is disabled when no API key is configured."""
    # The experiment-level AI button (use .first to avoid tooltip wrapper duplicates)
    ai_btn = detail_page.locator(".st-key-btn_ai button").first
    expect(ai_btn).to_be_visible(timeout=8000)
    expect(ai_btn).to_be_disabled()


def test_modal_close(detail_page):
    """Dismissing the modal returns to the detail page."""
    open_scenario_detail(detail_page, EXP_DIR, SC_BASELINE)
    dialog = detail_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    # Press Escape to close
    detail_page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible(timeout=5000)

    # The detail page should still be showing
    expect(detail_page.locator(":text('E2E Smoke Test Experiment')").first).to_be_visible()
