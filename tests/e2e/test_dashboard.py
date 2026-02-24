"""
E2E tests for the dashboard page: stats cards, experiment cards,
and the Create Experiment modal.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import open_experiment

pytestmark = pytest.mark.e2e


# ── Stats cards ────────────────────────────────────────


def test_stats_cards_render(app_page):
    """The dashboard shows 4 stat card columns."""
    # Stat cards are rendered as raw HTML via st.markdown(unsafe_allow_html=True).
    # Use locator() with :text() pseudo-class and .first to handle potential
    # multi-element matches from nested DOM structure.
    expect(app_page.locator(":text('Total Experiments')").first).to_be_visible(timeout=10000)
    expect(app_page.locator(":text('Total Scenarios')").first).to_be_visible(timeout=5000)
    expect(app_page.locator(":text('Completed Scenarios')").first).to_be_visible(timeout=5000)
    expect(app_page.locator(":text('Last Activity')").first).to_be_visible(timeout=5000)


# ── Create Experiment button & modal ───────────────────


def test_create_button_visible(app_page):
    """The Create Experiment button is present on the dashboard."""
    btn = app_page.locator(".st-key-btn_create_exp button")
    expect(btn).to_be_visible(timeout=8000)


def test_create_modal_opens(app_page):
    """Clicking Create Experiment opens a modal dialog."""
    app_page.locator(".st-key-btn_create_exp button").click()
    dialog = app_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)


def test_create_modal_has_name_input(app_page):
    """The create modal contains a text input for the experiment name."""
    app_page.locator(".st-key-btn_create_exp button").click()
    dialog = app_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)
    # The dialog should contain a text input (name field)
    name_input = dialog.locator("input[type='text']").first
    expect(name_input).to_be_visible()


def test_create_modal_has_network_config(app_page):
    """The create modal includes network type configuration."""
    app_page.locator(".st-key-btn_create_exp button").click()
    dialog = app_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)
    # Network type is configured via a selectbox with key="create_network_type"
    expect(app_page.locator(".st-key-create_network_type")).to_be_visible()


def test_create_modal_cancel_closes(app_page):
    """Closing/dismissing the create modal makes it disappear."""
    app_page.locator(".st-key-btn_create_exp button").click()
    dialog = app_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    # Press Escape to close the modal
    app_page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible(timeout=5000)


# ── Experiment cards ───────────────────────────────────


def test_experiment_card_renders(app_page):
    """The pre-seeded experiment card is visible on the dashboard."""
    card = app_page.locator(".st-key-exp_card_0")
    expect(card).to_be_visible(timeout=8000)


def test_experiment_card_shows_name(app_page):
    """The experiment card displays the experiment name."""
    # Use text-based lookup — card index can shift when other tests create experiments
    expect(app_page.get_by_text("E2E Smoke Test Experiment").first).to_be_visible(timeout=8000)


def test_experiment_card_shows_scenario_count(app_page):
    """The experiment card shows the correct number of scenarios."""
    # Find the card containing the pre-seeded experiment name, then check scenario count
    cards = app_page.locator("[class*='st-key-exp_card_']")
    card_count = cards.count()
    found = False
    for i in range(card_count):
        card = cards.nth(i)
        if "E2E Smoke Test Experiment" in (card.text_content() or ""):
            expect(card).to_contain_text("2")
            found = True
            break
    assert found, "Pre-seeded experiment card not found"


def test_experiment_card_clickable(app_page):
    """Clicking an experiment card navigates to the detail view."""
    # Find the card container that has the pre-seeded experiment, then click its button.
    # Card indices can shift when other tests create experiments, so we search by text.
    cards = app_page.locator("[class*='st-key-exp_card_']")
    card_count = cards.count()
    clicked = False
    for i in range(card_count):
        card = cards.nth(i)
        if "E2E Smoke Test Experiment" in (card.text_content() or ""):
            card.locator("button").click()
            clicked = True
            break
    assert clicked, "Pre-seeded experiment card not found"
    # Wait for the detail-specific back button to confirm navigation succeeded
    back_btn = app_page.locator(".st-key-detail_back_btn button")
    expect(back_btn).to_be_visible(timeout=15000)
    # The detail view should show the experiment name
    expect(app_page.locator(":text('E2E Smoke Test Experiment')").first).to_be_visible()


# ── Create experiment flow ─────────────────────────────


def test_create_experiment_flow(app_page):
    """Creating a new experiment adds a card to the dashboard."""
    # Open the create modal
    app_page.locator(".st-key-btn_create_exp button").click()
    dialog = app_page.locator("[data-testid='stDialog']")
    expect(dialog).to_be_visible(timeout=8000)

    # Fill in the experiment name
    name_input = dialog.locator("input[type='text']").first
    name_input.fill("E2E Created Experiment")

    # Submit — look for a Create/Save button inside the dialog
    create_btn = dialog.get_by_role("button", name="Create")
    expect(create_btn).to_be_visible(timeout=5000)
    create_btn.click()

    # Wait for the dialog to close and the page to rerender
    app_page.wait_for_timeout(2000)

    # The new experiment should appear somewhere on the page
    expect(app_page.get_by_text("E2E Created Experiment")).to_be_visible(timeout=8000)
