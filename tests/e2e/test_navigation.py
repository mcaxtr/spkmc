"""
E2E tests for sidebar navigation, page routing, and app chrome.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import navigate_to_dashboard, navigate_to_settings

pytestmark = pytest.mark.e2e


def test_app_loads(app_page):
    """The app loads without crash and the sidebar is visible."""
    sidebar = app_page.locator("[data-testid='stSidebar']")
    expect(sidebar).to_be_visible()


def test_page_title(app_page):
    """The browser tab title contains SPKMC."""
    assert "SPKMC" in app_page.title()


def test_dashboard_is_default_page(app_page):
    """The dashboard is shown by default on first load."""
    # Dashboard renders stat cards — use a specific stat card label as indicator.
    # Use .first to handle potential multi-element matches from nested HTML containers.
    expect(app_page.locator(":text('Total Experiments')").first).to_be_visible(timeout=10000)


def test_navigate_to_settings(app_page):
    """Clicking the Preferences button navigates to the settings page."""
    navigate_to_settings(app_page)
    # Settings page has a unique subtitle
    expect(app_page.get_by_text("Configure web interface and simulation defaults")).to_be_visible(
        timeout=8000
    )


def test_navigate_back_to_dashboard(app_page):
    """Clicking the Experiments button returns to the dashboard."""
    navigate_to_settings(app_page)
    navigate_to_dashboard(app_page)
    expect(app_page.locator(":text('Total Experiments')").first).to_be_visible(timeout=10000)


def test_sidebar_brand_visible(app_page):
    """The SPKMC brand text is visible in the sidebar."""
    sidebar = app_page.locator("[data-testid='stSidebar']")
    expect(sidebar.get_by_text("SPKMC")).to_be_visible()


def test_sidebar_version_visible(app_page):
    """The version footer is displayed in the sidebar."""
    sidebar = app_page.locator("[data-testid='stSidebar']")
    version = sidebar.locator(".sidebar-version-footer")
    expect(version).to_be_visible()
    # Version text should start with 'v'
    expect(version).to_contain_text("v")


def test_query_params_reflect_page(app_page):
    """URL query params reflect the current page after navigation."""
    navigate_to_settings(app_page)
    assert "page=settings" in app_page.url
