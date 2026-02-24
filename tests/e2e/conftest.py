"""
E2E test configuration and fixtures.

Manages the Streamlit server lifecycle, pre-seeds experiment fixture data,
and provides reusable page navigation helpers.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import pytest

# ── Paths ──────────────────────────────────────────────

E2E_DIR = Path(__file__).parent
FIXTURES_DIR = E2E_DIR / "fixtures"
PROJECT_ROOT = E2E_DIR.parent.parent


# ── SIR result data generator ──────────────────────────


def _make_sir_result(n: int = 50) -> Dict:
    """Generate synthetic SIR simulation result data.

    Produces mathematically plausible S/I/R curves that look like a
    real epidemic simulation result, suitable for chart rendering tests.
    """
    t = np.linspace(0, 10, n)
    s = np.exp(-t * 0.3)
    i = 0.3 * np.exp(-((t - 3) ** 2) / 4)
    r = 1.0 - s - i
    r = np.clip(r, 0, 1)
    err = np.ones(n) * 0.01
    return {
        "time": t.tolist(),
        "S_val": s.tolist(),
        "I_val": i.tolist(),
        "R_val": r.tolist(),
        "S_err": err.tolist(),
        "I_err": err.tolist(),
        "R_err": err.tolist(),
    }


# ── Fixture: temp environment ──────────────────────────


@pytest.fixture(scope="session")
def app_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary environment with fixture experiments and pre-seeded results.

    Returns the temp directory root that contains:
    - experiments/<exp_name>/data.json  (copied from fixtures)
    - experiments/<exp_name>/baseline.json  (generated SIR result)
    - experiments/<exp_name>/high_lambda.json  (generated SIR result)
    - web_config.json  (pointing experiments_directory to the temp experiments dir)
    """
    tmp = tmp_path_factory.mktemp("spkmc_e2e")

    # Copy fixture experiments
    src_experiments = FIXTURES_DIR / "experiments"
    dst_experiments = tmp / "experiments"
    shutil.copytree(src_experiments, dst_experiments)

    # Generate result JSONs for the smoke experiment
    smoke_exp_dir = dst_experiments / "e2e_smoke_exp"
    for filename in ("baseline.json", "high_lambda.json"):
        result_path = smoke_exp_dir / filename
        result_path.write_text(json.dumps(_make_sir_result(), indent=2))

    # Write web config pointing to the temp experiments directory
    config_path = tmp / "web_config.json"
    config = {
        "data_directory": str(tmp / "data"),
        "experiments_directory": str(dst_experiments),
        "default_nodes": 100,
        "default_k_avg": 5.0,
        "default_samples": 10,
        "default_num_runs": 1,
        "default_t_max": 5.0,
        "default_steps": 50,
        "default_initial_perc": 0.01,
        "chart_height": 500,
        "chart_template": "plotly_white",
        "chart_color_s": "#4477AA",
        "chart_color_i": "#EE6677",
        "chart_color_r": "#228833",
        "ai_model": "gpt-4o-mini",
    }
    config_path.write_text(json.dumps(config, indent=2))

    # Create data directory
    (tmp / "data").mkdir(exist_ok=True)

    return tmp


# ── Fixture: Streamlit server ──────────────────────────


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    """Block until a TCP port accepts connections or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Streamlit server did not start on {host}:{port} within {timeout}s")


@pytest.fixture(scope="session")
def app_url(app_env: Path) -> Generator[str, None, None]:
    """Start the Streamlit server and yield its base URL.

    The server runs with:
    - SPKMC_WEB_CONFIG_FILE pointing to the temp config
    - Headless mode enabled, file watcher disabled
    - Port 8502 to avoid conflicts with a dev server on 8501
    """
    port = 8502
    config_path = app_env / "web_config.json"

    env = {**os.environ}
    env["SPKMC_WEB_CONFIG_FILE"] = str(config_path)

    app_path = str(PROJECT_ROOT / "spkmc" / "web" / "app.py")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--server.fileWatcherType",
            "none",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(app_env),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_port("localhost", port, timeout=60)
        yield f"http://localhost:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


# ── Fixture: Playwright base URL ───────────────────────


@pytest.fixture(scope="session")
def base_url(app_url: str) -> str:
    """Provide the base URL for pytest-playwright's page.goto()."""
    return app_url


# ── Fixture: page with sidebar ready ──────────────────


@pytest.fixture
def app_page(page, app_url: str):
    """Navigate to the app root and wait for the sidebar to be ready.

    Returns the Playwright page object after Streamlit has fully loaded.
    """
    page.goto(app_url)
    page.wait_for_selector("[data-testid='stSidebar']", timeout=15000)
    # Wait a bit for Streamlit to settle its initial render
    page.wait_for_timeout(1000)
    return page


# ── Navigation helpers ─────────────────────────────────


def navigate_to_settings(page) -> None:
    """Click the Preferences nav button in the sidebar."""
    page.locator(".st-key-nav_settings button").click()
    page.wait_for_timeout(1500)


def navigate_to_dashboard(page) -> None:
    """Click the Experiments nav button in the sidebar."""
    page.locator(".st-key-nav_experiments button").click()
    page.wait_for_timeout(1500)


def open_experiment(page, idx: int = 0) -> None:
    """Click the experiment card at the given index to open its detail view."""
    btn = page.locator(f".st-key-exp_btn_{idx} button")
    btn.wait_for(state="visible", timeout=10000)
    btn.click()
    page.wait_for_timeout(1500)


def open_scenario_detail(page, experiment_dir: str, scenario_label: str) -> None:
    """Click a scenario card to open its detail modal.

    Args:
        page: Playwright page object
        experiment_dir: The experiment directory name (e.g. "e2e_smoke_exp")
        scenario_label: The normalized scenario label (e.g. "baseline", "high_lambda")
    """
    scenario_id = f"sim--{experiment_dir}--{scenario_label}"
    page.locator(f".st-key-sc_btn_{scenario_id} button").click()
    page.wait_for_timeout(1500)
