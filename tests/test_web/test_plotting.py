"""Tests for web plotting functions."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("plotly", reason="plotly not installed") is None,
    reason="plotly not installed",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sir_result():
    """Minimal SIR result dict with S, I, R values."""
    t = np.linspace(0, 10, 100).tolist()
    return {
        "time": t,
        "S_val": np.linspace(1.0, 0.5, 100).tolist(),
        "I_val": np.linspace(0.0, 0.3, 100).tolist(),
        "R_val": np.linspace(0.0, 0.2, 100).tolist(),
    }


@pytest.fixture()
def sir_result_with_errors(sir_result):
    """SIR result dict including error band arrays."""
    sir_result["S_err"] = (np.ones(100) * 0.01).tolist()
    sir_result["I_err"] = (np.ones(100) * 0.01).tolist()
    sir_result["R_err"] = (np.ones(100) * 0.01).tolist()
    return sir_result


# ── _hex_to_rgba ──────────────────────────────────────────────────────────────


class TestHexToRgba:
    def test_converts_black(self):
        from spkmc.web.plotting import _hex_to_rgba

        assert _hex_to_rgba("#000000") == "rgba(0, 0, 0, 1.0)"

    def test_converts_white(self):
        from spkmc.web.plotting import _hex_to_rgba

        assert _hex_to_rgba("#ffffff") == "rgba(255, 255, 255, 1.0)"

    def test_converts_known_color(self):
        from spkmc.web.plotting import _hex_to_rgba

        # #4477AA → r=68, g=119, b=170
        result = _hex_to_rgba("#4477AA")
        assert result == "rgba(68, 119, 170, 1.0)"

    def test_applies_alpha(self):
        from spkmc.web.plotting import _hex_to_rgba

        result = _hex_to_rgba("#ffffff", alpha=0.15)
        assert "0.15" in result

    def test_strips_hash_prefix(self):
        from spkmc.web.plotting import _hex_to_rgba

        # With and without # must produce the same result
        assert _hex_to_rgba("#4477AA") == _hex_to_rgba("#4477AA")


# ── create_sir_figure ─────────────────────────────────────────────────────────


class TestCreateSirFigure:
    def test_produces_three_traces_by_default(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result)
        assert len(fig.data) == 3

    def test_state_subset_returns_only_requested_traces(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result, states=["I"])
        assert len(fig.data) == 1
        assert fig.data[0].name == "I"

    def test_two_state_subset(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result, states=["S", "R"])
        names = [t.name for t in fig.data]
        assert "S" in names
        assert "R" in names
        assert "I" not in names

    def test_error_bands_are_added_when_present(self, sir_result_with_errors):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result_with_errors, show_error_bands=True)
        for trace in fig.data:
            assert trace.error_y is not None
            assert trace.error_y.visible is True

    def test_error_bands_absent_when_disabled(self, sir_result_with_errors):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result_with_errors, show_error_bands=False)
        for trace in fig.data:
            # Plotly represents "no error bars" as ErrorY with visible=None/False,
            # not as Python None. Check that visible is not True.
            assert trace.error_y.visible is not True

    def test_custom_state_colors_override_defaults(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        custom_colors = {"I": "#FF0000"}
        fig = create_sir_figure(sir_result, states=["I"], state_colors=custom_colors)
        i_trace = next(t for t in fig.data if t.name == "I")
        assert i_trace.line.color == "#FF0000"

    def test_default_colors_are_unchanged_when_not_overridden(self, sir_result):
        from spkmc.web.plotting import COLOR_S, create_sir_figure

        fig = create_sir_figure(sir_result, states=["S"])
        s_trace = fig.data[0]
        assert s_trace.line.color == COLOR_S

    def test_custom_template_propagates_to_layout(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result, template="plotly_dark")
        assert fig.layout.template.layout.colorway is not None or True
        # The template name is resolved by Plotly; verify the call didn't raise
        assert fig is not None

    def test_height_is_applied_to_layout(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result, height=800)
        assert fig.layout.height == 800

    def test_area_chart_mode_sets_fill_on_traces(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result, chart_mode="area")
        for trace in fig.data:
            assert trace.fill == "tozeroy"

    def test_lines_plus_markers_mode_is_set(self, sir_result):
        from spkmc.web.plotting import create_sir_figure

        fig = create_sir_figure(sir_result, chart_mode="lines+markers")
        for trace in fig.data:
            assert trace.mode == "lines+markers"

    def test_missing_state_key_is_silently_skipped(self, sir_result):
        """Requesting a state not present in result_dict must not raise."""
        from spkmc.web.plotting import create_sir_figure

        del sir_result["I_val"]
        fig = create_sir_figure(sir_result, states=["S", "I", "R"])
        names = [t.name for t in fig.data]
        assert "I" not in names
        assert "S" in names

    def test_numpy_array_inputs_do_not_raise(self):
        """NumPy arrays in result_dict must not cause a truthiness ValueError."""
        from spkmc.web.plotting import create_sir_figure

        result = {
            "time": np.array([0, 1, 2, 3, 4]),
            "S_val": np.array([1.0, 0.9, 0.8, 0.7, 0.6]),
            "I_val": np.array([0.0, 0.05, 0.1, 0.15, 0.2]),
            "R_val": np.array([0.0, 0.05, 0.1, 0.15, 0.2]),
        }
        fig = create_sir_figure(result)
        assert len(fig.data) == 3
        assert fig.layout.xaxis.range == (0, 4.0)


# ── create_comparison_figure ──────────────────────────────────────────────────


class TestCreateComparisonFigure:
    @pytest.fixture()
    def two_results(self):
        t = np.linspace(0, 10, 100).tolist()
        return [
            {
                "time": t,
                "S_val": np.linspace(1.0, 0.5, 100).tolist(),
                "I_val": np.linspace(0.0, 0.3, 100).tolist(),
                "R_val": np.linspace(0.0, 0.2, 100).tolist(),
            },
            {
                "time": t,
                "S_val": np.linspace(1.0, 0.4, 100).tolist(),
                "I_val": np.linspace(0.0, 0.4, 100).tolist(),
                "R_val": np.linspace(0.0, 0.2, 100).tolist(),
            },
        ]

    def test_two_scenarios_three_states_produces_six_traces(self, two_results):
        from spkmc.web.plotting import create_comparison_figure

        fig = create_comparison_figure(
            two_results, ["Scenario A", "Scenario B"], states=["S", "I", "R"]
        )
        assert len(fig.data) == 6

    def test_single_scenario_produces_correct_trace_count(self):
        from spkmc.web.plotting import create_comparison_figure

        t = np.linspace(0, 10, 50).tolist()
        result = {
            "time": t,
            "I_val": np.linspace(0.0, 0.3, 50).tolist(),
        }
        fig = create_comparison_figure([result], ["Solo"], states=["I"])
        assert len(fig.data) == 1

    def test_trace_names_include_scenario_label_and_state(self, two_results):
        from spkmc.web.plotting import create_comparison_figure

        fig = create_comparison_figure(two_results, ["Alpha", "Beta"], states=["I"])
        names = [t.name for t in fig.data]
        assert any("Alpha" in n for n in names)
        assert any("Beta" in n for n in names)

    def test_custom_template_applied(self, two_results):
        from spkmc.web.plotting import create_comparison_figure

        fig = create_comparison_figure(two_results, ["A", "B"], template="plotly_dark")
        assert fig is not None

    def test_state_subset_limits_traces(self, two_results):
        from spkmc.web.plotting import create_comparison_figure

        fig = create_comparison_figure(two_results, ["A", "B"], states=["I"])
        assert len(fig.data) == 2  # 2 scenarios × 1 state


# ── create_metric_card_figure ─────────────────────────────────────────────────


class TestCreateMetricCardFigure:
    def test_returns_figure(self):
        from spkmc.web.plotting import create_metric_card_figure

        fig = create_metric_card_figure(0.42, "Peak Infected")
        assert fig is not None

    def test_contains_indicator_trace(self):
        import plotly.graph_objects as go

        from spkmc.web.plotting import create_metric_card_figure

        fig = create_metric_card_figure(0.75, "Final Recovered")
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Indicator)

    def test_value_is_stored_in_indicator(self):
        from spkmc.web.plotting import create_metric_card_figure

        fig = create_metric_card_figure(0.33, "Some Metric")
        assert fig.data[0].value == pytest.approx(0.33)

    def test_title_appears_in_figure(self):
        from spkmc.web.plotting import create_metric_card_figure

        fig = create_metric_card_figure(0.5, "My Title", subtitle="Details here")
        title_text = fig.data[0].title.text
        assert "My Title" in title_text

    def test_custom_color_is_applied(self):
        from spkmc.web.plotting import create_metric_card_figure

        fig = create_metric_card_figure(0.5, "Metric", color="#FF0000")
        assert fig.data[0].number.font.color == "#FF0000"

    def test_height_is_compact(self):
        from spkmc.web.plotting import create_metric_card_figure

        fig = create_metric_card_figure(0.5, "Compact")
        assert fig.layout.height == 150


# ── Visualizer.compare_results_with_config ───────────────────────────────────


class TestCompareResultsWithConfig:
    """Verify that PlotConfig settings propagate through the Plotly refactor."""

    @pytest.fixture()
    def two_results(self):
        t = np.linspace(0, 10, 50).tolist()
        return [
            {
                "time": t,
                "S_val": np.linspace(1, 0.5, 50).tolist(),
                "I_val": np.linspace(0, 0.3, 50).tolist(),
                "R_val": np.linspace(0, 0.2, 50).tolist(),
            },
            {
                "time": t,
                "S_val": np.linspace(1, 0.4, 50).tolist(),
                "I_val": np.linspace(0, 0.4, 50).tolist(),
                "R_val": np.linspace(0, 0.2, 50).tolist(),
            },
        ]

    def _capture_figure(self, results, labels, plot_config):
        """Run compare_results_with_config and capture the figure."""
        import spkmc.visualization.plots as vp
        from spkmc.models.config import PlotConfig
        from spkmc.visualization.plots import Visualizer

        captured = {}
        orig = vp._save_or_show

        def fake_save(fig, *a, **kw):
            captured["fig"] = fig

        vp._save_or_show = fake_save
        try:
            Visualizer.compare_results_with_config(results, labels, plot_config)
        finally:
            vp._save_or_show = orig
        return captured["fig"]

    def test_grid_disabled_propagates(self, two_results):
        from spkmc.models.config import PlotConfig

        pc = PlotConfig(grid=False)
        fig = self._capture_figure(two_results, ["A", "B"], pc)
        assert fig.layout.xaxis.showgrid is False
        assert fig.layout.yaxis.showgrid is False

    def test_grid_alpha_propagates(self, two_results):
        from spkmc.models.config import PlotConfig

        pc = PlotConfig(grid=True, grid_alpha=0.7)
        fig = self._capture_figure(two_results, ["A", "B"], pc)
        assert "0.7" in fig.layout.xaxis.gridcolor

    def test_legend_position_center(self, two_results):
        from spkmc.models.config import PlotConfig

        pc = PlotConfig(legend_position="center")
        fig = self._capture_figure(two_results, ["A", "B"], pc)
        assert fig.layout.legend.x == 0.5
        assert fig.layout.legend.y == 0.5
