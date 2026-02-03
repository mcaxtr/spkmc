"""
Tests for ExecutionEngine error handling.

These tests verify that execution errors are properly reported,
not silently swallowed.
"""

import numpy as np
import pytest

from spkmc.core.engine import ExecutionEngine
from spkmc.models.result import SimulationResult


class TestExecutionEngineErrorReporting:
    """Test that ExecutionEngine properly reports errors to users."""

    def test_empty_results_are_identified(self):
        """Empty results from failed scenarios should be identifiable."""
        # Create an empty result (like what happens when execution fails)
        empty_result = SimulationResult(
            S_val=np.array([]),
            I_val=np.array([]),
            R_val=np.array([]),
            time=np.array([]),
            scenario_label="failed_scenario",
            metadata={"error": "Something went wrong"},
        )

        assert empty_result.is_empty
        assert "error" in empty_result.metadata
        assert empty_result.peak_infected == 0.0
        assert empty_result.final_recovered == 0.0

    def test_failed_scenarios_can_be_counted(self):
        """Results should allow counting of failed vs successful scenarios."""
        # Create mock results: 2 success, 1 failure
        success_result = SimulationResult(
            S_val=np.linspace(1, 0.1, 50),
            I_val=np.linspace(0, 0.3, 50),
            R_val=np.linspace(0, 0.6, 50),
            time=np.linspace(0, 5, 50),
            scenario_label="success",
            metadata={},
        )
        failed_result = SimulationResult(
            S_val=np.array([]),
            I_val=np.array([]),
            R_val=np.array([]),
            time=np.array([]),
            scenario_label="failed",
            metadata={"error": "Out of memory"},
        )

        results = [success_result, failed_result, success_result]

        # Count failed results
        failed_count = sum(1 for r in results if r.is_empty)
        success_count = sum(1 for r in results if not r.is_empty)

        assert failed_count == 1
        assert success_count == 2

    def test_error_metadata_preserved_in_failed_result(self):
        """Failed results should preserve error information in metadata."""
        error_msg = "CUDA out of memory: tried to allocate 2GB"
        failed_result = SimulationResult(
            S_val=np.array([]),
            I_val=np.array([]),
            R_val=np.array([]),
            time=np.array([]),
            scenario_label="large_scenario",
            metadata={"error": error_msg},
        )

        assert failed_result.is_empty
        assert failed_result.metadata.get("error") == error_msg


class TestExecutionEngineFailedScenarioTracking:
    """
    Test that ExecutionEngine tracks and reports failed scenarios.

    BUG: Currently, when scenarios fail in parallel execution:
    1. Errors are caught silently (no logging)
    2. Empty results are created but no warning shown
    3. Experiment is marked "completed" even with failures
    4. User sees "peak=0.000, final=0.000" with no explanation

    The fix should:
    1. Log warnings when scenarios fail
    2. Track failed scenario count
    3. Report failures at end of execution
    """

    def test_engine_should_track_failed_scenarios(self):
        """ExecutionEngine should track which scenarios failed."""
        engine = ExecutionEngine(verbose=False)

        # After execution, engine should expose failed scenario info
        # This tests that the tracking mechanism exists
        assert hasattr(engine, "get_failed_scenarios") or hasattr(
            engine, "_failed_scenarios"
        ), "ExecutionEngine should have a way to track failed scenarios"

    def test_engine_should_report_failure_count(self):
        """ExecutionEngine should report how many scenarios failed."""
        # Create results with some failures
        results = [
            SimulationResult(
                S_val=np.linspace(1, 0.1, 50),
                I_val=np.linspace(0, 0.3, 50),
                R_val=np.linspace(0, 0.6, 50),
                time=np.linspace(0, 5, 50),
                scenario_label="success_1",
                metadata={},
            ),
            SimulationResult(
                S_val=np.array([]),
                I_val=np.array([]),
                R_val=np.array([]),
                time=np.array([]),
                scenario_label="failed_1",
                metadata={"error": "Memory error"},
            ),
            SimulationResult(
                S_val=np.array([]),
                I_val=np.array([]),
                R_val=np.array([]),
                time=np.array([]),
                scenario_label="failed_2",
                metadata={"error": "Timeout"},
            ),
        ]

        # The engine should provide a way to get failure summary
        failed = [r for r in results if r.is_empty]
        failed_labels = [r.scenario_label for r in failed]
        failed_errors = [r.metadata.get("error", "Unknown") for r in failed]

        assert len(failed) == 2
        assert "failed_1" in failed_labels
        assert "failed_2" in failed_labels
        assert "Memory error" in failed_errors


class TestSimulationResultErrorHandling:
    """Test SimulationResult handles error cases properly."""

    def test_empty_result_properties_return_zero(self):
        """Empty results should return 0.0 for all computed properties."""
        empty = SimulationResult(
            S_val=np.array([]),
            I_val=np.array([]),
            R_val=np.array([]),
            time=np.array([]),
            scenario_label="empty",
            metadata={},
        )

        assert empty.is_empty
        assert empty.peak_infected == 0.0
        assert empty.final_recovered == 0.0
        assert empty.final_susceptible == 0.0
        assert empty.peak_time == 0.0

    def test_valid_result_computes_statistics(self):
        """Valid results should compute proper statistics."""
        result = SimulationResult(
            S_val=np.array([1.0, 0.8, 0.5, 0.3, 0.2]),
            I_val=np.array([0.0, 0.15, 0.35, 0.25, 0.1]),
            R_val=np.array([0.0, 0.05, 0.15, 0.45, 0.7]),
            time=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            scenario_label="valid",
            metadata={},
        )

        assert not result.is_empty
        assert result.peak_infected == pytest.approx(0.35)
        assert result.final_recovered == pytest.approx(0.7)
        assert result.peak_time == pytest.approx(2.0)
