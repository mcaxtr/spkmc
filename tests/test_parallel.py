"""
Tests for parallel execution utilities.
"""

import time
from typing import Any, Dict

import pytest

from spkmc.utils.hardware import ParallelizationStrategy
from spkmc.utils.parallel import ParallelBatchExecutor, ScenarioResult, run_scenarios_parallel


def simple_scenario_executor(scenario: Dict[str, Any], index: int) -> ScenarioResult:
    """Simple scenario executor for testing."""
    time.sleep(0.01)  # Simulate some work
    return ScenarioResult(
        scenario_index=index,
        label=scenario.get("label", f"scenario_{index}"),
        success=True,
        result_path=f"/tmp/result_{index}.json",
        execution_time=0.01,
    )


def failing_scenario_executor(scenario: Dict[str, Any], index: int) -> ScenarioResult:
    """Scenario executor that fails on specific indices."""
    if scenario.get("should_fail", False):
        raise ValueError(f"Simulated failure for scenario {index}")
    return simple_scenario_executor(scenario, index)


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_scenario_result_success(self):
        """Should create a successful result."""
        result = ScenarioResult(
            scenario_index=0,
            label="test_scenario",
            success=True,
            result_path="/tmp/result.json",
            execution_time=1.5,
        )

        assert result.scenario_index == 0
        assert result.label == "test_scenario"
        assert result.success is True
        assert result.result_path == "/tmp/result.json"
        assert result.error is None

    def test_scenario_result_failure(self):
        """Should create a failed result."""
        result = ScenarioResult(
            scenario_index=1, label="failed_scenario", success=False, error="Something went wrong"
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.result_path is None


class TestRunScenariosParallel:
    """Tests for run_scenarios_parallel function."""

    @pytest.fixture
    def sequential_strategy(self):
        """Strategy that forces sequential execution."""
        return ParallelizationStrategy(
            scenario_workers=1, simulation_workers=1, numba_threads=2, use_gpu=False
        )

    @pytest.fixture
    def parallel_strategy(self):
        """Strategy that enables parallel execution."""
        return ParallelizationStrategy(
            scenario_workers=2, simulation_workers=1, numba_threads=2, use_gpu=False
        )

    def test_sequential_execution(self, sequential_strategy):
        """Should execute scenarios sequentially with 1 worker."""
        scenarios = [{"id": i, "label": f"scenario_{i}"} for i in range(3)]

        results = run_scenarios_parallel(scenarios, simple_scenario_executor, sequential_strategy)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_parallel_execution(self, parallel_strategy):
        """Should execute scenarios in parallel with multiple workers."""
        scenarios = [{"id": i, "label": f"scenario_{i}"} for i in range(4)]

        results = run_scenarios_parallel(scenarios, simple_scenario_executor, parallel_strategy)

        assert len(results) == 4
        assert all(r.success for r in results)

    def test_preserves_order(self, parallel_strategy):
        """Results should be in original scenario order."""
        scenarios = [{"id": i, "label": f"scenario_{i}"} for i in range(5)]

        results = run_scenarios_parallel(scenarios, simple_scenario_executor, parallel_strategy)

        for i, result in enumerate(results):
            assert result.scenario_index == i

    def test_progress_callback(self, sequential_strategy):
        """Should call progress callback for each completed scenario."""
        scenarios = [{"id": i} for i in range(3)]
        progress_calls = []

        def on_progress(completed, total, label):
            progress_calls.append((completed, total, label))

        run_scenarios_parallel(
            scenarios, simple_scenario_executor, sequential_strategy, progress_callback=on_progress
        )

        assert len(progress_calls) == 3
        assert progress_calls[-1][0] == 3  # Last call should show all completed


class TestParallelBatchExecutor:
    """Tests for ParallelBatchExecutor class."""

    @pytest.fixture
    def sequential_strategy(self):
        return ParallelizationStrategy(
            scenario_workers=1, simulation_workers=1, numba_threads=2, use_gpu=False
        )

    @pytest.fixture
    def parallel_strategy(self):
        return ParallelizationStrategy(
            scenario_workers=2, simulation_workers=1, numba_threads=2, use_gpu=False
        )

    def test_is_parallel_property(self, parallel_strategy, sequential_strategy):
        """Should correctly report if parallel execution is enabled."""
        parallel_executor = ParallelBatchExecutor(parallel_strategy)
        sequential_executor = ParallelBatchExecutor(sequential_strategy)

        assert parallel_executor.is_parallel is True
        assert sequential_executor.is_parallel is False

    def test_worker_count_property(self, parallel_strategy):
        """Should report correct worker count."""
        executor = ParallelBatchExecutor(parallel_strategy)
        assert executor.worker_count == 2

    def test_execute_sequential(self, sequential_strategy):
        """Should execute scenarios sequentially."""
        scenarios = [{"id": i, "label": f"scenario_{i}"} for i in range(3)]
        executor = ParallelBatchExecutor(sequential_strategy)

        def scenario_executor(scenario, index, strategy):
            return ScenarioResult(
                scenario_index=index, label=scenario.get("label", f"scenario_{index}"), success=True
            )

        results = executor.execute(scenarios, scenario_executor)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_execute_with_errors(self, sequential_strategy):
        """Should handle errors gracefully."""
        scenarios = [
            {"id": 0, "label": "ok_1"},
            {"id": 1, "label": "fail", "should_fail": True},
            {"id": 2, "label": "ok_2"},
        ]
        executor = ParallelBatchExecutor(sequential_strategy)
        errors = []

        def scenario_executor(scenario, index, strategy):
            if scenario.get("should_fail"):
                raise ValueError("Simulated failure")
            return ScenarioResult(scenario_index=index, label=scenario.get("label"), success=True)

        def on_error(idx, label, exc):
            errors.append((idx, label, str(exc)))

        results = executor.execute(scenarios, scenario_executor, on_error=on_error)

        # Should have 3 results (2 success, 1 failure)
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

        # Should have recorded the error
        assert len(errors) == 1
        assert errors[0][1] == "fail"

    def test_get_summary(self, sequential_strategy):
        """Should provide correct execution summary."""
        scenarios = [
            {"id": 0},
            {"id": 1, "should_fail": True},
            {"id": 2},
        ]
        executor = ParallelBatchExecutor(sequential_strategy)

        def scenario_executor(scenario, index, strategy):
            if scenario.get("should_fail"):
                raise ValueError("Fail")
            return ScenarioResult(scenario_index=index, label=f"scenario_{index}", success=True)

        executor.execute(scenarios, scenario_executor)
        total, succeeded, failed = executor.get_summary()

        assert total == 3
        assert succeeded == 2
        assert failed == 1

    def test_progress_callback(self, sequential_strategy):
        """Should call progress callback during execution."""
        scenarios = [{"id": i} for i in range(3)]
        executor = ParallelBatchExecutor(sequential_strategy)
        progress_calls = []

        def on_progress(completed, total, label):
            progress_calls.append((completed, total))

        def scenario_executor(scenario, index, strategy):
            return ScenarioResult(scenario_index=index, label=f"scenario_{index}", success=True)

        executor.execute(scenarios, scenario_executor, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]
