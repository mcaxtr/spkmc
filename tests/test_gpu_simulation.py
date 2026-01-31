"""
Tests for GPU-accelerated simulation functionality.

These tests verify that GPU results match CPU results (statistical equivalence)
and that GPU-specific features work correctly.
"""

import numpy as np
import pytest

# Skip entire module if GPU is not available
try:
    import cudf as _cudf  # noqa: F401 - imported for availability check
    import cugraph as _cugraph  # noqa: F401 - imported for availability check
    import cupy as cp

    GPU_AVAILABLE = True
    del _cudf, _cugraph  # Clean up namespace
except ImportError:
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU dependencies not available")


class TestGPUUtilsAvailability:
    """Tests for GPU utilities availability and configuration."""

    def test_is_gpu_available(self):
        """Should detect GPU availability correctly."""
        from spkmc.utils.gpu_utils import is_gpu_available, reset_gpu_cache

        reset_gpu_cache()
        result = is_gpu_available()
        assert isinstance(result, bool)
        # If we got here (not skipped), GPU should be available
        assert result is True

    def test_configure_memory_pool(self):
        """Should configure memory pool without errors."""
        from spkmc.utils.gpu_utils import configure_gpu_memory_pool, reset_gpu_cache

        reset_gpu_cache()
        result = configure_gpu_memory_pool(fraction=0.5)
        assert result is True

    def test_memory_pool_caches_result(self):
        """Should return True on second call (cached)."""
        from spkmc.utils.gpu_utils import configure_gpu_memory_pool, reset_gpu_cache

        reset_gpu_cache()
        configure_gpu_memory_pool(fraction=0.5)
        # Second call should also return True (already configured)
        result = configure_gpu_memory_pool(
            fraction=0.8
        )  # Different fraction, but already configured
        assert result is True


class TestBatchedGPUSimulator:
    """Tests for BatchedGPUSimulator class."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        # Small complete graph for testing
        N = 100
        edges = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    edges.append([i, j])
        return N, np.array(edges)

    @pytest.fixture
    def time_steps(self):
        """Create time steps array."""
        return np.linspace(0, 30, 100).astype(np.float32)

    @pytest.fixture
    def gamma_params(self):
        """Gamma distribution parameters."""
        return {"distribution": "gamma", "shape": 2.0, "scale": 1.0, "lambda_val": 0.5}

    @pytest.fixture
    def exponential_params(self):
        """Exponential distribution parameters."""
        return {"distribution": "exponential", "mu": 1.0, "lambda_val": 0.5}

    def test_initialization(self, simple_graph, time_steps):
        """Should initialize without errors."""
        from spkmc.utils.gpu_utils import BatchedGPUSimulator

        N, edges = simple_graph
        simulator = BatchedGPUSimulator(N, edges, time_steps)
        assert simulator._N == N
        assert simulator._num_edges == len(edges)
        assert simulator._num_steps == len(time_steps)

    def test_run_samples_gamma(self, simple_graph, time_steps, gamma_params):
        """Should run samples with gamma distribution."""
        from spkmc.utils.gpu_utils import BatchedGPUSimulator

        N, edges = simple_graph
        sources = np.array([0, 1, 2])
        samples = 5

        simulator = BatchedGPUSimulator(N, edges, time_steps)
        S, I, R = simulator.run_samples(samples, sources, gamma_params)

        # Check shapes
        assert S.shape == (len(time_steps),)
        assert I.shape == (len(time_steps),)
        assert R.shape == (len(time_steps),)

        # Check SIR sums to 1
        total = S + I + R
        np.testing.assert_array_almost_equal(total, np.ones_like(total), decimal=5)

        # Check boundary conditions
        assert S[0] > 0.9  # Most should be susceptible at start
        assert R[-1] > 0.5  # Many should be recovered at end

    def test_run_samples_exponential(self, simple_graph, time_steps, exponential_params):
        """Should run samples with exponential distribution."""
        from spkmc.utils.gpu_utils import BatchedGPUSimulator

        N, edges = simple_graph
        sources = np.array([0])
        samples = 5

        simulator = BatchedGPUSimulator(N, edges, time_steps)
        S, I, R = simulator.run_samples(samples, sources, exponential_params)

        # Check shapes
        assert S.shape == (len(time_steps),)
        assert I.shape == (len(time_steps),)
        assert R.shape == (len(time_steps),)

        # Check SIR sums to 1
        total = S + I + R
        np.testing.assert_array_almost_equal(total, np.ones_like(total), decimal=5)

    def test_progress_callback(self, simple_graph, time_steps, gamma_params):
        """Should call progress callback for each sample."""
        from spkmc.utils.gpu_utils import BatchedGPUSimulator

        N, edges = simple_graph
        sources = np.array([0])
        samples = 10

        callback_count = [0]

        def progress_callback(advance):
            callback_count[0] += advance

        simulator = BatchedGPUSimulator(N, edges, time_steps, progress_callback=progress_callback)
        simulator.run_samples(samples, sources, gamma_params)

        # Callback should be called once per sample
        assert callback_count[0] == samples

    def test_single_source_produces_valid_results(self, simple_graph, time_steps, gamma_params):
        """Single source should produce valid epidemic dynamics."""
        from spkmc.utils.gpu_utils import BatchedGPUSimulator

        N, edges = simple_graph
        sources = np.array([0])  # Single source
        samples = 10

        simulator = BatchedGPUSimulator(N, edges, time_steps)
        S, I, R = simulator.run_samples(samples, sources, gamma_params)

        # Initial infection should be small (1/N)
        # Due to averaging and small initial condition, S should be high initially
        assert S[0] > 0.9

        # S should be monotonically non-increasing (people only leave S)
        for i in range(1, len(S)):
            assert S[i] <= S[i - 1] + 1e-5  # Small tolerance for numerical error

        # R should be monotonically non-decreasing (people only enter R)
        for i in range(1, len(R)):
            assert R[i] >= R[i - 1] - 1e-5


class TestGetDistGPU:
    """Tests for get_dist_gpu function."""

    def test_returns_correct_shapes(self):
        """Should return arrays with correct shapes."""
        from spkmc.utils.gpu_utils import get_dist_gpu

        N = 50
        # Simple linear graph
        edges = np.array([[i, i + 1] for i in range(N - 1)])
        sources = np.array([0])
        params = {"distribution": "exponential", "mu": 1.0, "lambda_val": 0.5}

        distances, recovery_times = get_dist_gpu(N, edges, sources, params)

        assert distances.shape == (N,)
        assert recovery_times.shape == (N,)

    def test_source_has_zero_distance(self):
        """Source nodes should have zero infection time."""
        from spkmc.utils.gpu_utils import get_dist_gpu

        N = 50
        edges = np.array([[i, i + 1] for i in range(N - 1)])
        sources = np.array([0])
        params = {"distribution": "exponential", "mu": 1.0, "lambda_val": 0.5}

        distances, _ = get_dist_gpu(N, edges, sources, params)

        # Source should have distance 0
        assert distances[0] == 0.0

    def test_unreachable_nodes_have_inf_distance(self):
        """Nodes unreachable from sources should have inf distance."""
        from spkmc.utils.gpu_utils import get_dist_gpu

        N = 50
        # Disconnected graph: two components
        edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],  # Component 1
                [10, 11],
                [11, 12],  # Component 2 (disconnected)
            ]
        )
        sources = np.array([0])  # Source in component 1
        params = {"distribution": "exponential", "mu": 1.0, "lambda_val": 0.5}

        distances, _ = get_dist_gpu(N, edges, sources, params)

        # Nodes in component 2 should be unreachable
        assert np.isinf(distances[10])
        assert np.isinf(distances[11])
        assert np.isinf(distances[12])


class TestCalculateGPU:
    """Tests for calculate_gpu function."""

    def test_returns_correct_shapes(self):
        """Should return SIR arrays with correct shapes."""
        from spkmc.utils.gpu_utils import calculate_gpu

        N = 100
        time_to_infect = np.random.exponential(5, size=N).astype(np.float32)
        recovery_times = np.random.exponential(1, size=N).astype(np.float32)
        time_steps = np.linspace(0, 30, 50).astype(np.float32)

        S, I, R = calculate_gpu(N, time_to_infect, recovery_times, time_steps)

        assert S.shape == (len(time_steps),)
        assert I.shape == (len(time_steps),)
        assert R.shape == (len(time_steps),)

    def test_sir_sums_to_one(self):
        """S + I + R should equal 1 at all time steps."""
        from spkmc.utils.gpu_utils import calculate_gpu

        N = 100
        time_to_infect = np.random.exponential(5, size=N).astype(np.float32)
        recovery_times = np.random.exponential(1, size=N).astype(np.float32)
        time_steps = np.linspace(0, 30, 50).astype(np.float32)

        S, I, R = calculate_gpu(N, time_to_infect, recovery_times, time_steps)

        total = S + I + R
        np.testing.assert_array_almost_equal(total, np.ones_like(total), decimal=5)

    def test_all_susceptible_at_t0(self):
        """At t=0, all nodes with positive infection time should be susceptible."""
        from spkmc.utils.gpu_utils import calculate_gpu

        N = 100
        # All nodes have infection time > 0
        time_to_infect = np.full(N, 10.0, dtype=np.float32)
        recovery_times = np.ones(N, dtype=np.float32)
        time_steps = np.array([0.0], dtype=np.float32)

        S, I, R = calculate_gpu(N, time_to_infect, recovery_times, time_steps)

        assert S[0] == 1.0
        assert I[0] == 0.0
        assert R[0] == 0.0

    def test_validates_input_shapes(self):
        """Should raise ValueError for mismatched input shapes."""
        from spkmc.utils.gpu_utils import calculate_gpu

        N = 100
        time_to_infect = np.zeros(50)  # Wrong size
        recovery_times = np.zeros(N)
        time_steps = np.linspace(0, 30, 50)

        with pytest.raises(ValueError, match="time_to_infect shape mismatch"):
            calculate_gpu(N, time_to_infect, recovery_times, time_steps)


class TestGPUvsCPUEquivalence:
    """Tests comparing GPU and CPU results for statistical equivalence."""

    @pytest.fixture
    def network_params(self):
        """Common network parameters for comparison tests."""
        return {"N": 500, "k_avg": 10, "samples": 20, "num_runs": 1}

    def test_erdos_renyi_equivalence(self, network_params):
        """GPU and CPU should produce statistically similar results for ER networks."""
        import os as os_module

        from spkmc.core.distributions import create_distribution
        from spkmc.core.simulation import SPKMC

        N = network_params["N"]
        time_steps = np.linspace(0, 30, 100)

        # Create distribution
        dist = create_distribution("exponential", mu=1.0, lambda_param=0.3)

        # Set same random seed for reproducibility
        np.random.seed(42)
        cp.random.seed(42)

        # Force GPU mode even for small graphs
        os_module.environ["SPKMC_FORCE_GPU"] = "1"
        os_module.environ["SPKMC_BATCH_GPU"] = "1"

        try:
            # Run with GPU
            sim_gpu = SPKMC(dist, use_gpu=True)
            S_gpu, I_gpu, R_gpu, _, _, _ = sim_gpu.simulate_erdos_renyi(
                num_runs=1,
                time_steps=time_steps,
                N=N,
                k_avg=network_params["k_avg"],
                samples=network_params["samples"],
                load_if_exists=False,
                show_progress=False,
            )

            # Reset seed
            np.random.seed(42)

            # Run with CPU
            os_module.environ["SPKMC_NO_GPU"] = "1"
            sim_cpu = SPKMC(dist, use_gpu=False)
            S_cpu, I_cpu, R_cpu, _, _, _ = sim_cpu.simulate_erdos_renyi(
                num_runs=1,
                time_steps=time_steps,
                N=N,
                k_avg=network_params["k_avg"],
                samples=network_params["samples"],
                load_if_exists=False,
                show_progress=False,
            )

            # Results should be similar (not identical due to different RNG)
            # Check that curves have similar shape and magnitude
            assert np.corrcoef(S_gpu, S_cpu)[0, 1] > 0.95
            assert np.corrcoef(I_gpu, I_cpu)[0, 1] > 0.90
            assert np.corrcoef(R_gpu, R_cpu)[0, 1] > 0.95

            # Check that final values are within reasonable range
            assert abs(R_gpu[-1] - R_cpu[-1]) < 0.2

        finally:
            # Clean up environment variables
            os_module.environ.pop("SPKMC_FORCE_GPU", None)
            os_module.environ.pop("SPKMC_BATCH_GPU", None)
            os_module.environ.pop("SPKMC_NO_GPU", None)


class TestEnvironmentVariables:
    """Tests for GPU-related environment variables."""

    def test_spkmc_no_gpu_disables_gpu(self):
        """SPKMC_NO_GPU=1 should force CPU mode."""
        import os as os_module

        from spkmc.core.distributions import create_distribution
        from spkmc.core.simulation import SPKMC

        os_module.environ["SPKMC_NO_GPU"] = "1"
        try:
            dist = create_distribution("exponential", mu=1.0, lambda_param=0.3)
            sim = SPKMC(dist, use_gpu=True)  # Request GPU
            assert sim.use_gpu is False  # But should be forced to CPU
        finally:
            os_module.environ.pop("SPKMC_NO_GPU", None)

    def test_spkmc_force_gpu_uses_gpu_for_small_graphs(self):
        """SPKMC_FORCE_GPU=1 should use GPU even for small graphs."""
        import os as os_module

        from spkmc.core.distributions import create_distribution
        from spkmc.core.simulation import SPKMC

        os_module.environ["SPKMC_FORCE_GPU"] = "1"
        try:
            dist = create_distribution("exponential", mu=1.0, lambda_param=0.3)
            sim = SPKMC(dist, use_gpu=True)
            # Small graph (below threshold)
            N = 100
            should_use = sim._should_use_batched_gpu(N)
            assert should_use is True
        finally:
            os_module.environ.pop("SPKMC_FORCE_GPU", None)

    def test_spkmc_batch_gpu_can_be_disabled(self):
        """SPKMC_BATCH_GPU=0 should disable batched GPU mode."""
        import os as os_module

        from spkmc.core.distributions import create_distribution
        from spkmc.core.simulation import SPKMC

        os_module.environ["SPKMC_BATCH_GPU"] = "0"
        os_module.environ["SPKMC_FORCE_GPU"] = "1"
        try:
            dist = create_distribution("exponential", mu=1.0, lambda_param=0.3)
            sim = SPKMC(dist, use_gpu=True)
            # Should not use batched GPU mode
            N = 100
            should_use = sim._should_use_batched_gpu(N)
            assert should_use is False
        finally:
            os_module.environ.pop("SPKMC_BATCH_GPU", None)
            os_module.environ.pop("SPKMC_FORCE_GPU", None)
