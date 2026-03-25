"""
Tests for the SPKMC simulation module.

This module contains tests for the SPKMC class and its functionality.
"""

from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from spkmc.core.distributions import ExponentialDistribution, GammaDistribution, WeibullDistribution
from spkmc.core.simulation import SPKMC


@pytest.fixture
def gamma_distribution():
    """Fixture for a Gamma distribution."""
    return GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)


@pytest.fixture
def exponential_distribution():
    """Fixture for an Exponential distribution."""
    return ExponentialDistribution(mu=1.0, lmbd=1.0)


@pytest.fixture
def weibull_distribution():
    """Fixture for a Weibull distribution."""
    return WeibullDistribution(shape=2.0, scale=1.0, lmbd=1.0)


@pytest.fixture
def small_network():
    """Fixture for a small network."""
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    G.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]
    )
    return G


@pytest.fixture
def time_steps():
    """Fixture for time steps."""
    return np.linspace(0, 10, 11)


def test_spkmc_initialization(gamma_distribution):
    """Test SPKMC simulator initialization."""
    simulator = SPKMC(gamma_distribution)
    assert simulator.distribution == gamma_distribution


def test_get_dist_sparse(gamma_distribution):
    """Test shortest distance calculation."""
    simulator = SPKMC(gamma_distribution)

    # Create a simple network
    N = 5
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    sources = np.array([0])

    # Calculate distances
    dist, recovery_weights = simulator.get_dist_sparse(N, edges, sources)

    # Verify results
    assert isinstance(dist, np.ndarray)
    assert isinstance(recovery_weights, np.ndarray)
    assert dist.shape == (N,)
    assert recovery_weights.shape == (N,)
    assert dist[0] == 0  # Distance from node 0 to itself is 0
    assert np.all(dist[1:] > 0)  # Distances to other nodes are positive


def test_run_single_simulation(gamma_distribution, small_network, time_steps):
    """Test running a single simulation."""
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    N = small_network.number_of_nodes()
    edges = np.array(list(small_network.edges()))
    sources = np.array([0])  # Node 0 initially infected

    # Run the simulation
    S, I, R = simulator.run_single_simulation(N, edges, sources, time_steps)

    # Verify results
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # Sum must be 1
    assert S[0] < 1.0  # There should be at least one initially infected node
    assert I[0] > 0.0  # There should be at least one initially infected node
    assert R[0] == 0.0  # There should be no initially recovered nodes


def test_run_multiple_simulations(gamma_distribution, small_network, time_steps):
    """Test running multiple simulations."""
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    sources = np.array([0])  # Node 0 initially infected
    samples = 3

    # Run simulations
    S, I, R = simulator.run_multiple_simulations(
        small_network, sources, time_steps, samples, show_progress=False
    )

    # Verify results
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # Sum must be 1
    assert S[0] < 1.0  # There should be at least one initially infected node
    assert I[0] > 0.0  # There should be at least one initially infected node
    assert R[0] == 0.0  # There should be no initially recovered nodes


def test_simulate_erdos_renyi(gamma_distribution, time_steps):
    """Test simulation on Erdos-Renyi networks."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    N = 20
    k_avg = 4
    samples = 2
    num_runs = 2
    initial_perc = 0.1

    # Run the simulation with a reduced number of nodes/samples
    S, I, R, S_err, I_err, R_err = simulator.simulate_erdos_renyi(
        num_runs=num_runs,
        time_steps=time_steps,
        N=N,
        k_avg=k_avg,
        samples=samples,
        initial_perc=initial_perc,
    )

    # Verify results
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(S_err, np.ndarray)
    assert isinstance(I_err, np.ndarray)
    assert isinstance(R_err, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert S_err.shape == time_steps.shape
    assert I_err.shape == time_steps.shape
    assert R_err.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # Sum must be 1


def test_simulate_scale_free_network(gamma_distribution, time_steps):
    """Test simulation on scale-free networks."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    N = 20
    k_avg = 4
    samples = 2
    num_runs = 2
    initial_perc = 0.1
    exponent = 2.5

    # Run the simulation with a reduced number of nodes/samples
    S, I, R, S_err, I_err, R_err = simulator.simulate_scale_free_network(
        num_runs=num_runs,
        exponent=exponent,
        time_steps=time_steps,
        N=N,
        k_avg=k_avg,
        samples=samples,
        initial_perc=initial_perc,
    )

    # Verify results
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(S_err, np.ndarray)
    assert isinstance(I_err, np.ndarray)
    assert isinstance(R_err, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert S_err.shape == time_steps.shape
    assert I_err.shape == time_steps.shape
    assert R_err.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # Sum must be 1


def test_simulate_complete_graph(gamma_distribution, time_steps):
    """Test simulation on complete graphs."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    N = 10  # Use a small value for the test
    samples = 2
    initial_perc = 0.1
    num_runs = 2

    # Run the simulation with a reduced number of nodes/samples
    S, I, R, S_err, I_err, R_err = simulator.simulate_complete_graph(
        num_runs=num_runs,
        time_steps=time_steps,
        N=N,
        samples=samples,
        initial_perc=initial_perc,
    )

    # Verify results
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(S_err, np.ndarray)
    assert isinstance(I_err, np.ndarray)
    assert isinstance(R_err, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert S_err.shape == time_steps.shape
    assert I_err.shape == time_steps.shape
    assert R_err.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # Sum must be 1


def test_run_simulation_er(gamma_distribution, time_steps):
    """Test run_simulation for Erdos-Renyi networks."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    with patch.object(simulator, "simulate_erdos_renyi") as mock_simulate:
        # Configure the mock to return valid values
        mock_simulate.return_value = (
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
        )

        # Run the simulation
        result = simulator.run_simulation(
            network_type="er",
            time_steps=time_steps,
            N=100,
            k_avg=5,
            samples=10,
            initial_perc=0.01,
            num_runs=2,
            overwrite=True,
        )

        # Verify simulate_erdos_renyi was called with correct parameters
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["num_runs"] == 2
        assert kwargs["N"] == 100
        assert kwargs["k_avg"] == 5
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01

        # Verify the result
        assert "S_val" in result
        assert "I_val" in result
        assert "R_val" in result
        assert "S_err" in result
        assert "I_err" in result
        assert "R_err" in result
        assert "time" in result
        assert "has_error" in result
        assert result["has_error"] is True


def test_run_simulation_sf(gamma_distribution, time_steps):
    """Test run_simulation for scale-free networks."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    with patch.object(simulator, "simulate_scale_free_network") as mock_simulate:
        # Configure the mock to return valid values
        mock_simulate.return_value = (
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
        )

        # Run the simulation
        result = simulator.run_simulation(
            network_type="sf",
            time_steps=time_steps,
            N=100,
            k_avg=5,
            samples=10,
            initial_perc=0.01,
            num_runs=2,
            exponent=2.5,
            overwrite=True,
        )

        # Verify simulate_scale_free_network was called with correct parameters
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["num_runs"] == 2
        assert kwargs["exponent"] == 2.5
        assert kwargs["N"] == 100
        assert kwargs["k_avg"] == 5
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01

        # Verify the result
        assert "S_val" in result
        assert "I_val" in result
        assert "R_val" in result
        assert "S_err" in result
        assert "I_err" in result
        assert "R_err" in result
        assert "time" in result
        assert "has_error" in result
        assert result["has_error"] is True


def test_run_simulation_cg(gamma_distribution, time_steps):
    """Test run_simulation for complete graphs."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    with patch.object(simulator, "simulate_complete_graph") as mock_simulate:
        # Configure the mock to return valid values (now includes error arrays)
        mock_simulate.return_value = (
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
        )

        # Run the simulation
        result = simulator.run_simulation(
            network_type="cg",
            time_steps=time_steps,
            N=100,
            samples=10,
            initial_perc=0.01,
            num_runs=2,
            overwrite=True,
        )

        # Verify simulate_complete_graph was called with correct parameters
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["N"] == 100
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01
        assert kwargs["num_runs"] == 2

        # Verify the result
        assert "S_val" in result
        assert "I_val" in result
        assert "R_val" in result
        assert "S_err" in result
        assert "I_err" in result
        assert "R_err" in result
        assert "time" in result
        assert "has_error" in result
        assert result["has_error"] is True


def test_run_simulation_invalid_network(gamma_distribution, time_steps):
    """Test run_simulation with an invalid network type."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Run the simulation with an invalid network type
    with pytest.raises(ValueError):
        simulator.run_simulation(
            network_type="invalid", time_steps=time_steps, N=100, samples=10, initial_perc=0.01
        )


def test_spkmc_initialization_weibull(weibull_distribution):
    """Test SPKMC simulator initialization with Weibull distribution."""
    simulator = SPKMC(weibull_distribution)
    assert simulator.distribution == weibull_distribution


def test_get_dist_sparse_weibull(weibull_distribution):
    """Test shortest distance calculation with Weibull distribution."""
    simulator = SPKMC(weibull_distribution)

    N = 5
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    sources = np.array([0])

    dist, recovery_weights = simulator.get_dist_sparse(N, edges, sources)

    assert isinstance(dist, np.ndarray)
    assert isinstance(recovery_weights, np.ndarray)
    assert dist.shape == (N,)
    assert recovery_weights.shape == (N,)
    assert dist[0] == 0
    assert np.all(dist[1:] > 0)


def test_run_single_simulation_weibull(weibull_distribution, small_network, time_steps):
    """Test running a single simulation with Weibull distribution."""
    simulator = SPKMC(weibull_distribution)

    N = small_network.number_of_nodes()
    edges = np.array(list(small_network.edges()))
    sources = np.array([0])

    S, I, R = simulator.run_single_simulation(N, edges, sources, time_steps)

    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()
    assert S[0] < 1.0
    assert I[0] > 0.0
    assert R[0] == 0.0


def test_simulate_erdos_renyi_weibull(weibull_distribution, time_steps):
    """Test simulation on Erdos-Renyi networks with Weibull distribution."""
    simulator = SPKMC(weibull_distribution)

    N = 20
    k_avg = 4
    samples = 2
    num_runs = 2
    initial_perc = 0.1

    S, I, R, S_err, I_err, R_err = simulator.simulate_erdos_renyi(
        num_runs=num_runs,
        time_steps=time_steps,
        N=N,
        k_avg=k_avg,
        samples=samples,
        initial_perc=initial_perc,
    )

    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()


def test_batched_gpu_execution_chunks_samples(monkeypatch, gamma_distribution, time_steps):
    """Batched GPU mode should split samples into cleanup-friendly chunks."""
    import spkmc.utils.gpu_utils as gpu_utils

    simulator = SPKMC(gamma_distribution, use_gpu=True)
    simulator._gpu_available = True
    simulator._use_batched_gpu = True

    cleanup_calls = []

    class FakeBatchedGPUSimulator:
        def __init__(self, N, edges, time_steps, progress_callback=None):
            self.progress_callback = progress_callback

        def run_samples(self, samples, sources, params):
            for _ in range(samples):
                if self.progress_callback is not None:
                    self.progress_callback(1)
            return (
                np.full(len(time_steps), samples, dtype=np.float64),
                np.full(len(time_steps), samples * 2, dtype=np.float64),
                np.full(len(time_steps), samples * 3, dtype=np.float64),
            )

        def close(self):
            cleanup_calls.append("close")

    monkeypatch.setattr(gpu_utils, "BatchedGPUSimulator", FakeBatchedGPUSimulator)
    monkeypatch.setattr(gpu_utils, "get_gpu_sample_batch_size", lambda samples: 2)
    monkeypatch.setattr(gpu_utils, "is_gpu_oom_error", lambda exc: False)
    monkeypatch.setattr(
        gpu_utils,
        "cleanup_gpu_memory",
        lambda reset_rmm=False, allocator_mode=None: cleanup_calls.append(
            (reset_rmm, allocator_mode)
        ),
    )

    progress_updates = []
    S, I, R = simulator._run_multiple_simulations_batched_gpu(
        N=10,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
        sources=np.array([0], dtype=np.int32),
        time_steps=time_steps,
        samples=5,
        progress_callback=lambda completed, total: progress_updates.append((completed, total)),
    )

    expected_weighted_mean = (2 * 2 + 2 * 2 + 1 * 1) / 5
    np.testing.assert_allclose(S, np.full(len(time_steps), expected_weighted_mean))
    np.testing.assert_allclose(I, np.full(len(time_steps), expected_weighted_mean * 2))
    np.testing.assert_allclose(R, np.full(len(time_steps), expected_weighted_mean * 3))
    assert progress_updates == [(2, 5), (2, 5), (1, 5)]
    assert cleanup_calls.count("close") == 3
    assert cleanup_calls.count((True, None)) == 2


def test_batched_gpu_execution_retries_oom_with_direct_rmm(
    monkeypatch, gamma_distribution, time_steps
):
    """Batched GPU mode should retry fragmented OOMs with direct RMM allocation."""
    import spkmc.utils.gpu_utils as gpu_utils

    simulator = SPKMC(gamma_distribution, use_gpu=True)
    simulator._gpu_available = True
    simulator._use_batched_gpu = True

    cleanup_calls = []
    attempts = {"count": 0}

    class FakeBatchedGPUSimulator:
        def __init__(self, N, edges, time_steps, progress_callback=None):
            self.progress_callback = progress_callback

        def run_samples(self, samples, sources, params):
            attempts["count"] += 1
            if attempts["count"] == 1:
                if self.progress_callback is not None:
                    self.progress_callback(1)
                raise MemoryError("std::bad_alloc: out_of_memory: RMM failed to allocate")

            for _ in range(samples):
                if self.progress_callback is not None:
                    self.progress_callback(1)
            return (
                np.ones(len(time_steps), dtype=np.float64),
                np.ones(len(time_steps), dtype=np.float64) * 2,
                np.ones(len(time_steps), dtype=np.float64) * 3,
            )

        def close(self):
            cleanup_calls.append("close")

    monkeypatch.setattr(gpu_utils, "BatchedGPUSimulator", FakeBatchedGPUSimulator)
    monkeypatch.setattr(gpu_utils, "get_gpu_sample_batch_size", lambda samples: samples)
    monkeypatch.setattr(gpu_utils, "is_gpu_oom_error", lambda exc: True)
    monkeypatch.setattr(
        gpu_utils,
        "cleanup_gpu_memory",
        lambda reset_rmm=False, allocator_mode=None: cleanup_calls.append(
            (reset_rmm, allocator_mode)
        ),
    )

    progress_updates = []
    S, I, R = simulator._run_multiple_simulations_batched_gpu(
        N=10,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
        sources=np.array([0], dtype=np.int32),
        time_steps=time_steps,
        samples=3,
        progress_callback=lambda completed, total: progress_updates.append((completed, total)),
    )

    np.testing.assert_allclose(S, np.ones(len(time_steps)))
    np.testing.assert_allclose(I, np.ones(len(time_steps)) * 2)
    np.testing.assert_allclose(R, np.ones(len(time_steps)) * 3)
    assert attempts["count"] == 2
    assert progress_updates == [(3, 3)]
    assert cleanup_calls.count("close") == 2
    assert (True, "direct") in cleanup_calls


# --- Microscopic data tests ---


def test_run_single_simulation_microscopic(gamma_distribution, small_network, time_steps):
    """Test run_single_simulation returns microscopic data when requested."""
    simulator = SPKMC(gamma_distribution)
    edges = np.array(small_network.edges())
    N = small_network.number_of_nodes()
    sources = np.array([0])

    # Without microscopic - returns 3-tuple
    result = simulator.run_single_simulation(N, edges, sources, time_steps)
    assert len(result) == 3

    # With microscopic - returns 5-tuple
    result = simulator.run_single_simulation(N, edges, sources, time_steps, return_microscopic=True)
    assert len(result) == 5
    S, I, R, time_to_infect, recovery_times = result
    assert S.shape == time_steps.shape
    assert time_to_infect.shape == (N,)
    assert recovery_times.shape == (N,)


def test_multiple_simulations_standard_microscopic(gamma_distribution, time_steps):
    """Test _run_multiple_simulations_standard captures first sample's microscopic data."""
    simulator = SPKMC(gamma_distribution)
    N = 20
    G = nx.erdos_renyi_graph(N, 0.3, directed=True)
    edges = np.array(G.edges())
    sources = np.array([0])

    # Without microscopic - returns 3-tuple
    result = simulator._run_multiple_simulations_standard(
        N,
        edges,
        sources,
        time_steps,
        samples=3,
        show_progress=False,
    )
    assert len(result) == 3

    # With microscopic - returns 4-tuple
    result = simulator._run_multiple_simulations_standard(
        N,
        edges,
        sources,
        time_steps,
        samples=3,
        show_progress=False,
        microscopic=True,
    )
    assert len(result) == 4
    S_mean, I_mean, R_mean, micro_data = result
    assert S_mean.shape == time_steps.shape
    assert "time_to_infect" in micro_data
    assert "recovery_times" in micro_data
    assert micro_data["time_to_infect"].shape == (N,)
    assert micro_data["recovery_times"].shape == (N,)


def test_simulate_erdos_renyi_microscopic(gamma_distribution, time_steps):
    """Test simulate_erdos_renyi returns microscopic data per run."""
    simulator = SPKMC(gamma_distribution)

    # Without microscopic - returns 6-tuple
    result = simulator.simulate_erdos_renyi(
        num_runs=2,
        time_steps=time_steps,
        N=50,
        k_avg=5,
        samples=3,
        initial_perc=0.1,
        show_progress=False,
    )
    assert len(result) == 6

    # With microscopic - returns 7-tuple
    result = simulator.simulate_erdos_renyi(
        num_runs=2,
        time_steps=time_steps,
        N=50,
        k_avg=5,
        samples=3,
        initial_perc=0.1,
        show_progress=False,
        microscopic=True,
    )
    assert len(result) == 7
    S, I, R, S_err, I_err, R_err, micro_runs = result
    assert len(micro_runs) == 2  # one per run
    for run_data in micro_runs:
        assert "time_to_infect" in run_data
        assert "recovery_times" in run_data
        assert "generation" in run_data
        assert "sources" in run_data
        assert run_data["time_to_infect"].shape == (50,)
        assert run_data["recovery_times"].shape == (50,)
        assert run_data["generation"].shape == (50,)


def test_run_simulation_microscopic(gamma_distribution, time_steps):
    """Test run_simulation includes microscopic data in result dict."""
    simulator = SPKMC(gamma_distribution)

    # Without microscopic
    result = simulator.run_simulation(
        "er",
        time_steps,
        N=50,
        k_avg=5,
        samples=3,
        num_runs=1,
        initial_perc=0.1,
        show_progress=False,
    )
    assert "microscopic" not in result

    # With microscopic
    result = simulator.run_simulation(
        "er",
        time_steps,
        N=50,
        k_avg=5,
        samples=3,
        num_runs=1,
        initial_perc=0.1,
        show_progress=False,
        microscopic=True,
    )
    assert "microscopic" in result
    assert len(result["microscopic"]) == 1  # 1 run


def test_bfs_generations():
    """Test _compute_bfs_generations computes correct hop distances."""
    from spkmc.core.simulation import _compute_bfs_generations

    # Simple chain: 0->1->2->3->4
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    sources = np.array([0])
    gen = _compute_bfs_generations(5, edges, sources)
    np.testing.assert_array_equal(gen, [0, 1, 2, 3, 4])

    # Two sources
    gen = _compute_bfs_generations(5, edges, np.array([0, 2]))
    np.testing.assert_array_equal(gen, [0, 1, 0, 1, 2])
