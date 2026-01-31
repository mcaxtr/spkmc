"""
Tests for the SPKMC simulation module.

This module contains tests for the SPKMC class and its functionality.
"""

from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from spkmc.core.distributions import ExponentialDistribution, GammaDistribution
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


@patch("os.path.exists")
def test_simulate_erdos_renyi(mock_exists, gamma_distribution, time_steps):
    """Test simulation on Erdos-Renyi networks."""
    # Configure mock to simulate missing file
    mock_exists.return_value = False

    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    N = 20
    k_avg = 4
    samples = 2
    num_runs = 2
    initial_perc = 0.1

    # Run the simulation with a reduced number of nodes/samples
    with patch("spkmc.io.results.ResultManager.save_result") as mock_save:
        S, I, R, S_err, I_err, R_err = simulator.simulate_erdos_renyi(
            num_runs=num_runs,
            time_steps=time_steps,
            N=N,
            k_avg=k_avg,
            samples=samples,
            initial_perc=initial_perc,
            load_if_exists=False,
        )

        # Verify save_result was called
        assert mock_save.called

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


@patch("os.path.exists")
def test_simulate_complex_network(mock_exists, gamma_distribution, time_steps):
    """Test simulation on complex networks."""
    # Configure mock to simulate missing file
    mock_exists.return_value = False

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
    with patch("spkmc.io.results.ResultManager.save_result") as mock_save:
        S, I, R, S_err, I_err, R_err = simulator.simulate_complex_network(
            num_runs=num_runs,
            exponent=exponent,
            time_steps=time_steps,
            N=N,
            k_avg=k_avg,
            samples=samples,
            initial_perc=initial_perc,
            load_if_exists=False,
        )

        # Verify save_result was called
        assert mock_save.called

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


@patch("os.path.exists")
def test_simulate_complete_graph(mock_exists, gamma_distribution, time_steps):
    """Test simulation on complete graphs."""
    # Configure mock to simulate missing file
    mock_exists.return_value = False

    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    N = 10  # Use a small value for the test
    samples = 2
    initial_perc = 0.1

    # Run the simulation with a reduced number of nodes/samples
    with patch("spkmc.io.results.ResultManager.save_result") as mock_save:
        S, I, R = simulator.simulate_complete_graph(
            time_steps=time_steps, N=N, samples=samples, initial_perc=initial_perc, overwrite=True
        )

        # Verify save_result was called
        assert mock_save.called

    # Verify results
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
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
        assert kwargs["load_if_exists"] is False

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


def test_run_simulation_cn(gamma_distribution, time_steps):
    """Test run_simulation for complex networks."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Configure the simulation
    with patch.object(simulator, "simulate_complex_network") as mock_simulate:
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
            network_type="cn",
            time_steps=time_steps,
            N=100,
            k_avg=5,
            samples=10,
            initial_perc=0.01,
            num_runs=2,
            exponent=2.5,
            overwrite=True,
        )

        # Verify simulate_complex_network was called with correct parameters
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["num_runs"] == 2
        assert kwargs["exponent"] == 2.5
        assert kwargs["N"] == 100
        assert kwargs["k_avg"] == 5
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01
        assert kwargs["load_if_exists"] is False

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
        # Configure the mock to return valid values
        mock_simulate.return_value = (
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
            overwrite=True,
        )

        # Verify simulate_complete_graph was called with correct parameters
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["N"] == 100
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01
        assert kwargs["overwrite"] is True

        # Verify the result
        assert "S_val" in result
        assert "I_val" in result
        assert "R_val" in result
        assert "time" in result
        assert "has_error" in result
        assert result["has_error"] is False


def test_run_simulation_invalid_network(gamma_distribution, time_steps):
    """Test run_simulation with an invalid network type."""
    # Create the simulator
    simulator = SPKMC(gamma_distribution)

    # Run the simulation with an invalid network type
    with pytest.raises(ValueError):
        simulator.run_simulation(
            network_type="invalid", time_steps=time_steps, N=100, samples=10, initial_perc=0.01
        )
