"""
Integration tests for SPKMC.

This module contains integration tests to verify the full SPKMC flow,
from creating distributions and networks to running simulations and exporting results.
"""

import os
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

from spkmc.cli.commands import cli
from spkmc.core.distributions import create_distribution
from spkmc.core.networks import NetworkFactory
from spkmc.core.simulation import SPKMC
from spkmc.io.export import ExportManager
from spkmc.io.results import ResultManager


@pytest.fixture
def runner():
    """Fixture for Click's CliRunner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    # Clean up the directory after the test
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

    # Remove the directory
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_full_workflow_programmatic(temp_dir):
    """
    Test the full workflow programmatically.

    This test verifies integration between different SPKMC components.
    """
    # 1. Create distribution
    distribution = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)

    # 2. Create simulator
    simulator = SPKMC(distribution)

    # 3. Configure parameters
    N = 20  # Use a small value for the test
    k_avg = 4
    samples = 2
    initial_perc = 0.1
    t_max = 5.0
    steps = 6
    time_steps = np.linspace(0, t_max, steps)

    # 4. Run simulation
    result = simulator.run_simulation(
        network_type="er",
        time_steps=time_steps,
        N=N,
        k_avg=k_avg,
        samples=samples,
        initial_perc=initial_perc,
        num_runs=2,
        overwrite=True,
    )

    # 5. Verify results
    assert "S_val" in result
    assert "I_val" in result
    assert "R_val" in result
    assert "time" in result
    assert "has_error" in result
    assert result["has_error"] is True
    assert len(result["S_val"]) == steps
    assert len(result["I_val"]) == steps
    assert len(result["R_val"]) == steps
    assert len(result["time"]) == steps
    assert np.isclose(result["S_val"] + result["I_val"] + result["R_val"], 1.0).all()

    # 6. Export results in different formats
    output_base = os.path.join(temp_dir, "test_result")

    # 6.1. Export to JSON
    json_path = ExportManager.export_to_json(result, f"{output_base}.json")
    assert os.path.exists(json_path)

    # 6.2. Export to CSV
    csv_path = ExportManager.export_to_csv(result, f"{output_base}.csv")
    assert os.path.exists(csv_path)

    # 6.3. Export to Excel
    excel_path = ExportManager.export_to_excel(result, f"{output_base}.xlsx")
    assert os.path.exists(excel_path)

    # 6.4. Export to Markdown
    md_path = ExportManager.export_to_markdown(result, f"{output_base}.md", include_plot=True)
    assert os.path.exists(md_path)
    assert os.path.exists(f"{output_base}.png")  # Verify the plot was generated

    # 6.5. Export to HTML
    html_path = ExportManager.export_to_html(result, f"{output_base}.html", include_plot=True)
    assert os.path.exists(html_path)

    # 7. Load results
    loaded_result = ResultManager.load_result(json_path)
    # Verify the core result data structure is preserved after save/load
    assert "S_val" in loaded_result
    assert "I_val" in loaded_result
    assert "R_val" in loaded_result
    assert "time" in loaded_result
    assert "has_error" in loaded_result
    assert len(loaded_result["S_val"]) == steps
    assert len(loaded_result["I_val"]) == steps
    assert len(loaded_result["R_val"]) == steps


def test_cli_integration(runner, temp_dir, monkeypatch):
    """
    Test CLI integration.

    This test verifies integration between different SPKMC CLI commands.
    """

    # Mock to avoid displaying plots
    def mock_plot(*args, **kwargs):
        pass

    # Apply the mock
    from spkmc.visualization.plots import Visualizer

    monkeypatch.setattr(Visualizer, "plot_result", mock_plot)
    monkeypatch.setattr(Visualizer, "plot_result_with_error", mock_plot)
    monkeypatch.setattr(Visualizer, "compare_results", mock_plot)

    # 1. Run simulation with an Erdos-Renyi network and Gamma distribution
    output_path = os.path.join(temp_dir, "er_gamma.json")
    result = runner.invoke(
        cli,
        [
            "run",
            "--network-type",
            "er",
            "--dist-type",
            "gamma",
            "--shape",
            "2.0",
            "--scale",
            "1.0",
            "--nodes",
            "20",
            "--k-avg",
            "4",
            "--samples",
            "2",
            "--num-runs",
            "2",
            "--initial-perc",
            "0.1",
            "--t-max",
            "5.0",
            "--steps",
            "6",
            "--output",
            output_path,
            "--no-plot",
        ],
    )

    # Verify the simulation ran successfully
    assert result.exit_code == 0
    assert "Simulation completed successfully" in result.output
    assert os.path.exists(output_path)

    # 2. Run simulation with a complex network and Exponential distribution
    output_path2 = os.path.join(temp_dir, "cn_exp.json")
    result = runner.invoke(
        cli,
        [
            "run",
            "--network-type",
            "cn",
            "--dist-type",
            "exponential",
            "--mu",
            "1.0",
            "--lambda",
            "1.0",
            "--exponent",
            "2.5",
            "--nodes",
            "20",
            "--k-avg",
            "4",
            "--samples",
            "2",
            "--num-runs",
            "2",
            "--initial-perc",
            "0.1",
            "--t-max",
            "5.0",
            "--steps",
            "6",
            "--output",
            output_path2,
            "--no-plot",
        ],
    )

    # Verify the simulation ran successfully
    assert result.exit_code == 0
    assert "Simulation completed successfully" in result.output
    assert os.path.exists(output_path2)

    # 3. Run simulation with a random regular network and Gamma distribution
    output_path3 = os.path.join(temp_dir, "rrn_gamma.json")
    result = runner.invoke(
        cli,
        [
            "run",
            "--network-type",
            "rrn",
            "--dist-type",
            "gamma",
            "--shape",
            "2.0",
            "--scale",
            "1.0",
            "--nodes",
            "20",
            "--k-avg",
            "4",
            "--samples",
            "2",
            "--num-runs",
            "2",
            "--initial-perc",
            "0.1",
            "--t-max",
            "5.0",
            "--steps",
            "6",
            "--output",
            output_path3,
            "--no-plot",
        ],
    )

    # Verify the simulation ran successfully
    assert result.exit_code == 0
    assert "Simulation completed successfully" in result.output
    assert os.path.exists(output_path3)

    # 4. Visualize results
    result = runner.invoke(cli, ["plot", output_path, "--with-error"])

    # Verify the visualization ran successfully
    assert result.exit_code == 0

    # 5. Get information about results
    result = runner.invoke(cli, ["info", "--result-file", output_path])

    # Verify info was displayed successfully
    assert result.exit_code == 0
    assert "Simulation Parameters" in result.output
    assert "network_type: er" in result.output.lower()

    # 6. Compare results
    result = runner.invoke(
        cli,
        [
            "compare",
            output_path,
            output_path2,
            output_path3,
            "--labels",
            "ER-Gamma",
            "CN-Exp",
            "RRN-Gamma",
        ],
    )

    # Verify the comparison ran successfully
    assert result.exit_code == 0


def test_network_distribution_integration():
    """
    Test integration between networks and distributions.

    This test verifies integration between the networks and distributions modules.
    """
    # 1. Create distributions
    gamma_dist = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    exp_dist = create_distribution("exponential", mu=1.0, lambda_=1.0)

    # 2. Create networks
    er_network = NetworkFactory.create_erdos_renyi(N=20, k_avg=4)
    cn_network = NetworkFactory.create_complex_network(N=20, exponent=2.5, k_avg=4)
    cg_network = NetworkFactory.create_complete_graph(N=10)
    rrn_network = NetworkFactory.create_random_regular_network(N=20, k_avg=4)

    # 3. Verify network properties
    assert er_network.number_of_nodes() == 20
    assert er_network.number_of_edges() > 0

    assert cn_network.number_of_nodes() == 20
    assert cn_network.number_of_edges() > 0

    assert cg_network.number_of_nodes() == 10
    assert cg_network.number_of_edges() == 10 * 9  # Directed complete graph

    assert rrn_network.number_of_nodes() == 20
    assert rrn_network.number_of_edges() > 0
    # For DiGraph from undirected RRN, each node has out_degree == k_avg
    out_degrees = dict(rrn_network.out_degree()).values()
    assert all(d == 4 for d in out_degrees)

    # 4. Create simulators
    gamma_simulator = SPKMC(gamma_dist)
    exp_simulator = SPKMC(exp_dist)

    # 5. Configure simulation
    time_steps = np.linspace(0, 5.0, 6)
    sources = np.array([0])  # Node 0 initially infected

    # 6. Run simulations with different combinations
    # 6.1. Gamma + ER
    S1, I1, R1 = gamma_simulator.run_multiple_simulations(
        er_network, sources, time_steps, samples=2, show_progress=False
    )

    # 6.2. Exponential + ER
    S2, I2, R2 = exp_simulator.run_multiple_simulations(
        er_network, sources, time_steps, samples=2, show_progress=False
    )

    # 6.3. Gamma + CN
    S3, I3, R3 = gamma_simulator.run_multiple_simulations(
        cn_network, sources, time_steps, samples=2, show_progress=False
    )

    # 6.4. Exponential + CN
    S4, I4, R4 = exp_simulator.run_multiple_simulations(
        cn_network, sources, time_steps, samples=2, show_progress=False
    )

    # 6.5. Gamma + RRN
    S5, I5, R5 = gamma_simulator.run_multiple_simulations(
        rrn_network, sources, time_steps, samples=2, show_progress=False
    )

    # 6.6. Exponential + RRN
    S6, I6, R6 = exp_simulator.run_multiple_simulations(
        rrn_network, sources, time_steps, samples=2, show_progress=False
    )

    # 7. Verify results
    for S, I, R in [
        (S1, I1, R1),
        (S2, I2, R2),
        (S3, I3, R3),
        (S4, I4, R4),
        (S5, I5, R5),
        (S6, I6, R6),
    ]:
        assert isinstance(S, np.ndarray)
        assert isinstance(I, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert S.shape == time_steps.shape
        assert I.shape == time_steps.shape
        assert R.shape == time_steps.shape
        assert np.isclose(S + I + R, 1.0).all()
        assert S[0] < 1.0  # There should be at least one initially infected node
        assert I[0] > 0.0  # There should be at least one initially infected node
        assert R[0] == 0.0  # There should be no initially recovered nodes
