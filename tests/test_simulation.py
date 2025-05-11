"""
Testes para o módulo de simulação do SPKMC.

Este módulo contém testes para a classe SPKMC e suas funcionalidades.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock

from spkmc.core.simulation import SPKMC
from spkmc.core.distributions import GammaDistribution, ExponentialDistribution
from spkmc.core.networks import NetworkFactory


@pytest.fixture
def gamma_distribution():
    """Fixture para uma distribuição Gamma."""
    return GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)


@pytest.fixture
def exponential_distribution():
    """Fixture para uma distribuição Exponencial."""
    return ExponentialDistribution(mu=1.0, lmbd=1.0)


@pytest.fixture
def small_network():
    """Fixture para uma rede pequena."""
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
    return G


@pytest.fixture
def time_steps():
    """Fixture para passos de tempo."""
    return np.linspace(0, 10, 11)


def test_spkmc_initialization(gamma_distribution):
    """Testa a inicialização do simulador SPKMC."""
    simulator = SPKMC(gamma_distribution)
    assert simulator.distribution == gamma_distribution


def test_get_dist_sparse(gamma_distribution):
    """Testa o cálculo de distâncias mínimas."""
    simulator = SPKMC(gamma_distribution)
    
    # Cria uma rede simples
    N = 5
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    sources = np.array([0])
    
    # Calcula as distâncias
    dist, recovery_weights = simulator.get_dist_sparse(N, edges, sources)
    
    # Verifica os resultados
    assert isinstance(dist, np.ndarray)
    assert isinstance(recovery_weights, np.ndarray)
    assert dist.shape == (N,)
    assert recovery_weights.shape == (N,)
    assert dist[0] == 0  # A distância do nó 0 para ele mesmo é 0
    assert np.all(dist[1:] > 0)  # As distâncias para os outros nós são positivas


def test_run_single_simulation(gamma_distribution, small_network, time_steps):
    """Testa a execução de uma única simulação."""
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    N = small_network.number_of_nodes()
    edges = np.array(list(small_network.edges()))
    sources = np.array([0])  # Nó 0 inicialmente infectado
    
    # Executa a simulação
    S, I, R = simulator.run_single_simulation(N, edges, sources, time_steps)
    
    # Verifica os resultados
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # A soma deve ser 1
    assert S[0] < 1.0  # Deve haver pelo menos um nó infectado inicialmente
    assert I[0] > 0.0  # Deve haver pelo menos um nó infectado inicialmente
    assert R[0] == 0.0  # Não deve haver nós recuperados inicialmente


def test_run_multiple_simulations(gamma_distribution, small_network, time_steps):
    """Testa a execução de múltiplas simulações."""
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    sources = np.array([0])  # Nó 0 inicialmente infectado
    samples = 3
    
    # Executa as simulações
    S, I, R = simulator.run_multiple_simulations(small_network, sources, time_steps, samples, show_progress=False)
    
    # Verifica os resultados
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # A soma deve ser 1
    assert S[0] < 1.0  # Deve haver pelo menos um nó infectado inicialmente
    assert I[0] > 0.0  # Deve haver pelo menos um nó infectado inicialmente
    assert R[0] == 0.0  # Não deve haver nós recuperados inicialmente


@patch('os.path.exists')
def test_simulate_erdos_renyi(mock_exists, gamma_distribution, time_steps):
    """Testa a simulação em redes Erdos-Renyi."""
    # Configura o mock para simular que o arquivo não existe
    mock_exists.return_value = False
    
    # Cria o simulador
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    N = 20
    k_avg = 4
    samples = 2
    num_runs = 2
    initial_perc = 0.1
    
    # Executa a simulação com um número reduzido de nós e amostras para o teste
    with patch('spkmc.io.results.ResultManager.save_result') as mock_save:
        S, I, R, S_err, I_err, R_err = simulator.simulate_erdos_renyi(
            num_runs=num_runs,
            time_steps=time_steps,
            N=N,
            k_avg=k_avg,
            samples=samples,
            initial_perc=initial_perc,
            load_if_exists=False
        )
        
        # Verifica se o método save_result foi chamado
        assert mock_save.called
    
    # Verifica os resultados
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
    assert np.isclose(S + I + R, 1.0).all()  # A soma deve ser 1


@patch('os.path.exists')
def test_simulate_complex_network(mock_exists, gamma_distribution, time_steps):
    """Testa a simulação em redes complexas."""
    # Configura o mock para simular que o arquivo não existe
    mock_exists.return_value = False
    
    # Cria o simulador
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    N = 20
    k_avg = 4
    samples = 2
    num_runs = 2
    initial_perc = 0.1
    exponent = 2.5
    
    # Executa a simulação com um número reduzido de nós e amostras para o teste
    with patch('spkmc.io.results.ResultManager.save_result') as mock_save:
        S, I, R, S_err, I_err, R_err = simulator.simulate_complex_network(
            num_runs=num_runs,
            exponent=exponent,
            time_steps=time_steps,
            N=N,
            k_avg=k_avg,
            samples=samples,
            initial_perc=initial_perc,
            load_if_exists=False
        )
        
        # Verifica se o método save_result foi chamado
        assert mock_save.called
    
    # Verifica os resultados
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
    assert np.isclose(S + I + R, 1.0).all()  # A soma deve ser 1


@patch('os.path.exists')
def test_simulate_complete_graph(mock_exists, gamma_distribution, time_steps):
    """Testa a simulação em grafos completos."""
    # Configura o mock para simular que o arquivo não existe
    mock_exists.return_value = False
    
    # Cria o simulador
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    N = 10  # Usa um valor pequeno para o teste
    samples = 2
    initial_perc = 0.1
    
    # Executa a simulação com um número reduzido de nós e amostras para o teste
    with patch('spkmc.io.results.ResultManager.save_result') as mock_save:
        S, I, R = simulator.simulate_complete_graph(
            time_steps=time_steps,
            N=N,
            samples=samples,
            initial_perc=initial_perc,
            overwrite=True
        )
        
        # Verifica se o método save_result foi chamado
        assert mock_save.called
    
    # Verifica os resultados
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == time_steps.shape
    assert I.shape == time_steps.shape
    assert R.shape == time_steps.shape
    assert np.isclose(S + I + R, 1.0).all()  # A soma deve ser 1


def test_run_simulation_er(gamma_distribution, time_steps):
    """Testa o método run_simulation para redes Erdos-Renyi."""
    # Cria o simulador
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    with patch.object(simulator, 'simulate_erdos_renyi') as mock_simulate:
        # Configura o mock para retornar valores válidos
        mock_simulate.return_value = (
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps)
        )
        
        # Executa a simulação
        result = simulator.run_simulation(
            network_type="er",
            time_steps=time_steps,
            N=100,
            k_avg=5,
            samples=10,
            initial_perc=0.01,
            num_runs=2,
            overwrite=True
        )
        
        # Verifica se o método simulate_erdos_renyi foi chamado com os parâmetros corretos
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["num_runs"] == 2
        assert kwargs["N"] == 100
        assert kwargs["k_avg"] == 5
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01
        assert kwargs["load_if_exists"] is False
        
        # Verifica o resultado
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
    """Testa o método run_simulation para redes complexas."""
    # Cria o simulador
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    with patch.object(simulator, 'simulate_complex_network') as mock_simulate:
        # Configura o mock para retornar valores válidos
        mock_simulate.return_value = (
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps)
        )
        
        # Executa a simulação
        result = simulator.run_simulation(
            network_type="cn",
            time_steps=time_steps,
            N=100,
            k_avg=5,
            samples=10,
            initial_perc=0.01,
            num_runs=2,
            exponent=2.5,
            overwrite=True
        )
        
        # Verifica se o método simulate_complex_network foi chamado com os parâmetros corretos
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["num_runs"] == 2
        assert kwargs["exponent"] == 2.5
        assert kwargs["N"] == 100
        assert kwargs["k_avg"] == 5
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01
        assert kwargs["load_if_exists"] is False
        
        # Verifica o resultado
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
    """Testa o método run_simulation para grafos completos."""
    # Cria o simulador
    simulator = SPKMC(gamma_distribution)
    
    # Configura a simulação
    with patch.object(simulator, 'simulate_complete_graph') as mock_simulate:
        # Configura o mock para retornar valores válidos
        mock_simulate.return_value = (
            np.zeros_like(time_steps),
            np.zeros_like(time_steps),
            np.zeros_like(time_steps)
        )
        
        # Executa a simulação
        result = simulator.run_simulation(
            network_type="cg",
            time_steps=time_steps,
            N=100,
            samples=10,
            initial_perc=0.01,
            overwrite=True
        )
        
        # Verifica se o método simulate_complete_graph foi chamado com os parâmetros corretos
        mock_simulate.assert_called_once()
        args, kwargs = mock_simulate.call_args
        assert kwargs["N"] == 100
        assert kwargs["samples"] == 10
        assert kwargs["initial_perc"] == 0.01
        assert kwargs["overwrite"] is True
        
        # Verifica o resultado
        assert "S_val" in result
        assert "I_val" in result
        assert "R_val" in result
        assert "time" in result
        assert "has_error" in result
        assert result["has_error"] is False


def test_run_simulation_invalid_network(gamma_distribution, time_steps):
    """Testa o método run_simulation com um tipo de rede inválido."""
    # Cria o simulador
    simulator = SPKMC(gamma_distribution)
    
    # Executa a simulação com um tipo de rede inválido
    with pytest.raises(ValueError):
        simulator.run_simulation(
            network_type="invalid",
            time_steps=time_steps,
            N=100,
            samples=10,
            initial_perc=0.01
        )