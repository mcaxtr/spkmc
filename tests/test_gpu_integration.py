"""
Testes de integração para a funcionalidade GPU no SPKMC.

Este módulo contém testes de integração para verificar se a simulação funciona
corretamente com a opção GPU, incluindo a integração com a CLI.
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from spkmc.core.distributions import create_distribution
from spkmc.core.networks import NetworkFactory
from spkmc.core.simulation import SPKMC
from spkmc.utils.gpu_utils import is_gpu_available
from spkmc.cli.commands import cli


@pytest.fixture
def runner():
    """Fixture para o CliRunner do Click."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Fixture para criar um diretório temporário."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Limpa o diretório após o teste
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    
    # Remove o diretório
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


@pytest.fixture
def small_network():
    """Fixture para uma rede pequena."""
    G = NetworkFactory.create_erdos_renyi(N=10, k_avg=3)
    return G


@pytest.fixture
def time_steps():
    """Fixture para passos de tempo."""
    return np.linspace(0, 5, 6)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_spkmc_with_gpu():
    """
    Testa a classe SPKMC com a opção GPU.
    
    Este teste verifica se a classe SPKMC funciona corretamente com a opção GPU.
    O teste é pulado se a GPU não estiver disponível.
    """
    # Cria a distribuição
    distribution = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    
    # Cria o simulador com GPU
    simulator = SPKMC(distribution, use_gpu=True)
    
    # Verifica se a opção GPU está ativada
    assert simulator.use_gpu is True
    
    # Cria uma rede simples
    N = 5
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    sources = np.array([0])
    time_steps = np.linspace(0, 5, 6)
    
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


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_spkmc_gpu_vs_cpu(small_network, time_steps):
    """
    Testa a consistência entre as implementações GPU e CPU.
    
    Este teste verifica se os resultados das implementações GPU e CPU são consistentes.
    O teste é pulado se a GPU não estiver disponível.
    """
    # Cria a distribuição
    distribution = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    
    # Cria os simuladores
    simulator_cpu = SPKMC(distribution, use_gpu=False)
    simulator_gpu = SPKMC(distribution, use_gpu=True)
    
    # Configura a simulação
    N = small_network.number_of_nodes()
    edges = np.array(list(small_network.edges()))
    sources = np.array([0])  # Nó 0 inicialmente infectado
    
    # Define uma semente aleatória para garantir resultados comparáveis
    np.random.seed(42)
    
    # Executa a simulação com CPU
    S_cpu, I_cpu, R_cpu = simulator_cpu.run_single_simulation(N, edges, sources, time_steps)
    
    # Redefine a semente aleatória
    np.random.seed(42)
    
    # Executa a simulação com GPU
    S_gpu, I_gpu, R_gpu = simulator_gpu.run_single_simulation(N, edges, sources, time_steps)
    
    # Verifica se os resultados são aproximadamente iguais
    # Nota: Devido a diferenças de precisão entre CPU e GPU, usamos uma tolerância
    assert np.allclose(S_cpu, S_gpu, rtol=1e-2, atol=1e-2)
    assert np.allclose(I_cpu, I_gpu, rtol=1e-2, atol=1e-2)
    assert np.allclose(R_cpu, R_gpu, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_cli_with_gpu_flag(runner, temp_dir, monkeypatch):
    """
    Testa a CLI com a flag --gpu.
    
    Este teste verifica se a CLI funciona corretamente com a flag --gpu.
    O teste é pulado se a GPU não estiver disponível.
    """
    # Mock para evitar a exibição de gráficos
    def mock_plot(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot)
    monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot)
    
    # Executa a simulação com a flag --gpu
    output_path = os.path.join(temp_dir, "gpu_test.json")
    result = runner.invoke(cli, [
        '--gpu',
        'run',
        '--network-type', 'er',
        '--dist-type', 'gamma',
        '--shape', '2.0',
        '--scale', '1.0',
        '--nodes', '20',
        '--k-avg', '4',
        '--samples', '2',
        '--num-runs', '2',
        '--initial-perc', '0.1',
        '--t-max', '5.0',
        '--steps', '6',
        '--output', output_path,
        '--no-plot'
    ])
    
    # Verifica se a simulação foi executada com sucesso
    assert result.exit_code == 0
    assert "Aceleração GPU ativada" in result.output
    assert "Simulação concluída com sucesso" in result.output
    assert os.path.exists(output_path)


def test_gpu_fallback_when_not_available():
    """
    Testa o fallback para CPU quando a GPU não está disponível.
    
    Este teste verifica se o SPKMC recorre automaticamente à implementação CPU
    quando a GPU é solicitada mas não está disponível.
    """
    # Cria a distribuição
    distribution = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    
    # Simula que a GPU não está disponível
    with patch('spkmc.utils.gpu_utils.is_gpu_available', return_value=False):
        # Cria o simulador com GPU (que deve recorrer à CPU)
        simulator = SPKMC(distribution, use_gpu=True)
        
        # Verifica se a opção GPU está desativada
        assert simulator.use_gpu is False


def test_cli_gpu_fallback_global(runner, temp_dir, monkeypatch):
    """
    Testa o fallback da CLI para CPU quando a GPU não está disponível (opção global).
    
    Este teste verifica se a CLI recorre automaticamente à implementação CPU
    quando a GPU é solicitada como opção global mas não está disponível.
    """
    # Mock para evitar a exibição de gráficos
    def mock_plot(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot)
    monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot)
    
    # Simula que a GPU não está disponível
    with patch('spkmc.utils.gpu_utils.is_gpu_available', return_value=False):
        # Executa a simulação com a flag --gpu
        output_path = os.path.join(temp_dir, "gpu_fallback_test.json")
        result = runner.invoke(cli, [
            '--gpu',
            'run',
            '--network-type', 'er',
            '--dist-type', 'gamma',
            '--shape', '2.0',
            '--scale', '1.0',
            '--nodes', '20',
            '--k-avg', '4',
            '--samples', '2',
            '--num-runs', '2',
            '--initial-perc', '0.1',
            '--t-max', '5.0',
            '--steps', '6',
            '--output', output_path,
            '--no-plot'
        ])
        
        # Verifica se a simulação foi executada com sucesso
        assert result.exit_code == 0
        assert "Aceleração GPU solicitada, mas não disponível" in result.output
        assert "Simulação concluída com sucesso" in result.output
        assert os.path.exists(output_path)


def test_cli_gpu_fallback_command_option(runner, temp_dir, monkeypatch):
    """
    Testa o fallback da CLI para CPU quando a GPU não está disponível (opção de comando).
    
    Este teste verifica se a CLI recorre automaticamente à implementação CPU
    quando a GPU é solicitada como opção específica do comando mas não está disponível.
    """
    # Mock para evitar a exibição de gráficos
    def mock_plot(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot)
    monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot)
    
    # Simula que a GPU não está disponível
    with patch('spkmc.utils.gpu_utils.is_gpu_available', return_value=False):
        # Executa a simulação com a flag --gpu como opção específica
        output_path = os.path.join(temp_dir, "gpu_fallback_command_option_test.json")
        result = runner.invoke(cli, [
            'run',
            '--gpu',  # Agora a flag está após o comando
            '--network-type', 'er',
            '--dist-type', 'gamma',
            '--shape', '2.0',
            '--scale', '1.0',
            '--nodes', '20',
            '--k-avg', '4',
            '--samples', '2',
            '--num-runs', '2',
            '--initial-perc', '0.1',
            '--t-max', '5.0',
            '--steps', '6',
            '--output', output_path,
            '--no-plot'
        ])
        
        # Verifica se a simulação foi executada com sucesso
        assert result.exit_code == 0
        assert "Aceleração GPU solicitada, mas não disponível" in result.output
        assert "Simulação concluída com sucesso" in result.output
        assert os.path.exists(output_path)


def test_cli_batch_gpu_fallback_command_option(runner, temp_dir, monkeypatch):
    """
    Testa o fallback do comando batch para CPU quando a GPU não está disponível (opção de comando).
    
    Este teste verifica se o comando batch recorre automaticamente à implementação CPU
    quando a GPU é solicitada como opção específica mas não está disponível.
    """
    # Mock para evitar a exibição de gráficos
    def mock_plot(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot)
    monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot)
    monkeypatch.setattr(Visualizer, 'compare_results', mock_plot)
    
    # Cria um arquivo de cenários temporário
    scenarios_file = os.path.join(temp_dir, "test_fallback_scenarios.json")
    scenarios = [
        {
            "network_type": "er",
            "dist_type": "gamma",
            "nodes": 20,
            "k_avg": 4,
            "shape": 2.0,
            "scale": 1.0,
            "lambda_val": 1.0,
            "samples": 2,
            "num_runs": 2,
            "initial_perc": 0.1,
            "t_max": 5.0,
            "steps": 6
        }
    ]
    
    with open(scenarios_file, 'w') as f:
        import json
        json.dump(scenarios, f)
    
    # Cria diretório de saída
    output_dir = os.path.join(temp_dir, "batch_fallback_results")
    
    # Simula que a GPU não está disponível
    with patch('spkmc.utils.gpu_utils.is_gpu_available', return_value=False):
        # Executa o comando batch com a flag --gpu como opção específica
        result = runner.invoke(cli, [
            'batch',
            '--gpu',  # Agora a flag está após o comando
            scenarios_file,
            '--output-dir', output_dir,
            '--prefix', 'fallback_',
            '--no-plot'
        ])
        
        # Verifica se a simulação foi executada com sucesso
        assert result.exit_code == 0
        assert "Aceleração GPU solicitada, mas não disponível" in result.output
        assert "Execução em lote concluída" in result.output
        
        # Verifica se o arquivo de resultado foi criado
        result_file = os.path.join(output_dir, "fallback_scenario_001.json")
        assert os.path.exists(result_file)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_cli_with_gpu_flag_as_command_option(runner, temp_dir, monkeypatch):
    """
    Testa a CLI com a flag --gpu como opção específica do comando.
    
    Este teste verifica se a CLI funciona corretamente quando a flag --gpu
    é usada como uma opção específica após o comando 'run'.
    O teste é pulado se a GPU não estiver disponível.
    """
    # Mock para evitar a exibição de gráficos
    def mock_plot(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot)
    monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot)
    
    # Executa a simulação com a flag --gpu como opção específica do comando
    output_path = os.path.join(temp_dir, "gpu_test_command_option.json")
    result = runner.invoke(cli, [
        'run',
        '--gpu',  # Agora a flag está após o comando
        '--network-type', 'er',
        '--dist-type', 'gamma',
        '--shape', '2.0',
        '--scale', '1.0',
        '--nodes', '20',
        '--k-avg', '4',
        '--samples', '2',
        '--num-runs', '2',
        '--initial-perc', '0.1',
        '--t-max', '5.0',
        '--steps', '6',
        '--output', output_path,
        '--no-plot'
    ])
    
    # Verifica se a simulação foi executada com sucesso
    assert result.exit_code == 0
    assert "Aceleração GPU ativada para esta simulação" in result.output
    assert "Simulação concluída com sucesso" in result.output
    assert os.path.exists(output_path)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_cli_batch_with_gpu_flag_as_command_option(runner, temp_dir, monkeypatch):
    """
    Testa a CLI com a flag --gpu como opção específica do comando batch.
    
    Este teste verifica se a CLI funciona corretamente quando a flag --gpu
    é usada como uma opção específica após o comando 'batch'.
    O teste é pulado se a GPU não estiver disponível.
    """
    # Mock para evitar a exibição de gráficos
    def mock_plot(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot)
    monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot)
    monkeypatch.setattr(Visualizer, 'compare_results', mock_plot)
    
    # Cria um arquivo de cenários temporário
    scenarios_file = os.path.join(temp_dir, "test_scenarios.json")
    scenarios = [
        {
            "network_type": "er",
            "dist_type": "gamma",
            "nodes": 20,
            "k_avg": 4,
            "shape": 2.0,
            "scale": 1.0,
            "lambda_val": 1.0,
            "samples": 2,
            "num_runs": 2,
            "initial_perc": 0.1,
            "t_max": 5.0,
            "steps": 6
        }
    ]
    
    with open(scenarios_file, 'w') as f:
        import json
        json.dump(scenarios, f)
    
    # Cria diretório de saída
    output_dir = os.path.join(temp_dir, "batch_results")
    
    # Executa o comando batch com a flag --gpu como opção específica
    result = runner.invoke(cli, [
        'batch',
        '--gpu',  # Agora a flag está após o comando
        scenarios_file,
        '--output-dir', output_dir,
        '--prefix', 'test_',
        '--no-plot'
    ])
    
    # Verifica se a simulação foi executada com sucesso
    assert result.exit_code == 0
    assert "Aceleração GPU ativada para este lote de simulações" in result.output
    assert "Execução em lote concluída" in result.output
    
    # Verifica se o arquivo de resultado foi criado
    result_file = os.path.join(output_dir, "test_scenario_001.json")
    assert os.path.exists(result_file)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_run_multiple_simulations_with_gpu(small_network, time_steps):
    """
    Testa o método run_multiple_simulations com GPU.
    
    Este teste verifica se o método run_multiple_simulations funciona corretamente
    com a opção GPU. O teste é pulado se a GPU não estiver disponível.
    """
    # Cria a distribuição
    distribution = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    
    # Cria o simulador com GPU
    simulator = SPKMC(distribution, use_gpu=True)
    
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


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_run_simulation_with_gpu():
    """
    Testa o método run_simulation com GPU.
    
    Este teste verifica se o método run_simulation funciona corretamente
    com a opção GPU. O teste é pulado se a GPU não estiver disponível.
    """
    # Cria a distribuição
    distribution = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    
    # Cria o simulador com GPU
    simulator = SPKMC(distribution, use_gpu=True)
    
    # Configura a simulação
    time_steps = np.linspace(0, 5, 6)
    
    # Executa a simulação
    result = simulator.run_simulation(
        network_type="er",
        time_steps=time_steps,
        N=20,
        k_avg=4,
        samples=2,
        initial_perc=0.1,
        num_runs=2,
        overwrite=True
    )
    
    # Verifica o resultado
    assert "S_val" in result
    assert "I_val" in result
    assert "R_val" in result
    assert "time" in result
    assert "has_error" in result
    assert result["has_error"] is True