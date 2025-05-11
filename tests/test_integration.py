"""
Testes de integração para o SPKMC.

Este módulo contém testes de integração para verificar o fluxo completo do SPKMC,
desde a criação de distribuições e redes até a execução de simulações e exportação de resultados.
"""

import os
import tempfile
import pytest
import numpy as np
from click.testing import CliRunner

from spkmc.core.distributions import create_distribution
from spkmc.core.networks import NetworkFactory
from spkmc.core.simulation import SPKMC
from spkmc.io.results import ResultManager
from spkmc.io.export import ExportManager
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


def test_full_workflow_programmatic(temp_dir):
    """
    Testa o fluxo completo de forma programática.
    
    Este teste verifica a integração entre os diferentes componentes do SPKMC,
    desde a criação de distribuições e redes até a execução de simulações e exportação de resultados.
    """
    # 1. Criar distribuição
    distribution = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    
    # 2. Criar simulador
    simulator = SPKMC(distribution)
    
    # 3. Configurar parâmetros
    N = 20  # Usa um valor pequeno para o teste
    k_avg = 4
    samples = 2
    initial_perc = 0.1
    t_max = 5.0
    steps = 6
    time_steps = np.linspace(0, t_max, steps)
    
    # 4. Executar simulação
    result = simulator.run_simulation(
        network_type="er",
        time_steps=time_steps,
        N=N,
        k_avg=k_avg,
        samples=samples,
        initial_perc=initial_perc,
        num_runs=2,
        overwrite=True
    )
    
    # 5. Verificar resultados
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
    
    # 6. Exportar resultados em diferentes formatos
    output_base = os.path.join(temp_dir, "test_result")
    
    # 6.1. Exportar para JSON
    json_path = ExportManager.export_to_json(result, f"{output_base}.json")
    assert os.path.exists(json_path)
    
    # 6.2. Exportar para CSV
    csv_path = ExportManager.export_to_csv(result, f"{output_base}.csv")
    assert os.path.exists(csv_path)
    
    # 6.3. Exportar para Excel
    excel_path = ExportManager.export_to_excel(result, f"{output_base}.xlsx")
    assert os.path.exists(excel_path)
    
    # 6.4. Exportar para Markdown
    md_path = ExportManager.export_to_markdown(result, f"{output_base}.md", include_plot=True)
    assert os.path.exists(md_path)
    assert os.path.exists(f"{output_base}.png")  # Verifica se o gráfico foi gerado
    
    # 6.5. Exportar para HTML
    html_path = ExportManager.export_to_html(result, f"{output_base}.html", include_plot=True)
    assert os.path.exists(html_path)
    
    # 7. Carregar resultados
    loaded_result = ResultManager.load_result(json_path)
    assert loaded_result["metadata"]["network_type"] == "er"
    assert loaded_result["metadata"]["distribution"] == "gamma"
    assert loaded_result["metadata"]["N"] == N
    assert loaded_result["metadata"]["k_avg"] == k_avg
    assert loaded_result["metadata"]["samples"] == samples
    assert loaded_result["metadata"]["initial_perc"] == initial_perc


def test_cli_integration(runner, temp_dir, monkeypatch):
    """
    Testa a integração da CLI.
    
    Este teste verifica a integração entre os diferentes comandos da CLI do SPKMC.
    """
    # Mock para evitar a exibição de gráficos
    def mock_plot(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot)
    monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot)
    monkeypatch.setattr(Visualizer, 'compare_results', mock_plot)
    
    # 1. Executar simulação com rede Erdos-Renyi e distribuição Gamma
    output_path = os.path.join(temp_dir, "er_gamma.json")
    result = runner.invoke(cli, [
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
    assert "Simulação concluída com sucesso" in result.output
    assert os.path.exists(output_path)
    
    # 2. Executar simulação com rede complexa e distribuição Exponencial
    output_path2 = os.path.join(temp_dir, "cn_exp.json")
    result = runner.invoke(cli, [
        'run',
        '--network-type', 'cn',
        '--dist-type', 'exponential',
        '--mu', '1.0',
        '--lambda', '1.0',
        '--exponent', '2.5',
        '--nodes', '20',
        '--k-avg', '4',
        '--samples', '2',
        '--num-runs', '2',
        '--initial-perc', '0.1',
        '--t-max', '5.0',
        '--steps', '6',
        '--output', output_path2,
        '--no-plot'
    ])
    
    # Verifica se a simulação foi executada com sucesso
    assert result.exit_code == 0
    assert "Simulação concluída com sucesso" in result.output
    assert os.path.exists(output_path2)
    
    # 3. Visualizar resultados
    result = runner.invoke(cli, [
        'plot',
        output_path,
        '--with-error'
    ])
    
    # Verifica se a visualização foi executada com sucesso
    assert result.exit_code == 0
    
    # 4. Obter informações sobre os resultados
    result = runner.invoke(cli, [
        'info',
        '--result-file', output_path
    ])
    
    # Verifica se as informações foram exibidas com sucesso
    assert result.exit_code == 0
    assert "Parâmetros da Simulação" in result.output
    assert "network_type: er" in result.output.lower()
    
    # 5. Comparar resultados
    result = runner.invoke(cli, [
        'compare',
        output_path, output_path2,
        '--labels', 'ER-Gamma', 'CN-Exp'
    ])
    
    # Verifica se a comparação foi executada com sucesso
    assert result.exit_code == 0


def test_network_distribution_integration():
    """
    Testa a integração entre redes e distribuições.
    
    Este teste verifica a integração entre os módulos de redes e distribuições.
    """
    # 1. Criar distribuições
    gamma_dist = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    exp_dist = create_distribution("exponential", mu=1.0, lambda_=1.0)
    
    # 2. Criar redes
    er_network = NetworkFactory.create_erdos_renyi(N=20, k_avg=4)
    cn_network = NetworkFactory.create_complex_network(N=20, exponent=2.5, k_avg=4)
    cg_network = NetworkFactory.create_complete_graph(N=10)
    
    # 3. Verificar propriedades das redes
    assert er_network.number_of_nodes() == 20
    assert er_network.number_of_edges() > 0
    
    assert cn_network.number_of_nodes() == 20
    assert cn_network.number_of_edges() > 0
    
    assert cg_network.number_of_nodes() == 10
    assert cg_network.number_of_edges() == 10 * 9  # Grafo direcionado completo
    
    # 4. Criar simuladores
    gamma_simulator = SPKMC(gamma_dist)
    exp_simulator = SPKMC(exp_dist)
    
    # 5. Configurar simulação
    time_steps = np.linspace(0, 5.0, 6)
    sources = np.array([0])  # Nó 0 inicialmente infectado
    
    # 6. Executar simulações com diferentes combinações
    # 6.1. Gamma + ER
    S1, I1, R1 = gamma_simulator.run_multiple_simulations(er_network, sources, time_steps, samples=2, show_progress=False)
    
    # 6.2. Exponential + ER
    S2, I2, R2 = exp_simulator.run_multiple_simulations(er_network, sources, time_steps, samples=2, show_progress=False)
    
    # 6.3. Gamma + CN
    S3, I3, R3 = gamma_simulator.run_multiple_simulations(cn_network, sources, time_steps, samples=2, show_progress=False)
    
    # 6.4. Exponential + CN
    S4, I4, R4 = exp_simulator.run_multiple_simulations(cn_network, sources, time_steps, samples=2, show_progress=False)
    
    # 7. Verificar resultados
    for S, I, R in [(S1, I1, R1), (S2, I2, R2), (S3, I3, R3), (S4, I4, R4)]:
        assert isinstance(S, np.ndarray)
        assert isinstance(I, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert S.shape == time_steps.shape
        assert I.shape == time_steps.shape
        assert R.shape == time_steps.shape
        assert np.isclose(S + I + R, 1.0).all()
        assert S[0] < 1.0  # Deve haver pelo menos um nó infectado inicialmente
        assert I[0] > 0.0  # Deve haver pelo menos um nó infectado inicialmente
        assert R[0] == 0.0  # Não deve haver nós recuperados inicialmente