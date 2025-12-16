"""
Testes para as melhorias no comando plot do SPKMC.

Este módulo contém testes específicos para as novas funcionalidades do comando plot,
incluindo suporte a diretórios, filtros de estados e múltiplos cenários.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from click.testing import CliRunner
import numpy as np

from spkmc.cli.commands import cli
from spkmc.io.results import ResultManager
from spkmc.visualization.plots import Visualizer


@pytest.fixture
def runner():
    """Fixture para o CliRunner do Click."""
    return CliRunner()


@pytest.fixture
def sample_result_data():
    """Dados de exemplo para resultados de simulação."""
    return {
        "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
        "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
        "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
        "time": [0.0, 2.5, 5.0, 7.5, 10.0],
        "metadata": {
            "network_type": "er",
            "distribution": "gamma",
            "N": 100,
            "k_avg": 5,
            "samples": 10,
            "initial_perc": 0.01
        }
    }


@pytest.fixture
def temp_results_dir(sample_result_data):
    """Cria um diretório temporário com múltiplos arquivos de resultados."""
    temp_dir = tempfile.mkdtemp()
    
    # Cria 3 arquivos de resultados com pequenas variações
    for i in range(3):
        result = sample_result_data.copy()
        # Modifica ligeiramente os dados para cada cenário
        result["I_val"] = [v * (1 + i * 0.1) for v in result["I_val"]]
        result["metadata"]["scenario"] = f"scenario_{i+1}"
        
        file_path = os.path.join(temp_dir, f"scenario_{i+1:03d}.json")
        with open(file_path, 'w') as f:
            json.dump(result, f)
    
    yield temp_dir
    
    # Limpa o diretório
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_single_result(sample_result_data):
    """Cria um arquivo temporário com resultado único."""
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    
    with open(path, 'w') as f:
        json.dump(sample_result_data, f)
    
    yield path
    
    if os.path.exists(path):
        os.remove(path)


def test_plot_single_file(runner, temp_single_result, monkeypatch):
    """Testa o plot de um arquivo único (comportamento original)."""
    # Mock para evitar exibição de gráficos
    def mock_plot_result(*args, **kwargs):
        pass
    
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot_result)
    
    result = runner.invoke(cli, ['plot', temp_single_result])
    
    assert result.exit_code == 0
    assert "Estatísticas da Simulação" in result.output
    assert "Máximo de infectados" in result.output


def test_plot_directory(runner, temp_results_dir, monkeypatch):
    """Testa o plot de um diretório com múltiplos arquivos."""
    # Mock para evitar exibição de gráficos
    def mock_compare_results(*args, **kwargs):
        pass
    
    monkeypatch.setattr(Visualizer, 'compare_results', mock_compare_results)
    
    result = runner.invoke(cli, ['plot', temp_results_dir])
    
    assert result.exit_code == 0
    assert "Encontrados 3 arquivos JSON no diretório" in result.output
    assert "Gerando visualização comparativa de 3 cenários" in result.output


def test_plot_with_states_filter(runner, temp_single_result, monkeypatch):
    """Testa o plot com filtro de estados específicos."""
    # Mock para capturar os argumentos passados
    plot_calls = []
    
    def mock_plot_result(S, I, R, time, title, save_path, states_to_plot):
        plot_calls.append({
            'states_to_plot': states_to_plot
        })
    
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot_result)
    
    # Testa com apenas infectados
    result = runner.invoke(cli, ['plot', temp_single_result, '--states', 'infected'])
    assert result.exit_code == 0
    assert plot_calls[-1]['states_to_plot'] == {'I'}
    
    # Testa com infectados e recuperados
    result = runner.invoke(cli, ['plot', temp_single_result, '--states', 'infected', '--states', 'recovered'])
    assert result.exit_code == 0
    assert plot_calls[-1]['states_to_plot'] == {'I', 'R'}
    
    # Testa com abreviações
    result = runner.invoke(cli, ['plot', temp_single_result, '--states', 's', '--states', 'i'])
    assert result.exit_code == 0
    assert plot_calls[-1]['states_to_plot'] == {'S', 'I'}


def test_plot_directory_separate(runner, temp_results_dir, monkeypatch):
    """Testa o plot de diretório com gráficos separados."""
    plot_calls = []
    
    def mock_plot_result(*args, **kwargs):
        plot_calls.append(kwargs.get('save_path') or args[-2] if len(args) > 2 else None)
    
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot_result)
    
    result = runner.invoke(cli, ['plot', temp_results_dir, '--separate', '--output', 'test.png'])
    
    assert result.exit_code == 0
    assert "Processando 3 arquivos de resultados" in result.output
    assert len(plot_calls) == 3
    # Verifica se os nomes dos arquivos de saída são únicos
    assert all('scenario_' in str(path) for path in plot_calls if path)


def test_plot_invalid_state(runner, temp_single_result, monkeypatch):
    """Testa o plot com estado inválido."""
    def mock_plot_result(*args, **kwargs):
        pass
    
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot_result)
    
    result = runner.invoke(cli, ['plot', temp_single_result, '--states', 'invalid_state'])
    
    assert result.exit_code == 0
    assert "Estado inválido ignorado: invalid_state" in result.output


def test_plot_empty_directory(runner):
    """Testa o plot de um diretório vazio."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(cli, ['plot', temp_dir])
        
        assert result.exit_code != 0
        assert "Nenhum arquivo JSON encontrado no diretório" in result.output


def test_plot_nonexistent_path(runner):
    """Testa o plot com caminho inexistente."""
    result = runner.invoke(cli, ['plot', '/path/that/does/not/exist'])
    
    assert result.exit_code != 0
    assert "Caminho não encontrado" in result.output


def test_plot_with_error_bars(runner, monkeypatch):
    """Testa o plot com barras de erro."""
    # Cria resultado com dados de erro
    result_with_error = {
        "S_val": [0.99, 0.95, 0.90, 0.85, 0.80],
        "I_val": [0.01, 0.04, 0.05, 0.05, 0.04],
        "R_val": [0.00, 0.01, 0.05, 0.10, 0.16],
        "S_err": [0.001, 0.002, 0.003, 0.004, 0.005],
        "I_err": [0.001, 0.002, 0.003, 0.002, 0.001],
        "R_err": [0.000, 0.001, 0.002, 0.003, 0.004],
        "time": [0.0, 2.5, 5.0, 7.5, 10.0],
        "metadata": {
            "network_type": "er",
            "distribution": "gamma",
            "N": 100
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(result_with_error, f)
        temp_file = f.name
    
    try:
        plot_calls = []
        
        def mock_plot_with_error(*args, **kwargs):
            plot_calls.append('with_error')
        
        monkeypatch.setattr(Visualizer, 'plot_result_with_error', mock_plot_with_error)
        
        result = runner.invoke(cli, ['plot', temp_file, '--with-error'])
        
        assert result.exit_code == 0
        assert len(plot_calls) == 1
        assert plot_calls[0] == 'with_error'
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_load_results_from_directory():
    """Testa a função auxiliar load_results_from_directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Cria alguns arquivos JSON
        for i in range(3):
            data = {
                "S_val": [0.99],
                "I_val": [0.01],
                "R_val": [0.00],
                "time": [0.0],
                "metadata": {"id": i}
            }
            
            file_path = os.path.join(temp_dir, f"result_{i}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f)
        
        # Cria um arquivo não-JSON que deve ser ignorado
        with open(os.path.join(temp_dir, "readme.txt"), 'w') as f:
            f.write("This is not a JSON file")
        
        # Testa a função
        results = ResultManager.load_results_from_directory(temp_dir)
        
        assert len(results) == 3
        assert all(isinstance(r[0], Path) for r in results)
        assert all(isinstance(r[1], dict) for r in results)
        assert all("metadata" in r[1] for r in results)


def test_plot_directory_with_export(runner, temp_results_dir, monkeypatch):
    """Testa o plot de diretório com exportação adicional."""
    def mock_compare_results(*args, **kwargs):
        pass
    
    def mock_export(*args, **kwargs):
        return "exported_file.csv"
    
    monkeypatch.setattr(Visualizer, 'compare_results', mock_compare_results)
    
    # Não podemos mockar diretamente ExportManager porque ele é importado no topo
    # Então vamos apenas testar se o comando é executado sem erros
    result = runner.invoke(cli, ['plot', temp_results_dir, '--export', 'csv'])
    
    # O comando deve executar sem erros, mas a exportação real pode falhar
    # porque estamos trabalhando com múltiplos arquivos
    assert result.exit_code == 0


def test_visualizer_states_filter():
    """Testa diretamente as funções do Visualizer com filtro de estados."""
    import matplotlib.pyplot as plt
    
    # Dados de teste
    S = np.array([0.99, 0.95, 0.90, 0.85, 0.80])
    I = np.array([0.01, 0.04, 0.05, 0.05, 0.04])
    R = np.array([0.00, 0.01, 0.05, 0.10, 0.16])
    time = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    
    # Testa plot_result com diferentes filtros
    # Como não podemos verificar visualmente, apenas garantimos que não há erros
    
    # Todos os estados
    Visualizer.plot_result(S, I, R, time, "Test", save_path="test_all.png", states_to_plot={'S', 'I', 'R'})
    plt.close()
    
    # Apenas infectados
    Visualizer.plot_result(S, I, R, time, "Test", save_path="test_i.png", states_to_plot={'I'})
    plt.close()
    
    # Infectados e recuperados
    Visualizer.plot_result(S, I, R, time, "Test", save_path="test_ir.png", states_to_plot={'I', 'R'})
    plt.close()
    
    # Remove arquivos de teste se foram criados
    for file in ['test_all.png', 'test_i.png', 'test_ir.png']:
        if os.path.exists(file):
            os.remove(file)


def test_plot_help(runner):
    """Testa a mensagem de ajuda do comando plot."""
    result = runner.invoke(cli, ['plot', '--help'])
    
    assert result.exit_code == 0
    assert "--states" in result.output
    assert "--separate" in result.output
    # Verifica partes da mensagem para evitar problemas de codificação
    assert "Estados" in result.output and "plotar" in result.output
    assert "separados" in result.output