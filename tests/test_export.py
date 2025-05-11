"""
Testes para o módulo de exportação do SPKMC.

Este módulo contém testes para a classe ExportManager e suas funcionalidades.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from spkmc.io.export import ExportManager


@pytest.fixture
def sample_result():
    """Fixture para um resultado de exemplo."""
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
def sample_result_with_error():
    """Fixture para um resultado de exemplo com dados de erro."""
    return {
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
            "N": 100,
            "k_avg": 5,
            "samples": 10,
            "initial_perc": 0.01
        }
    }


def test_export_to_csv(sample_result):
    """Testa a exportação para CSV."""
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name
    
    try:
        # Exporta os resultados
        exported_path = ExportManager.export_to_csv(sample_result, path)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(exported_path)
        assert exported_path == path
        
        # Carrega o CSV e verifica
        df = pd.read_csv(path)
        
        assert "Time" in df.columns
        assert "Susceptible" in df.columns
        assert "Infected" in df.columns
        assert "Recovered" in df.columns
        assert len(df) == len(sample_result["time"])
        assert df["Time"].tolist() == sample_result["time"]
        assert df["Susceptible"].tolist() == sample_result["S_val"]
        assert df["Infected"].tolist() == sample_result["I_val"]
        assert df["Recovered"].tolist() == sample_result["R_val"]
    finally:
        # Remove o arquivo
        if os.path.exists(path):
            os.remove(path)


def test_export_to_csv_with_error(sample_result_with_error):
    """Testa a exportação para CSV com dados de erro."""
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name
    
    try:
        # Exporta os resultados
        exported_path = ExportManager.export_to_csv(sample_result_with_error, path)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(exported_path)
        
        # Carrega o CSV e verifica
        df = pd.read_csv(path)
        
        assert "Time" in df.columns
        assert "Susceptible" in df.columns
        assert "Infected" in df.columns
        assert "Recovered" in df.columns
        assert "Susceptible_Error" in df.columns
        assert "Infected_Error" in df.columns
        assert "Recovered_Error" in df.columns
        assert len(df) == len(sample_result_with_error["time"])
        assert df["Susceptible_Error"].tolist() == sample_result_with_error["S_err"]
        assert df["Infected_Error"].tolist() == sample_result_with_error["I_err"]
        assert df["Recovered_Error"].tolist() == sample_result_with_error["R_err"]
    finally:
        # Remove o arquivo
        if os.path.exists(path):
            os.remove(path)


def test_export_to_excel(sample_result):
    """Testa a exportação para Excel."""
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        path = f.name
    
    try:
        # Exporta os resultados
        exported_path = ExportManager.export_to_excel(sample_result, path)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(exported_path)
        
        # Carrega o Excel e verifica
        with pd.ExcelFile(path) as xls:
            # Verifica as planilhas
            assert "Data" in xls.sheet_names
            assert "Metadata" in xls.sheet_names
            assert "Statistics" in xls.sheet_names
            
            # Verifica a planilha de dados
            df_data = pd.read_excel(xls, "Data")
            assert "Time" in df_data.columns
            assert "Susceptible" in df_data.columns
            assert "Infected" in df_data.columns
            assert "Recovered" in df_data.columns
            assert len(df_data) == len(sample_result["time"])
            
            # Verifica a planilha de metadados
            df_metadata = pd.read_excel(xls, "Metadata")
            assert "Parameter" in df_metadata.columns
            assert "Value" in df_metadata.columns
            assert len(df_metadata) == len(sample_result["metadata"])
            
            # Verifica a planilha de estatísticas
            df_stats = pd.read_excel(xls, "Statistics")
            assert "Statistic" in df_stats.columns
            assert "Value" in df_stats.columns
            assert "Max Infected" in df_stats["Statistic"].values
            assert "Final Recovered" in df_stats["Statistic"].values
            assert "Time to Peak" in df_stats["Statistic"].values
    finally:
        # Remove o arquivo
        if os.path.exists(path):
            os.remove(path)


def test_export_to_json(sample_result):
    """Testa a exportação para JSON."""
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    
    try:
        # Exporta os resultados
        exported_path = ExportManager.export_to_json(sample_result, path)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(exported_path)
        
        # Carrega o JSON e verifica
        with open(path, 'r') as f:
            loaded_result = json.load(f)
        
        assert loaded_result["S_val"] == sample_result["S_val"]
        assert loaded_result["I_val"] == sample_result["I_val"]
        assert loaded_result["R_val"] == sample_result["R_val"]
        assert loaded_result["time"] == sample_result["time"]
        assert loaded_result["metadata"] == sample_result["metadata"]
    finally:
        # Remove o arquivo
        if os.path.exists(path):
            os.remove(path)


def test_export_to_markdown(sample_result, monkeypatch):
    """Testa a exportação para Markdown."""
    # Mock para Visualizer.plot_result para evitar a geração de gráficos
    def mock_plot_result(*args, **kwargs):
        pass
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot_result)
    
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
        path = f.name
    
    try:
        # Exporta os resultados
        exported_path = ExportManager.export_to_markdown(sample_result, path, include_plot=False)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(exported_path)
        
        # Lê o conteúdo do arquivo
        with open(path, 'r') as f:
            content = f.read()
        
        # Verifica o conteúdo
        assert "# Relatório de Simulação SPKMC" in content
        assert "## Parâmetros da Simulação" in content
        assert "## Estatísticas" in content
        assert "## Dados da Simulação" in content
        assert "| Tempo | Suscetíveis | Infectados | Recuperados |" in content
        assert "Tipo de Rede | ER" in content
        assert "Distribuição | Gamma" in content
        assert "Número de Nós (N) | 100" in content
    finally:
        # Remove o arquivo
        if os.path.exists(path):
            os.remove(path)


def test_export_to_html(sample_result, monkeypatch):
    """Testa a exportação para HTML."""
    # Mock para Visualizer.plot_result para evitar a geração de gráficos
    def mock_plot_result(*args, **kwargs):
        pass
    
    # Mock para export_to_markdown para evitar a geração de arquivos Markdown
    def mock_export_to_markdown(result, output_path, include_plot=True):
        # Cria um arquivo Markdown de exemplo
        with open(output_path, 'w') as f:
            f.write("# Relatório de Simulação SPKMC\n\n## Parâmetros da Simulação\n\n| Parâmetro | Valor |\n|-----------|-------|\n| Tipo de Rede | ER |\n")
        return output_path
    
    # Aplica os mocks
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot_result)
    monkeypatch.setattr(ExportManager, 'export_to_markdown', mock_export_to_markdown)
    
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        path = f.name
    
    try:
        # Exporta os resultados
        exported_path = ExportManager.export_to_html(sample_result, path, include_plot=False)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(exported_path)
        
        # Lê o conteúdo do arquivo
        with open(path, 'r') as f:
            content = f.read()
        
        # Verifica o conteúdo
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "Relatório de Simulação SPKMC" in content
        assert "Parâmetros da Simulação" in content
        assert "Tipo de Rede" in content
        assert "ER" in content
    finally:
        # Remove o arquivo
        if os.path.exists(path):
            os.remove(path)


def test_export_plot(sample_result, monkeypatch):
    """Testa a exportação de gráficos."""
    # Mock para Visualizer.plot_result para evitar a geração de gráficos
    def mock_plot_result(*args, **kwargs):
        # Cria um arquivo vazio no caminho especificado
        with open(kwargs.get('save_path', args[-1]), 'w') as f:
            f.write('')
    
    # Aplica o mock
    from spkmc.visualization.plots import Visualizer
    monkeypatch.setattr(Visualizer, 'plot_result', mock_plot_result)
    
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    
    try:
        # Exporta o gráfico
        exported_path = ExportManager.export_plot(sample_result, path, format="png", dpi=300)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(exported_path)
    finally:
        # Remove o arquivo
        if os.path.exists(path):
            os.remove(path)


def test_export_results_invalid_format(sample_result):
    """Testa a exportação com um formato inválido."""
    with pytest.raises(ValueError):
        ExportManager.export_results(sample_result, "output.xyz", format="invalid")


def test_export_results_valid_formats(sample_result, monkeypatch):
    """Testa a exportação com formatos válidos."""
    # Mock para os métodos de exportação
    def mock_export(*args, **kwargs):
        return "mock_path.ext"
    
    # Aplica os mocks
    monkeypatch.setattr(ExportManager, 'export_to_json', mock_export)
    monkeypatch.setattr(ExportManager, 'export_to_csv', mock_export)
    monkeypatch.setattr(ExportManager, 'export_to_excel', mock_export)
    monkeypatch.setattr(ExportManager, 'export_to_markdown', mock_export)
    monkeypatch.setattr(ExportManager, 'export_to_html', mock_export)
    
    # Testa os formatos válidos
    assert ExportManager.export_results(sample_result, "output.json", format="json") == "mock_path.ext"
    assert ExportManager.export_results(sample_result, "output.csv", format="csv") == "mock_path.ext"
    assert ExportManager.export_results(sample_result, "output.xlsx", format="excel") == "mock_path.ext"
    assert ExportManager.export_results(sample_result, "output.md", format="md") == "mock_path.ext"
    assert ExportManager.export_results(sample_result, "output.md", format="markdown") == "mock_path.ext"
    assert ExportManager.export_results(sample_result, "output.html", format="html") == "mock_path.ext"