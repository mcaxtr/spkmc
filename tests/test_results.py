"""
Testes para o módulo de resultados do SPKMC.

Este módulo contém testes para a classe ResultManager e suas funcionalidades.
"""

import os
import json
import tempfile
import pytest
import numpy as np
from pathlib import Path

from spkmc.io.results import ResultManager
from spkmc.core.distributions import GammaDistribution, ExponentialDistribution


@pytest.fixture
def gamma_distribution():
    """Fixture para uma distribuição Gamma."""
    return GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)


@pytest.fixture
def exponential_distribution():
    """Fixture para uma distribuição Exponencial."""
    return ExponentialDistribution(mu=1.0, lmbd=1.0)


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
def temp_result_file(sample_result):
    """Fixture para criar um arquivo de resultados temporário."""
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        # Salva os dados no arquivo
        json.dump(sample_result, f)
        
        # Retorna o caminho do arquivo
        path = f.name
    
    yield path
    
    # Remove o arquivo após o teste
    if os.path.exists(path):
        os.remove(path)


def test_get_result_path_er(gamma_distribution):
    """Testa a geração de caminho para resultados de rede Erdos-Renyi."""
    path = ResultManager.get_result_path("er", gamma_distribution, 1000, 50)
    
    assert isinstance(path, str)
    assert "er" in path.lower()
    assert "gamma" in path.lower()
    assert "1000" in path
    assert "50" in path
    assert path.endswith(".json")


def test_get_result_path_cn(gamma_distribution):
    """Testa a geração de caminho para resultados de rede complexa."""
    path = ResultManager.get_result_path("cn", gamma_distribution, 1000, 50, 2.5)
    
    assert isinstance(path, str)
    assert "cn" in path.lower()
    assert "gamma" in path.lower()
    assert "1000" in path
    assert "50" in path
    assert "2.5" in path or "25" in path  # Pode ser formatado como 2.5 ou 25
    assert path.endswith(".json")


def test_load_result(temp_result_file, sample_result):
    """Testa o carregamento de resultados."""
    result = ResultManager.load_result(temp_result_file)
    
    assert isinstance(result, dict)
    assert "S_val" in result
    assert "I_val" in result
    assert "R_val" in result
    assert "time" in result
    assert "metadata" in result
    assert result["metadata"]["network_type"] == sample_result["metadata"]["network_type"]
    assert result["metadata"]["distribution"] == sample_result["metadata"]["distribution"]
    assert result["metadata"]["N"] == sample_result["metadata"]["N"]


def test_load_result_nonexistent():
    """Testa o carregamento de um arquivo inexistente."""
    with pytest.raises(FileNotFoundError):
        ResultManager.load_result("nonexistent_file.json")


def test_save_result(sample_result):
    """Testa o salvamento de resultados."""
    # Cria um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    
    try:
        # Salva os resultados
        ResultManager.save_result(path, sample_result)
        
        # Verifica se o arquivo foi criado
        assert os.path.exists(path)
        
        # Carrega os resultados e verifica
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


def test_list_results(monkeypatch):
    """Testa a listagem de resultados."""
    # Mock para Path.exists
    def mock_exists(self):
        return True
    
    # Mock para Path.iterdir
    def mock_iterdir(self):
        paths = [
            Path("data/spkmc/gamma"),
            Path("data/spkmc/exponential")
        ]
        for path in paths:
            yield path
    
    # Mock para o segundo nível de iterdir
    def mock_iterdir_level2(self):
        if str(self).endswith("gamma"):
            paths = [
                Path("data/spkmc/gamma/er"),
                Path("data/spkmc/gamma/cn")
            ]
        else:
            paths = [
                Path("data/spkmc/exponential/er")
            ]
        for path in paths:
            yield path
    
    # Mock para o terceiro nível de iterdir
    def mock_iterdir_level3(self):
        if str(self).endswith("er"):
            paths = [
                Path("data/spkmc/gamma/er/results_1000_50_2.0.json"),
                Path("data/spkmc/gamma/er/results_2000_100_2.0.json")
            ]
        else:
            paths = [
                Path("data/spkmc/gamma/cn/results_25_1000_50_2.0.json")
            ]
        for path in paths:
            yield path
    
    # Mock para Path.glob
    def mock_glob(self, pattern):
        if str(self).endswith("er"):
            paths = [
                Path("data/spkmc/gamma/er/results_1000_50_2.0.json"),
                Path("data/spkmc/gamma/er/results_2000_100_2.0.json")
            ]
        else:
            paths = [
                Path("data/spkmc/gamma/cn/results_25_1000_50_2.0.json")
            ]
        for path in paths:
            yield path
    
    # Aplica os mocks
    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "iterdir", mock_iterdir)
    monkeypatch.setattr(Path, "glob", mock_glob)
    
    # Substitui o método iterdir para diferentes instâncias
    original_iterdir = Path.iterdir
    def patched_iterdir(self):
        if str(self).endswith("spkmc"):
            return mock_iterdir(self)
        elif str(self).endswith("gamma") or str(self).endswith("exponential"):
            return mock_iterdir_level2(self)
        else:
            return mock_iterdir_level3(self)
    
    monkeypatch.setattr(Path, "iterdir", patched_iterdir)
    
    # Testa a listagem de resultados
    results = ResultManager.list_results()
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("gamma/er" in r for r in results)
    assert any("gamma/cn" in r for r in results)


def test_get_metadata_from_path():
    """Testa a extração de metadados do caminho do arquivo."""
    path = "data/spkmc/gamma/er/results_1000_50_2.0.json"
    metadata = ResultManager.get_metadata_from_path(path)
    
    assert isinstance(metadata, dict)
    assert metadata["distribution"] == "gamma"
    assert metadata["network_type"] == "er"
    assert metadata["N"] == 1000
    assert metadata["samples"] == 50


def test_get_metadata_from_path_cn():
    """Testa a extração de metadados do caminho do arquivo para rede complexa."""
    path = "data/spkmc/gamma/cn/results_25_1000_50_2.0.json"
    metadata = ResultManager.get_metadata_from_path(path)
    
    assert isinstance(metadata, dict)
    assert metadata["distribution"] == "gamma"
    assert metadata["network_type"] == "cn"
    assert metadata["exponent"] == 2.5
    assert metadata["N"] == 1000
    assert metadata["samples"] == 50


def test_format_result_for_cli(sample_result):
    """Testa a formatação de resultados para a CLI."""
    formatted = ResultManager.format_result_for_cli(sample_result)
    
    assert isinstance(formatted, dict)
    assert "metadata" in formatted
    assert "max_infected" in formatted
    assert "final_recovered" in formatted
    assert "data_points" in formatted
    assert "has_error_data" in formatted
    assert formatted["max_infected"] == 0.05
    assert formatted["final_recovered"] == 0.16
    assert formatted["data_points"] == 5
    assert formatted["has_error_data"] is False