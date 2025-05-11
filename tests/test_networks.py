"""
Testes para o módulo de redes.

Este módulo contém testes para as classes de redes do SPKMC.
"""

import pytest
import networkx as nx
import numpy as np

from spkmc.core.networks import NetworkFactory


def test_create_erdos_renyi():
    """Testa a criação de uma rede Erdos-Renyi."""
    N = 100
    k_avg = 5
    
    G = NetworkFactory.create_erdos_renyi(N, k_avg)
    
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == N
    assert G.number_of_edges() > 0
    
    # Verifica se o grau médio está próximo do esperado
    avg_degree = sum(dict(G.degree()).values()) / N
    assert abs(avg_degree - k_avg) < k_avg * 0.3  # Tolerância de 30%


def test_create_complex_network():
    """Testa a criação de uma rede complexa."""
    N = 100
    exponent = 2.5
    k_avg = 5
    
    G = NetworkFactory.create_complex_network(N, exponent, k_avg)
    
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == N
    assert G.number_of_edges() > 0
    
    # Verifica se o grau médio está próximo do esperado
    avg_degree = sum(dict(G.degree()).values()) / N
    assert abs(avg_degree - k_avg) < k_avg * 0.3  # Tolerância de 30%


def test_create_complete_graph():
    """Testa a criação de um grafo completo."""
    N = 10
    
    G = NetworkFactory.create_complete_graph(N)
    
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == N
    assert G.number_of_edges() == N * (N - 1)  # Grafo direcionado completo


def test_generate_discrete_power_law():
    """Testa a geração de uma sequência de lei de potência discreta."""
    n = 100
    alpha = 2.5
    xmin = 2
    xmax = 10
    
    seq = NetworkFactory.generate_discrete_power_law(n, alpha, xmin, xmax)
    
    assert isinstance(seq, np.ndarray)
    assert len(seq) == n
    assert np.all(seq >= xmin)
    assert np.all(seq <= xmax)
    assert sum(seq) % 2 == 0  # A soma deve ser par


def test_create_network_er():
    """Testa a função create_network para rede Erdos-Renyi."""
    G = NetworkFactory.create_network("er", N=100, k_avg=5)
    
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 100


def test_create_network_cn():
    """Testa a função create_network para rede complexa."""
    G = NetworkFactory.create_network("cn", N=100, exponent=2.5, k_avg=5)
    
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 100


def test_create_network_cg():
    """Testa a função create_network para grafo completo."""
    G = NetworkFactory.create_network("cg", N=10)
    
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 10 * 9  # Grafo direcionado completo


def test_create_network_invalid():
    """Testa a função create_network com um tipo inválido."""
    with pytest.raises(ValueError):
        NetworkFactory.create_network("invalid_type")


def test_get_network_info():
    """Testa a função get_network_info."""
    # Rede Erdos-Renyi
    info_er = NetworkFactory.get_network_info("er", N=100, k_avg=5)
    assert info_er["type"] == "er"
    assert info_er["N"] == 100
    assert info_er["k_avg"] == 5
    
    # Rede complexa
    info_cn = NetworkFactory.get_network_info("cn", N=100, k_avg=5, exponent=2.5)
    assert info_cn["type"] == "cn"
    assert info_cn["N"] == 100
    assert info_cn["k_avg"] == 5
    assert info_cn["exponent"] == 2.5
    
    # Grafo completo
    info_cg = NetworkFactory.get_network_info("cg", N=10)
    assert info_cg["type"] == "cg"
    assert info_cg["N"] == 10