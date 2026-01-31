"""
Tests for the networks module.

This module contains tests for SPKMC network classes.
"""

import networkx as nx
import numpy as np
import pytest

from spkmc.core.networks import NetworkFactory


def test_create_erdos_renyi():
    """Test creating an Erdos-Renyi network."""
    N = 100
    k_avg = 5

    G = NetworkFactory.create_erdos_renyi(N, k_avg)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == N
    assert G.number_of_edges() > 0

    # For DiGraph created from undirected ER, G.degree() returns in+out degree
    # which is ~2*k_avg. We check out_degree which should be ~k_avg
    avg_out_degree = sum(dict(G.out_degree()).values()) / N
    assert abs(avg_out_degree - k_avg) < k_avg * 0.3  # 30% tolerance


def test_create_complex_network():
    """Test creating a complex network."""
    N = 100
    exponent = 2.5
    k_avg = 5

    G = NetworkFactory.create_complex_network(N, exponent, k_avg)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == N
    assert G.number_of_edges() > 0

    # For DiGraph created from undirected CN, G.degree() returns in+out degree
    # which is ~2*k_avg. We check out_degree which should be ~k_avg
    avg_out_degree = sum(dict(G.out_degree()).values()) / N
    assert abs(avg_out_degree - k_avg) < k_avg * 0.3  # 30% tolerance


def test_create_complete_graph():
    """Test creating a complete graph."""
    N = 10

    G = NetworkFactory.create_complete_graph(N)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == N
    assert G.number_of_edges() == N * (N - 1)  # Directed complete graph


def test_create_random_regular_network():
    """Test creating a random regular network."""
    N = 100
    k_avg = 4  # Must be even for random_regular_graph

    G = NetworkFactory.create_random_regular_network(N, k_avg)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == N
    assert G.number_of_edges() > 0

    # For DiGraph from undirected RRN, each node has out_degree == k_avg
    # G.degree() returns in+out which is 2*k_avg for each node
    out_degrees = dict(G.out_degree()).values()
    assert all(d == k_avg for d in out_degrees)


def test_generate_discrete_power_law():
    """Test generating a discrete power-law sequence."""
    n = 100
    alpha = 2.5
    xmin = 2
    xmax = 10

    seq = NetworkFactory.generate_discrete_power_law(n, alpha, xmin, xmax)

    assert isinstance(seq, np.ndarray)
    assert len(seq) == n
    assert np.all(seq >= xmin)
    assert np.all(seq <= xmax)
    assert sum(seq) % 2 == 0  # Sum must be even


def test_create_network_er():
    """Test create_network for an Erdos-Renyi network."""
    G = NetworkFactory.create_network("er", N=100, k_avg=5)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 100


def test_create_network_cn():
    """Test create_network for a complex network."""
    G = NetworkFactory.create_network("cn", N=100, exponent=2.5, k_avg=5)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 100


def test_create_network_cg():
    """Test create_network for a complete graph."""
    G = NetworkFactory.create_network("cg", N=10)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 10 * 9  # Directed complete graph


def test_create_network_rrn():
    """Test create_network for a random regular network."""
    G = NetworkFactory.create_network("rrn", N=100, k_avg=4)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 100

    # For DiGraph from undirected RRN, each node has out_degree == k_avg
    out_degrees = dict(G.out_degree()).values()
    assert all(d == 4 for d in out_degrees)


def test_create_network_invalid():
    """Test create_network with an invalid type."""
    with pytest.raises(ValueError):
        NetworkFactory.create_network("invalid_type")


def test_get_network_info():
    """Test get_network_info."""
    # Erdos-Renyi network
    info_er = NetworkFactory.get_network_info("er", N=100, k_avg=5)
    assert info_er["type"] == "er"
    assert info_er["N"] == 100
    assert info_er["k_avg"] == 5

    # Complex network
    info_cn = NetworkFactory.get_network_info("cn", N=100, k_avg=5, exponent=2.5)
    assert info_cn["type"] == "cn"
    assert info_cn["N"] == 100
    assert info_cn["k_avg"] == 5
    assert info_cn["exponent"] == 2.5

    # Complete graph
    info_cg = NetworkFactory.get_network_info("cg", N=10)
    assert info_cg["type"] == "cg"
    assert info_cg["N"] == 10

    # Random regular network
    info_rrn = NetworkFactory.get_network_info("rrn", N=100, k_avg=4)
    assert info_rrn["type"] == "rrn"
    assert info_rrn["N"] == 100
    assert info_rrn["k_avg"] == 4
