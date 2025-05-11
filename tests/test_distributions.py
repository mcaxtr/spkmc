"""
Testes para o módulo de distribuições.

Este módulo contém testes para as classes de distribuição do SPKMC.
"""

import pytest
import numpy as np

from spkmc.core.distributions import (
    Distribution,
    GammaDistribution,
    ExponentialDistribution,
    create_distribution
)


def test_gamma_distribution_creation():
    """Testa a criação de uma distribuição Gamma."""
    shape = 2.0
    scale = 1.0
    lmbd = 1.0
    
    dist = GammaDistribution(shape=shape, scale=scale, lmbd=lmbd)
    
    assert dist.shape == shape
    assert dist.scale == scale
    assert dist.lmbd == lmbd
    assert dist.get_distribution_name() == "gamma"
    assert dist.get_params_string() == "2.0"


def test_exponential_distribution_creation():
    """Testa a criação de uma distribuição Exponencial."""
    mu = 1.0
    lmbd = 1.0
    
    dist = ExponentialDistribution(mu=mu, lmbd=lmbd)
    
    assert dist.mu == mu
    assert dist.lmbd == lmbd
    assert dist.get_distribution_name() == "exponential"
    assert dist.get_params_string() == ""


def test_create_distribution_gamma():
    """Testa a função create_distribution para distribuição Gamma."""
    dist = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)
    
    assert isinstance(dist, GammaDistribution)
    assert dist.shape == 2.0
    assert dist.scale == 1.0
    assert dist.lmbd == 1.0


def test_create_distribution_exponential():
    """Testa a função create_distribution para distribuição Exponencial."""
    dist = create_distribution("exponential", mu=1.0, lambda_=1.0)
    
    assert isinstance(dist, ExponentialDistribution)
    assert dist.mu == 1.0
    assert dist.lmbd == 1.0


def test_create_distribution_invalid():
    """Testa a função create_distribution com um tipo inválido."""
    with pytest.raises(ValueError):
        create_distribution("invalid_type")


def test_gamma_distribution_recovery_weights():
    """Testa a geração de pesos de recuperação para distribuição Gamma."""
    dist = GammaDistribution(shape=2.0, scale=1.0)
    size = 10
    
    weights = dist.get_recovery_weights(size)
    
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (size,)
    assert np.all(weights >= 0)


def test_exponential_distribution_recovery_weights():
    """Testa a geração de pesos de recuperação para distribuição Exponencial."""
    dist = ExponentialDistribution(mu=1.0, lmbd=1.0)
    size = 10
    
    weights = dist.get_recovery_weights(size)
    
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (size,)
    assert np.all(weights >= 0)


def test_distribution_params_dict():
    """Testa a geração de dicionário de parâmetros."""
    gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)
    exp_dist = ExponentialDistribution(mu=1.0, lmbd=1.0)
    
    gamma_params = gamma_dist.get_params_dict()
    exp_params = exp_dist.get_params_dict()
    
    assert gamma_params["type"] == "gamma"
    assert gamma_params["shape"] == 2.0
    assert gamma_params["scale"] == 1.0
    assert gamma_params["lambda"] == 1.0
    
    assert exp_params["type"] == "exponential"
    assert exp_params["mu"] == 1.0
    assert exp_params["lambda"] == 1.0