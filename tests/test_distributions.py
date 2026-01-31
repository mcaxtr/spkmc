"""
Tests for the distributions module.

This module contains tests for SPKMC distribution classes.
"""

import numpy as np
import pytest

from spkmc.core.distributions import ExponentialDistribution, GammaDistribution, create_distribution


def test_gamma_distribution_creation():
    """Test creation of a Gamma distribution."""
    shape = 2.0
    scale = 1.0
    lmbd = 1.0

    dist = GammaDistribution(shape=shape, scale=scale, lmbd=lmbd)

    assert dist.shape == shape
    assert dist.scale == scale
    assert dist.lmbd == lmbd
    assert dist.get_distribution_name() == "gamma"
    assert dist.get_params_string() == "sh2_sc1_l1"


def test_exponential_distribution_creation():
    """Test creation of an Exponential distribution."""
    mu = 1.0
    lmbd = 1.0

    dist = ExponentialDistribution(mu=mu, lmbd=lmbd)

    assert dist.mu == mu
    assert dist.lmbd == lmbd
    assert dist.get_distribution_name() == "exponential"
    assert dist.get_params_string() == "mu1_l1"


def test_create_distribution_gamma():
    """Test create_distribution for a Gamma distribution."""
    dist = create_distribution("gamma", shape=2.0, scale=1.0, lambda_=1.0)

    assert isinstance(dist, GammaDistribution)
    assert dist.shape == 2.0
    assert dist.scale == 1.0
    assert dist.lmbd == 1.0


def test_create_distribution_exponential():
    """Test create_distribution for an Exponential distribution."""
    dist = create_distribution("exponential", mu=1.0, lambda_=1.0)

    assert isinstance(dist, ExponentialDistribution)
    assert dist.mu == 1.0
    assert dist.lmbd == 1.0


def test_create_distribution_invalid():
    """Test create_distribution with an invalid type."""
    with pytest.raises(ValueError):
        create_distribution("invalid_type")


def test_gamma_distribution_recovery_weights():
    """Test recovery weight generation for the Gamma distribution."""
    dist = GammaDistribution(shape=2.0, scale=1.0)
    size = 10

    weights = dist.get_recovery_weights(size)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == (size,)
    assert np.all(weights >= 0)


def test_exponential_distribution_recovery_weights():
    """Test recovery weight generation for the Exponential distribution."""
    dist = ExponentialDistribution(mu=1.0, lmbd=1.0)
    size = 10

    weights = dist.get_recovery_weights(size)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == (size,)
    assert np.all(weights >= 0)


def test_distribution_params_dict():
    """Test generation of parameter dictionaries."""
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
