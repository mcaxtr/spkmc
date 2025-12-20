"""
Core SPKMC algorithm components.

This package contains the main implementation of the SPKMC algorithm,
including simulation logic, probability distributions, network factories,
and transmissibility calculations.
"""

from spkmc.core.distributions import (
    Distribution,
    GammaDistribution,
    ExponentialDistribution,
    create_distribution,
)

from spkmc.core.networks import NetworkFactory

from spkmc.core.simulation import SPKMC

from spkmc.core.transmissibility import (
    TransmissibilityCalculator,
    ExponentialExponentialTransmissibility,
    GammaExponentialTransmissibility,
    NumericalTransmissibility,
    EpidemicThreshold,
    create_transmissibility_calculator,
    calculate_empirical_transmissibility,
    poisson_poisson_transmissibility,
    gamma_exponential_transmissibility,
    epidemic_threshold_regular,
    epidemic_threshold_er,
)

__all__ = [
    # Distributions
    "Distribution",
    "GammaDistribution",
    "ExponentialDistribution",
    "create_distribution",
    # Networks
    "NetworkFactory",
    # Simulation
    "SPKMC",
    # Transmissibility
    "TransmissibilityCalculator",
    "ExponentialExponentialTransmissibility",
    "GammaExponentialTransmissibility",
    "NumericalTransmissibility",
    "EpidemicThreshold",
    "create_transmissibility_calculator",
    "calculate_empirical_transmissibility",
    "poisson_poisson_transmissibility",
    "gamma_exponential_transmissibility",
    "epidemic_threshold_regular",
    "epidemic_threshold_er",
]
