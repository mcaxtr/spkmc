"""
Core SPKMC algorithm components.

This package contains the main implementation of the SPKMC algorithm,
including simulation logic, probability distributions, network factories,
and transmissibility calculations.

Usage:
    from spkmc.core.simulation import SPKMC
    from spkmc.core.distributions import create_distribution
"""

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


def __getattr__(name: str) -> object:
    """Lazy import for core modules to avoid slow startup."""
    # Distributions
    if name in (
        "Distribution",
        "GammaDistribution",
        "ExponentialDistribution",
        "create_distribution",
    ):
        from spkmc.core.distributions import (
            Distribution,
            ExponentialDistribution,
            GammaDistribution,
            create_distribution,
        )

        globals().update(
            {
                "Distribution": Distribution,
                "GammaDistribution": GammaDistribution,
                "ExponentialDistribution": ExponentialDistribution,
                "create_distribution": create_distribution,
            }
        )
        return globals()[name]
    # Networks
    elif name == "NetworkFactory":
        from spkmc.core.networks import NetworkFactory

        globals()["NetworkFactory"] = NetworkFactory
        return NetworkFactory
    # Simulation
    elif name == "SPKMC":
        from spkmc.core.simulation import SPKMC

        globals()["SPKMC"] = SPKMC
        return SPKMC
    # Transmissibility
    elif name in (
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
    ):
        from spkmc.core import transmissibility as trans

        globals().update(
            {
                "TransmissibilityCalculator": trans.TransmissibilityCalculator,
                "ExponentialExponentialTransmissibility": (
                    trans.ExponentialExponentialTransmissibility
                ),
                "GammaExponentialTransmissibility": trans.GammaExponentialTransmissibility,
                "NumericalTransmissibility": trans.NumericalTransmissibility,
                "EpidemicThreshold": trans.EpidemicThreshold,
                "create_transmissibility_calculator": trans.create_transmissibility_calculator,
                "calculate_empirical_transmissibility": trans.calculate_empirical_transmissibility,
                "poisson_poisson_transmissibility": trans.poisson_poisson_transmissibility,
                "gamma_exponential_transmissibility": trans.gamma_exponential_transmissibility,
                "epidemic_threshold_regular": trans.epidemic_threshold_regular,
                "epidemic_threshold_er": trans.epidemic_threshold_er,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'spkmc.core' has no attribute '{name}'")
