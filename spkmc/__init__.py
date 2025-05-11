"""
SPKMC - Shortest Path Kinetic Monte Carlo

Este pacote implementa o algoritmo SPKMC para simulação de propagação de epidemias em redes,
utilizando o modelo SIR (Susceptible-Infected-Recovered).

A implementação é baseada em classes e interfaces que permitem a simulação em diferentes
tipos de redes e com diferentes distribuições de probabilidade.
"""

__version__ = "1.0.0"

from spkmc.core.distributions import (
    Distribution,
    GammaDistribution,
    ExponentialDistribution,
    create_distribution
)
from spkmc.core.networks import NetworkFactory
from spkmc.core.simulation import SPKMC
from spkmc.io.results import ResultManager
from spkmc.visualization.plots import Visualizer

__all__ = [
    "Distribution",
    "GammaDistribution",
    "ExponentialDistribution",
    "create_distribution",
    "NetworkFactory",
    "SPKMC",
    "ResultManager",
    "Visualizer",
]