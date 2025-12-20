"""
SPKMC - Shortest Path Kinetic Monte Carlo

Este pacote implementa o algoritmo SPKMC para simulação de propagação de epidemias em redes,
utilizando o modelo SIR (Susceptible-Infected-Recovered).

A implementação é baseada em classes e interfaces que permitem a simulação em diferentes
tipos de redes e com diferentes distribuições de probabilidade.
"""

# Suppress OpenMP deprecation warning (must be set before Numba imports)
# KMP_WARNINGS=0 suppresses Intel OpenMP informational messages
# OMP_MAX_ACTIVE_LEVELS replaces the deprecated omp_set_nested
import os as _os
_os.environ.setdefault('KMP_WARNINGS', '0')
_os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

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