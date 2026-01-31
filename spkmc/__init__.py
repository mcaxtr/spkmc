"""
SPKMC - Shortest Path Kinetic Monte Carlo

Este pacote implementa o algoritmo SPKMC para simulação de propagação de epidemias em redes,
utilizando o modelo SIR (Susceptible-Infected-Recovered).

A implementação é baseada em classes e interfaces que permitem a simulação em diferentes
tipos de redes e com diferentes distribuições de probabilidade.

Usage:
    # Import specific modules directly for faster startup:
    from spkmc.core.simulation import SPKMC
    from spkmc.core.distributions import create_distribution

    # Or use lazy imports (triggers JIT compilation on first use):
    import spkmc
    sim = spkmc.SPKMC(...)
"""

# Suppress OpenMP deprecation warning (must be set before Numba imports)
# KMP_WARNINGS=0 suppresses Intel OpenMP informational messages
# OMP_MAX_ACTIVE_LEVELS replaces the deprecated omp_set_nested
import os as _os
_os.environ.setdefault('KMP_WARNINGS', '0')
_os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

__version__ = "1.0.0"

# Lazy imports to avoid slow startup (Numba JIT compilation takes ~60s)
# Heavy modules are only imported when accessed via __getattr__
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


def __getattr__(name: str):
    """Lazy import for heavy modules to speed up CLI startup."""
    if name in ("Distribution", "GammaDistribution", "ExponentialDistribution", "create_distribution"):
        from spkmc.core.distributions import Distribution, GammaDistribution, ExponentialDistribution, create_distribution
        globals().update({
            "Distribution": Distribution,
            "GammaDistribution": GammaDistribution,
            "ExponentialDistribution": ExponentialDistribution,
            "create_distribution": create_distribution,
        })
        return globals()[name]
    elif name == "NetworkFactory":
        from spkmc.core.networks import NetworkFactory
        globals()["NetworkFactory"] = NetworkFactory
        return NetworkFactory
    elif name == "SPKMC":
        from spkmc.core.simulation import SPKMC
        globals()["SPKMC"] = SPKMC
        return SPKMC
    elif name == "ResultManager":
        from spkmc.io.results import ResultManager
        globals()["ResultManager"] = ResultManager
        return ResultManager
    elif name == "Visualizer":
        from spkmc.visualization.plots import Visualizer
        globals()["Visualizer"] = Visualizer
        return Visualizer
    raise AttributeError(f"module 'spkmc' has no attribute '{name}'")