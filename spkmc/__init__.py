"""
SPKMC - Shortest Path Kinetic Monte Carlo

This package implements the SPKMC algorithm to simulate epidemic spread on networks,
using the SIR model (Susceptible-Infected-Recovered).

The implementation is based on classes and interfaces that enable simulation across
different network types and probability distributions.

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

_os.environ.setdefault("KMP_WARNINGS", "0")
_os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

# Dynamic version from setuptools-scm (Git tags)
try:
    from spkmc._version import __version__
except ImportError:
    # Fallback for when package is not installed (e.g., editable install without build)
    from importlib.metadata import version as _get_version

    try:
        __version__ = _get_version("spkmc")
    except Exception:
        __version__ = "0.0.0.dev0"

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


def __getattr__(name: str) -> object:
    """Lazy import for heavy modules to speed up CLI startup."""
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
