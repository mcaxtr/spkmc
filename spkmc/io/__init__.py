"""
Módulo de entrada/saída para o algoritmo SPKMC.

Este módulo contém classes e funções para gerenciamento de resultados,
experimentos e exportação de dados.

Usage:
    from spkmc.io.results import ResultManager
    from spkmc.io.experiments import ExperimentManager
"""

__all__ = [
    "ResultManager",
    "ExportManager",
    "ExperimentManager",
    "Experiment",
    "PlotConfig",
]


def __getattr__(name: str) -> object:
    """Lazy import for IO modules."""
    if name == "ResultManager":
        from spkmc.io.results import ResultManager

        globals()["ResultManager"] = ResultManager
        return ResultManager
    elif name == "ExportManager":
        from spkmc.io.export import ExportManager

        globals()["ExportManager"] = ExportManager
        return ExportManager
    elif name in ("ExperimentManager", "Experiment", "PlotConfig"):
        from spkmc.io.experiments import Experiment, ExperimentManager, PlotConfig

        globals().update(
            {
                "ExperimentManager": ExperimentManager,
                "Experiment": Experiment,
                "PlotConfig": PlotConfig,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'spkmc.io' has no attribute '{name}'")
