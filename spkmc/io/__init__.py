"""
Módulo de entrada/saída para o algoritmo SPKMC.

Este módulo contém classes e funções para gerenciamento de resultados,
experimentos e exportação de dados.
"""

from spkmc.io.results import ResultManager
from spkmc.io.export import ExportManager
from spkmc.io.experiments import ExperimentManager, Experiment, PlotConfig

__all__ = [
    "ResultManager",
    "ExportManager",
    "ExperimentManager",
    "Experiment",
    "PlotConfig",
]
