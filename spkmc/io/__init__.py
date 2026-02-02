"""
Input/output module for the SPKMC algorithm.

This module contains classes and functions for managing data I/O,
experiments, and result storage.

Usage:
    from spkmc.io import DataManager, Scenario, Experiment
    from spkmc.io import ExperimentManager, PlotConfig
"""

__all__ = [
    "DataManager",
    "Scenario",
    "ScenarioOverride",
    "Experiment",
    "ExperimentConfig",
    "ExperimentManager",
    "PlotConfig",
]


def __getattr__(name: str) -> object:
    """Lazy import for IO modules."""
    if name == "DataManager":
        from spkmc.io.data_manager import DataManager

        globals()["DataManager"] = DataManager
        return DataManager
    elif name in (
        "Scenario",
        "ScenarioOverride",
        "Experiment",
        "ExperimentConfig",
        "ExperimentManager",
        "PlotConfig",
    ):
        from spkmc.io.experiments import (
            Experiment,
            ExperimentConfig,
            ExperimentManager,
            PlotConfig,
            Scenario,
            ScenarioOverride,
        )

        globals().update(
            {
                "Scenario": Scenario,
                "ScenarioOverride": ScenarioOverride,
                "Experiment": Experiment,
                "ExperimentConfig": ExperimentConfig,
                "ExperimentManager": ExperimentManager,
                "PlotConfig": PlotConfig,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'spkmc.io' has no attribute '{name}'")
