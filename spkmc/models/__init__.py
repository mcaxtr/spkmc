"""
SPKMC domain models package.

This package contains Pydantic models and dataclasses for:
- Scenario and ScenarioOverride: Simulation configuration
- Experiment and ExperimentConfig: Multi-scenario experiment management
- SimulationResult: Unified result container
- PlotConfig: Visualization configuration

These models are used by both the `run` and `experiments` CLI commands.
"""

from spkmc.models.config import PlotConfig
from spkmc.models.experiment import Experiment, ExperimentConfig
from spkmc.models.result import SimulationResult
from spkmc.models.scenario import Scenario, ScenarioOverride

__all__ = [
    "Scenario",
    "ScenarioOverride",
    "Experiment",
    "ExperimentConfig",
    "SimulationResult",
    "PlotConfig",
]
