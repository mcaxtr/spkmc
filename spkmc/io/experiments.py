"""
Experiment management and I/O operations for the SPKMC algorithm.

This module contains the ExperimentManager class for loading and managing
experiments from the filesystem.

Domain models (Scenario, Experiment, etc.) are now in spkmc.models.
This module re-exports them for backward compatibility.
"""

import json
from pathlib import Path
from typing import List, Optional

# Re-export models for backward compatibility
from spkmc.models import (
    Experiment,
    ExperimentConfig,
    PlotConfig,
    Scenario,
    ScenarioOverride,
    SimulationResult,
)

__all__ = [
    "Scenario",
    "ScenarioOverride",
    "PlotConfig",
    "ExperimentConfig",
    "Experiment",
    "SimulationResult",
    "ExperimentManager",
]


class ExperimentManager:
    """Manage SPKMC experiments."""

    DEFAULT_EXPERIMENTS_DIR = "experiments"
    DATA_FILE_NAME = "data.json"

    def __init__(self, experiments_dir: Optional[str] = None):
        """
        Initialize the experiment manager.

        Args:
            experiments_dir: Base directory for experiments (optional)
        """
        import os

        self.experiments_dir = Path(
            experiments_dir
            or os.environ.get("SPKMC_EXPERIMENTS_DIR")
            or self.DEFAULT_EXPERIMENTS_DIR
        )

    def list_experiments(self) -> List[Experiment]:
        """
        List all available experiments.

        Returns:
            List of Experiment objects
        """
        experiments: List[Experiment] = []

        if not self.experiments_dir.exists():
            return experiments

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if exp_dir.is_dir():
                data_file = exp_dir / self.DATA_FILE_NAME
                if data_file.exists():
                    try:
                        experiment = self.load_experiment(exp_dir.name)
                        experiments.append(experiment)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Skip invalid experiments
                        continue

        return experiments

    def load_experiment(self, experiment_name: str) -> Experiment:
        """
        Load an experiment by name.

        Args:
            experiment_name: Experiment directory name

        Returns:
            Experiment object

        Raises:
            FileNotFoundError: If the experiment does not exist
            ValueError: If data.json is invalid
        """
        exp_path = self.experiments_dir / experiment_name
        data_file = exp_path / self.DATA_FILE_NAME

        if not data_file.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_name}")

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        if "name" not in data:
            raise ValueError(f"Required field 'name' missing in {data_file}")
        if "scenarios" not in data or not data["scenarios"]:
            raise ValueError(f"Required field 'scenarios' missing or empty in {data_file}")

        # Create config and convert to experiment
        config = ExperimentConfig.from_dict(data)
        return Experiment.from_config(config, path=exp_path)

    def get_experiment_by_index(self, index: int) -> Optional[Experiment]:
        """
        Get an experiment by index in the list.

        Args:
            index: Experiment index (1-based)

        Returns:
            Experiment object or None if not found
        """
        experiments = self.list_experiments()
        if 1 <= index <= len(experiments):
            return experiments[index - 1]
        return None

    def experiment_exists(self, experiment_name: str) -> bool:
        """
        Check whether an experiment exists.

        Args:
            experiment_name: Experiment directory name

        Returns:
            True if the experiment exists
        """
        exp_path = self.experiments_dir / experiment_name
        data_file = exp_path / self.DATA_FILE_NAME
        return data_file.exists()
