"""
Experiment management for the SPKMC algorithm.

This module contains functions and classes for managing experiments,
including discovery, loading, validation, and execution.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PlotConfig:
    """Configuration for plotting results."""

    title: Optional[str] = None
    xlabel: str = "Time"
    ylabel: str = "Proportion of Individuals"
    legend_position: str = "best"
    figsize: Tuple[float, float] = (10, 6)
    colors: Dict[str, str] = field(default_factory=lambda: {"S": "blue", "I": "red", "R": "green"})
    states_to_plot: List[str] = field(default_factory=lambda: ["S", "I", "R"])
    dpi: int = 300
    grid: bool = True
    grid_alpha: float = 0.3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotConfig":
        """
        Create PlotConfig from a dictionary.

        Args:
            data: Dictionary with plot settings

        Returns:
            PlotConfig instance
        """
        figsize = data.get("figsize", [10, 6])
        return cls(
            title=data.get("title"),
            xlabel=data.get("xlabel", "Time"),
            ylabel=data.get("ylabel", "Proportion of Individuals"),
            legend_position=data.get("legend_position", "best"),
            figsize=tuple(figsize) if isinstance(figsize, list) else figsize,
            colors=data.get("colors", {"S": "blue", "I": "red", "R": "green"}),
            states_to_plot=data.get("states_to_plot", ["S", "I", "R"]),
            dpi=data.get("dpi", 300),
            grid=data.get("grid", True),
            grid_alpha=data.get("grid_alpha", 0.3),
        )


@dataclass
class Experiment:
    """Represents an SPKMC experiment."""

    name: str
    path: Path
    description: Optional[str] = None
    plot_config: PlotConfig = field(default_factory=PlotConfig)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def results_dir(self) -> Path:
        """Return the results directory path."""
        return self.path / "results"

    @property
    def has_results(self) -> bool:
        """Check whether the experiment has results."""
        return self.results_dir.exists() and any(self.results_dir.glob("*.json"))

    @property
    def result_count(self) -> int:
        """Return the number of result files."""
        if not self.results_dir.exists():
            return 0
        # Count all JSON files except comparison metadata
        return len(
            [f for f in self.results_dir.glob("*.json") if not f.name.startswith("comparison")]
        )

    def clean_results(self) -> None:
        """Remove all results from the experiment."""
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)

    def ensure_results_dir(self) -> Path:
        """Ensure the results directory exists."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir


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

        # Filter out comment objects from scenarios
        scenarios = [s for s in data["scenarios"] if not s.get("_comment")]

        # Parse plot config
        plot_config = PlotConfig.from_dict(data.get("plot", {}))

        # Extract global parameters (used as defaults for scenarios)
        global_params = data.get("parameters", {})

        # Normalize parameter key names (data.json format -> internal format)
        key_mapping = {
            "time_max": "t_max",
            "time_points": "steps",
        }

        def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize parameter keys to internal format."""
            normalized = {}
            for key, value in params.items():
                normalized_key = key_mapping.get(key, key)
                normalized[normalized_key] = value
            return normalized

        normalized_global = normalize_params(global_params)

        # Merge global parameters into each scenario (scenario values override global)
        merged_scenarios = []
        for scenario in scenarios:
            normalized_scenario = normalize_params(scenario)
            merged = {**normalized_global, **normalized_scenario}
            merged_scenarios.append(merged)

        return Experiment(
            name=data["name"],
            path=exp_path,
            description=data.get("description"),
            plot_config=plot_config,
            scenarios=merged_scenarios,
            parameters=global_params,
        )

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
