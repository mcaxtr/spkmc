"""
Experiment models for SPKMC.

This module contains the Experiment and ExperimentConfig classes
for managing multi-scenario experiment configurations.
"""

import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from spkmc.models.config import PlotConfig
from spkmc.models.scenario import Scenario, ScenarioOverride


class ExperimentConfig(BaseModel):
    """
    Raw experiment configuration as loaded from data.json.

    This represents the structure of the data.json file before
    parameters are merged into scenarios.
    """

    name: str
    description: Optional[str] = None
    plot: PlotConfig = Field(default_factory=PlotConfig)
    parameters: Dict[str, Any]  # Global default parameters
    scenarios: List[ScenarioOverride] = Field(min_length=1)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """
        Create ExperimentConfig from a dictionary (loaded from data.json).

        Handles key normalization and comment filtering.

        Args:
            data: Dictionary loaded from data.json

        Returns:
            ExperimentConfig instance
        """
        # Key mapping for parameter normalization
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

        # Normalize global parameters
        parameters = normalize_params(data.get("parameters", {}))

        # Filter out comment objects and normalize scenario parameters
        raw_scenarios = data.get("scenarios", [])
        scenarios = []
        for s in raw_scenarios:
            if not s.get("_comment"):
                normalized_s = normalize_params(s)
                scenarios.append(ScenarioOverride(**normalized_s))

        # Parse plot config
        plot = PlotConfig.from_dict(data.get("plot", {}))

        return cls(
            name=data["name"],
            description=data.get("description"),
            plot=plot,
            parameters=parameters,
            scenarios=scenarios,
        )


class Experiment(BaseModel):
    """
    Resolved experiment with fully merged scenarios.

    This is the result of merging ExperimentConfig.parameters into each scenario.
    """

    name: str
    description: Optional[str] = None
    plot: PlotConfig = Field(default_factory=PlotConfig)
    plot_config: Optional[PlotConfig] = Field(
        default=None, exclude=True
    )  # Alias for backward compat
    scenarios: List[Any] = Field(min_length=1)  # Accept raw dicts or Scenario objects
    path: Optional[Path] = None
    results_base_dir: Optional[Path] = None  # Custom results base directory

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        # Handle plot_config alias
        if "plot_config" in data and "plot" not in data:
            data["plot"] = data.pop("plot_config")
        elif "plot_config" in data:
            data.pop("plot_config")  # Remove if plot is also provided
        super().__init__(**data)

    @property
    def normalized_name(self) -> str:
        """Name normalized for directory (lowercase, spaces to underscores)."""
        normalized = self.name.lower().strip()
        normalized = re.sub(r"[\s\-]+", "_", normalized)
        normalized = re.sub(r"[^\w]", "", normalized)
        return normalized

    @property
    def results_dir(self) -> Path:
        """Return the results directory path."""
        base = self.results_base_dir or Path(Scenario.EXPERIMENTS_BASE)
        return base / self.normalized_name

    # Supported result file extensions (class variable, not a Pydantic field)
    SUPPORTED_EXTENSIONS: ClassVar[List[str]] = ["*.json", "*.csv", "*.xlsx", "*.md", "*.html"]

    @property
    def has_results(self) -> bool:
        """Check whether the experiment has results."""
        if not self.results_dir.exists():
            return False
        for ext in self.SUPPORTED_EXTENSIONS:
            if any(self.results_dir.glob(ext)):
                return True
        return False

    @property
    def result_count(self) -> int:
        """Return the number of result files."""
        if not self.results_dir.exists():
            return 0
        # Count all result files except comparison metadata
        count = 0
        for ext in self.SUPPORTED_EXTENSIONS:
            count += len(
                [f for f in self.results_dir.glob(ext) if not f.name.startswith("comparison")]
            )
        return count

    def total_samples(self) -> int:
        """
        Calculate total samples across all scenarios for progress tracking.

        Returns:
            Total number of samples to be executed
        """
        total = 0
        for scenario in self.scenarios:
            if isinstance(scenario, Scenario):
                total += scenario.total_samples()
            elif isinstance(scenario, dict):
                # Handle dict scenarios
                num_runs = scenario.get("num_runs", 1)
                samples = scenario.get("samples", 0)
                network_type = scenario.get("network", "")
                if network_type == "cg":
                    total += samples
                else:
                    total += num_runs * samples
        return total

    def ensure_results_dir(self) -> Path:
        """Ensure the results directory exists."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir

    def clean_results(self) -> None:
        """Remove all results from the experiment."""
        import shutil

        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)

    @classmethod
    def from_config(cls, config: ExperimentConfig, path: Optional[Path] = None) -> "Experiment":
        """
        Create Experiment from config, merging parameters into each scenario.

        Args:
            config: ExperimentConfig with raw data
            path: Optional path to the experiment directory

        Returns:
            Experiment with fully resolved scenarios
        """
        scenarios = [
            Scenario.from_merged(config.parameters, override) for override in config.scenarios
        ]
        return cls(
            name=config.name,
            description=config.description,
            plot=config.plot,
            scenarios=scenarios,
            path=path,
        )

    @classmethod
    def from_single_scenario(
        cls,
        scenario: Scenario,
        name: Optional[str] = None,
        results_base_dir: Optional[Path] = None,
    ) -> "Experiment":
        """
        Create an Experiment from a single Scenario (for unified run command).

        Args:
            scenario: The single scenario to run
            name: Optional experiment name (defaults to "single_run")
            results_base_dir: Custom base directory for results

        Returns:
            Experiment with single scenario
        """
        return cls(
            name=name or "single_run",
            scenarios=[scenario],
            results_base_dir=results_base_dir,
        )
