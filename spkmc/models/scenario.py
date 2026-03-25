"""
Scenario models for SPKMC simulations.

This module contains the Scenario and ScenarioOverride Pydantic models
used for configuring and executing simulations.
"""

import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScenarioOverride(BaseModel):
    """
    Partial scenario - just label and overrides (used in data.json).

    Used to define scenario-specific overrides that merge with global parameters.
    All fields except label are optional - they override the global parameters.
    """

    label: str
    network: Optional[Literal["er", "sf", "cg", "rrn"]] = None
    distribution: Optional[Literal["gamma", "exponential", "weibull"]] = None
    nodes: Optional[int] = Field(gt=0, default=None)
    samples: Optional[int] = Field(gt=0, default=None)
    k_avg: Optional[float] = Field(gt=0, default=None)
    exponent: Optional[float] = Field(gt=0, default=None)
    num_runs: Optional[int] = Field(gt=0, default=None)
    shape: Optional[float] = Field(gt=0, default=None)
    scale: Optional[float] = Field(gt=0, default=None)
    mu: Optional[float] = Field(gt=0, default=None)
    lambda_param: Optional[float] = Field(alias="lambda", gt=0, default=None)
    t_max: Optional[float] = Field(gt=0, default=None)
    steps: Optional[int] = Field(gt=0, default=None)
    initial_perc: Optional[float] = Field(gt=0, le=1, default=None)
    microscopic: Optional[bool] = None

    model_config = ConfigDict(populate_by_name=True)


class Scenario(BaseModel):
    """
    A complete simulation scenario with all required parameters.

    Used by:
    - `spkmc run` command (single scenario)
    - `spkmc experiments` command (merged from parameters + scenario override)

    This class also provides path generation methods for consistent file storage.
    """

    RUNS_BASE: ClassVar[str] = "data/runs"
    EXPERIMENTS_BASE: ClassVar[str] = "data/experiments"

    label: str
    network: Literal["er", "sf", "cg", "rrn"]
    distribution: Literal["gamma", "exponential", "weibull"]
    nodes: int = Field(gt=0)
    samples: int = Field(gt=0)

    # Network parameters
    k_avg: Optional[float] = Field(gt=0, default=None)
    exponent: Optional[float] = Field(gt=0, default=None)
    num_runs: int = Field(gt=0, default=1)

    # Distribution parameters
    shape: Optional[float] = Field(gt=0, default=None)
    scale: Optional[float] = Field(gt=0, default=None)
    mu: Optional[float] = Field(gt=0, default=None)
    lambda_param: float = Field(alias="lambda", gt=0)

    # Simulation parameters
    t_max: float = Field(gt=0)
    steps: int = Field(gt=0)
    initial_perc: float = Field(gt=0, le=1)

    # Context fields (for execution engine)
    experiment_name: Optional[str] = None
    output_path: Optional[str] = None
    microscopic: bool = False

    model_config = ConfigDict(populate_by_name=True)

    @staticmethod
    def normalize_label(label: str) -> str:
        """Normalize a label for filename (lowercase, spaces/dashes to underscores)."""
        normalized = label.lower().strip()
        normalized = re.sub(r"[\s\-]+", "_", normalized)
        normalized = re.sub(r"[^\w]", "", normalized)
        return normalized

    @property
    def normalized_label(self) -> str:
        """Label normalized for filename (lowercase, spaces/dashes to underscores)."""
        return self.normalize_label(self.label)

    @model_validator(mode="after")
    def validate_network_params(self) -> "Scenario":
        """Validate network-specific parameters."""
        if self.network in ("er", "sf", "rrn") and self.k_avg is None:
            raise ValueError(f"k_avg required for network type '{self.network}'")
        if self.network == "sf" and self.exponent is None:
            raise ValueError("exponent required for scale-free network")
        return self

    @model_validator(mode="after")
    def validate_distribution_params(self) -> "Scenario":
        """Validate distribution-specific parameters."""
        if self.distribution == "gamma":
            if self.shape is None or self.scale is None:
                raise ValueError("shape and scale required for gamma distribution")
        elif self.distribution == "exponential":
            if self.mu is None:
                raise ValueError("mu required for exponential distribution")
        elif self.distribution == "weibull":
            if self.shape is None or self.scale is None:
                raise ValueError("shape and scale required for weibull distribution")
        return self

    def total_samples(self) -> int:
        """
        Calculate total number of samples for progress tracking.

        All network types (including complete graphs) now run num_runs * samples.

        Returns:
            Total number of samples to be executed
        """
        return self.num_runs * self.samples

    @classmethod
    def from_cli_args(
        cls,
        network_type: str,
        distribution: str,
        nodes: int,
        samples: int,
        num_runs: int = 1,
        k_avg: Optional[float] = None,
        exponent: Optional[float] = None,
        shape: Optional[float] = None,
        scale: Optional[float] = None,
        mu: Optional[float] = None,
        lambda_param: Optional[float] = None,
        t_max: float = 10.0,
        steps: int = 100,
        initial_perc: float = 0.01,
        experiment_name: Optional[str] = None,
        output_path: Optional[str] = None,
        microscopic: bool = False,
        **kwargs: Any,
    ) -> "Scenario":
        """
        Create Scenario from CLI arguments.

        Args:
            network_type: Network type (er, sf, cg, rrn)
            distribution: Distribution type (gamma, exponential)
            nodes: Number of nodes
            samples: Number of samples
            num_runs: Number of runs for averaging
            k_avg: Average degree
            exponent: Power-law exponent (for sf networks)
            shape: Gamma distribution shape
            scale: Gamma distribution scale
            mu: Exponential distribution mu
            lambda_param: Infection rate lambda
            t_max: Maximum simulation time
            steps: Number of time steps
            initial_perc: Initial percentage of infected
            experiment_name: Optional experiment name for context
            output_path: Optional explicit output path
            **kwargs: Additional parameters (ignored)

        Returns:
            A fully configured Scenario instance
        """
        label = f"{network_type}_{distribution}_{nodes}n_{samples}s"

        # Build kwargs for Pydantic model
        scenario_kwargs: Dict[str, Any] = {
            "label": label,
            "network": network_type,
            "distribution": distribution,
            "nodes": nodes,
            "samples": samples,
            "num_runs": num_runs,
            "t_max": t_max,
            "steps": steps,
            "initial_perc": initial_perc,
        }

        # Add lambda (required)
        if lambda_param is not None:
            scenario_kwargs["lambda"] = lambda_param
        else:
            raise ValueError("lambda_param is required")

        # Add optional parameters
        if k_avg is not None:
            scenario_kwargs["k_avg"] = k_avg
        if exponent is not None:
            scenario_kwargs["exponent"] = exponent
        if shape is not None:
            scenario_kwargs["shape"] = shape
        if scale is not None:
            scenario_kwargs["scale"] = scale
        if mu is not None:
            scenario_kwargs["mu"] = mu
        if experiment_name is not None:
            scenario_kwargs["experiment_name"] = experiment_name
        if output_path is not None:
            scenario_kwargs["output_path"] = output_path
        if microscopic:
            scenario_kwargs["microscopic"] = microscopic

        return cls(**scenario_kwargs)

    @classmethod
    def from_merged(cls, parameters: Dict[str, Any], override: ScenarioOverride) -> "Scenario":
        """
        Create Scenario by merging global parameters with scenario override.

        Args:
            parameters: Global default parameters from experiment config
            override: Scenario-specific overrides

        Returns:
            A fully configured Scenario with merged parameters
        """
        merged = {**parameters}  # Start with global parameters
        # Override with scenario-specific values (exclude None values)
        for key, value in override.model_dump(by_alias=True).items():
            if value is not None:
                merged[key] = value
        return cls(**merged)

    def get_run_path(self, format: str = "json") -> str:
        """
        Generate path for individual run.

        Example: data/runs/gamma/ER/results_1000_50_k10_sh2_sc05_lam05.json

        Args:
            format: Output format (json, csv, excel, md, html)

        Returns:
            Full path string for the result file
        """
        ext_map = {"json": ".json", "csv": ".csv", "excel": ".xlsx", "md": ".md", "html": ".html"}
        ext = ext_map.get(format, ".json")
        base = Path(self.RUNS_BASE) / self.distribution / self.network.upper()
        params = self._build_params_string()
        return str(base / f"results_{params}{ext}")

    def get_experiment_path(self, experiment_name: str, format: str = "json") -> str:
        """
        Generate path for experiment scenario.

        Example: data/experiments/infection_rate_study/high_infection.json

        Args:
            experiment_name: Name of the experiment
            format: Output format (json, csv, excel, md, html)

        Returns:
            Full path string for the result file
        """
        ext_map = {"json": ".json", "csv": ".csv", "excel": ".xlsx", "md": ".md", "html": ".html"}
        ext = ext_map.get(format, ".json")
        exp_normalized = self._normalize(experiment_name)
        base = Path(self.EXPERIMENTS_BASE) / exp_normalized
        return str(base / f"{self.normalized_label}{ext}")

    def _build_params_string(self) -> str:
        """Build parameter string for run filename.

        Uses 'p' to represent decimal point for unambiguous encoding.
        Example: 0.05 -> 0p05, 2.75 -> 2p75
        """
        parts = [str(self.nodes), str(self.samples)]
        if self.k_avg is not None:
            parts.append(f"k{int(self.k_avg)}")
        if self.exponent is not None:
            parts.append(f"exp{str(self.exponent).replace('.', 'p')}")
        if self.shape is not None:
            parts.append(f"sh{str(self.shape).replace('.', 'p')}")
        if self.scale is not None:
            parts.append(f"sc{str(self.scale).replace('.', 'p')}")
        if self.mu is not None:
            parts.append(f"mu{str(self.mu).replace('.', 'p')}")
        parts.append(f"lam{str(self.lambda_param).replace('.', 'p')}")
        return "_".join(parts)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for filename/directory."""
        normalized = text.lower().strip()
        normalized = re.sub(r"[\s\-]+", "_", normalized)
        return re.sub(r"[^\w]", "", normalized)

    def to_simulation_params(self) -> Dict[str, Any]:
        """
        Convert scenario to simulation parameters dict.

        Returns:
            Dictionary of parameters suitable for SPKMC.run_simulation()
        """
        params: Dict[str, Any] = {
            "N": self.nodes,
            "samples": self.samples,
            "initial_perc": self.initial_perc,
            "num_runs": self.num_runs,
        }

        if self.k_avg is not None:
            params["k_avg"] = self.k_avg

        if self.exponent is not None:
            params["exponent"] = self.exponent

        return params

    def to_distribution_params(self) -> Dict[str, Any]:
        """
        Get distribution parameters dict.

        Returns:
            Dictionary of distribution parameters for create_distribution()
        """
        return {
            "shape": self.shape,
            "scale": self.scale,
            "mu": self.mu,
            "lambda": self.lambda_param,
        }

    def to_metadata(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate metadata dict for result storage.

        Args:
            experiment_name: Optional experiment name to include

        Returns:
            Dictionary of metadata suitable for result files
        """
        # Use the experiment_name field if not explicitly provided
        exp_name = experiment_name or self.experiment_name

        metadata: Dict[str, Any] = {
            "network": self.network,
            "distribution": self.distribution,
            "N": self.nodes,
            "samples": self.samples,
            "num_runs": self.num_runs,
            "t_max": self.t_max,
            "steps": self.steps,
            "initial_perc": self.initial_perc,
            "lambda": self.lambda_param,
            "scenario_label": self.label,
        }

        if self.k_avg is not None:
            metadata["k_avg"] = self.k_avg
        if self.exponent is not None:
            metadata["exponent"] = self.exponent
        if self.shape is not None:
            metadata["shape"] = self.shape
        if self.scale is not None:
            metadata["scale"] = self.scale
        if self.mu is not None:
            metadata["mu"] = self.mu
        if exp_name:
            metadata["experiment_name"] = exp_name

        return metadata
