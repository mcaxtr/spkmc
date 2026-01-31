"""
Metrics extraction for SPKMC experiment analysis.

This module provides dataclasses and functions to extract key epidemic metrics
from simulation results for AI-powered analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ScenarioMetrics:
    """Extracted metrics from a single scenario result."""

    # Identification
    label: str
    network_type: str
    distribution: str

    # Network parameters
    nodes: int
    k_avg: Optional[float] = None
    exponent: Optional[float] = None

    # Distribution parameters
    shape: Optional[float] = None
    scale: Optional[float] = None
    mu: Optional[float] = None
    lambda_param: Optional[float] = None

    # Simulation parameters
    initial_perc: float = 0.01
    samples: int = 100
    num_runs: int = 1
    t_max: float = 20.0
    steps: int = 500

    # Key epidemic metrics (derived from S_val, I_val, R_val, time)
    peak_infection: float = 0.0
    peak_infection_time: float = 0.0
    final_outbreak_size: float = 0.0
    epidemic_duration: float = 0.0
    attack_rate: float = 0.0

    # Error bounds (if available)
    peak_infection_err: Optional[float] = None
    final_outbreak_size_err: Optional[float] = None


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an entire experiment."""

    name: str
    description: str  # The hypothesis being tested
    scenarios: List[ScenarioMetrics] = field(default_factory=list)

    # Cross-scenario comparisons (computed after scenarios are added)
    max_peak_scenario: str = ""
    min_peak_scenario: str = ""
    max_final_size_scenario: str = ""
    min_final_size_scenario: str = ""
    peak_variation_range: float = 0.0
    final_size_variation_range: float = 0.0

    def compute_comparisons(self) -> None:
        """Compute cross-scenario comparison metrics."""
        if not self.scenarios:
            return

        # Find max/min peak infection
        peaks = [(s.label, s.peak_infection) for s in self.scenarios]
        peaks_sorted = sorted(peaks, key=lambda x: x[1])
        self.min_peak_scenario = peaks_sorted[0][0]
        self.max_peak_scenario = peaks_sorted[-1][0]
        self.peak_variation_range = peaks_sorted[-1][1] - peaks_sorted[0][1]

        # Find max/min final outbreak size
        final_sizes = [(s.label, s.final_outbreak_size) for s in self.scenarios]
        final_sorted = sorted(final_sizes, key=lambda x: x[1])
        self.min_final_size_scenario = final_sorted[0][0]
        self.max_final_size_scenario = final_sorted[-1][0]
        self.final_size_variation_range = final_sorted[-1][1] - final_sorted[0][1]


def extract_scenario_metrics(result: Dict[str, Any]) -> ScenarioMetrics:
    """
    Extract key epidemic metrics from a scenario result.

    Args:
        result: Loaded JSON result with S_val, I_val, R_val, time, metadata

    Returns:
        ScenarioMetrics dataclass with computed metrics
    """
    metadata = result.get("metadata", {})

    # Extract time series data
    I_val = np.array(result.get("I_val", []))
    R_val = np.array(result.get("R_val", []))
    time = np.array(result.get("time", []))

    # Extract error data if available
    I_err = result.get("I_err")
    R_err = result.get("R_err")

    # Compute key metrics
    peak_infection = float(np.max(I_val)) if len(I_val) > 0 else 0.0
    peak_idx = int(np.argmax(I_val)) if len(I_val) > 0 else 0
    peak_infection_time = float(time[peak_idx]) if len(time) > peak_idx else 0.0
    final_outbreak_size = float(R_val[-1]) if len(R_val) > 0 else 0.0

    # Compute epidemic duration (time to reach 99% of final size)
    epidemic_duration = 0.0
    if len(R_val) > 0 and final_outbreak_size > 0:
        threshold = 0.99 * final_outbreak_size
        above_threshold = np.where(R_val >= threshold)[0]
        if len(above_threshold) > 0:
            epidemic_duration = float(time[above_threshold[0]])

    # Compute attack rate
    initial_perc = metadata.get("initial_perc", 0.01)
    susceptible_fraction = 1.0 - initial_perc
    attack_rate = final_outbreak_size / susceptible_fraction if susceptible_fraction > 0 else 0.0

    # Error bounds at peak and final
    peak_infection_err = None
    final_outbreak_size_err = None
    if I_err is not None and len(I_err) > peak_idx:
        peak_infection_err = float(I_err[peak_idx])
    if R_err is not None and len(R_err) > 0:
        final_outbreak_size_err = float(R_err[-1])

    return ScenarioMetrics(
        label=metadata.get("scenario_label", metadata.get("label", "Unknown")),
        network_type=metadata.get("network_type", "unknown"),
        distribution=metadata.get("distribution", "unknown"),
        nodes=metadata.get("N", 0),
        k_avg=metadata.get("k_avg"),
        exponent=metadata.get("exponent"),
        shape=metadata.get("shape"),
        scale=metadata.get("scale"),
        mu=metadata.get("mu"),
        lambda_param=metadata.get("lambda"),
        initial_perc=initial_perc,
        samples=metadata.get("samples", 100),
        num_runs=metadata.get("num_runs", 1),
        t_max=metadata.get("t_max", float(time[-1]) if len(time) > 0 else 20.0),
        steps=metadata.get("steps", len(time)),
        peak_infection=peak_infection,
        peak_infection_time=peak_infection_time,
        final_outbreak_size=final_outbreak_size,
        epidemic_duration=epidemic_duration,
        attack_rate=attack_rate,
        peak_infection_err=peak_infection_err,
        final_outbreak_size_err=final_outbreak_size_err,
    )


def extract_experiment_metrics(
    experiment_name: str, experiment_description: str, results: List[Dict[str, Any]]
) -> ExperimentMetrics:
    """
    Extract and aggregate metrics across all scenarios in an experiment.

    Args:
        experiment_name: Name of the experiment
        experiment_description: The hypothesis being tested
        results: List of loaded result dictionaries

    Returns:
        ExperimentMetrics with all scenario metrics and comparisons
    """
    metrics = ExperimentMetrics(
        name=experiment_name,
        description=experiment_description,
    )

    for result in results:
        try:
            scenario_metrics = extract_scenario_metrics(result)
            metrics.scenarios.append(scenario_metrics)
        except Exception:
            # Skip malformed results
            continue

    # Compute cross-scenario comparisons
    metrics.compute_comparisons()

    return metrics
