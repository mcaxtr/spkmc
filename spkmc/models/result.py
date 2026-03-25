"""
Simulation result models for SPKMC.

This module contains the SimulationResult dataclass for storing
and working with simulation output data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SimulationResult:
    """
    Unified result container for simulation output.

    This class stores the results from a single scenario execution,
    including SIR values, error data (if available), metadata, and execution info.
    """

    # Core SIR data (required)
    S_val: np.ndarray
    I_val: np.ndarray
    R_val: np.ndarray
    time: np.ndarray

    # Error data (optional, for runs with multiple samples)
    S_err: Optional[np.ndarray] = None
    I_err: Optional[np.ndarray] = None
    R_err: Optional[np.ndarray] = None

    # Metadata and context
    metadata: Dict[str, Any] = field(default_factory=dict)
    scenario_label: str = ""
    output_path: Optional[str] = None
    execution_time: float = 0.0

    # Microscopic data (per-node infection/recovery times, saved separately as .npz)
    microscopic_data: Optional[List[Dict[str, Any]]] = None

    @property
    def has_error(self) -> bool:
        """Check if error data is available."""
        return self.S_err is not None and self.I_err is not None and self.R_err is not None

    @property
    def is_empty(self) -> bool:
        """Check if result has no data (failed scenario)."""
        return len(self.I_val) == 0 or len(self.time) == 0

    @property
    def peak_infected(self) -> float:
        """Get maximum infected proportion. Returns 0.0 for empty results."""
        if self.is_empty:
            return 0.0
        return float(np.max(self.I_val))

    @property
    def peak_time(self) -> float:
        """Get time at which peak infection occurs. Returns 0.0 for empty results."""
        if self.is_empty:
            return 0.0
        return float(self.time[np.argmax(self.I_val)])

    @property
    def final_recovered(self) -> float:
        """Get final recovered proportion. Returns 0.0 for empty results."""
        if self.is_empty:
            return 0.0
        return float(self.R_val[-1])

    @property
    def is_equilibrated(self) -> bool:
        """Check if the simulation reached equilibrium.

        Compares the R(t) curve over its final 10% of time steps.
        Returns False when R is still changing significantly,
        which usually means t_max is too short for the chosen parameters.
        """
        if self.is_empty or len(self.R_val) < 10:
            return True
        n = max(2, len(self.R_val) // 10)
        delta = abs(float(self.R_val[-1]) - float(self.R_val[-n]))
        return delta < 0.01

    @property
    def final_susceptible(self) -> float:
        """Get final susceptible proportion. Returns 0.0 for empty results."""
        if self.is_empty:
            return 0.0
        return float(self.S_val[-1])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format for serialization.

        Returns:
            Dictionary suitable for JSON serialization
        """
        result: Dict[str, Any] = {
            "S_val": (
                self.S_val.tolist() if isinstance(self.S_val, np.ndarray) else list(self.S_val)
            ),
            "I_val": (
                self.I_val.tolist() if isinstance(self.I_val, np.ndarray) else list(self.I_val)
            ),
            "R_val": (
                self.R_val.tolist() if isinstance(self.R_val, np.ndarray) else list(self.R_val)
            ),
            "time": self.time.tolist() if isinstance(self.time, np.ndarray) else list(self.time),
            "metadata": self.metadata,
        }

        if (
            self.has_error
            and self.S_err is not None
            and self.I_err is not None
            and self.R_err is not None
        ):
            result["S_err"] = (
                self.S_err.tolist() if isinstance(self.S_err, np.ndarray) else list(self.S_err)
            )
            result["I_err"] = (
                self.I_err.tolist() if isinstance(self.I_err, np.ndarray) else list(self.I_err)
            )
            result["R_err"] = (
                self.R_err.tolist() if isinstance(self.R_err, np.ndarray) else list(self.R_err)
            )

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], scenario_label: str = "") -> "SimulationResult":
        """
        Create SimulationResult from dictionary (loaded from JSON).

        Args:
            data: Dictionary with result data
            scenario_label: Optional label for the scenario

        Returns:
            SimulationResult instance
        """
        # Extract error data if present
        S_err = np.array(data["S_err"]) if "S_err" in data else None
        I_err = np.array(data["I_err"]) if "I_err" in data else None
        R_err = np.array(data["R_err"]) if "R_err" in data else None

        # Get scenario label from metadata if not provided
        metadata = data.get("metadata", {})
        if not scenario_label:
            scenario_label = metadata.get("scenario_label", "")

        return cls(
            S_val=np.array(data["S_val"]),
            I_val=np.array(data["I_val"]),
            R_val=np.array(data["R_val"]),
            time=np.array(data["time"]),
            S_err=S_err,
            I_err=I_err,
            R_err=R_err,
            metadata=metadata,
            scenario_label=scenario_label,
        )

    @classmethod
    def from_simulation_output(
        cls,
        result: Dict[str, Any],
        time_steps: np.ndarray,
        scenario_label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        output_path: Optional[str] = None,
    ) -> "SimulationResult":
        """
        Create SimulationResult from raw simulation output.

        Args:
            result: Raw output from SPKMC.run_simulation()
            time_steps: Time step array
            scenario_label: Label for this scenario
            metadata: Optional metadata dict
            execution_time: Execution time in seconds
            output_path: Path where result was saved

        Returns:
            SimulationResult instance
        """
        # Extract error data if present
        has_error = result.get("has_error", False)
        S_err = result.get("S_err") if has_error else None
        I_err = result.get("I_err") if has_error else None
        R_err = result.get("R_err") if has_error else None

        # Extract microscopic data if present (not serialized to JSON)
        microscopic_runs = result.get("microscopic")

        return cls(
            S_val=np.array(result["S_val"]),
            I_val=np.array(result["I_val"]),
            R_val=np.array(result["R_val"]),
            time=time_steps,
            S_err=np.array(S_err) if S_err is not None else None,
            I_err=np.array(I_err) if I_err is not None else None,
            R_err=np.array(R_err) if R_err is not None else None,
            metadata=metadata or {},
            scenario_label=scenario_label,
            execution_time=execution_time,
            output_path=output_path,
            microscopic_data=microscopic_runs,
        )

    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate and return key statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "peak_infected": self.peak_infected,
            "peak_time": self.peak_time,
            "final_recovered": self.final_recovered,
            "final_susceptible": self.final_susceptible,
            "time_points": len(self.time),
        }
