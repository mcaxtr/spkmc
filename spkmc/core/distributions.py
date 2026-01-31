"""
Distribution classes for the SPKMC algorithm.

This module contains implementations of different probability distributions
used by the SPKMC algorithm to model recovery and infection times.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from spkmc.core.transmissibility import TransmissibilityCalculator

from spkmc.utils.numba_utils import (
    compute_infection_times_exponential,
    gamma_sampling,
    get_weight_exponential,
)


class Distribution(ABC):
    """Abstract base class for probability distributions used in SPKMC."""

    @abstractmethod
    def get_recovery_weights(self, size: int) -> np.ndarray:
        """
        Generate recovery weights for each node.

        Args:
            size: Number of nodes

        Returns:
            Array with recovery weights
        """
        pass

    @abstractmethod
    def get_infection_times(self, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Calculate infection times for each edge.

        Args:
            recovery_times: Recovery times for each node
            edges: Graph edges as a matrix (u, v)

        Returns:
            Array with infection times
        """
        pass

    @abstractmethod
    def get_distribution_name(self) -> str:
        """
        Return the distribution name.

        Returns:
            Distribution name
        """
        pass

    @abstractmethod
    def get_params_string(self) -> str:
        """
        Return a string with distribution parameters for filenames.

        Returns:
            String with parameters
        """
        pass

    def get_params_dict(self) -> dict:
        """
        Return a dictionary of distribution parameters.

        Returns:
            Dictionary with parameters
        """
        return {}

    def get_transmissibility_calculator(self) -> TransmissibilityCalculator:
        """
        Return a transmissibility calculator for this distribution.

        The calculator can be used to compute mean transmissibility T̄,
        critical thresholds, and perform epidemic analysis.

        Returns:
            TransmissibilityCalculator instance configured for this distribution
        """
        raise NotImplementedError(
            f"get_transmissibility_calculator not implemented for {self.__class__.__name__}"
        )


class GammaDistribution(Distribution):
    """Gamma distribution implementation for SPKMC."""

    def __init__(self, shape: float, scale: float, lmbd: float = 1.0):
        """
        Initialize the Gamma distribution.

        Args:
            shape: Shape parameter of the Gamma distribution
            scale: Scale parameter of the Gamma distribution
            lmbd: Lambda parameter for infection times (default: 1.0)
        """
        self.shape = shape
        self.scale = scale
        self.lmbd = lmbd

    def get_recovery_weights(self, size: int) -> np.ndarray:
        """
        Generate recovery weights using the Gamma distribution.

        Args:
            size: Number of nodes

        Returns:
            Array with recovery weights
        """
        result: np.ndarray = np.asarray(gamma_sampling(self.shape, self.scale, size))
        return result

    def get_infection_times(self, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Calculate infection times using the Gamma distribution.

        Args:
            recovery_times: Recovery times for each node
            edges: Graph edges as a matrix (u, v)

        Returns:
            Array with infection times
        """
        # Note: Currently using exponential infection times even with gamma recovery
        result: np.ndarray = np.asarray(
            compute_infection_times_exponential(self.lmbd, recovery_times, edges)
        )
        return result

    def get_distribution_name(self) -> str:
        """
        Return the distribution name.

        Returns:
            Distribution name
        """
        return "gamma"

    def get_params_string(self) -> str:
        """
        Return a string with distribution parameters for filenames.

        Returns:
            String with parameters (shape, scale, and lambda formatted for filenames)
        """
        shape_str = f"{self.shape:.4f}".rstrip("0").rstrip(".")
        scale_str = f"{self.scale:.4f}".rstrip("0").rstrip(".")
        lmbd_str = f"{self.lmbd:.4f}".rstrip("0").rstrip(".")
        return f"sh{shape_str}_sc{scale_str}_l{lmbd_str}"

    def get_params_dict(self) -> dict:
        """
        Return a dictionary with distribution parameters.

        Returns:
            Dictionary with parameters
        """
        return {
            "type": "gamma",
            "shape": self.shape,
            "scale": self.scale,
            "lambda": self.lmbd,
        }

    def get_transmissibility_calculator(self) -> TransmissibilityCalculator:
        """
        Return a transmissibility calculator for Gamma recovery + Exponential infection.

        Returns:
            GammaExponentialTransmissibility instance
        """
        from spkmc.core.transmissibility import GammaExponentialTransmissibility

        return GammaExponentialTransmissibility(shape=self.shape, scale=self.scale, beta=self.lmbd)


class ExponentialDistribution(Distribution):
    """Exponential distribution implementation for SPKMC."""

    def __init__(self, mu: float, lmbd: float):
        """
        Initialize the Exponential distribution.

        Args:
            mu: Mu parameter for recovery times
            lmbd: Lambda parameter for infection times
        """
        self.mu = mu
        self.lmbd = lmbd

    def get_recovery_weights(self, size: int) -> np.ndarray:
        """
        Generate recovery weights using the Exponential distribution.

        Args:
            size: Number of nodes

        Returns:
            Array with recovery weights
        """
        result: np.ndarray = np.asarray(get_weight_exponential(self.mu, size))
        return result

    def get_infection_times(self, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Calculate infection times using the Exponential distribution.

        Args:
            recovery_times: Recovery times for each node
            edges: Graph edges as a matrix (u, v)

        Returns:
            Array with infection times
        """
        result: np.ndarray = np.asarray(
            compute_infection_times_exponential(self.lmbd, recovery_times, edges)
        )
        return result

    def get_distribution_name(self) -> str:
        """
        Return the distribution name.

        Returns:
            Distribution name
        """
        return "exponential"

    def get_params_string(self) -> str:
        """
        Return a string with distribution parameters for filenames.

        Returns:
            String with parameters (mu and lambda formatted for filenames)
        """
        mu_str = f"{self.mu:.4f}".rstrip("0").rstrip(".")
        lmbd_str = f"{self.lmbd:.4f}".rstrip("0").rstrip(".")
        return f"mu{mu_str}_l{lmbd_str}"

    def get_params_dict(self) -> dict:
        """
        Return a dictionary with distribution parameters.

        Returns:
            Dictionary with parameters
        """
        return {
            "type": "exponential",
            "mu": self.mu,
            "lambda": self.lmbd,
        }

    def get_transmissibility_calculator(self) -> TransmissibilityCalculator:
        """
        Return a transmissibility calculator for Exponential recovery + Exponential infection.

        Returns:
            ExponentialExponentialTransmissibility instance
        """
        from spkmc.core.transmissibility import ExponentialExponentialTransmissibility

        return ExponentialExponentialTransmissibility(beta=self.lmbd, gamma=self.mu)


def create_distribution(dist_type: str, **kwargs: Any) -> Distribution:
    """
    Create a distribution instance based on the given type and parameters.

    Args:
        dist_type: Distribution type ('gamma' or 'exponential')
        **kwargs: Distribution-specific parameters

    Returns:
        Instance of the requested distribution

    Raises:
        ValueError: If the distribution type is unknown
    """
    if dist_type.lower() == "gamma":
        shape = kwargs.get("shape", 2.0)
        scale = kwargs.get("scale", 1.0)
        lmbd = kwargs.get("lambda", 1.0)
        return GammaDistribution(shape=shape, scale=scale, lmbd=lmbd)

    elif dist_type.lower() == "exponential":
        mu = kwargs.get("mu", 1.0)
        lmbd = kwargs.get("lambda", 1.0)
        return ExponentialDistribution(mu=mu, lmbd=lmbd)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
