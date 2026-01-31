"""
Transmissibility calculations for SPKMC epidemic simulations.

This module implements the mean transmissibility T̄ calculations from:
Böttcher & Antulov-Fantulin, "Unifying continuous, discrete, and hybrid
susceptible-infected-recovered processes on networks",
Phys. Rev. Research 2, 033121 (2020).

Mean transmissibility T̄ is the probability that an infected node transmits
the disease to an adjacent susceptible node before recovering. It provides
a unifying description of epidemic dynamics across different distributions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.stats import gamma as gamma_dist


@dataclass
class EpidemicThreshold:
    """Results of epidemic threshold analysis."""

    transmissibility: float  # T̄
    critical_transmissibility: float  # T̄_c
    is_supercritical: bool  # T̄ > T̄_c
    expected_outbreak_size: Optional[float] = None  # S(T̄) if supercritical

    def __str__(self) -> str:
        status = (
            "SUPERCRITICAL (epidemic spreads)"
            if self.is_supercritical
            else "SUBCRITICAL (epidemic dies out)"
        )
        result = f"T̄ = {self.transmissibility:.4f}, T̄_c = {self.critical_transmissibility:.4f}\n"
        result += f"Status: {status}"
        if self.expected_outbreak_size is not None:
            result += f"\nExpected outbreak size: {self.expected_outbreak_size:.2%}"
        return result


class TransmissibilityCalculator(ABC):
    """
    Abstract base class for transmissibility calculations.

    Mean transmissibility is defined as (Paper Eq. 6):

        T̄ = ∫₀^∞ φ(τ) [∫₀^τ ψ(τ')dτ'] dτ

    Where:
        φ(τ) = recovery time PDF
        ψ(τ) = infection time PDF
        ∫₀^τ ψ(τ')dτ' = Ψ(τ) = infection CDF (prob. infection occurred before τ)
    """

    @abstractmethod
    def calculate(self) -> float:
        """
        Calculate the mean transmissibility T̄.

        Returns:
            Mean transmissibility value in [0, 1]
        """
        pass

    @abstractmethod
    def get_params_dict(self) -> Dict[str, Any]:
        """Return dictionary of distribution parameters."""
        pass

    def get_critical_transmissibility(self, k_mean: float, k2_mean: float) -> float:
        """
        Calculate the critical transmissibility T̄_c for epidemic spread.

        From bond percolation theory (Paper Eq. 7):
            T̄_c = ⟨k⟩ / (⟨k²⟩ - ⟨k⟩)

        Args:
            k_mean: Mean degree ⟨k⟩ of the network
            k2_mean: Second moment ⟨k²⟩ of the degree distribution

        Returns:
            Critical transmissibility threshold

        Raises:
            ValueError: If k2_mean <= k_mean (invalid degree distribution)
        """
        if k2_mean <= k_mean:
            raise ValueError(f"Invalid degree moments: ⟨k²⟩={k2_mean} must be > ⟨k⟩={k_mean}")
        return k_mean / (k2_mean - k_mean)

    def get_critical_for_regular_network(self, k: int) -> float:
        """
        Critical transmissibility for a k-regular network.

        For regular networks: ⟨k⟩ = k, ⟨k²⟩ = k²
        So: T̄_c = k / (k² - k) = 1 / (k - 1)

        Args:
            k: Degree of each node

        Returns:
            Critical transmissibility
        """
        if k <= 1:
            raise ValueError(f"Degree must be > 1, got {k}")
        return 1.0 / (k - 1)

    def get_critical_for_erdos_renyi(self, k_mean: float) -> float:
        """
        Critical transmissibility for Erdős-Rényi network.

        For ER networks with Poisson degree distribution:
            ⟨k⟩ = k_mean
            ⟨k²⟩ = k_mean² + k_mean (variance = mean for Poisson)
        So: T̄_c = k_mean / (k_mean² + k_mean - k_mean) = 1 / k_mean

        Args:
            k_mean: Mean degree

        Returns:
            Critical transmissibility
        """
        if k_mean <= 0:
            raise ValueError(f"Mean degree must be > 0, got {k_mean}")
        return 1.0 / k_mean

    def get_critical_for_power_law(
        self, k_min: int, exponent: float, k_max: Optional[int] = None
    ) -> float:
        """
        Critical transmissibility for scale-free network with power-law degree distribution.

        P(k) ∝ k^(-γ) for k ≥ k_min

        For γ ≤ 3, ⟨k²⟩ diverges → T̄_c → 0 (epidemic always spreads)
        For γ > 3, finite threshold exists

        Args:
            k_min: Minimum degree
            exponent: Power-law exponent γ
            k_max: Maximum degree (for finite-size networks)

        Returns:
            Critical transmissibility (0 if scale-free with γ ≤ 3)
        """
        if exponent <= 2:
            raise ValueError(f"Exponent must be > 2 for normalizable distribution, got {exponent}")

        # For infinite networks with γ ≤ 3, threshold is 0
        if exponent <= 3 and k_max is None:
            return 0.0

        # Calculate moments numerically for finite networks
        if k_max is None:
            k_max = 10000  # Practical cutoff

        k_values = np.arange(k_min, k_max + 1)
        p_k = k_values.astype(float) ** (-exponent)
        p_k = p_k / np.sum(p_k)  # Normalize

        k_mean: float = float(np.sum(k_values * p_k))
        k2_mean: float = float(np.sum(k_values**2 * p_k))

        return self.get_critical_transmissibility(k_mean, k2_mean)

    def analyze_epidemic(self, k_mean: float, k2_mean: float) -> EpidemicThreshold:
        """
        Perform complete epidemic threshold analysis.

        Args:
            k_mean: Mean degree of network
            k2_mean: Second moment of degree distribution

        Returns:
            EpidemicThreshold with analysis results
        """
        T_bar = self.calculate()
        T_c = self.get_critical_transmissibility(k_mean, k2_mean)
        is_super = T_bar > T_c

        # Calculate expected outbreak size if supercritical
        outbreak_size = None
        if is_super:
            outbreak_size = self._calculate_outbreak_size(T_bar, k_mean, k2_mean)

        return EpidemicThreshold(
            transmissibility=T_bar,
            critical_transmissibility=T_c,
            is_supercritical=is_super,
            expected_outbreak_size=outbreak_size,
        )

    def analyze_regular_network(self, k: int) -> EpidemicThreshold:
        """
        Analyze epidemic on a k-regular network.

        Args:
            k: Degree of each node

        Returns:
            EpidemicThreshold with analysis results
        """
        return self.analyze_epidemic(k_mean=float(k), k2_mean=float(k**2))

    def analyze_erdos_renyi(self, k_mean: float) -> EpidemicThreshold:
        """
        Analyze epidemic on an Erdős-Rényi network.

        Args:
            k_mean: Mean degree

        Returns:
            EpidemicThreshold with analysis results
        """
        # For Poisson: ⟨k²⟩ = ⟨k⟩² + ⟨k⟩
        k2_mean = k_mean**2 + k_mean
        return self.analyze_epidemic(k_mean=k_mean, k2_mean=k2_mean)

    def _calculate_outbreak_size(self, T_bar: float, k_mean: float, k2_mean: float) -> float:
        """
        Calculate expected outbreak size using generating function approach.

        From Paper Eqs. 12-13:
            u = G₁(u; T̄)  [self-consistency equation]
            S(T̄) = 1 - G₀(u; T̄)

        For Poisson degree distribution (ER networks):
            G₀(x) = G₁(x) = exp(⟨k⟩(x-1))

        Args:
            T_bar: Mean transmissibility
            k_mean: Mean degree
            k2_mean: Second moment (used for more accurate calculation)

        Returns:
            Expected fraction of recovered nodes (outbreak size)
        """
        # Solve self-consistency equation iteratively for Poisson approximation
        # u = exp(k_mean * T_bar * (u - 1))

        u = 0.5  # Initial guess
        for _ in range(100):
            u_new = np.exp(k_mean * T_bar * (u - 1))
            if abs(u_new - u) < 1e-10:
                break
            u = u_new

        # S = probability of not being in giant component
        # Outbreak size = 1 - S
        S = np.exp(k_mean * (u - 1))
        return float(1 - S)


class ExponentialExponentialTransmissibility(TransmissibilityCalculator):
    """
    Transmissibility for Poisson-Poisson (fully Markovian) SIR dynamics.

    Recovery: φ(τ) = γ·e^(-γτ)  (exponential with rate γ)
    Infection: ψ(τ) = β·e^(-βτ)  (exponential with rate β)

    Analytical result (Paper Eq. C1):
        T̄_PP = λ/(1+λ)  where λ = β/γ

    Also written as: T̄_PP = β/(β+γ)
    """

    def __init__(self, beta: float, gamma: float):
        """
        Initialize Poisson-Poisson transmissibility calculator.

        Args:
            beta: Infection rate (λ in some notations)
            gamma: Recovery rate (μ in some notations)

        Raises:
            ValueError: If rates are not positive
        """
        if beta <= 0:
            raise ValueError(f"Infection rate beta must be > 0, got {beta}")
        if gamma <= 0:
            raise ValueError(f"Recovery rate gamma must be > 0, got {gamma}")

        self.beta = beta
        self.gamma = gamma
        self.lambda_eff = beta / gamma

    def calculate(self) -> float:
        """
        Calculate T̄_PP = λ/(1+λ) = β/(β+γ).

        Derivation:
            T̄ = ∫₀^∞ γe^(-γτ) [1 - e^(-βτ)] dτ
              = 1 - γ∫₀^∞ e^(-(γ+β)τ) dτ
              = 1 - γ/(γ+β)
              = β/(γ+β)
              = λ/(1+λ)
        """
        return self.lambda_eff / (1 + self.lambda_eff)

    def get_params_dict(self) -> Dict[str, Any]:
        return {
            "type": "exponential-exponential",
            "beta": self.beta,
            "gamma": self.gamma,
            "lambda_effective": self.lambda_eff,
        }

    def get_effective_infection_rate(self) -> float:
        """Return λ = β/γ."""
        return self.lambda_eff


class GammaExponentialTransmissibility(TransmissibilityCalculator):
    """
    Transmissibility for Gamma/Erlang recovery with exponential infection.

    Recovery: φ(τ) = γⁿτⁿ⁻¹e^(-γτ)/(n-1)!  (Gamma/Erlang)
    Infection: ψ(τ) = β·e^(-βτ)  (exponential)

    This is computed numerically via integration.
    For integer shape n (Erlang), analytical form exists but numerical is general.
    """

    def __init__(self, shape: float, scale: float, beta: float):
        """
        Initialize Gamma-Exponential transmissibility calculator.

        Args:
            shape: Gamma shape parameter (n for Erlang, must be positive)
            scale: Gamma scale parameter (1/γ, mean = shape*scale)
            beta: Infection rate

        Raises:
            ValueError: If parameters are invalid
        """
        if shape <= 0:
            raise ValueError(f"Shape must be > 0, got {shape}")
        if scale <= 0:
            raise ValueError(f"Scale must be > 0, got {scale}")
        if beta <= 0:
            raise ValueError(f"Infection rate beta must be > 0, got {beta}")

        self.shape = shape
        self.scale = scale
        self.gamma = 1 / scale  # Rate parameter
        self.beta = beta
        self._cached_value: Optional[float] = None

    def calculate(self) -> float:
        """
        Calculate T̄ via numerical integration.

        T̄ = ∫₀^∞ φ(τ) Ψ(τ) dτ

        Where:
            φ(τ) = Gamma PDF with shape and scale
            Ψ(τ) = 1 - e^(-βτ) = exponential CDF
        """
        if self._cached_value is not None:
            return self._cached_value

        def integrand(tau: float) -> float:
            if tau <= 0:
                return 0.0
            # Gamma PDF for recovery time
            phi = gamma_dist.pdf(tau, a=self.shape, scale=self.scale)
            # Exponential CDF for infection (probability infected before tau)
            psi_cdf = 1 - np.exp(-self.beta * tau)
            return float(phi * psi_cdf)

        result, _ = integrate.quad(integrand, 0, np.inf)
        self._cached_value = float(result)
        return float(result)

    def calculate_analytical_erlang(self) -> Optional[float]:
        """
        Analytical formula for integer shape (Erlang distribution).

        For Erlang(n, γ) recovery and Exp(β) infection:
        T̄ = 1 - (γ/(γ+β))^n

        Returns:
            Analytical result if shape is integer, None otherwise
        """
        n = self.shape
        if not np.isclose(n, round(n)):
            return None

        n = int(round(n))
        ratio = self.gamma / (self.gamma + self.beta)
        return float(1 - ratio**n)

    def get_params_dict(self) -> Dict[str, Any]:
        return {
            "type": "gamma-exponential",
            "shape": self.shape,
            "scale": self.scale,
            "gamma": self.gamma,
            "beta": self.beta,
            "mean_recovery_time": self.shape * self.scale,
        }


class NumericalTransmissibility(TransmissibilityCalculator):
    """
    General numerical transmissibility calculator for arbitrary distributions.

    Uses scipy.integrate for numerical integration of the transmissibility formula.
    """

    def __init__(
        self,
        recovery_pdf: Callable[[float], float],
        infection_cdf: Callable[[float], float],
    ):
        """
        Initialize with custom distribution functions.

        Args:
            recovery_pdf: Callable(tau) -> float, the recovery time PDF φ(τ)
            infection_cdf: Callable(tau) -> float, the infection time CDF Ψ(τ)
        """
        self.recovery_pdf = recovery_pdf
        self.infection_cdf = infection_cdf
        self._cached_value: Optional[float] = None

    def calculate(self) -> float:
        """Calculate T̄ via numerical integration."""
        if self._cached_value is not None:
            return self._cached_value

        def integrand(tau: float) -> float:
            if tau <= 0:
                return 0.0
            return self.recovery_pdf(tau) * self.infection_cdf(tau)

        result, _ = integrate.quad(integrand, 0, np.inf)
        self._cached_value = float(result)
        return float(result)

    def get_params_dict(self) -> Dict[str, Any]:
        return {"type": "numerical-custom"}


def calculate_empirical_transmissibility(
    edge_weights: np.ndarray,
    num_edges: Optional[int] = None,
) -> float:
    """
    Calculate empirical transmissibility from edge weights.

    T̄ ≈ (# finite edge weights) / (# total edges)

    An edge weight is finite if infection can occur along that edge
    (i.e., infection time < recovery time of source node).

    Args:
        edge_weights: Weight of each edge (inf if no transmission possible)
        num_edges: Total number of edges (if different from len(edge_weights))

    Returns:
        Empirical transmissibility estimate
    """
    if num_edges is None:
        num_edges = len(edge_weights)

    if num_edges == 0:
        return 0.0

    finite_edges: int = int(np.sum(np.isfinite(edge_weights)))
    return float(finite_edges) / float(num_edges)


def calculate_transmissibility_from_outbreak(
    infection_times: np.ndarray,
    total_nodes: int,
) -> Tuple[float, float]:
    """
    Estimate transmissibility from outbreak size using inverse percolation.

    Given the final outbreak size r = R/N, estimate T̄ that would produce it.

    Args:
        infection_times: Array of infection times (inf for never infected)
        total_nodes: Total number of nodes N

    Returns:
        Tuple of (outbreak_fraction, estimated_transmissibility)
        Note: T̄ estimation is approximate and requires network info
    """
    # Count infected nodes (finite infection time)
    infected_count: int = int(np.sum(np.isfinite(infection_times)))
    outbreak_fraction = float(infected_count) / float(total_nodes)

    # T̄ estimation from outbreak size is complex and network-dependent
    # For now, return NaN for the estimate
    return outbreak_fraction, np.nan


def create_transmissibility_calculator(
    recovery_type: str,
    infection_type: str,
    **params: Any,
) -> TransmissibilityCalculator:
    """
    Factory function to create appropriate transmissibility calculator.

    Args:
        recovery_type: Type of recovery distribution ("exponential", "gamma")
        infection_type: Type of infection distribution ("exponential")
        **params: Distribution parameters
            For exponential recovery: gamma (rate) or mu
            For gamma recovery: shape, scale
            For exponential infection: beta (rate) or lambda/lmbd

    Returns:
        Appropriate TransmissibilityCalculator instance

    Raises:
        ValueError: If distribution combination is not supported
    """
    recovery_type = recovery_type.lower()
    infection_type = infection_type.lower()

    if infection_type != "exponential":
        raise ValueError(
            f"Infection type '{infection_type}' not yet supported. "
            "Only 'exponential' is implemented."
        )

    # Get infection rate (try multiple parameter names)
    beta = params.get("beta") or params.get("lambda") or params.get("lmbd")
    if beta is None:
        raise ValueError("Infection rate 'beta' (or 'lambda'/'lmbd') is required")

    if recovery_type == "exponential":
        gamma = params.get("gamma") or params.get("mu")
        if gamma is None:
            raise ValueError("Recovery rate 'gamma' (or 'mu') is required for exponential recovery")
        return ExponentialExponentialTransmissibility(beta=beta, gamma=gamma)

    elif recovery_type == "gamma":
        shape = params.get("shape")
        scale = params.get("scale")
        if shape is None or scale is None:
            raise ValueError("Parameters 'shape' and 'scale' are required for gamma recovery")
        return GammaExponentialTransmissibility(shape=shape, scale=scale, beta=beta)

    else:
        raise ValueError(
            f"Recovery type '{recovery_type}' not supported. " "Use 'exponential' or 'gamma'."
        )


# Convenience functions for quick calculations


def poisson_poisson_transmissibility(beta: float, gamma: float) -> float:
    """
    Quick calculation of T̄ for Poisson-Poisson dynamics.

    T̄ = β/(β+γ)

    Args:
        beta: Infection rate
        gamma: Recovery rate

    Returns:
        Mean transmissibility
    """
    return beta / (beta + gamma)


def gamma_exponential_transmissibility(shape: float, scale: float, beta: float) -> float:
    """
    Quick calculation of T̄ for Gamma recovery with exponential infection.

    Args:
        shape: Gamma shape parameter
        scale: Gamma scale parameter
        beta: Infection rate

    Returns:
        Mean transmissibility (computed numerically)
    """
    calc = GammaExponentialTransmissibility(shape, scale, beta)
    return calc.calculate()


def epidemic_threshold_regular(k: int) -> float:
    """
    Critical transmissibility for k-regular network.

    T̄_c = 1/(k-1)

    Args:
        k: Degree of each node

    Returns:
        Critical transmissibility
    """
    if k <= 1:
        raise ValueError(f"Degree must be > 1, got {k}")
    return 1.0 / (k - 1)


def epidemic_threshold_er(k_mean: float) -> float:
    """
    Critical transmissibility for Erdős-Rényi network.

    T̄_c = 1/⟨k⟩

    Args:
        k_mean: Mean degree

    Returns:
        Critical transmissibility
    """
    if k_mean <= 0:
        raise ValueError(f"Mean degree must be > 0, got {k_mean}")
    return 1.0 / k_mean
