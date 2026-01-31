"""
Tests for transmissibility calculations.

These tests validate the implementation against analytical results from:
Böttcher & Antulov-Fantulin, "Unifying continuous, discrete, and hybrid
susceptible-infected-recovered processes on networks",
Phys. Rev. Research 2, 033121 (2020).
"""

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from spkmc.core.distributions import ExponentialDistribution, GammaDistribution
from spkmc.core.transmissibility import (
    ExponentialExponentialTransmissibility,
    GammaExponentialTransmissibility,
    NumericalTransmissibility,
    calculate_empirical_transmissibility,
    create_transmissibility_calculator,
    epidemic_threshold_er,
    epidemic_threshold_regular,
    gamma_exponential_transmissibility,
    poisson_poisson_transmissibility,
)


class TestExponentialExponentialTransmissibility:
    """Tests for Poisson-Poisson (Exp-Exp) transmissibility."""

    def test_basic_calculation(self):
        """Test T̄ = β/(β+γ) formula."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)
        T_bar = calc.calculate()

        # T̄ = 0.5 / (0.5 + 0.1) = 0.5 / 0.6 = 0.8333...
        expected = 0.5 / 0.6
        assert abs(T_bar - expected) < 1e-10

    def test_lambda_formulation(self):
        """Test T̄ = λ/(1+λ) where λ = β/γ."""
        beta, gamma = 0.3, 0.2
        calc = ExponentialExponentialTransmissibility(beta=beta, gamma=gamma)

        lambda_eff = beta / gamma  # 1.5
        expected = lambda_eff / (1 + lambda_eff)  # 1.5 / 2.5 = 0.6

        assert abs(calc.calculate() - expected) < 1e-10
        assert abs(calc.get_effective_infection_rate() - lambda_eff) < 1e-10

    def test_boundary_cases(self):
        """Test behavior at extreme parameter values."""
        # High infection rate → T̄ → 1
        calc_high = ExponentialExponentialTransmissibility(beta=100.0, gamma=0.1)
        assert calc_high.calculate() > 0.99

        # Low infection rate → T̄ → 0
        calc_low = ExponentialExponentialTransmissibility(beta=0.01, gamma=10.0)
        assert calc_low.calculate() < 0.01

    def test_equal_rates(self):
        """When β = γ, T̄ = 0.5."""
        calc = ExponentialExponentialTransmissibility(beta=1.0, gamma=1.0)
        assert abs(calc.calculate() - 0.5) < 1e-10

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            ExponentialExponentialTransmissibility(beta=-0.1, gamma=0.1)

        with pytest.raises(ValueError):
            ExponentialExponentialTransmissibility(beta=0.1, gamma=0)

    def test_params_dict(self):
        """Test parameter dictionary."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)
        params = calc.get_params_dict()

        assert params["type"] == "exponential-exponential"
        assert params["beta"] == 0.5
        assert params["gamma"] == 0.1
        assert abs(params["lambda_effective"] - 5.0) < 1e-10


class TestGammaExponentialTransmissibility:
    """Tests for Gamma recovery + Exponential infection transmissibility."""

    def test_erlang_n1_matches_exponential(self):
        """Erlang(1, γ) = Exponential(γ), results should match."""
        beta, gamma = 0.5, 0.2
        scale = 1.0 / gamma  # scale = 1/γ for Erlang parameterization

        calc_gamma = GammaExponentialTransmissibility(shape=1.0, scale=scale, beta=beta)
        calc_exp = ExponentialExponentialTransmissibility(beta=beta, gamma=gamma)

        T_gamma = calc_gamma.calculate()
        T_exp = calc_exp.calculate()

        assert abs(T_gamma - T_exp) < 1e-6

    def test_erlang_analytical_formula(self):
        """Test analytical Erlang formula: T̄ = 1 - (γ/(γ+β))^n."""
        shape = 2
        gamma = 0.4
        beta = 0.3
        scale = 1.0 / gamma

        calc = GammaExponentialTransmissibility(shape=float(shape), scale=scale, beta=beta)

        # Analytical: T̄ = 1 - (γ/(γ+β))^n = 1 - (0.4/0.7)^2
        ratio = gamma / (gamma + beta)
        expected = 1 - ratio**shape

        # Numerical result
        T_numerical = calc.calculate()

        # Analytical result from method
        T_analytical = calc.calculate_analytical_erlang()

        assert T_analytical is not None
        assert abs(T_numerical - expected) < 1e-6
        assert abs(T_analytical - expected) < 1e-10

    def test_higher_shape_increases_transmissibility(self):
        """Higher shape (more concentrated recovery) increases T̄."""
        beta = 0.5
        mean_recovery = 2.0  # Keep mean constant

        # For Gamma: mean = shape * scale
        T_values = []
        for shape in [1, 2, 4, 8]:
            scale = mean_recovery / shape
            calc = GammaExponentialTransmissibility(shape=float(shape), scale=scale, beta=beta)
            T_values.append(calc.calculate())

        # T̄ should increase with shape (more concentrated recovery = higher transmission prob)
        for i in range(len(T_values) - 1):
            assert T_values[i] < T_values[i + 1]

    def test_caching(self):
        """Test that results are cached."""
        calc = GammaExponentialTransmissibility(shape=2.0, scale=0.5, beta=0.3)

        T1 = calc.calculate()
        T2 = calc.calculate()

        assert T1 == T2
        assert calc._cached_value is not None

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            GammaExponentialTransmissibility(shape=0, scale=0.5, beta=0.3)

        with pytest.raises(ValueError):
            GammaExponentialTransmissibility(shape=2.0, scale=-0.5, beta=0.3)

        with pytest.raises(ValueError):
            GammaExponentialTransmissibility(shape=2.0, scale=0.5, beta=0)

    def test_non_integer_shape(self):
        """Test with non-integer shape (true Gamma, not Erlang)."""
        calc = GammaExponentialTransmissibility(shape=2.5, scale=0.5, beta=0.3)
        T = calc.calculate()

        assert 0 < T < 1
        assert calc.calculate_analytical_erlang() is None  # No analytical for non-integer


class TestCriticalThreshold:
    """Tests for epidemic threshold calculations."""

    def test_regular_network_k5(self):
        """For k=5 regular network, T̄_c = 1/(5-1) = 0.25."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)
        T_c = calc.get_critical_for_regular_network(k=5)

        assert abs(T_c - 0.25) < 1e-10

    def test_regular_network_formula(self):
        """Test T̄_c = 1/(k-1) for various k."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)

        for k in [3, 5, 10, 20]:
            T_c = calc.get_critical_for_regular_network(k)
            expected = 1.0 / (k - 1)
            assert abs(T_c - expected) < 1e-10

    def test_erdos_renyi_formula(self):
        """Test T̄_c = 1/⟨k⟩ for ER networks."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)

        for k_mean in [5.0, 10.0, 20.0]:
            T_c = calc.get_critical_for_erdos_renyi(k_mean)
            expected = 1.0 / k_mean
            assert abs(T_c - expected) < 1e-10

    def test_general_threshold_formula(self):
        """Test general formula T̄_c = ⟨k⟩/(⟨k²⟩ - ⟨k⟩)."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)

        k_mean = 10.0
        k2_mean = 150.0  # Some arbitrary second moment > k_mean²

        T_c = calc.get_critical_transmissibility(k_mean, k2_mean)
        expected = k_mean / (k2_mean - k_mean)

        assert abs(T_c - expected) < 1e-10

    def test_invalid_degree_moments(self):
        """Test error for invalid degree moments."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)

        with pytest.raises(ValueError):
            calc.get_critical_transmissibility(k_mean=10.0, k2_mean=5.0)

    def test_power_law_exponent_leq_3(self):
        """For scale-free with γ ≤ 3, T̄_c → 0."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)

        # γ = 2.5 < 3: infinite ⟨k²⟩, threshold is 0
        T_c = calc.get_critical_for_power_law(k_min=2, exponent=2.5)
        assert T_c == 0.0

        # γ = 3.0: boundary case
        T_c = calc.get_critical_for_power_law(k_min=2, exponent=3.0)
        assert T_c == 0.0

    def test_power_law_exponent_gt_3(self):
        """For γ > 3, finite threshold exists."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)

        T_c = calc.get_critical_for_power_law(k_min=2, exponent=3.5, k_max=1000)
        assert T_c > 0


class TestEpidemicAnalysis:
    """Tests for epidemic threshold analysis."""

    def test_supercritical_detection(self):
        """Test detection of supercritical epidemics."""
        # High transmissibility: T̄ = 0.833, much greater than T̄_c = 0.1 for ER with k=10
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)
        result = calc.analyze_erdos_renyi(k_mean=10.0)

        assert result.is_supercritical
        assert result.transmissibility > result.critical_transmissibility
        assert result.expected_outbreak_size is not None
        assert result.expected_outbreak_size > 0.5  # Large outbreak expected

    def test_subcritical_detection(self):
        """Test detection of subcritical epidemics."""
        # Low transmissibility: T̄ = 0.05 < T̄_c = 0.2 for k=6 regular
        calc = ExponentialExponentialTransmissibility(beta=0.05, gamma=0.95)
        result = calc.analyze_regular_network(k=6)

        assert not result.is_supercritical
        assert result.transmissibility < result.critical_transmissibility
        assert result.expected_outbreak_size is None

    def test_outbreak_size_estimation(self):
        """Test that outbreak size is reasonable."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)
        result = calc.analyze_erdos_renyi(k_mean=10.0)

        # Outbreak size should be between 0 and 1
        assert 0 < result.expected_outbreak_size <= 1

    def test_str_representation(self):
        """Test string representation of EpidemicThreshold."""
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)
        result = calc.analyze_erdos_renyi(k_mean=10.0)

        s = str(result)
        assert "SUPERCRITICAL" in s
        assert "T̄" in s


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_poisson_poisson_transmissibility(self):
        """Test quick calculation function."""
        T = poisson_poisson_transmissibility(beta=0.5, gamma=0.1)
        expected = 0.5 / 0.6
        assert abs(T - expected) < 1e-10

    def test_gamma_exponential_transmissibility(self):
        """Test Gamma-Exp convenience function."""
        T = gamma_exponential_transmissibility(shape=2.0, scale=0.5, beta=0.3)
        assert 0 < T < 1

    def test_epidemic_threshold_regular(self):
        """Test regular network threshold function."""
        T_c = epidemic_threshold_regular(k=5)
        assert abs(T_c - 0.25) < 1e-10

    def test_epidemic_threshold_er(self):
        """Test ER network threshold function."""
        T_c = epidemic_threshold_er(k_mean=10.0)
        assert abs(T_c - 0.1) < 1e-10


class TestFactoryFunction:
    """Tests for create_transmissibility_calculator factory."""

    def test_create_exponential_exponential(self):
        """Test creating Exp-Exp calculator."""
        calc = create_transmissibility_calculator(
            recovery_type="exponential",
            infection_type="exponential",
            beta=0.5,
            gamma=0.1,
        )

        assert isinstance(calc, ExponentialExponentialTransmissibility)
        assert abs(calc.calculate() - 0.5 / 0.6) < 1e-10

    def test_create_gamma_exponential(self):
        """Test creating Gamma-Exp calculator."""
        calc = create_transmissibility_calculator(
            recovery_type="gamma",
            infection_type="exponential",
            shape=2.0,
            scale=0.5,
            beta=0.3,
        )

        assert isinstance(calc, GammaExponentialTransmissibility)
        assert 0 < calc.calculate() < 1

    def test_alternative_param_names(self):
        """Test with alternative parameter names."""
        # Use 'mu' instead of 'gamma'
        calc1 = create_transmissibility_calculator(
            recovery_type="exponential",
            infection_type="exponential",
            beta=0.5,
            mu=0.1,
        )
        assert isinstance(calc1, ExponentialExponentialTransmissibility)

        # Use 'lambda' instead of 'beta'
        calc2 = create_transmissibility_calculator(
            recovery_type="exponential",
            infection_type="exponential",
            **{"lambda": 0.5, "gamma": 0.1},
        )
        assert isinstance(calc2, ExponentialExponentialTransmissibility)

    def test_unsupported_infection_type(self):
        """Test error for unsupported infection type."""
        with pytest.raises(ValueError, match="not yet supported"):
            create_transmissibility_calculator(
                recovery_type="exponential",
                infection_type="geometric",
                beta=0.5,
                gamma=0.1,
            )

    def test_unsupported_recovery_type(self):
        """Test error for unsupported recovery type."""
        with pytest.raises(ValueError, match="not supported"):
            create_transmissibility_calculator(
                recovery_type="weibull",
                infection_type="exponential",
                beta=0.5,
            )

    def test_missing_parameters(self):
        """Test error when required parameters are missing."""
        with pytest.raises(ValueError):
            create_transmissibility_calculator(
                recovery_type="gamma",
                infection_type="exponential",
                beta=0.5,
                # Missing shape and scale
            )


class TestNumericalTransmissibility:
    """Tests for custom numerical transmissibility."""

    def test_matches_analytical_exp_exp(self):
        """Test that numerical matches analytical for Exp-Exp."""
        beta, gamma = 0.5, 0.2

        # Define PDFs/CDFs manually
        def recovery_pdf(tau):
            return gamma * np.exp(-gamma * tau)

        def infection_cdf(tau):
            return 1 - np.exp(-beta * tau)

        calc_num = NumericalTransmissibility(recovery_pdf, infection_cdf)
        calc_ana = ExponentialExponentialTransmissibility(beta=beta, gamma=gamma)

        assert abs(calc_num.calculate() - calc_ana.calculate()) < 1e-6

    def test_with_scipy_distributions(self):
        """Test using scipy distribution functions."""
        shape, scale = 2.0, 0.5
        beta = 0.3

        def recovery_pdf(tau):
            return gamma_dist.pdf(tau, a=shape, scale=scale)

        def infection_cdf(tau):
            return 1 - np.exp(-beta * tau)

        calc_num = NumericalTransmissibility(recovery_pdf, infection_cdf)
        calc_gamma = GammaExponentialTransmissibility(shape=shape, scale=scale, beta=beta)

        assert abs(calc_num.calculate() - calc_gamma.calculate()) < 1e-6


class TestEmpiricalTransmissibility:
    """Tests for empirical transmissibility calculation."""

    def test_all_finite_weights(self):
        """All finite weights → T̄ = 1."""
        edge_weights = np.array([0.5, 1.0, 2.0, 0.1])
        T = calculate_empirical_transmissibility(edge_weights)
        assert abs(T - 1.0) < 1e-10

    def test_all_infinite_weights(self):
        """All infinite weights → T̄ = 0."""
        edge_weights = np.array([np.inf, np.inf, np.inf])
        T = calculate_empirical_transmissibility(edge_weights)
        assert abs(T - 0.0) < 1e-10

    def test_mixed_weights(self):
        """Mix of finite and infinite weights."""
        edge_weights = np.array([0.5, np.inf, 1.0, np.inf])
        T = calculate_empirical_transmissibility(edge_weights)
        assert abs(T - 0.5) < 1e-10

    def test_empty_array(self):
        """Empty array → T̄ = 0."""
        edge_weights = np.array([])
        T = calculate_empirical_transmissibility(edge_weights)
        assert T == 0.0


class TestDistributionIntegration:
    """Tests for integration with Distribution classes."""

    def test_gamma_distribution_calculator(self):
        """Test GammaDistribution.get_transmissibility_calculator()."""
        dist = GammaDistribution(shape=2.0, scale=0.5, lmbd=0.3)
        calc = dist.get_transmissibility_calculator()

        assert isinstance(calc, GammaExponentialTransmissibility)
        assert calc.shape == 2.0
        assert calc.scale == 0.5
        assert calc.beta == 0.3

    def test_exponential_distribution_calculator(self):
        """Test ExponentialDistribution.get_transmissibility_calculator()."""
        dist = ExponentialDistribution(mu=0.2, lmbd=0.5)
        calc = dist.get_transmissibility_calculator()

        assert isinstance(calc, ExponentialExponentialTransmissibility)
        assert calc.beta == 0.5
        assert calc.gamma == 0.2

    def test_calculator_gives_correct_results(self):
        """Test that calculator from distribution gives correct T̄."""
        dist = ExponentialDistribution(mu=0.1, lmbd=0.5)
        calc = dist.get_transmissibility_calculator()

        expected = 0.5 / (0.5 + 0.1)
        assert abs(calc.calculate() - expected) < 1e-10


class TestPaperValidation:
    """
    Tests that validate results against the original paper.

    Reference: Böttcher & Antulov-Fantulin, Phys. Rev. Research 2, 033121 (2020)
    """

    def test_figure_2a_poisson_poisson(self):
        """
        Validate against Figure 2(a): Poisson-Poisson SIR.

        For λ_PP = β/γ varying, T̄ = λ/(1+λ) should match.
        """
        # Test a few points from the expected curve
        test_cases = [
            (0.2, 0.1667),  # λ=0.2 → T̄ ≈ 0.167
            (0.5, 0.3333),  # λ=0.5 → T̄ ≈ 0.333
            (1.0, 0.5),  # λ=1.0 → T̄ = 0.5
            (2.0, 0.6667),  # λ=2.0 → T̄ ≈ 0.667
        ]

        for lambda_eff, expected_T in test_cases:
            gamma = 1.0
            beta = lambda_eff * gamma
            calc = ExponentialExponentialTransmissibility(beta=beta, gamma=gamma)
            T = calc.calculate()
            assert abs(T - expected_T) < 0.01, f"λ={lambda_eff}: T̄={T}, expected={expected_T}"

    def test_figure_6b_critical_threshold(self):
        """
        Validate against Figure 6(b): Critical threshold for k=5 regular network.

        λ^c_PP = 1/3 corresponds to T̄_c = 0.25 for k=5 regular network.
        """
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)
        T_c = calc.get_critical_for_regular_network(k=5)

        # Paper states λ^c = 1/3, which gives T̄_c = (1/3)/(1 + 1/3) = 0.25
        assert abs(T_c - 0.25) < 1e-10

    def test_equation_c1_transmissibility(self):
        """
        Validate Equation (C1): T̄_PP = λ_PP/(λ_PP + 1) = β/(β+γ).
        """
        beta, gamma = 0.3, 0.15
        calc = ExponentialExponentialTransmissibility(beta=beta, gamma=gamma)

        # Method 1: β/(β+γ)
        T1 = beta / (beta + gamma)

        # Method 2: λ/(λ+1)
        lambda_eff = beta / gamma
        T2 = lambda_eff / (lambda_eff + 1)

        # Should match
        assert abs(calc.calculate() - T1) < 1e-10
        assert abs(calc.calculate() - T2) < 1e-10

    def test_equation_7_threshold(self):
        """
        Validate Equation (7): p_c = ⟨k⟩ / (⟨k²⟩ - ⟨k⟩).
        """
        calc = ExponentialExponentialTransmissibility(beta=0.5, gamma=0.1)

        # For Poisson degree distribution: ⟨k²⟩ = ⟨k⟩² + ⟨k⟩
        k_mean = 8.0
        k2_mean = k_mean**2 + k_mean  # 72

        T_c = calc.get_critical_transmissibility(k_mean, k2_mean)
        expected = k_mean / (k2_mean - k_mean)  # 8 / (72 - 8) = 8/64 = 0.125

        assert abs(T_c - expected) < 1e-10
        assert abs(T_c - 1.0 / k_mean) < 1e-10  # For Poisson: T_c = 1/⟨k⟩
