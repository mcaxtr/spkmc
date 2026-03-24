"""
Compute matched Gamma and Weibull scenarios where T-bar is held constant
while CV varies. For each CV level, solve for lambda (infection rate) that
produces the exact same T-bar.
"""

import numpy as np
from scipy import integrate, special
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import gamma as gamma_dist
from scipy.stats import weibull_min

# --- T-bar computation functions ---


def tbar_gamma(shape, scale, beta):
    """Compute T-bar for Gamma recovery with exponential infection."""

    def integrand(tau):
        if tau <= 0:
            return 0.0
        phi = gamma_dist.pdf(tau, a=shape, scale=scale)
        psi_cdf = 1 - np.exp(-beta * tau)
        return phi * psi_cdf

    result, _ = integrate.quad(integrand, 0, np.inf)
    return result


def tbar_weibull(shape_w, scale_w, beta):
    """Compute T-bar for Weibull recovery with exponential infection."""

    def integrand(tau):
        if tau <= 0:
            return 0.0
        phi = weibull_min.pdf(tau, c=shape_w, scale=scale_w)
        psi_cdf = 1 - np.exp(-beta * tau)
        return phi * psi_cdf

    result, _ = integrate.quad(integrand, 0, np.inf)
    return result


# --- Weibull parameter solving ---


def weibull_shape_from_cv(target_cv):
    """Solve for Weibull shape parameter k given a target CV."""

    def cv_error(k):
        g1 = special.gamma(1 + 1.0 / k)
        g2 = special.gamma(1 + 2.0 / k)
        cv = np.sqrt(g2 / g1**2 - 1)
        return cv - target_cv

    # CV is monotonically decreasing in k; large k -> small CV
    # For CV=1.0, k~1; for CV=0.1, k~large
    k_lo, k_hi = 0.1, 200.0
    shape_w = brentq(cv_error, k_lo, k_hi, xtol=1e-12)
    return shape_w


def weibull_scale_from_shape(shape_w, target_mean=1.0):
    """Compute Weibull scale given shape and target mean."""
    return target_mean / special.gamma(1 + 1.0 / shape_w)


# --- Main computation ---


def main():
    # Define CV levels and corresponding Gamma shapes
    cv_levels = [0.20, 0.3162, 0.50, 0.7071, 1.00]
    # n = 1/CV^2
    gamma_shapes = [1.0 / cv**2 for cv in cv_levels]

    # Reference scenario: Gamma(shape=4, scale=0.25) with lambda=0.4
    ref_shape = 4.0
    ref_scale = 0.25
    ref_lambda = 0.4
    target_tbar = tbar_gamma(ref_shape, ref_scale, ref_lambda)

    print(f"Reference scenario: Gamma(n={ref_shape}, theta={ref_scale}), lambda={ref_lambda}")
    print(f"Target T-bar = {target_tbar:.10f}")
    print()

    # Storage for results
    gamma_results = []
    weibull_results = []

    for cv, n in zip(cv_levels, gamma_shapes):
        theta = 1.0 / n  # scale = 1/shape to keep mean=1

        # --- Gamma: solve for lambda ---
        def gamma_tbar_residual(beta, n=n, theta=theta):
            return tbar_gamma(n, theta, beta) - target_tbar

        # T-bar is monotonically increasing in beta (0 -> 1), so brentq works
        lam_g = brentq(gamma_tbar_residual, 1e-6, 100.0, xtol=1e-12)

        # Verify
        tbar_verify_g = tbar_gamma(n, theta, lam_g)

        # Gamma moments
        skewness_g = 2.0 / np.sqrt(n)
        kurtosis_g = 6.0 / n

        gamma_results.append(
            {
                "cv": cv,
                "shape": n,
                "scale": theta,
                "lam": lam_g,
                "tbar": tbar_verify_g,
                "skewness": skewness_g,
                "kurtosis": kurtosis_g,
                "mean": n * theta,
                "variance": n * theta**2,
            }
        )

        # --- Weibull: solve for shape, scale, then lambda ---
        k_w = weibull_shape_from_cv(cv)
        s_w = weibull_scale_from_shape(k_w, target_mean=1.0)

        # Verify Weibull mean and CV
        g1 = special.gamma(1 + 1.0 / k_w)
        g2 = special.gamma(1 + 2.0 / k_w)
        g3 = special.gamma(1 + 3.0 / k_w)
        g4 = special.gamma(1 + 4.0 / k_w)
        weibull_mean = s_w * g1
        weibull_cv = np.sqrt(g2 / g1**2 - 1)

        # Weibull skewness and excess kurtosis (standardized)
        mu = s_w * g1
        sigma = s_w * np.sqrt(g2 - g1**2)
        skewness_w = (s_w**3 * (g3 - 3 * g1 * g2 + 2 * g1**3)) / sigma**3
        kurtosis_w = (s_w**4 * (g4 - 4 * g1 * g3 + 6 * g1**2 * g2 - 3 * g1**4)) / sigma**4 - 3

        def weibull_tbar_residual(beta, k_w=k_w, s_w=s_w):
            return tbar_weibull(k_w, s_w, beta) - target_tbar

        lam_w = brentq(weibull_tbar_residual, 1e-6, 100.0, xtol=1e-12)
        tbar_verify_w = tbar_weibull(k_w, s_w, lam_w)

        weibull_results.append(
            {
                "cv": cv,
                "shape": k_w,
                "scale": s_w,
                "lam": lam_w,
                "tbar": tbar_verify_w,
                "skewness": skewness_w,
                "kurtosis": kurtosis_w,
                "mean": weibull_mean,
                "variance": sigma**2,
                "cv_verify": weibull_cv,
            }
        )

    # --- Print Gamma table ---
    print("=" * 110)
    print("GAMMA DISTRIBUTION SCENARIOS (mean recovery = 1.0, T-bar matched)")
    print("=" * 110)
    header = f"{'CV':>8} {'shape(n)':>12} {'scale(θ)':>12} {'lambda(β)':>12} {'T-bar':>12} {'skewness':>10} {'ex.kurt':>10} {'mean':>8}"
    print(header)
    print("-" * 110)
    for r in gamma_results:
        print(
            f"{r['cv']:8.4f} {r['shape']:12.6f} {r['scale']:12.6f} {r['lam']:12.6f} "
            f"{r['tbar']:12.10f} {r['skewness']:10.6f} {r['kurtosis']:10.6f} {r['mean']:8.4f}"
        )
    print()

    # --- Print Weibull table ---
    print("=" * 120)
    print("WEIBULL DISTRIBUTION SCENARIOS (mean recovery = 1.0, T-bar matched)")
    print("=" * 120)
    header = f"{'CV':>8} {'shape(k)':>12} {'scale(s)':>12} {'lambda(β)':>12} {'T-bar':>12} {'skewness':>10} {'ex.kurt':>10} {'mean':>8} {'CV_chk':>8}"
    print(header)
    print("-" * 120)
    for r in weibull_results:
        print(
            f"{r['cv']:8.4f} {r['shape']:12.6f} {r['scale']:12.6f} {r['lam']:12.6f} "
            f"{r['tbar']:12.10f} {r['skewness']:10.6f} {r['kurtosis']:10.6f} {r['mean']:8.4f} {r['cv_verify']:8.4f}"
        )
    print()

    # --- Print combined summary for easy copy-paste ---
    print("=" * 100)
    print("COMBINED SUMMARY: lambda values for each CV level")
    print("=" * 100)
    print(
        f"{'CV':>8} | {'Gamma lambda':>14} | {'Weibull lambda':>14} | {'Gamma n':>10} | {'Weibull k':>12} | {'Weibull s':>12}"
    )
    print("-" * 100)
    for g, w in zip(gamma_results, weibull_results):
        print(
            f"{g['cv']:8.4f} | {g['lam']:14.6f} | {w['lam']:14.6f} | {g['shape']:10.6f} | {w['shape']:12.6f} | {w['scale']:12.6f}"
        )
    print()

    # --- Verification: show T-bar deviation ---
    print("=" * 80)
    print("T-BAR VERIFICATION (deviation from target)")
    print("=" * 80)
    print(f"Target T-bar: {target_tbar:.12f}")
    print()
    for g, w in zip(gamma_results, weibull_results):
        g_err = abs(g["tbar"] - target_tbar)
        w_err = abs(w["tbar"] - target_tbar)
        print(f"CV={g['cv']:.4f}:  Gamma error={g_err:.2e},  Weibull error={w_err:.2e}")


if __name__ == "__main__":
    main()
