"""Hard mathematical invariants — the gold standard for the numerical core.

Each test is an identity that must hold *exactly* (to floating-point) if the
implementation is right: closed-form moments vs numerical quadrature, Lagrange
interpolation vs direct evaluation, Tweedie's formula vs the analytic Gaussian
mixture score, and the EDM preconditioning identities.
"""

from __future__ import annotations

import numpy as np
import pytest
from diffuforge.data.densities import analytic_gmm_logpdf, analytic_gmm_score
from diffuforge.samplers.dpm import (
    lagrange_power_coeffs,
    lambda_moments,
    poly_update,
    sigma_moments,
)
from diffuforge.samplers.schedule import karras_sigma, steps_for_budget, uniform_log_sigma
from diffuforge.score.precond import edm_coefficients
from scipy.integrate import quad

# --------------------------------------------------------------- quadrature


def _quad_moment(sig_i: float, sig_j: float, k: int) -> float:
    """Numerical value of int_{lam_i}^{lam_j} e^{-2 lam} (lam - lam_i)^k dlam."""
    lam_i, lam_j = -np.log(sig_i), -np.log(sig_j)
    f = lambda t: np.exp(-2.0 * t) * (t - lam_i) ** k  # noqa: E731
    return float(quad(f, lam_i, lam_j, limit=200)[0])


@pytest.mark.parametrize("sig_i", [4.0, 1.0, 0.3])
@pytest.mark.parametrize("ratio", [0.5, 0.1, 0.01])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_lambda_moments_match_quadrature(sig_i: float, ratio: float, k: int) -> None:
    sig_j = sig_i * ratio
    closed = float(lambda_moments(sig_i, sig_j)[k])
    numeric = _quad_moment(sig_i, sig_j, k)
    assert closed == pytest.approx(numeric, rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("sig_i", [4.0, 0.7, 0.05])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_lambda_moments_terminal_limit(sig_i: float, k: int) -> None:
    """sigma_j -> 0 must agree with the limit of the closed form."""
    limit = float(lambda_moments(sig_i, 0.0)[k])
    near = float(lambda_moments(sig_i, sig_i * 1e-9)[k])
    assert limit == pytest.approx(near, rel=1e-6)


def test_lambda_moment0_is_variance_gap() -> None:
    si, sj = 3.0, 0.5
    assert float(lambda_moments(si, sj)[0]) == pytest.approx(0.5 * (si * si - sj * sj))


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_sigma_moments_are_monomials(k: int) -> None:
    si, sj = 3.0, 1.0
    expect = (sj - si) ** (k + 1) / (k + 1)
    assert float(sigma_moments(si, sj)[k]) == pytest.approx(expect)


# ------------------------------------------------------- Lagrange machinery


def test_lagrange_basis_is_a_partition_of_unity() -> None:
    taus = np.array([0.0, -0.4, -0.9])
    coeffs = lagrange_power_coeffs(taus)
    for tau in np.linspace(-1.5, 1.5, 11):
        total = sum(sum(c * tau**p for p, c in enumerate(co)) for co in coeffs)
        assert total == pytest.approx(1.0, abs=1e-12)


def test_lagrange_basis_is_nodal() -> None:
    taus = np.array([0.0, -0.3, -0.8, -1.4])
    coeffs = lagrange_power_coeffs(taus)
    for j, tau_j in enumerate(taus):
        for i, co in enumerate(coeffs):
            val = sum(c * tau_j**p for p, c in enumerate(co))
            assert val == pytest.approx(1.0 if i == j else 0.0, abs=1e-10)


def test_poly_update_integrates_a_constant_field_exactly() -> None:
    """A constant score must integrate to I0 * s (DPM-Solver-1)."""
    x = np.zeros((5, 2))
    s = np.full((1, 2), 0.7) * np.array([[1.0, -2.0]])
    mom = lambda_moments(2.0, 0.5)
    out = poly_update(x, np.array([0.0]), [s[0]], mom)
    assert np.allclose(out, s[0] * mom[0])


def test_poly_update_is_exact_for_a_linear_field() -> None:
    """If s(lambda) is linear, the 2-node update is exact (order-2 collocation)."""
    si, sj = 2.0, 0.5
    lam_i, lam_j = -np.log(si), -np.log(sj)
    a = np.array([[0.3, -0.2]])
    b = np.array([[1.1, 0.7]])
    s_of = lambda lam: a + b * (lam - lam_i)  # noqa: E731
    exact = a * float(lambda_moments(si, sj)[0]) + b * float(lambda_moments(si, sj)[1])
    got = poly_update(
        np.zeros((1, 2)),
        np.array([0.0, lam_j - lam_i]),
        [s_of(lam_i)[0], s_of(lam_j)[0]],
        lambda_moments(si, sj),
    )
    assert np.allclose(got, exact, atol=1e-12)


def test_lagrange_rejects_degenerate_and_out_of_range_orders() -> None:
    from diffuforge.core.errors import SamplerError

    with pytest.raises(SamplerError):
        lagrange_power_coeffs(np.array([0.0, 0.0]))
    with pytest.raises(SamplerError):
        lagrange_power_coeffs(np.array([0.0, -1.0, -2.0, -3.0, -4.0]))


# ------------------------------------------------------------------ EDM


@pytest.mark.parametrize("sigma", [0.01, 0.5, 1.0, 4.0, 20.0])
def test_edm_preconditioning_identities(sigma: float) -> None:
    sd = 1.3
    c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, sd)
    # Karras et al. 2022, Table 1
    assert float(c_skip) == pytest.approx(sd**2 / (sigma**2 + sd**2))
    assert float(c_out) == pytest.approx(sigma * sd / np.sqrt(sd**2 + sigma**2))
    assert float(c_in) == pytest.approx(1.0 / np.sqrt(sigma**2 + sd**2))
    assert float(c_noise) == pytest.approx(0.25 * np.log(sigma))
    # c_skip -> 1 (identity map) as sigma -> 0 and -> 0 as sigma -> inf
    assert float(edm_coefficients(1e-6, sd)[0]) == pytest.approx(1.0, abs=1e-9)
    assert float(edm_coefficients(1e6, sd)[0]) == pytest.approx(0.0, abs=1e-9)


def test_analytic_gmm_score_is_the_gradient_of_the_log_density() -> None:
    """``analytic_gmm_score`` must equal the finite-difference gradient.

    Convolving an isotropic Gaussian mixture with ``N(0, sigma^2 I)`` yields
    another mixture whose per-component std is ``sqrt(std^2 + sigma^2)``, so the
    closed-form score is checkable against a central difference of the analytic
    log-density — an independent, implementation-free gold standard.
    """
    rng = np.random.default_rng(11)
    centers = np.array([[-2.0, 0.0], [2.0, 1.0], [0.0, -2.5]])
    std, sigma = 0.4, 0.9
    conv_std = float(np.sqrt(std**2 + sigma**2))
    x = rng.normal(scale=1.5, size=(20, 2))

    eps = 1e-6
    fd = np.zeros_like(x)
    for d in range(2):
        step = np.zeros_like(x)
        step[:, d] = eps
        fd[:, d] = (
            analytic_gmm_logpdf(x + step, centers, conv_std, None)
            - analytic_gmm_logpdf(x - step, centers, conv_std, None)
        ) / (2.0 * eps)
    closed = analytic_gmm_score(x, centers, std, sigma)
    assert np.abs(closed - fd).max() < 1e-6


def test_analytic_gmm_score_single_component_is_quadratic() -> None:
    """A one-component mixture has score ``-(x - c) / (std^2 + sigma^2)``."""
    c = np.array([[1.0, -1.0]])
    std, sigma = 0.4, 0.9
    x = np.array([[0.3, 0.2], [-1.7, 1.1], [2.2, -0.4]])
    got = analytic_gmm_score(x, c, std, sigma)
    assert np.allclose(got, -(x - c[0]) / (std**2 + sigma**2))


def test_score_field_has_zero_mean_under_its_own_density() -> None:
    """E_p[grad log p] = 0 — a strong global invariant (Stein's identity)."""
    centers = np.array([[-2.0, 0.0], [2.0, 1.0], [0.0, -2.5]])
    std, sigma = 0.4, 0.9
    conv_std = float(np.sqrt(std**2 + sigma**2))
    # Draw *directly* from the convolved mixture (importance weighting from a
    # broad Gaussian proposal has too small an effective sample size here).
    rng = np.random.default_rng(3)
    n = 300_000
    idx = rng.integers(0, len(centers), size=n)
    grid = centers[idx] + conv_std * rng.normal(size=(n, 2))
    sc = analytic_gmm_score(grid, centers, std, sigma)
    mean_score = sc.mean(axis=0)
    stderr = sc.std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(mean_score) < 6.0 * stderr)


# ---------------------------------------------------------------- ladders


def test_karras_ladder_properties() -> None:
    sig = karras_sigma(20, 0.01, 5.0, rho=7.0)
    assert sig.shape == (21,)
    assert sig[0] == pytest.approx(5.0)
    assert sig[-2] == pytest.approx(0.01)
    assert sig[-1] == 0.0
    assert np.all(np.diff(sig) < 0)  # strictly decreasing down to 0


def test_uniform_log_ladder_is_geometric() -> None:
    sig = uniform_log_sigma(10, 0.1, 8.0)
    ratios = sig[:-2] / sig[1:-1]
    assert np.allclose(ratios, ratios[0])
    assert sig[-1] == 0.0


def test_steps_for_budget() -> None:
    assert steps_for_budget(20, 1) == 20
    assert steps_for_budget(20, 2) == 10
    assert steps_for_budget(5, 2) == 3
    from diffuforge.core.errors import SamplerError

    with pytest.raises(SamplerError):
        steps_for_budget(10, 3)
