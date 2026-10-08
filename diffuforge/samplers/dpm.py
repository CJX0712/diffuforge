"""DPM-Solver family (Lu et al. 2022/2023) in the EDM variance-exploding gauge.

Probability-flow ODE in log-sigma coordinates.  With ``u = log sigma`` and
``lambda = -u``, Tweedie's formula gives ``score = (D - x)/sigma^2`` and

    dx/dlambda = exp(-2 lambda) * s(x, lambda).

DPM-Solver integrates the exponential factor *exactly* and approximates ``s`` by
a polynomial in ``lambda`` built from past evaluations — which is precisely why
it is sample-quality-optimal at a fixed NFE budget.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import SamplerError
from .base import BaseSampler, final_denoise, init_noise
from .schedule import karras_sigma, steps_for_budget


def lambda_moments(sig_i: float, sig_j: float) -> np.ndarray:
    """Exact integrals ``I_k = int_{lam_i}^{lam_j} e^{-2 lam} (lam - lam_i)^k dlam``.

    Returns ``[I0, I1, I2, I3]``; handles the terminal ``sig_j -> 0`` limit.
    """
    s2i = sig_i * sig_i
    s2j = sig_j * sig_j
    i0 = 0.5 * (s2i - s2j)
    if sig_j <= 0.0:
        return np.array([i0, 0.25 * s2i, 0.25 * s2i, 0.375 * s2i])
    h = float(np.log(sig_i / sig_j))
    h2 = h * h
    i1 = 0.25 * s2i - s2j * (0.25 + 0.5 * h)
    i2 = 0.25 * s2i - s2j * (0.25 + 0.5 * h + 0.5 * h2)
    i3 = 0.375 * s2i - s2j * 0.375 * (1.0 + 2.0 * h + 2.0 * h2 + (4.0 / 3.0) * h2 * h)
    return np.array([i0, i1, i2, i3])


def sigma_moments(sig_i: float, sig_j: float) -> np.ndarray:
    """Plain monomial moments ``int_{sig_i}^{sig_j} (sig - sig_i)^k dsig``.

    These are the counterparts of :func:`lambda_moments` for the Karras
    sigma-space gauge ``dx/dsig = (x - D(x; sig)) / sig``.
    """
    d = float(sig_j) - float(sig_i)
    return np.array([d, d * d / 2.0, d**3 / 3.0, d**4 / 4.0])


def _poly_from_roots(roots: np.ndarray) -> list[float]:
    """Ascending-power coefficients of ``prod (tau - r)``."""
    coeffs = [1.0]
    for r in roots:
        new = [0.0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i + 1] += c
            new[i] += -float(r) * c
        coeffs = new
    return coeffs


def lagrange_power_coeffs(taus) -> list[tuple[float, ...]]:
    """Lagrange basis polynomials expanded in powers of ``tau = lam - lam_i``.

    ``taus[j]`` are the (pairwise distinct) node offsets; the j-th returned tuple
    holds the ascending-power coefficients of the j-th Lagrange basis polynomial.
    """
    taus = np.asarray(taus, dtype=np.float64)
    k = len(taus)
    if k < 1 or k > 4:
        raise SamplerError("only orders 1..4 supported", order=k)
    out: list[tuple[float, ...]] = []
    for j in range(k):
        others = np.delete(taus, j)
        denom = float(np.prod(taus[j] - others))
        if abs(denom) < 1e-14:
            raise SamplerError("degenerate (repeated) interpolation nodes", taus=taus)
        num = _poly_from_roots(others)
        out.append(tuple(c / denom for c in num))
    return out


def poly_update(
    x: np.ndarray,
    taus: np.ndarray,
    scores: list[np.ndarray],
    mom: np.ndarray,
) -> np.ndarray:
    """Advance the solution by integrating a polynomial interpolant of the field.

    ``taus[j]`` are the interpolation node offsets relative to the left endpoint
    (they may include the right endpoint — that is what turns an
    Adams-Bashforth predictor into an Adams-Moulton style corrector); ``mom[k]``
    is the exact moment ``int t^k`` over the step in the chosen gauge.
    """
    coeffs = lagrange_power_coeffs(taus)
    delta = np.zeros_like(x)
    for coef, s in zip(coeffs, scores, strict=False):
        acc = 0.0
        for p, c in enumerate(coef):
            if p < len(mom):
                acc += c * mom[p]
        delta += s * acc
    return x + delta


class DPM1Sampler(BaseSampler):
    """DPM-Solver-1: exponential integrator with a piecewise-constant score."""

    name = "dpm1"
    nfe_per_step = 1

    def __init__(self, rho: float = 7.0) -> None:
        self.rho = float(rho)

    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:
        n_steps = steps_for_budget(nfe_budget, 1)
        sig = karras_sigma(n_steps, model.sigma_min, model.sigma_max, self.rho)
        x = init_noise(model, n, rng)
        for i in range(n_steps):
            if sig[i + 1] <= 0.0:
                x = final_denoise(model, x, sig[i])
                break
            d = model.denoise(x, sig[i])
            s = (d - x) / (sig[i] ** 2)
            x = x + s * lambda_moments(sig[i], sig[i + 1])[0]
        return x


class DPM2SSampler(BaseSampler):
    """DPM-Solver-2 (single-step): midpoint evaluation, 2 NFE per step."""

    name = "dpm2s"
    nfe_per_step = 2

    def __init__(self, rho: float = 7.0) -> None:
        self.rho = float(rho)

    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:
        n_steps = steps_for_budget(nfe_budget, 2)
        sig = karras_sigma(n_steps, model.sigma_min, model.sigma_max, self.rho)
        x = init_noise(model, n, rng)
        for i in range(n_steps):
            if sig[i + 1] <= 0.0:
                x = final_denoise(model, x, sig[i])
                break
            si, sj = sig[i], sig[i + 1]
            d = model.denoise(x, si)
            s_i = (d - x) / (si * si)
            sm = float(np.sqrt(max(si * sj, 0.0)))
            if sm <= 0.0:  # pragma: no cover - guarded by the branch above
                x = final_denoise(model, x, si)
                break
            xm = x + s_i * lambda_moments(si, sm)[0]
            dm = model.denoise(xm, sm)
            s_m = (dm - xm) / (sm * sm)
            mom = lambda_moments(si, sj)
            lam_i, lam_m = -np.log(si), -np.log(sm)
            c = (s_m - s_i) / (lam_m - lam_i)
            x = x + s_i * mom[0] + c * mom[1]
        return x


class _MultistepSampler(BaseSampler):
    """Base for DPM-Solver++ multistep variants (1 NFE per step)."""

    name = "dpmms"
    nfe_per_step = 1
    max_order = 2

    def __init__(self, rho: float = 7.0) -> None:
        self.rho = float(rho)

    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:
        n_steps = steps_for_budget(nfe_budget, 1)
        sig = karras_sigma(n_steps, model.sigma_min, model.sigma_max, self.rho)
        x = init_noise(model, n, rng)
        hist_lam: list[float] = []
        hist_s: list[np.ndarray] = []
        for i in range(n_steps):
            if sig[i + 1] <= 0.0:
                x = final_denoise(model, x, sig[i])
                break
            d = model.denoise(x, sig[i])
            s = (d - x) / (sig[i] ** 2)
            hist_lam.insert(0, -float(np.log(sig[i])))
            hist_s.insert(0, s)
            if len(hist_s) > 4:
                hist_lam.pop()
                hist_s.pop()
            order = int(min(i + 1, self.max_order))
            taus = np.array(hist_lam[:order]) - hist_lam[0]
            x = poly_update(x, taus, hist_s[:order], lambda_moments(sig[i], sig[i + 1]))
        return x


class DPM2MSampler(_MultistepSampler):
    """DPM-Solver++ 2M: second-order multistep (the classic speed champion)."""

    name = "dpm2m"
    max_order = 2


class DPM3MSampler(_MultistepSampler):
    """DPM-Solver++ 3M: third-order multistep."""

    name = "dpm3m"
    max_order = 3


class DPM4MSampler(_MultistepSampler):
    """Fourth-order multistep exponential integrator (beyond the published 3M)."""

    name = "dpm4m"
    max_order = 4


__all__ = [
    "DPM1Sampler",
    "DPM2MSampler",
    "DPM2SSampler",
    "DPM3MSampler",
    "DPM4MSampler",
    "lagrange_power_coeffs",
    "lambda_moments",
    "poly_update",
    "sigma_moments",
]
