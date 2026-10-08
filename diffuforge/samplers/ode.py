"""Karras et al. (2022) deterministic samplers (Algorithm 1), sigma-space.

The probability-flow ODE in the EDM (variance-exploding) parameterisation is

    dx/dsigma = (x - D(x; sigma)) / sigma        (sigma decreasing)

and in log-sigma coordinates ``u = log sigma`` it becomes the well-conditioned

    dx/du = x - D(x; sigma),      sigma = exp(u).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import SamplerError
from .base import BaseSampler, final_denoise, init_noise
from .schedule import karras_sigma, steps_for_budget


class EulerSampler(BaseSampler):
    """Karras first-order deterministic sampler (1 NFE per step)."""

    name = "euler"
    nfe_per_step = 1

    def __init__(self, rho: float = 7.0) -> None:
        self.rho = float(rho)

    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:
        n_steps = steps_for_budget(nfe_budget, 1)
        sig = karras_sigma(n_steps, model.sigma_min, model.sigma_max, self.rho)
        x = init_noise(model, n, rng)
        for i in range(n_steps):
            d = (x - model.denoise(x, sig[i])) / sig[i]
            x = x + (sig[i + 1] - sig[i]) * d
        return x


class HeunSampler(BaseSampler):
    """Karras second-order deterministic sampler (2 NFE per step, 1 on the last)."""

    name = "heun"
    nfe_per_step = 2

    def __init__(self, rho: float = 7.0) -> None:
        self.rho = float(rho)

    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:
        n_steps = steps_for_budget(nfe_budget, 2)
        sig = karras_sigma(n_steps, model.sigma_min, model.sigma_max, self.rho)
        if sig[-1] != 0.0:  # pragma: no cover - karras_sigma always ends at 0
            raise SamplerError("ladder must terminate at sigma=0")
        x = init_noise(model, n, rng)
        for i in range(n_steps):
            # sigma -> 0: the exact terminal move is x_0 = D(x; sigma_last);
            # checking *before* the slope evaluation avoids a wasted NFE.
            if sig[i + 1] <= 0.0:
                x = final_denoise(model, x, sig[i])
                break
            dsig = sig[i + 1] - sig[i]
            d = (x - model.denoise(x, sig[i])) / sig[i]
            x2 = x + dsig * d
            d2 = (x2 - model.denoise(x2, sig[i + 1])) / sig[i + 1]
            x = x + dsig * 0.5 * (d + d2)
        return x


class LogEulerSampler(BaseSampler):
    """First-order Euler in log-sigma coordinates (1 NFE per step).

    Included as a control: it isolates the effect of the *coordinate system*
    from the effect of the *integration order*.
    """

    name = "logeuler"
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
            f = x - model.denoise(x, sig[i])
            du = float(np.log(sig[i + 1]) - np.log(sig[i]))
            x = x + du * f
        return x


__all__ = ["EulerSampler", "HeunSampler", "LogEulerSampler"]
