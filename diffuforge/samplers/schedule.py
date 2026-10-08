"""Time discretisations shared by every sampler.

Karras et al. (2022) propose a polynomially spaced sigma ladder:

    sigma_i = ( sigma_max^{1/rho} + i/(N-1) * (sigma_min^{1/rho} - sigma_max^{1/rho}) )^rho

with the final step landing exactly on ``sigma = 0``.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import SamplerError


def karras_sigma(n_steps: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> np.ndarray:
    """Return ``(n_steps + 1,)`` sigmas from ``sigma_max`` down to ``0``."""
    if n_steps < 1:
        raise SamplerError("n_steps must be >= 1", n_steps=n_steps)
    if sigma_min <= 0 or sigma_max <= sigma_min:
        raise SamplerError(
            "require 0 < sigma_min < sigma_max",
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    t_min, t_max = float(sigma_min), float(sigma_max)
    inv_rho = 1.0 / float(rho)
    ramp = np.linspace(0.0, 1.0, int(n_steps))
    sig = (t_max**inv_rho + ramp * (t_min**inv_rho - t_max**inv_rho)) ** float(rho)
    sig = np.concatenate([sig, np.array([0.0])])
    return sig


def uniform_log_sigma(n_steps: int, sigma_min: float, sigma_max: float) -> np.ndarray:
    """Geometrically spaced ladder, ending at 0."""
    if n_steps < 1:
        raise SamplerError("n_steps must be >= 1", n_steps=n_steps)
    sig = np.exp(np.linspace(float(np.log(sigma_max)), float(np.log(sigma_min)), int(n_steps)))
    return np.concatenate([sig, np.array([0.0])])


def steps_for_budget(nfe_budget: int, nfe_per_step: int) -> int:
    """Largest step count whose NFE cost fits in the budget.

    ``euler``/multistep samplers cost ``1`` NFE per step; the Karras Heun scheme
    costs ``2`` per step except for the final (sigma -> 0) step which costs ``1``.
    """
    if nfe_per_step == 1:
        return max(1, int(nfe_budget))
    if nfe_per_step == 2:
        return max(1, (int(nfe_budget) + 1) // 2)
    raise SamplerError("unsupported nfe_per_step", value=nfe_per_step)


__all__ = ["karras_sigma", "steps_for_budget", "uniform_log_sigma"]
