"""Shared plumbing: initial noise and the terminal sigma -> 0 step."""

from __future__ import annotations

import numpy as np

from ..core.errors import SamplerError


class BaseSampler:
    """Common scaffolding for every sampler.

    Sub-classes implement :meth:`_run` and declare :attr:`name`.
    """

    name = "base"
    nfe_per_step = 1

    # ------------------------------------------------------------- helpers
    @staticmethod
    def init_noise(model, n: int, rng: np.random.Generator) -> np.ndarray:
        return init_noise(model, n, rng)

    @staticmethod
    def final_denoise(model, x: np.ndarray, sigma: float) -> np.ndarray:
        return final_denoise(model, x, sigma)

    # ------------------------------------------------------------- contract
    def sample(self, model, n: int, nfe_budget: int, rng) -> tuple[np.ndarray, int]:
        model.reset_nfe()
        out = self._run(model, int(n), int(nfe_budget), rng)
        used = model.nfe()
        if used > int(nfe_budget):
            raise SamplerError(
                "sampler exceeded its NFE budget",
                sampler=self.name,
                used=used,
                budget=int(nfe_budget),
            )
        return out, used

    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def init_noise(model, n: int, rng: np.random.Generator) -> np.ndarray:
    """Start point of the VE reverse process at ``sigma = sigma_max``.

    The textbook choice is ``N(0, sigma_max^2 I)``, but the true perturbed
    marginal is ``p_data * N(0, sigma_max^2 I)`` whose mean and covariance are
    ``mu_data`` and ``cov_data + sigma_max^2 I``.  Matching those two moments is
    free (they are training-set statistics) and removes the transient the
    sampler would otherwise spend walking from the origin to the data cloud.
    """
    d = model.net.out_dim
    n = int(n)
    z = rng.standard_normal((n, d))
    if not getattr(model, "exact_moment_init", False):
        return z * float(model.sigma_max)
    mu = getattr(model, "mu_data", None)
    cov = getattr(model, "cov_data", None)
    if mu is None or cov is None:
        return z * float(model.sigma_max)
    s2 = float(model.sigma_max) ** 2
    total = cov + s2 * np.eye(d)
    total = 0.5 * (total + total.T)
    evals, evecs = np.linalg.eigh(total)
    evals = np.clip(evals, 0.0, None)
    return np.asarray(mu, dtype=np.float64)[None, :] + (z * np.sqrt(evals)[None, :]) @ evecs.T


def final_denoise(model, x: np.ndarray, sigma: float) -> np.ndarray:
    """Terminal step of the Karras ladder: ``sigma -> 0`` gives ``x_0 = D(x; sigma)``."""
    if sigma <= 0:
        raise SamplerError("terminal sigma must be > 0", sigma=sigma)
    return model.denoise(x, sigma)


__all__ = ["BaseSampler", "final_denoise", "init_noise"]
