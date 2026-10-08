"""EDM preconditioning (Karras et al. 2022, "Elucidating the Design Space of
Diffusion-Based Generative Models", NeurIPS 2022 outstanding paper).

    c_skip(sigma)  = sigma_data^2 / (sigma^2 + sigma_data^2)
    c_out(sigma)   = sigma * sigma_data / sqrt(sigma_data^2 + sigma^2)
    c_in(sigma)    = 1 / sqrt(sigma^2 + sigma_data^2)
    c_noise(sigma) = log(sigma) / 4

    D(x; sigma) = c_skip * x + c_out * F(c_in * x ; c_noise)
    score(x; sigma) = (D(x; sigma) - x) / sigma^2        (Tweedie 1956)
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ModelError
from .net import MLP


def edm_coefficients(
    sigma: np.ndarray | float, sigma_data: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(c_skip, c_out, c_in, c_noise)`` for the given sigma."""
    s = np.asarray(sigma, dtype=np.float64)
    var_d = float(sigma_data) ** 2
    denom = s * s + var_d
    c_skip = var_d / denom
    c_out = s * float(sigma_data) / np.sqrt(denom)
    c_in = 1.0 / np.sqrt(denom)
    c_noise = 0.25 * np.log(np.maximum(s, 1e-300))
    return c_skip, c_out, c_in, c_noise


class EDMDenoiser:
    """Wraps an :class:`MLP` into the EDM denoiser interface."""

    def __init__(
        self,
        net: MLP,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        mu_data: np.ndarray | None = None,
        cov_data: np.ndarray | None = None,
        exact_moment_init: bool = False,
    ) -> None:
        if sigma_data <= 0:
            raise ModelError("sigma_data must be > 0", sigma_data=sigma_data)
        if not 0.0 <= sigma_min < sigma_max:
            raise ModelError(
                "require 0 <= sigma_min < sigma_max",
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
        self.net = net
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.mu_data = None if mu_data is None else np.asarray(mu_data, dtype=np.float64)
        self.cov_data = None if cov_data is None else np.asarray(cov_data, dtype=np.float64)
        self.exact_moment_init = bool(exact_moment_init)
        self._nfe = 0

    # ------------------------------------------------------------- interface
    def denoise(self, x: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        s = np.asarray(sigma, dtype=np.float64)
        if s.ndim == 0:
            s = np.full(x.shape[0], float(s), dtype=np.float64)
        if s.shape[0] != x.shape[0]:
            raise ModelError("sigma batch mismatch", sigma_shape=s.shape, x_shape=x.shape)
        self._nfe += 1
        c_skip, c_out, c_in, c_noise = edm_coefficients(s, self.sigma_data)
        f = self.net.forward(x * c_in[:, None], c_noise)
        return c_skip[:, None] * x + c_out[:, None] * f

    def score(self, x: np.ndarray, sigma: float) -> np.ndarray:
        s = float(sigma)
        if s <= 0:
            raise ModelError("sigma must be > 0 for score()", sigma=s)
        d = self.denoise(x, s)
        return (d - x) / (s * s)

    def nfe(self) -> int:
        return self._nfe

    def reset_nfe(self) -> None:
        self._nfe = 0

    def clone(self) -> EDMDenoiser:

        net = MLP(
            self.net.in_dim,
            self.net.out_dim,
            hidden=self.net.hidden,
            n_layers=self.net.n_layers,
            fourier_dim=self.net.fourier_dim,
            fourier_max_freq=self.net.fourier_max_freq,
        )
        net.weights = [w.copy() for w in self.net.weights]
        net.biases = [b.copy() for b in self.net.biases]
        return EDMDenoiser(net, self.sigma_data, self.sigma_min, self.sigma_max)


__all__ = ["EDMDenoiser", "edm_coefficients"]
