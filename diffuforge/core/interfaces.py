"""Protocol contracts.  Layers above depend only on these, never on concretions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Denoiser(Protocol):
    """EDM-preconditioned denoiser ``D(x; sigma)``.

    Contract: ``denoise(x, sigma) -> D`` with ``D`` the MMSE estimate of the
    clean sample; ``score = (D - x) / sigma**2`` (Tweedie, 1956).
    """

    sigma_data: float
    sigma_min: float
    sigma_max: float

    def denoise(self, x: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
        """Return ``D(x; sigma)``; broadcasting over a leading batch axis."""
        ...

    def score(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Return ``grad_x log p_sigma(x)``."""
        ...

    def nfe(self) -> int:
        """Number of network forward evaluations performed so far."""
        ...

    def reset_nfe(self) -> None:
        """Zero the NFE counter."""
        ...


@runtime_checkable
class Sampler(Protocol):
    """Deterministic (or seeded-stochastic) sampler operating under an NFE budget."""

    name: str

    def sample(
        self,
        model: Denoiser,
        n: int,
        nfe_budget: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Return ``(samples, nfe_used)``; ``nfe_used <= nfe_budget``."""
        ...


__all__ = ["Denoiser", "Sampler"]
