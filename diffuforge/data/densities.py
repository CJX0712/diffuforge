"""Six 2-D densities used as the benchmark suite.

Every density exposes an exact sampler so that the reference distribution is
available in closed form — no dataset download, no network, no leakage.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import DensitySpec


def _gmm8(n: int, rng: np.random.Generator) -> np.ndarray:
    k = rng.integers(0, 8, size=n)
    ang = 2.0 * np.pi * k / 8.0
    c = np.stack([3.0 * np.cos(ang), 3.0 * np.sin(ang)], axis=1)
    return c + 0.25 * rng.standard_normal((n, 2))


def _rings(n: int, rng: np.random.Generator) -> np.ndarray:
    j = rng.integers(0, 3, size=n)
    r = np.array([1.5, 3.0, 4.5])[j]
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    return pts + 0.12 * rng.standard_normal((n, 2))


def _moons(n: int, rng: np.random.Generator) -> np.ndarray:
    half = n // 2
    rest = n - half
    out = np.empty((n, 2))
    for start, count, sign in ((0, half, 1.0), (half, rest, -1.0)):
        th = rng.uniform(0.0, np.pi, size=count)
        x = np.stack([np.cos(th), np.sin(th)], axis=1)
        rot = np.array(
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]]
        )
        x = x @ rot.T
        x[:, 1] *= sign
        x = x + np.array([0.0, 0.35 * sign]) + np.array([0.0, 0.0])
        x = x * 2.0
        out[start : start + count] = x + 0.10 * rng.standard_normal((count, 2))
    return out


def _spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    half = n // 2
    rest = n - half
    out = np.empty((n, 2))
    for start, count, phase in ((0, half, 0.0), (half, rest, np.pi)):
        t = np.sqrt(rng.uniform(0.0, 1.0, size=count)) * 3.2
        th = 2.4 * np.pi * (t / 3.2) + phase
        pts = np.stack([t * np.cos(th), t * np.sin(th)], axis=1)
        out[start : start + count] = pts + 0.09 * rng.standard_normal((count, 2))
    return out


def _checker(n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi, cell = -4.0, 4.0, 2.0
    out = np.empty((0, 2))
    while out.shape[0] < n:
        cand = rng.uniform(lo, hi, size=(2 * n, 2))
        ix = np.floor((cand[:, 0] - lo) / cell)
        iy = np.floor((cand[:, 1] - lo) / cell)
        keep = np.asarray((ix + iy) % 2 == 0)
        out = cand[keep] if out.shape[0] == 0 else np.vstack([out, cand[keep]])
    return out[:n]


def _banana(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((n, 2))
    z[:, 1] *= 1.4
    x1 = z[:, 0] * 1.6
    x2 = z[:, 1] + 0.28 * (z[:, 0] ** 2 - 2.0)
    return np.stack([x1, x2], axis=1)


DENSITIES: dict[str, DensitySpec] = {
    "gmm8": DensitySpec(
        name="gmm8",
        sampler=_gmm8,
        mode_radius=0.6,
        description="8 isotropic Gaussians on a circle of radius 3 (std 0.25)",
    ),
    "rings": DensitySpec(
        name="rings",
        sampler=_rings,
        mode_radius=0.4,
        description="three concentric rings r=1.5/3.0/4.5 with 0.12 jitter",
    ),
    "moons": DensitySpec(
        name="moons",
        sampler=_moons,
        mode_radius=0.5,
        description="two interleaving half-moons, 0.10 jitter",
    ),
    "spiral": DensitySpec(
        name="spiral",
        sampler=_spiral,
        mode_radius=0.4,
        description="two-arm Archimedean spiral, 0.09 jitter",
    ),
    "checker": DensitySpec(
        name="checker",
        sampler=_checker,
        mode_radius=1.0,
        description="8x8 checkerboard on [-4,4]^2",
    ),
    "banana": DensitySpec(
        name="banana",
        sampler=_banana,
        mode_radius=0.8,
        description="curved banana (quadratic warping of a Gaussian)",
    ),
}


def list_densities() -> list[str]:
    return sorted(DENSITIES)


def get_density(name: str) -> DensitySpec:
    if name not in DENSITIES:
        raise DataError("unknown density", name=name, available=sorted(DENSITIES))
    return DENSITIES[name]


# --------------------------------------------------------------------------
# Analytic Gaussian-mixture quantities — the gold standard for the score head.
# --------------------------------------------------------------------------


def analytic_gmm_logpdf(
    x: np.ndarray, centers: np.ndarray, std: float, weights: np.ndarray | None = None
) -> np.ndarray:
    """log density of a Gaussian mixture at *x* (shape ``(n, d)``)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    k = centers.shape[0]
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    d = x.shape[1]
    # (n, k) squared distances
    diff = x[:, None, :] - centers[None, :, :]
    q = np.einsum("nkd,nkd->nk", diff, diff) / (2.0 * std * std)
    lp = np.log(w)[None, :] - q - 0.5 * d * np.log(2.0 * np.pi * std * std)
    m = lp.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))


def analytic_gmm_score(
    x: np.ndarray,
    centers: np.ndarray,
    std: float,
    sigma: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """``grad_x log p_sigma(x)`` where ``p_sigma`` = GMM convolved with N(0, sigma^2 I).

    Convolving a GMM with isotropic Gaussian noise keeps it a GMM with the same
    centres, weights and per-component variance ``std**2 + sigma**2``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    k = centers.shape[0]
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    var = std * std + sigma * sigma
    diff = x[:, None, :] - centers[None, :, :]  # (n, k, d)
    q = np.einsum("nkd,nkd->nk", diff, diff) / (2.0 * var)
    lp = np.log(w)[None, :] - q
    m = lp.max(axis=1, keepdims=True)
    e = np.exp(lp - m)
    resp = e / e.sum(axis=1, keepdims=True)  # (n, k)
    # grad log p = - sum_k resp_k (x - c_k) / var
    return -np.einsum("nk,nkd->nd", resp, diff) / var


__all__ = [
    "DENSITIES",
    "analytic_gmm_logpdf",
    "analytic_gmm_score",
    "get_density",
    "list_densities",
]
