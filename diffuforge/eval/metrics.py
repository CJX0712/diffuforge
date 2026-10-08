"""Sample-quality metrics.  Convention: **every metric is "smaller is better"**.

* ``w2_sq``     — squared 2-Wasserstein via *exact* optimal transport (Hungarian).
* ``mmd2``      — unbiased squared MMD with a multi-scale RBF kernel.
* ``coverage``  — 1 - (probability mass of the reference covered by the samples);
                  reported as a *miss* rate so that smaller is better.
* ``nn_tst``    — 1-NN two-sample test: ``2 * |accuracy - 0.5|``; 0 == indistinguishable.
"""

from __future__ import annotations

import numpy as np

from ..core.config import DiffusionConfig
from ..core.errors import EvalError

try:  # Tier-0: exact assignment via SciPy (Hungarian, O(n^3)).
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:  # pragma: no cover - Tier-1 fallback
    _lsa = None

_MMD_DEFAULT = (0.1, 0.25, 0.5, 1.0, 2.0)


def _sinkhorn_cost(cost: np.ndarray, reg: float = 0.05, iters: int = 400) -> float:
    """Pure-NumPy entropic OT fallback (used when SciPy is unavailable)."""
    n = cost.shape[0]
    k = np.exp(-cost / max(reg, 1e-12))
    k = np.maximum(k, 1e-300)
    u = np.full(n, 1.0 / n)
    v = np.full(n, 1.0 / n)
    for _ in range(iters):
        u = 1.0 / (n * (k @ v))
        v = 1.0 / (n * (k.T @ u))
    plan = np.outer(u, v) * k
    return float((plan * cost).sum())


def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise EvalError("metric input must be 2-D", shape=a.shape)
    return a


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.einsum("ij,ij->i", a, a)[:, None]
    bb = np.einsum("ij,ij->i", b, b)[None, :]
    d = aa + bb - 2.0 * (a @ b.T)
    np.maximum(d, 0.0, out=d)
    return d


def w2_sq(
    ref: np.ndarray, gen: np.ndarray, max_n: int = 256, rng: np.random.Generator | None = None
) -> float:
    """Squared 2-Wasserstein distance between two equal-sized empirical measures.

    Uses the exact assignment solution (no entropic bias); both clouds are
    subsampled to the same size so the marginals are uniform.
    """
    ref, gen = _as2d(ref), _as2d(gen)
    m = int(min(len(ref), len(gen), max_n))
    if m < 2:
        raise EvalError("need at least 2 points per cloud", m=m)
    # Both clouds are i.i.d. draws, so taking a common prefix is an unbiased
    # equal-sized subsample — and it keeps the metric free of extra RNG noise.
    cost = _sq_dist(ref[:m], gen[:m])
    if _lsa is None:  # pragma: no cover - Tier-1 offline fallback
        return _sinkhorn_cost(cost)
    rows, cols = _lsa(cost)
    return float(cost[rows, cols].mean())


def mmd2(ref: np.ndarray, gen: np.ndarray, bandwidths: tuple[float, ...] = _MMD_DEFAULT) -> float:
    """Unbiased squared MMD with a mixture of RBF kernels (diag removed)."""
    ref, gen = _as2d(ref), _as2d(gen)
    n, m = len(ref), len(gen)
    if n < 2 or m < 2:
        raise EvalError("need at least 2 points per cloud", n=n, m=m)
    total = 0.0
    for bw in bandwidths:
        g = 1.0 / (2.0 * float(bw) ** 2)
        kxx = np.exp(-g * _sq_dist(ref, ref))
        kyy = np.exp(-g * _sq_dist(gen, gen))
        kxy = np.exp(-g * _sq_dist(ref, gen))
        sxx = kxx.sum() - np.trace(kxx)
        syy = kyy.sum() - np.trace(kyy)
        total += sxx / (n * (n - 1)) + syy / (m * (m - 1)) - 2.0 * kxy.mean()
    return float(total / len(bandwidths))


def coverage(ref: np.ndarray, gen: np.ndarray, bins: int = 16) -> float:
    """Missed reference mass: ``1 - sum of reference mass in occupied cells``.

    Both clouds are binned on the common axis-aligned bounding box of *ref*.
    """
    ref, gen = _as2d(ref), _as2d(gen)
    lo = ref.min(axis=0)
    hi = ref.max(axis=0)
    span = hi - lo
    span = np.where(span < 1e-9, 1.0, span)
    edges = [np.linspace(lo[d] - 1e-9, hi[d] + 1e-9, bins + 1) for d in range(ref.shape[1])]
    idx_ref = np.stack(
        [np.clip(np.digitize(ref[:, d], edges[d]) - 1, 0, bins - 1) for d in range(ref.shape[1])],
        axis=1,
    )
    idx_gen = np.stack(
        [np.clip(np.digitize(gen[:, d], edges[d]) - 1, 0, bins - 1) for d in range(gen.shape[1])],
        axis=1,
    )
    flat_r = idx_ref[:, 0] * bins + idx_ref[:, 1]
    flat_g = idx_gen[:, 0] * bins + idx_gen[:, 1]
    counts_r = np.bincount(flat_r, minlength=bins * bins).astype(np.float64)
    counts_r /= counts_r.sum()
    occupied = np.zeros(bins * bins, dtype=bool)
    occupied[np.unique(flat_g)] = True
    return float(1.0 - counts_r[occupied].sum())


def nn_tst(ref: np.ndarray, gen: np.ndarray) -> float:
    """1-NN two-sample test statistic ``2|acc - 0.5|`` (0 == indistinguishable)."""
    ref, gen = _as2d(ref), _as2d(gen)
    n, m = len(ref), len(gen)
    if n < 2 or m < 2:
        raise EvalError("need at least 2 points per cloud", n=n, m=m)
    x = np.vstack([ref, gen])
    labels = np.concatenate([np.zeros(n), np.ones(m)])
    d = _sq_dist(x, x)
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argmin(d, axis=1)
    acc = float(np.mean(labels[nn_idx] == labels))
    return float(abs(acc - 0.5) * 2.0)


def all_metrics(
    ref: np.ndarray,
    gen: np.ndarray,
    cfg: DiffusionConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    cfg = cfg or DiffusionConfig()
    return {
        "w2_sq": w2_sq(ref, gen, int(cfg.eval_ot_size), rng),
        "mmd2": mmd2(ref, gen, tuple(cfg.mmd_bandwidths)),
        "coverage": coverage(ref, gen),
        "nn_tst": nn_tst(ref, gen),
    }


__all__ = ["all_metrics", "coverage", "mmd2", "nn_tst", "w2_sq"]
