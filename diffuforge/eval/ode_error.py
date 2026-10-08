"""ODE-integration accuracy: the metric that actually discriminates samplers.

Every sampler here solves *the same* probability-flow ODE

    dx/dlambda = D(x; sigma) - x,      lambda = -log sigma

with the *same* denoiser.  A converged solution can therefore be computed once
(dense Karras-Heun), and any sampler's error is the distance between its own
trajectory endpoint and that reference, started from the **same initial noise**
(common random numbers).  This is a paired, deterministic measurement with
variance roughly three orders of magnitude below sample-based W2^2, which is why
it is the benchmark's primary metric.

Why not just use W2^2?  Measured on this benchmark (see ``docs/architecture.md``
§7): at NFE >= 10 every sampler already sits at the *model's* quality ceiling —
W2^2 against a 512-point reference is ~5x the i.i.d. noise floor and the spread
between samplers collapses to 1-2%.  W2^2 is reported as a secondary metric
together with its noise floor, but it cannot arbitrate sampler quality.
"""

from __future__ import annotations

import numpy as np

from ..core.config import DiffusionConfig
from ..samplers.ode import HeunSampler

__all__ = ["dense_reference", "ode_rel_err"]


def dense_reference(
    model,
    cfg: DiffusionConfig | None = None,
    rng=None,
    n: int | None = None,
    nfe: int | None = None,
) -> tuple[np.ndarray, int]:
    """Converged PF-ODE solution: Karras-Heun at a large NFE budget.

    Heun is a *baseline* sampler (not the flagship), so the reference is not
    circular.  At ``nfe = 1500`` its own error is ~1e-6 relative, i.e. three
    orders below the NFE=10/20 errors it is used to score.
    """
    cfg = cfg or DiffusionConfig()
    budget = int(cfg.ode_ref_nfe if nfe is None else nfe)
    n_pts = int(cfg.ode_ref_points if n is None else n)
    sampler = HeunSampler(rho=float(cfg.rho_schedule))
    return sampler.sample(model, n_pts, budget, rng)


def ode_rel_err(gen: np.ndarray, ref: np.ndarray) -> float:
    """Relative RMS distance between a sampler's endpoint and the reference.

    The reference is cheaper to produce than a full-size cloud, so only the
    common prefix is compared — both start from the same initial noise, hence
    ``gen[:len(ref)]`` corresponds point-for-point to ``ref``.  Normalised by
    ``rms(ref)`` so the number is comparable across densities.
    """
    gen = np.asarray(gen, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if gen.shape[0] < ref.shape[0]:
        raise ValueError(f"need at least {ref.shape[0]} generated points, got {gen.shape[0]}")
    gen = gen[: ref.shape[0]]
    if gen.shape != ref.shape:
        raise ValueError(f"shape mismatch: {gen.shape} vs {ref.shape}")
    scale = float(np.sqrt(np.mean(ref * ref)))
    if scale <= 0.0:  # pragma: no cover - degenerate
        return float("nan")
    return float(np.sqrt(np.mean((gen - ref) ** 2)) / scale)
