"""Turn a set of generated samples into an :class:`EvalResult`."""

from __future__ import annotations

import numpy as np

from ..core.config import DiffusionConfig
from ..core.types import EvalResult
from .metrics import all_metrics


def evaluate_samples(
    ref: np.ndarray,
    gen: np.ndarray,
    *,
    dataset: str,
    seed: int,
    sampler: str,
    nfe_budget: int,
    nfe_used: int,
    cfg: DiffusionConfig | None = None,
    rng: np.random.Generator | None = None,
    elapsed_sec: float = 0.0,
) -> EvalResult:
    cfg = cfg or DiffusionConfig()
    m = all_metrics(ref, gen, cfg, rng)
    return EvalResult(
        dataset=dataset,
        seed=int(seed),
        sampler=sampler,
        nfe_budget=int(nfe_budget),
        nfe_used=int(nfe_used),
        w2_sq=float(m["w2_sq"]),
        mmd2=float(m["mmd2"]),
        coverage=float(m["coverage"]),
        nn_tst=float(m["nn_tst"]),
        elapsed_sec=float(elapsed_sec),
    )


__all__ = ["evaluate_samples"]
