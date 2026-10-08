"""Typed value objects passed between layers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DensitySpec:
    """A 2-D target density with an analytic sampler and (optionally) modes."""

    name: str
    sampler: Callable[[int, np.random.Generator], np.ndarray]
    modes: np.ndarray | None = None  # (K, 2) mode centres, for mode coverage
    mode_radius: float = 0.5
    description: str = ""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.asarray(self.sampler(n, rng), dtype=np.float64)
        if out.ndim != 2 or out.shape[1] != 2 or out.shape[0] != n:
            raise ValueError(f"sampler for {self.name} returned shape {out.shape}")
        return out


@dataclass
class TrainResult:
    """Outcome of fitting a denoiser on one density / seed."""

    dataset: str
    seed: int
    sigma_data: float
    final_loss: float
    loss_trace: list[float] = field(default_factory=list)
    n_iters: int = 0
    elapsed_sec: float = 0.0


@dataclass
class SampleResult:
    """Samples produced by one sampler at one NFE budget."""

    dataset: str
    seed: int
    sampler: str
    nfe_budget: int
    nfe_used: int
    samples: np.ndarray
    elapsed_sec: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Distribution-level quality of one sample set against a reference."""

    dataset: str
    seed: int
    sampler: str
    nfe_budget: int
    nfe_used: int
    w2_sq: float
    mmd2: float
    coverage: float
    nn_tst: float  # 1-NN two-sample test AUC; 0.5 == indistinguishable
    elapsed_sec: float = 0.0


@dataclass
class BenchmarkRow:
    """One (dataset, seed, sampler, nfe) cell of the benchmark table."""

    dataset: str
    seed: int
    sampler: str
    nfe_budget: int
    nfe_used: int
    w2_sq: float
    mmd2: float
    coverage: float
    nn_tst: float
    ode_err: float = float("nan")  # PF-ODE trajectory error vs dense reference


__all__ = [
    "BenchmarkRow",
    "DensitySpec",
    "EvalResult",
    "SampleResult",
    "TrainResult",
]
