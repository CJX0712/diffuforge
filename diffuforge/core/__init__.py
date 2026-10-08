"""Core primitives: errors, config, seed, types, interfaces."""

from __future__ import annotations

from .config import ConfigError, DiffusionConfig, config_from_env
from .errors import (
    DataError,
    DiffuError,
    EvalError,
    ModelError,
    SamplerError,
)
from .interfaces import Denoiser, Sampler
from .seed import get_rng, set_all
from .types import BenchmarkRow, DensitySpec, EvalResult, SampleResult, TrainResult

__all__ = [
    "BenchmarkRow",
    "ConfigError",
    "DataError",
    "Denoiser",
    "DensitySpec",
    "DiffuError",
    "DiffusionConfig",
    "EvalError",
    "EvalResult",
    "ModelError",
    "SampleResult",
    "Sampler",
    "SamplerError",
    "TrainResult",
    "config_from_env",
    "get_rng",
    "set_all",
]
