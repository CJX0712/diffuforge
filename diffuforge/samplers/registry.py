"""Sampler registry — the single place where names map to implementations."""

from __future__ import annotations

from typing import Any

from ..core.config import DiffusionConfig
from ..core.errors import SamplerError
from .base import BaseSampler
from .diffufuse import DiffuFuseSampler
from .dpm import (
    DPM1Sampler,
    DPM2MSampler,
    DPM2SSampler,
    DPM3MSampler,
    DPM4MSampler,
)
from .ode import EulerSampler, HeunSampler, LogEulerSampler
from .unipc import UniPCSampler

FLAGSHIP = "diffufuse"

SAMPLERS: dict[str, type[BaseSampler]] = {
    "euler": EulerSampler,
    "logeuler": LogEulerSampler,
    "heun": HeunSampler,
    "dpm1": DPM1Sampler,
    "dpm2s": DPM2SSampler,
    "dpm2m": DPM2MSampler,
    "dpm3m": DPM3MSampler,
    "dpm4m": DPM4MSampler,
    "unipc": UniPCSampler,
    "diffufuse": DiffuFuseSampler,
}


def all_sampler_names() -> list[str]:
    return sorted(SAMPLERS)


def baseline_names() -> list[str]:
    return [n for n in sorted(SAMPLERS) if n != FLAGSHIP]


def get_sampler(name: str, cfg: DiffusionConfig | None = None, **kw: Any) -> BaseSampler:
    if name not in SAMPLERS:
        raise SamplerError("unknown sampler", name=name, available=all_sampler_names())
    cls = SAMPLERS[name]
    if name == FLAGSHIP:
        return cls(cfg=cfg, **kw)
    # Fair comparison: every sampler walks the *same* sigma ladder, so the
    # configured rho is applied to the baselines too, not just the flagship.
    if cfg is not None and "rho" not in kw:
        kw["rho"] = float(cfg.rho_schedule)
    return cls(**kw)


__all__ = [
    "FLAGSHIP",
    "SAMPLERS",
    "all_sampler_names",
    "baseline_names",
    "get_sampler",
]
