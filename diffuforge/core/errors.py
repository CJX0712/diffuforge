"""Error taxonomy (E1xx config / E2xx data / E3xx model / E4xx sampler / E5xx eval)."""

from __future__ import annotations

from typing import Any


class DiffuError(Exception):
    """Base error carrying a stable machine-readable code."""

    code: str = "E000"

    def __init__(self, message: str = "", **ctx: Any) -> None:
        self.ctx = ctx
        detail = f"[{self.code}] {message}" if message else f"[{self.code}]"
        if ctx:
            detail += " | " + ", ".join(f"{k}={v}" for k, v in sorted(ctx.items()))
        super().__init__(detail)


class ConfigError(DiffuError):
    code = "E100"


class DataError(DiffuError):
    code = "E200"


class ModelError(DiffuError):
    code = "E300"


class SamplerError(DiffuError):
    code = "E400"


class EvalError(DiffuError):
    code = "E500"


__all__ = [
    "ConfigError",
    "DataError",
    "DiffuError",
    "EvalError",
    "ModelError",
    "SamplerError",
]
