"""Global determinism: one entry point, all RNG streams pinned."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

_STATE: dict[str, Any] = {"seed": None}


def set_all(seed: int) -> None:
    """Pin every RNG stream used anywhere in the system."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    _STATE["seed"] = seed


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a ``np.random.Generator`` (PCG64) for the requested seed."""
    if seed is None:
        seed = _STATE.get("seed") or 0
    return np.random.default_rng(int(seed))


def current_seed() -> int | None:
    return _STATE.get("seed")


__all__ = ["current_seed", "get_rng", "set_all"]
