"""Shared pytest fixtures.

Every fixture is deliberately tiny (a few hundred training iterations, 32-point
clouds) so the suite stays a unit-test-speed safety net.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffuforge.core.config import DiffusionConfig  # noqa: E402
from diffuforge.core.seed import get_rng, set_all  # noqa: E402
from diffuforge.data.densities import get_density  # noqa: E402
from diffuforge.score.train import train_denoiser  # noqa: E402

os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture(scope="session")
def fast_cfg() -> DiffusionConfig:
    """A deliberately small configuration used by every test."""
    return DiffusionConfig(
        n_train=2_000,
        n_eval_ref=32,
        n_gen=32,
        n_iters=60,
        batch_size=64,
        eval_ot_size=32,
        ode_ref_nfe=200,
        ode_ref_points=32,
        seeds=(7, 17),
        nfe_budgets=(10, 20),
        datasets=("gmm8", "rings"),
    )


@pytest.fixture(scope="session")
def trained(fast_cfg: DiffusionConfig):
    """One trained denoiser per (dataset, seed) for the two fast datasets."""
    out = {}
    for ds in fast_cfg.datasets:
        for seed in fast_cfg.seeds:
            set_all(int(seed))
            model, result = train_denoiser(get_density(ds), int(seed), fast_cfg)
            out[(ds, int(seed))] = (model, result)
    return out


@pytest.fixture()
def rng():
    return get_rng(20240501)
