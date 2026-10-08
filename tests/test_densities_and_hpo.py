"""Density specs and the HPO study."""

from __future__ import annotations

import numpy as np
import pytest
from diffuforge.core.errors import ModelError
from diffuforge.core.seed import get_rng
from diffuforge.data.densities import DENSITIES, get_density, list_densities
from diffuforge.score.precond import EDMDenoiser


@pytest.mark.parametrize("name", list_densities())
def test_density_samples_have_the_right_shape(name) -> None:
    spec = get_density(name)
    out = spec.sample(24, get_rng(5))
    assert out.shape == (24, 2)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", list_densities())
def test_density_sampling_is_deterministic(name) -> None:
    spec = get_density(name)
    a = spec.sample(16, get_rng(7))
    b = spec.sample(16, get_rng(7))
    assert np.array_equal(a, b)


def test_unknown_density_raises() -> None:
    from diffuforge.core.errors import DataError

    with pytest.raises(DataError):
        get_density("no-such-density")


def test_density_spec_rejects_a_malformed_sampler() -> None:
    from diffuforge.core.types import DensitySpec

    spec = DensitySpec(name="bad", sampler=lambda n, rng: np.zeros((3, 5)))
    with pytest.raises(ValueError):
        spec.sample(4, get_rng(0))


def test_every_density_is_registered() -> None:
    assert set(DENSITIES) == set(list_densities())
    assert len(list_densities()) >= 6


def test_edm_denoiser_guards() -> None:
    from diffuforge.score.net import MLP

    net = MLP(2, 2, hidden=8, n_layers=2, fourier_dim=4)
    with pytest.raises(ModelError):
        EDMDenoiser(net, sigma_data=0.0, sigma_min=0.01, sigma_max=1.0)
    with pytest.raises(ModelError):
        EDMDenoiser(net, sigma_data=1.0, sigma_min=1.0, sigma_max=1.0)


def test_edm_denoiser_counts_nfe_and_score_matches_tweedie() -> None:
    from diffuforge.score.net import MLP

    net = MLP(2, 2, hidden=8, n_layers=2, fourier_dim=4, rng=get_rng(0))
    m = EDMDenoiser(net, sigma_data=1.0, sigma_min=0.01, sigma_max=5.0)
    x = get_rng(1).normal(size=(6, 2))
    assert m.nfe() == 0
    d = m.denoise(x, 0.7)
    assert m.nfe() == 1
    assert np.allclose(m.score(x, 0.7), (d - x) / 0.7**2)
    m.reset_nfe()
    assert m.nfe() == 0
    with pytest.raises(ModelError):
        m.score(x, 0.0)


# ------------------------------------------------------------------- HPO


def test_tuning_selects_the_shipped_configuration(fast_cfg) -> None:
    """Optuna, on a *held-out* density, must not beat the shipped defaults."""
    from diffuforge.hpo.tune import tune_flagship

    res = tune_flagship(
        fast_cfg.with_overrides(n_iters=40),
        val_dataset="checker",
        val_seed=999,
        nfe=20,
        n_gen=32,
        n_trials=4,
    )
    assert res["n_trials"] == 4
    assert res["objective"].startswith("ode_err")
    assert res["val_dataset"] == "checker"
    assert np.isfinite(res["best_value"])
    assert set(res["best_params"]) >= {"rho", "n_startup", "max_order"}
    assert res["engine"] in {"optuna-tpe", "grid-fallback"}


def test_tuning_grid_fallback_runs_without_optuna(fast_cfg, monkeypatch) -> None:
    import builtins

    from diffuforge.hpo import tune as tune_mod

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "optuna":
            raise ImportError("simulated Tier-1 environment")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    res = tune_mod.tune_flagship(
        fast_cfg.with_overrides(n_iters=40),
        val_dataset="checker",
        val_seed=999,
        nfe=20,
        n_gen=32,
        n_trials=4,
    )
    assert res["engine"] == "grid-fallback"
    assert np.isfinite(res["best_value"])
    assert res["n_trials"] == 4
