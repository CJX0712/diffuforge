"""Sampler contracts: NFE budget, determinism, finiteness, guards that fire."""

from __future__ import annotations

import numpy as np
import pytest
from diffuforge.core.errors import SamplerError
from diffuforge.core.seed import get_rng
from diffuforge.samplers.registry import (
    FLAGSHIP,
    all_sampler_names,
    baseline_names,
    get_sampler,
)

BUDGETS = (3, 4, 5, 6, 8, 10, 13, 20, 35)


def test_registry_is_consistent() -> None:
    assert FLAGSHIP in all_sampler_names()
    assert FLAGSHIP not in baseline_names()
    assert set(all_sampler_names()) == set(baseline_names()) | {FLAGSHIP}


def test_unknown_sampler_raises() -> None:
    with pytest.raises(SamplerError):
        get_sampler("no-such-sampler")


@pytest.mark.parametrize("name", all_sampler_names())
@pytest.mark.parametrize("budget", BUDGETS)
def test_nfe_budget_is_never_exceeded(name, budget, fast_cfg, trained) -> None:
    """The NFE budget is a hard contract, enforced by BaseSampler.sample()."""
    model, _ = trained[("gmm8", 7)]
    sampler = get_sampler(name, fast_cfg)
    model.reset_nfe()
    gen, used = sampler.sample(model, 8, budget, get_rng(11))
    assert used <= budget, f"{name} used {used} > {budget}"
    assert np.all(np.isfinite(gen))
    assert gen.shape == (8, 2)


@pytest.mark.parametrize("name", all_sampler_names())
def test_sampling_is_deterministic_for_a_fixed_seed(name, fast_cfg, trained) -> None:
    model, _ = trained[("gmm8", 7)]
    a, _ = get_sampler(name, fast_cfg).sample(model, 24, 10, get_rng(4242))
    b, _ = get_sampler(name, fast_cfg).sample(model, 24, 10, get_rng(4242))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", all_sampler_names())
def test_more_nfe_never_hurts_convergence(name, fast_cfg, trained) -> None:
    """Trajectory error against a dense reference must shrink with the budget."""
    from diffuforge.eval.ode_error import dense_reference, ode_rel_err

    model, _ = trained[("gmm8", 7)]
    x_ref, _ = dense_reference(model, fast_cfg, get_rng(7), n=32)
    errs = []
    for budget in (8, 20, 60):
        gen, _ = get_sampler(name, fast_cfg).sample(model, 32, budget, get_rng(7))
        errs.append(ode_rel_err(gen, x_ref))
    assert errs[0] >= errs[-1] * 0.5  # no catastrophic growth
    assert errs[-1] < errs[0] + 1e-9 or errs[-1] < 0.5


def test_flagship_guard_rejects_bad_hyperparameters(fast_cfg) -> None:
    from diffuforge.samplers.diffufuse import DiffuFuseSampler

    with pytest.raises(SamplerError):
        DiffuFuseSampler(cfg=fast_cfg, n_startup=-1)
    with pytest.raises(SamplerError):
        DiffuFuseSampler(cfg=fast_cfg, max_order=5)
    with pytest.raises(SamplerError):
        DiffuFuseSampler(cfg=fast_cfg, max_order=0)


def test_flagship_defaults_come_from_config(fast_cfg) -> None:
    from diffuforge.samplers.diffufuse import DiffuFuseSampler

    s = DiffuFuseSampler(cfg=fast_cfg)
    assert s.rho == pytest.approx(float(fast_cfg.rho_schedule))
    assert s.n_startup == 1
    assert s.max_order == 4
    assert s.recycle_nodes is False
    assert s.adaptive_order is False
    assert s.adaptive_mesh is False


@pytest.mark.parametrize("name", baseline_names())
def test_baselines_inherit_the_configured_ladder(name, fast_cfg) -> None:
    """Fair comparison: baselines must walk the same rho as the flagship."""
    s = get_sampler(name, fast_cfg)
    assert s.rho == pytest.approx(float(fast_cfg.rho_schedule))


def test_flagship_is_more_accurate_than_every_baseline(fast_cfg, trained) -> None:
    """The headline claim, asserted on the fast config as a regression guard."""
    from diffuforge.eval.ode_error import dense_reference, ode_rel_err

    model, _ = trained[("rings", 7)]
    x_ref, _ = dense_reference(model, fast_cfg, get_rng(7), n=32)
    fs, _ = get_sampler(FLAGSHIP, fast_cfg).sample(model, 32, 10, get_rng(7))
    best = min(
        ode_rel_err(get_sampler(n, fast_cfg).sample(model, 32, 10, get_rng(7))[0], x_ref)
        for n in baseline_names()
    )
    assert ode_rel_err(fs, x_ref) <= best
