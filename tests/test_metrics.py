"""Metric contracts, including the pure-NumPy Tier-1 fallback path."""

from __future__ import annotations

import numpy as np
import pytest
from diffuforge.core.errors import EvalError
from diffuforge.core.seed import get_rng
from diffuforge.eval.metrics import _sinkhorn_cost, coverage, mmd2, nn_tst, w2_sq
from diffuforge.eval.ode_error import dense_reference, ode_rel_err


def _clouds(n: int = 40, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 2))
    b = rng.normal(size=(n, 2)) + shift
    return a, b


def test_w2_is_zero_for_identical_and_grows_with_shift() -> None:
    a, _ = _clouds()
    assert w2_sq(a, a, 40) == pytest.approx(0.0, abs=1e-12)
    _, b = _clouds(shift=0.0)
    d0 = w2_sq(*_clouds(shift=0.0), 40)
    d1 = w2_sq(*_clouds(shift=3.0), 40)
    assert d1 > d0
    assert b.shape == (40, 2)


def test_w2_is_symmetric() -> None:
    a, b = _clouds()
    assert w2_sq(a, b, 40) == pytest.approx(w2_sq(b, a, 40), rel=1e-12)


def test_w2_respects_max_n() -> None:
    a, b = _clouds(n=64)
    assert w2_sq(a, b, 16) == pytest.approx(w2_sq(a[:16], b[:16], 16), rel=1e-12)


def test_w2_rejects_degenerate_input() -> None:
    with pytest.raises(EvalError):
        w2_sq(np.zeros((1, 2)), np.zeros((5, 2)), 32)
    with pytest.raises(EvalError):  # not 2-D
        w2_sq(np.zeros((5, 2, 1)), np.zeros((5, 2)), 32)


def test_tier1_sinkhorn_fallback_approximates_exact_ot() -> None:
    """Without SciPy the pure-NumPy path must still give a sane OT cost."""
    a, b = _clouds(n=32, shift=1.5)
    cost = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    exact = w2_sq(a, b, 32)
    approx = _sinkhorn_cost(cost) / cost.shape[0]
    # Entropic regularisation biases the value upward but it stays O(exact).
    assert 0.0 < approx < 8.0 * exact


def test_mmd_is_zero_on_identical_clouds_and_grows_with_shift() -> None:
    a, b = _clouds(n=60)
    near = mmd2(a, b)
    far = mmd2(a, b + 4.0)
    assert far > near
    # The *unbiased* estimator is not clamped at zero: for two identical clouds
    # it returns 2*(mean off-diagonal kernel - 1)/n, i.e. a small negative
    # number of order 1/n.  It must be small in magnitude, not exactly zero.
    assert abs(mmd2(a, a)) < 0.1


def test_coverage_is_a_rate_in_unit_interval() -> None:
    a, b = _clouds(n=200)
    c = coverage(a, b)
    assert 0.0 <= c <= 1.0
    assert coverage(a, a) < coverage(a, b + 50.0)


def test_nn_tst_in_unit_interval() -> None:
    a, b = _clouds()
    v = nn_tst(a, b)
    assert 0.0 <= v <= 1.0


# --------------------------------------------------------------- ode error


def test_ode_rel_err_zero_for_identical() -> None:
    x = np.random.default_rng(1).normal(size=(16, 2))
    assert ode_rel_err(x, x) == pytest.approx(0.0, abs=1e-15)


def test_ode_rel_err_compares_prefixes() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(8, 2))
    gen = rng.normal(size=(32, 2))
    assert ode_rel_err(gen, ref) == pytest.approx(ode_rel_err(gen[:8], ref), rel=1e-15)


def test_ode_rel_err_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        ode_rel_err(np.zeros((4, 2)), np.zeros((8, 2)))


def test_dense_reference_converges(fast_cfg, trained) -> None:
    """A denser reference must agree with a cheaper one to high precision."""
    model, _ = trained[("gmm8", 7)]
    a, _ = dense_reference(model, fast_cfg.with_overrides(ode_ref_nfe=120), get_rng(3), n=16)
    b, _ = dense_reference(model, fast_cfg.with_overrides(ode_ref_nfe=900), get_rng(3), n=16)
    assert ode_rel_err(a, b) < 0.05
