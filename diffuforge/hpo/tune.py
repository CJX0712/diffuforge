"""Optuna-driven tuning of the flagship sampler.

The validation protocol is deliberately disjoint from the benchmark protocol:

* a dedicated density / seed (``val_dataset`` / ``val_seed``) that never appears
  in the reported benchmark tables;
* a fixed NFE budget and a fixed sample count;
* a fixed sampler seed, so the objective is a deterministic function of the
  trial's hyper-parameters.

The objective is the **PF-ODE trajectory error** (``ode_err``), not W2^2: on the
validation density every sampler sits at the model's quality ceiling already, so
W2^2 differences are 1-2% and are dominated by reference-set noise.  ``ode_err``
against a dense reference, with common random numbers, is the low-variance
objective that the sampler is actually being tuned to improve.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.config import DiffusionConfig
from ..core.seed import get_rng
from ..data.densities import get_density
from ..eval.ode_error import dense_reference, ode_rel_err
from ..samplers.diffufuse import DiffuFuseSampler
from ..score.train import train_denoiser

VAL_SAMPLE_SEED = 555_002


def tune_flagship(
    cfg: DiffusionConfig | None = None,
    val_dataset: str = "checker",
    val_seed: int = 999,
    nfe: int = 20,
    n_gen: int = 256,
    n_trials: int = 8,
) -> dict[str, Any]:
    """Return ``{"best_params": ..., "best_value": ..., "n_trials": ..., ...}``."""
    cfg = (cfg or DiffusionConfig()).validate()
    spec = get_density(val_dataset)
    t0 = time.perf_counter()

    try:
        import optuna
    except ImportError:  # pragma: no cover - Tier-1 fallback
        return _grid_fallback(cfg, spec, val_seed, nfe, n_gen, t0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model, _ = train_denoiser(spec, val_seed, cfg)
    x_ref, ref_nfe = dense_reference(model, cfg, get_rng(VAL_SAMPLE_SEED), n=n_gen)

    def objective(trial: optuna.Trial) -> float:
        rho = trial.suggest_float("rho", 1.5, 9.0)
        n_startup = trial.suggest_int("n_startup", 0, 2)
        max_order = trial.suggest_int("max_order", 1, 4)
        recycle_nodes = trial.suggest_categorical("recycle_nodes", [True, False])
        adaptive_order = trial.suggest_categorical("adaptive_order", [True, False])
        order_gain = trial.suggest_float("order_gain", 0.5, 2.0)
        sampler = DiffuFuseSampler(
            cfg=cfg,
            rho=rho,
            n_startup=n_startup,
            max_order=max_order,
            recycle_nodes=recycle_nodes,
            adaptive_order=adaptive_order,
            order_gain=order_gain,
        )
        model.reset_nfe()
        gen, _ = sampler.sample(model, int(n_gen), int(nfe), get_rng(VAL_SAMPLE_SEED))
        if not np.all(np.isfinite(gen)):  # pragma: no cover - guard
            return 1e9
        return ode_rel_err(gen, x_ref)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=val_seed)
    )
    # Force the published-like configuration into the search so the study can
    # never report "worse than the obvious default" without having tried it.
    study.enqueue_trial(
        {
            "rho": float(cfg.rho_schedule),
            "n_startup": 1,
            "max_order": 4,
            "recycle_nodes": False,
            "adaptive_order": False,
            "order_gain": 1.0,
        }
    )
    study.optimize(objective, n_trials=int(n_trials))
    return {
        "engine": "optuna-tpe",
        "objective": "ode_err (PF-ODE trajectory error vs dense reference)",
        "val_dataset": val_dataset,
        "val_seed": int(val_seed),
        "val_nfe": int(nfe),
        "ode_ref_nfe": int(ref_nfe),
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "n_trials": int(n_trials),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }


def _grid_fallback(
    cfg: DiffusionConfig, spec, val_seed: int, nfe: int, n_gen: int, t0: float
) -> dict[str, Any]:
    """Deterministic grid search used when Optuna is unavailable."""
    model, _ = train_denoiser(spec, val_seed, cfg)
    x_ref, ref_nfe = dense_reference(model, cfg, get_rng(VAL_SAMPLE_SEED), n=n_gen)
    grid = [
        {"rho": float(cfg.rho_schedule), "n_startup": 1, "max_order": 4},
        {"rho": float(cfg.rho_schedule), "n_startup": 0, "max_order": 4},
        {"rho": 5.0, "n_startup": 1, "max_order": 3},
        {"rho": 7.0, "n_startup": 1, "max_order": 2},
    ]
    best: tuple[float, dict[str, Any]] = (float("inf"), {})
    for kw in grid:
        sampler = DiffuFuseSampler(cfg=cfg, **kw)
        model.reset_nfe()
        gen, _ = sampler.sample(model, int(n_gen), int(nfe), get_rng(VAL_SAMPLE_SEED))
        val = ode_rel_err(gen, x_ref)
        if val < best[0]:
            best = (val, dict(kw))
    return {
        "engine": "grid-fallback",
        "objective": "ode_err (PF-ODE trajectory error vs dense reference)",
        "val_dataset": spec.name,
        "val_seed": int(val_seed),
        "ode_ref_nfe": int(ref_nfe),
        "best_params": best[1],
        "best_value": float(best[0]),
        "n_trials": len(grid),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }


__all__ = ["tune_flagship"]
