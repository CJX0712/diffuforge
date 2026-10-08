"""Training loop for the EDM-preconditioned denoiser.

Loss (Karras et al. 2022, eq. 7 with the optimal weighting lambda = 1/c_out^2):

    L = E_{sigma ~ log-U(sigma_min, sigma_max), x0, n} [
        || D(x0 + n; sigma) - x0 ||^2 / c_out(sigma)^2 ]
"""

from __future__ import annotations

import time

import numpy as np

from ..core.config import DiffusionConfig
from ..core.errors import DataError
from ..core.seed import get_rng
from ..core.types import DensitySpec, TrainResult
from ..data.densities import get_density
from .net import MLP
from .precond import EDMDenoiser, edm_coefficients


def _resolve_density(source: str | DensitySpec) -> DensitySpec:
    if isinstance(source, DensitySpec):
        return source
    return get_density(source)


def train_denoiser(
    source: str | DensitySpec,
    seed: int,
    cfg: DiffusionConfig | None = None,
    x_train: np.ndarray | None = None,
) -> tuple[EDMDenoiser, TrainResult]:
    """Fit a denoiser on *source* (or on an explicit training matrix)."""
    cfg = cfg or DiffusionConfig()
    cfg.validate()
    spec = _resolve_density(source)
    rng = get_rng(seed)

    if x_train is None:
        x_train = spec.sample(int(cfg.n_train), get_rng(seed * 1000 + 11))
    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[0] < 2:
        raise DataError("x_train must be (n, d) with n >= 2", shape=x_train.shape)

    d = x_train.shape[1]
    # sigma_data = RMS of the (centred) training cloud, per Karras et al.
    sigma_data = float(np.sqrt(np.mean((x_train - x_train.mean(axis=0)) ** 2)))
    if sigma_data <= 1e-12:
        raise DataError("degenerate training data (sigma_data ~ 0)")
    sigma_min = cfg.sigma_min_rel * sigma_data
    sigma_max = cfg.sigma_max_rel * sigma_data

    net = MLP(
        in_dim=d,
        out_dim=d,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        fourier_dim=cfg.fourier_dim,
        fourier_max_freq=cfg.fourier_max_freq,
        rng=get_rng(seed * 1000 + 3),
    )
    mu_data = x_train.mean(axis=0)
    cov_data = np.cov(x_train.T) if x_train.shape[0] > 1 else np.eye(d)
    model = EDMDenoiser(
        net,
        sigma_data,
        sigma_min,
        sigma_max,
        mu_data=mu_data,
        cov_data=cov_data,
        exact_moment_init=bool(cfg.exact_moment_init),
    )

    ln_min, ln_max = float(np.log(sigma_min)), float(np.log(sigma_max))
    n = x_train.shape[0]
    batch = int(cfg.batch_size)
    trace: list[float] = []
    t0 = time.perf_counter()
    running = 0.0
    for it in range(int(cfg.n_iters)):
        idx = rng.integers(0, n, size=batch)
        x0 = x_train[idx]
        sigma = np.exp(rng.uniform(ln_min, ln_max, size=batch))
        noise = rng.standard_normal((batch, d)) * sigma[:, None]
        x = x0 + noise

        c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, sigma_data)
        f = net.forward(x * c_in[:, None], c_noise)
        dd = c_skip[:, None] * x + c_out[:, None] * f
        resid = dd - x0
        loss = float(np.mean(np.sum(resid * resid, axis=1) / (c_out * c_out)))
        running += loss

        # dL/dF = (1/B) * 2 * resid / c_out  (per-sample broadcasting)
        grad_f = (2.0 / batch) * resid / c_out[:, None]
        net.backward(grad_f)
        net.adam_step(cfg.lr, cfg.adam_b1, cfg.adam_b2, cfg.adam_eps, cfg.grad_clip)

        if (it + 1) % 200 == 0:
            trace.append(running / 200.0)
            running = 0.0
    elapsed = time.perf_counter() - t0

    result = TrainResult(
        dataset=spec.name,
        seed=int(seed),
        sigma_data=sigma_data,
        final_loss=trace[-1] if trace else running,
        loss_trace=trace,
        n_iters=int(cfg.n_iters),
        elapsed_sec=elapsed,
    )
    return model, result


__all__ = ["train_denoiser"]
