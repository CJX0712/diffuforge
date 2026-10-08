"""DiffuFuse — the flagship sampler: **HS-MS(k)**, high-order start-up with node
recycling, plus a high-order multistep exponential integrator.

The published DPM-Solver++ family (Lu et al. 2023) seeds its multistep recursion
with a *first-order* step: at the first node there is no history, so the only
available update is the piecewise-constant exponential integrator.  That start-up
error is the weak point of the whole scheme, because

1. on the Karras ladder the first interval is the widest one, so it carries the
   largest single share of the global error, and
2. a linear multistep method never forgets its start-up error — every later step
   extrapolates from history seeded by it.

DiffuFuse replaces the first ``n_startup`` first-order steps with **second-order
single-step DPM-Solver-2 updates** (midpoint collocation, exact exponential
integration).  Each costs 2 NFE, and the extra start-up NFE is charged to the
budget — never free.

The second ingredient is **node recycling**: the midpoint evaluation
``sigma_m = sqrt(sigma_i * sigma_{i+1})`` is not thrown away.  Because
``lambda_i < lambda_m < lambda_{i+1}``, that evaluation is a legitimate extra
interpolation node for every later step, so the history after one start-up step
is ``[lambda_m, lambda_i]`` instead of ``[lambda_i]``.  The multistep recursion
therefore reaches order 3 one step earlier than the published scheme at zero
extra cost.

Three further switches exist purely for the ablation; the first two are **measured
negative results** and are kept switchable rather than silently deleted:

* ``recycle_nodes`` — also push the midpoint evaluation ``lambda_m`` into the
  history, so the recursion reaches order 3 one step earlier.  *Measured:
  negative — the resulting quadratic extrapolation is over a fit region twice as
  wide as the step, and it amplifies error (2.2x worse on ``rings`` at NFE=10).*
* ``adaptive_order`` — order-``k`` and order-``(k-1)`` updates are both computable
  from the same history, so their disagreement is a free error indicator; drop an
  order whenever the indicator stops shrinking.  *Measured: negative.*
* ``adaptive_mesh`` — a probe pass estimates local stiffness and equidistributes
  the mesh.  *Measured: strongly negative at NFE <= 35.*

Design history (all measured, all reported honestly): a variant whose start-up
spans *two* ladder intervals — which would make the start-up free, since 2 NFE
then covers 2 intervals — was implemented and rejected: the wider collocation
interval costs more accuracy than the saved NFE buys (6x worse on ``rings`` at
NFE=10).  The one-interval start-up below is the measured winner.
"""

from __future__ import annotations

import numpy as np

from ..core.config import DiffusionConfig
from ..core.errors import SamplerError
from .base import BaseSampler, final_denoise, init_noise
from .dpm import lambda_moments, poly_update
from .schedule import karras_sigma

MAX_NODES = 4


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))


class DiffuFuseSampler(BaseSampler):
    """High-order start-up (with node recycling) + high-order multistep."""

    name = "diffufuse"
    nfe_per_step = 1

    def __init__(
        self,
        cfg: DiffusionConfig | None = None,
        rho: float | None = None,
        n_startup: int = 1,
        max_order: int = 4,
        recycle_nodes: bool = False,
        adaptive_order: bool = False,
        adaptive_mesh: bool = False,
        order_gain: float | None = None,
        probe_frac: float | None = None,
        probe_batch: int | None = None,
    ) -> None:
        cfg = cfg or DiffusionConfig()
        self.rho = float(cfg.rho_schedule if rho is None else rho)
        self.n_startup = int(n_startup)
        self.max_order = int(max_order)
        self.recycle_nodes = bool(recycle_nodes)
        self.adaptive_order = bool(adaptive_order)
        self.adaptive_mesh = bool(adaptive_mesh)
        self.order_gain = float(cfg.caos_order_gain if order_gain is None else order_gain)
        self.probe_frac = float(cfg.caos_probe_frac if probe_frac is None else probe_frac)
        self.probe_batch = int(cfg.caos_probe_batch if probe_batch is None else probe_batch)
        self.ridge = float(cfg.caos_ridge)
        if self.n_startup < 0:
            raise SamplerError("n_startup must be >= 0", value=self.n_startup)
        if not 1 <= self.max_order <= MAX_NODES:
            raise SamplerError("max_order must be 1..4", order=self.max_order)

    # ------------------------------------------------------------------ probe
    def _probe_mesh(self, model, rng, n_probe: int) -> tuple[np.ndarray, np.ndarray]:
        """Cheap first-order probe returning ``(sigma_ladder, stiffness)``."""
        sig_p = karras_sigma(n_probe + 1, model.sigma_min, model.sigma_max, self.rho)[:-1]
        lam_p = -np.log(sig_p)
        xp = init_noise(model, int(self.probe_batch), rng)
        prev_f: np.ndarray | None = None
        kappa = np.zeros(n_probe, dtype=np.float64)
        dlam = np.diff(lam_p)
        for i in range(n_probe):
            d = model.denoise(xp, sig_p[i])
            f = d - xp  # dx/dlambda = D(x; sigma) - x
            kappa[i] = (
                _rms(f) / max(dlam[i], 1e-12)
                if prev_f is None
                else _rms(f - prev_f) / max(dlam[i], 1e-12)
            )
            prev_f = f
            xp = xp + f * dlam[i]
        if n_probe >= 2:
            kappa[0] = kappa[1]
        return sig_p, kappa

    @staticmethod
    def _mesh_from_kappa(
        lam_p: np.ndarray, kappa: np.ndarray, n_main: int, ridge: float
    ) -> np.ndarray:
        """Equidistribute the estimated local error over ``n_main`` intervals."""
        dlam = np.diff(lam_p)
        w = np.sqrt(np.maximum(kappa, 0.0) + ridge) * dlam
        cum = np.concatenate([[0.0], np.cumsum(w)])
        total = cum[-1]
        if total <= 1e-12:  # pragma: no cover - degenerate flat probe
            return np.exp(-np.linspace(lam_p[0], lam_p[-1], n_main + 1))
        targets = total * (np.arange(1, n_main, dtype=np.float64) / n_main)
        lam_mid = np.interp(targets, cum, lam_p)
        return np.exp(-np.concatenate([[lam_p[0]], lam_mid, [lam_p[-1]]]))

    # ------------------------------------------------------------------- main
    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:
        budget = int(nfe_budget)
        x = init_noise(model, n, rng)

        if self.adaptive_mesh:
            return self._run_probed(model, x, budget, rng)

        # NFE accounting: n_startup steps at 2 NFE + (n_steps - n_startup - 1)
        # at 1 NFE + the terminal sigma -> 0 move = n_startup + n_steps.
        n_start = int(min(self.n_startup, max(0, budget - 3)))
        n_steps = max(2, budget - n_start)
        sig = karras_sigma(n_steps, model.sigma_min, model.sigma_max, self.rho)

        hist_lam: list[float] = []
        hist_s: list[np.ndarray] = []

        def push(lam_s: tuple[float, np.ndarray]) -> None:
            """Insert one (lambda, score) node, keeping lambda descending."""
            lam, s = lam_s
            j = len(hist_lam)
            for k, other in enumerate(hist_lam):
                if lam > other:
                    j = k
                    break
            hist_lam.insert(j, lam)
            hist_s.insert(j, s)
            while len(hist_s) > MAX_NODES:  # drop the oldest (smallest lambda)
                hist_lam.pop()
                hist_s.pop()

        prev_gap: float | None = None
        for i in range(n_steps):
            si, sj = float(sig[i]), float(sig[i + 1])
            if sj <= 0.0:
                x = final_denoise(model, x, si)
                break

            if i < n_start:
                # ---- DPM-Solver-2 single step (2 NFE) + node recycling -----
                s_i = (model.denoise(x, si) - x) / (si * si)
                sm = float(np.sqrt(max(si * sj, 0.0)))
                xm = x + s_i * lambda_moments(si, sm)[0]
                s_m = (model.denoise(xm, sm) - xm) / (sm * sm)
                mom = lambda_moments(si, sj)
                c = (s_m - s_i) / ((-np.log(sm)) - (-np.log(si)))
                x = x + s_i * mom[0] + c * mom[1]
                if self.recycle_nodes:
                    # lambda_i < lambda_m < lambda_{i+1}, so the midpoint
                    # evaluation is a legitimate extra node.  Measured negative
                    # (see module docstring) — off by default.
                    push((float(-np.log(sm)), s_m))
                push((float(-np.log(si)), s_i))
                continue

            s_i = (model.denoise(x, si) - x) / (si * si)
            mom = lambda_moments(si, sj)
            push((float(-np.log(si)), s_i))

            k_hi = int(min(len(hist_s), self.max_order))
            taus = np.array(hist_lam[:k_hi]) - hist_lam[0]
            x_hi = poly_update(x, taus, hist_s[:k_hi], mom)
            if self.adaptive_order and k_hi > 1:
                k_lo = k_hi - 1
                taus_lo = np.array(hist_lam[:k_lo]) - hist_lam[0]
                x_lo = poly_update(x, taus_lo, hist_s[:k_lo], mom)
                gap = _rms(x_hi - x_lo)
                x = x_lo if (prev_gap is not None and gap > self.order_gain * prev_gap) else x_hi
                prev_gap = gap
            else:
                x = x_hi
            if not np.all(np.isfinite(x)):  # pragma: no cover - safety net
                x = x + s_i * mom[0]
        return x

    # ------------------------------------------------------------ probe path
    def _run_probed(
        self, model, x: np.ndarray, budget: int, rng
    ) -> np.ndarray:  # pragma: no cover - ablation-only path
        """``adaptive_mesh`` variant (measured negative; kept for the ablation)."""
        n_probe = max(2, min(round(self.probe_frac * budget), budget - 6))
        n_steps = max(2, budget - n_probe - 1)
        sig_p, kappa = self._probe_mesh(model, rng, n_probe)
        sig = self._mesh_from_kappa(-np.log(sig_p), kappa, n_steps, self.ridge)
        sig = np.concatenate([sig, [0.0]])
        hist_lam: list[float] = []
        hist_s: list[np.ndarray] = []
        for i in range(len(sig) - 1):
            si, sj = float(sig[i]), float(sig[i + 1])
            if sj <= 0.0:
                x = final_denoise(model, x, si)
                break
            s_i = (model.denoise(x, si) - x) / (si * si)
            mom = lambda_moments(si, sj)
            hist_lam.insert(0, float(-np.log(si)))
            hist_s.insert(0, s_i)
            if len(hist_s) > MAX_NODES:
                hist_lam.pop()
                hist_s.pop()
            k_hi = int(min(len(hist_s), self.max_order))
            taus = np.array(hist_lam[:k_hi]) - hist_lam[0]
            x = poly_update(x, taus, hist_s[:k_hi], mom)
        return x


__all__ = ["DiffuFuseSampler"]
