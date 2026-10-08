"""UniPC — Unified Predictor-Corrector (Zhao et al., NeurIPS 2023).

UniPC's edge over plain DPM-Solver++ comes from one observation: the predictor
evaluates the score at ``x_tilde_{i+1}`` anyway, so that extra evaluation can be
*recycled* as an interpolation node at the **right** endpoint ``lambda_{i+1}``.
The corrector then integrates a one-order-higher interpolant (Adams-Moulton
style) for free.  Cost: 2 NFE per step, same as Karras Heun.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import SamplerError
from .base import BaseSampler, final_denoise, init_noise
from .dpm import lambda_moments, poly_update
from .schedule import karras_sigma

MAX_NODES = 4


class UniPCSampler(BaseSampler):
    """Fixed-order UniPC (2 NFE per step)."""

    name = "unipc"
    nfe_per_step = 2

    def __init__(
        self,
        rho: float = 7.0,
        predictor_order: int = 3,
        corrector_order: int = 4,
    ) -> None:
        self.rho = float(rho)
        self.predictor_order = int(predictor_order)
        self.corrector_order = int(corrector_order)
        if not 1 <= self.predictor_order <= MAX_NODES:
            raise SamplerError("predictor_order out of range", order=self.predictor_order)
        if not 2 <= self.corrector_order <= MAX_NODES:
            raise SamplerError("corrector_order out of range", order=self.corrector_order)

    def _run(self, model, n: int, nfe_budget: int, rng) -> np.ndarray:
        # 3 NFE on step 0 (s_i + predictor + corrector), 2 NFE on every later
        # step, 1 NFE on the terminal step  =>  n_steps = B // 2 is safe.
        n_steps = max(1, int(nfe_budget) // 2)
        sig = karras_sigma(n_steps, model.sigma_min, model.sigma_max, self.rho)
        x = init_noise(model, n, rng)
        hist_lam: list[float] = []
        hist_s: list[np.ndarray] = []
        s_cur: np.ndarray | None = None
        for i in range(n_steps):
            si, sj = float(sig[i]), float(sig[i + 1])
            if sj <= 0.0:
                x = final_denoise(model, x, si)
                break
            lam_i, lam_j = -np.log(si), -np.log(sj)
            if i == 0:
                d = model.denoise(x, si)
                s_i = (d - x) / (si * si)
            else:
                if s_cur is None:  # pragma: no cover - invariant
                    raise SamplerError("missing carried score")
                s_i = s_cur
            hist_lam.insert(0, lam_i)
            hist_s.insert(0, s_i)
            if len(hist_s) > MAX_NODES:
                hist_lam.pop()
                hist_s.pop()

            # ---- predictor (Adams-Bashforth: nodes at lam <= lam_i) ---------
            kp = int(min(len(hist_s), self.predictor_order))
            taus_p = np.array(hist_lam[:kp]) - lam_i
            x_pred = poly_update(x, taus_p, hist_s[:kp], lambda_moments(si, sj))

            # ---- evaluate at the predictor point: 1 NFE ---------------------
            dp = model.denoise(x_pred, sj)
            s_tilde = (dp - x_pred) / (sj * sj)

            # ---- corrector (nodes include the right endpoint lam_j) ---------
            kc = int(min(len(hist_s) + 1, self.corrector_order))
            nodes_lam = [lam_j, *hist_lam[: kc - 1]]
            nodes_s = [s_tilde, *hist_s[: kc - 1]]
            taus_c = np.array(nodes_lam) - lam_i
            x = poly_update(x, taus_c, nodes_s, lambda_moments(si, sj))

            # ---- score at the corrected point, reused next step: 1 NFE ------
            dc = model.denoise(x, sj)
            s_cur = (dc - x) / (sj * sj)
        return x


__all__ = ["MAX_NODES", "UniPCSampler"]
