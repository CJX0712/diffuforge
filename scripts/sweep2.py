"""Fair sweep: one shared sigma ladder (rho) for EVERY sampler, then compare.

Trajectory error against a dense reference (same initial noise) is the objective.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from diffuforge.core.config import DiffusionConfig
from diffuforge.core.seed import get_rng, set_all
from diffuforge.data.densities import get_density
from diffuforge.samplers.diffufuse import DiffuFuseSampler
from diffuforge.samplers.dpm import (
    DPM1Sampler,
    DPM2MSampler,
    DPM2SSampler,
    DPM3MSampler,
    DPM4MSampler,
)
from diffuforge.samplers.ode import EulerSampler, HeunSampler, LogEulerSampler
from diffuforge.samplers.unipc import UniPCSampler
from diffuforge.score.train import train_denoiser

NFE_REF = 4000
INIT_SEED = 12345
DATASETS = ("gmm8", "rings", "moons", "spiral")
NFE_TAB = (10, 20)
RHOS = (1.0, 2.0, 3.0, 4.0, 5.0, 7.0)


def baselines(rho: float) -> dict[str, object]:
    return {
        "dpm1": DPM1Sampler(rho=rho),
        "dpm2m": DPM2MSampler(rho=rho),
        "dpm2s": DPM2SSampler(rho=rho),
        "dpm3m": DPM3MSampler(rho=rho),
        "dpm4m": DPM4MSampler(rho=rho),
        "euler": EulerSampler(rho=rho),
        "heun": HeunSampler(rho=rho),
        "logeuler": LogEulerSampler(rho=rho),
        "unipc": UniPCSampler(rho=rho),
    }


def main() -> None:
    cfg = DiffusionConfig()
    n = 128
    for ds in DATASETS:
        spec = get_density(ds)
        set_all(7)
        model, _ = train_denoiser(spec, 7, cfg)
        x_ref, _ = HeunSampler(rho=7.0).sample(model, n, NFE_REF, get_rng(INIT_SEED))
        scale = float(np.sqrt(np.mean(x_ref * x_ref)))

        def relerr(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean((x - x_ref) ** 2))) / scale

        print(f"\n=== {ds}")
        for rho in RHOS:
            cell: dict[str, list[float]] = {}
            for name, s in baselines(rho).items():
                vals = []
                for nfe in NFE_TAB:
                    model.reset_nfe()
                    x, _ = s.sample(model, n, int(nfe), get_rng(INIT_SEED))
                    vals.append(relerr(x))
                cell[name] = vals
            best = {i: min(v[i] for v in cell.values()) for i in range(len(NFE_TAB))}
            bestname = {i: min(cell, key=lambda k: cell[k][i]) for i in range(len(NFE_TAB))}
            fs = DiffuFuseSampler(cfg=cfg, rho=rho, n_startup=1, max_order=4)
            fv = []
            for nfe in NFE_TAB:
                model.reset_nfe()
                x, _ = fs.sample(model, n, int(nfe), get_rng(INIT_SEED))
                fv.append(relerr(x))
            ratios = [fv[i] / best[i] for i in range(len(NFE_TAB))]
            print(
                f"  rho={rho:<4.1f} flag={fv[0]:.5f}/{fv[1]:.5f} "
                f"best0={bestname[0]}:{best[0]:.5f} best1={bestname[1]}:{best[1]:.5f} "
                f"-> ratio {ratios[0]:.3f}/{ratios[1]:.3f}  worst={max(ratios):.3f}"
            )


if __name__ == "__main__":
    main()
