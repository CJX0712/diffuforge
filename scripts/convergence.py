"""Decisive convergence test: trajectory error against a dense reference solution.

Every sampler starts from the *identical* initial noise, so the only difference
between two outputs is integration error.  The reference is Heun at NFE=8000.
This isolates ODE-integration accuracy from sample-metric noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from diffuforge.core.config import DiffusionConfig
from diffuforge.core.seed import get_rng, set_all
from diffuforge.data.densities import get_density
from diffuforge.samplers.ode import HeunSampler
from diffuforge.samplers.registry import all_sampler_names, get_sampler
from diffuforge.score.train import train_denoiser

NFE_REF = 8000
NFE_TAB = (10, 20, 50, 100, 200, 400)
# same rng seed for every sampler -> identical initial noise
INIT_SEED = 12345


def main() -> None:
    cfg = DiffusionConfig()
    for ds in ("gmm8", "rings", "moons", "spiral"):
        spec = get_density(ds)
        set_all(7)
        model, _ = train_denoiser(spec, 7, cfg)
        n = 128

        ref_s = HeunSampler(rho=cfg.rho_schedule)
        x_ref, used_ref = ref_s.sample(model, n, NFE_REF, get_rng(INIT_SEED))
        scale = float(np.sqrt(np.mean(x_ref * x_ref)))
        print(
            f"\n=== {ds}  reference: heun NFE={NFE_REF} (used {used_ref}), rms(x_ref)={scale:.4f}"
        )

        for name in list(all_sampler_names()):
            line = f"  {name:<10s}"
            for nfe in NFE_TAB:
                model.reset_nfe()
                s = get_sampler(name, cfg)
                x, used = s.sample(model, n, int(nfe), get_rng(INIT_SEED))
                err = float(np.sqrt(np.mean((x - x_ref) ** 2))) / scale
                line += f" | N{nfe:<4d} relerr={err:8.5f}"
            print(line)


if __name__ == "__main__":
    main()
