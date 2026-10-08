"""Does W2^2 with common random numbers actually track integration error?

For each (dataset, seed) every sampler starts from the SAME initial noise (CRN),
so the generated clouds are paired.  We print W2^2 next to the trajectory error
against a dense reference to see whether the sample metric is discriminative.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from diffuforge.core.config import DiffusionConfig
from diffuforge.core.seed import get_rng, set_all
from diffuforge.data.densities import get_density
from diffuforge.eval.metrics import w2_sq
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
NFE_TAB = (5, 10, 20)
DATASETS = ("gmm8", "rings", "moons", "spiral")
SEEDS = (7, 17)


def build(rho: float, cfg) -> dict[str, object]:
    out = {
        "dpm1": DPM1Sampler(rho=rho),
        "dpm2m": DPM2MSampler(rho=rho),
        "dpm2s": DPM2SSampler(rho=rho),
        "dpm3m": DPM3MSampler(rho=rho),
        "dpm4m": DPM4MSampler(rho=rho),
        "euler": EulerSampler(rho=rho),
        "heun": HeunSampler(rho=rho),
        "logeuler": LogEulerSampler(rho=rho),
        "unipc": UniPCSampler(rho=rho),
        "diffufuse": DiffuFuseSampler(cfg=cfg, rho=rho, n_startup=1, max_order=4),
    }
    return out


def main() -> None:
    for sigma_max_rel in (4.0, 10.0):
        cfg = DiffusionConfig(sigma_max_rel=sigma_max_rel)
        rho = 3.0
        print(f"\n############ sigma_max_rel={sigma_max_rel}  rho={rho}  (CRN, paired)")
        for ds in DATASETS:
            spec = get_density(ds)
            w2: dict[str, list[float]] = {}
            for seed in SEEDS:
                set_all(seed)
                model, _ = train_denoiser(spec, seed, cfg)
                ref = spec.sample(int(cfg.n_eval_ref), get_rng(999_000 + seed))
                # dense reference trajectory for the error column
                n_small = 128
                x_ref, _ = HeunSampler(rho=rho).sample(model, n_small, NFE_REF, get_rng(4242))
                scale = float(np.sqrt(np.mean(x_ref * x_ref)))
                for name, s in build(rho, cfg).items():
                    for nfe in NFE_TAB:
                        model.reset_nfe()
                        gen, _ = s.sample(model, int(cfg.n_gen), int(nfe), get_rng(777 + seed))
                        w2.setdefault(f"{name}", []).append(w2_sq(ref, gen, int(cfg.eval_ot_size)))
                    model.reset_nfe()
                    xs, _ = s.sample(model, n_small, int(NFE_TAB[1]), get_rng(4242))
                    w2.setdefault("ERR_" + name, []).append(
                        float(np.sqrt(np.mean((xs - x_ref) ** 2))) / scale
                    )
            print(f"\n  === {ds}")
            for name in build(rho, cfg):
                v = w2[name]
                means = [float(np.mean(v[i :: len(NFE_TAB)])) for i in range(len(NFE_TAB))]
                err = float(np.mean(w2["ERR_" + name]))
                print(
                    f"    {name:<10s} W2^2 "
                    + " ".join(f"N{n}={m:7.4f}" for n, m in zip(NFE_TAB, means))
                    + f"   relerr@N10={err:.5f}"
                )


if __name__ == "__main__":
    main()
