"""Validate the HS-MS flagship (node recycling): NFE contract + trajectory error."""

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

NFE_REF = 2000
INIT_SEED = 4242
DATASETS = ("gmm8", "rings", "moons", "spiral")
NFE_TAB = (5, 10, 20)
RHO = 3.0


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
    n = 96
    # ---- 1. NFE budget contract -------------------------------------------
    print("### NFE budget contract")
    spec = get_density("gmm8")
    set_all(7)
    model, _ = train_denoiser(spec, 7, cfg)
    for n_startup in (0, 1, 2, 3):
        row = []
        for b in (3, 4, 5, 6, 8, 10, 13, 20, 35):
            s = DiffuFuseSampler(cfg=cfg, rho=RHO, n_startup=n_startup, max_order=4)
            model.reset_nfe()
            try:
                _, used = s.sample(model, 8, b, get_rng(INIT_SEED))
                row.append(f"{b}:{used}{'' if used <= b else '!'}")
            except Exception:
                row.append(f"{b}:ERR")
        print(f"   n_startup={n_startup}  " + "  ".join(row))

    # ---- 2. trajectory error ----------------------------------------------
    print(f"\n### trajectory error, rho={RHO}")
    for ds in DATASETS:
        spec = get_density(ds)
        set_all(7)
        model, _ = train_denoiser(spec, 7, cfg)
        x_ref, _ = HeunSampler(rho=RHO).sample(model, n, NFE_REF, get_rng(INIT_SEED))
        scale = float(np.sqrt(np.mean(x_ref * x_ref)))

        def relerr(x):
            return float(np.sqrt(np.mean((x - x_ref) ** 2))) / scale

        base = {}
        for name, s in baselines(RHO).items():
            vals = []
            for nfe in NFE_TAB:
                model.reset_nfe()
                x, _ = s.sample(model, n, int(nfe), get_rng(INIT_SEED))
                vals.append(relerr(x))
            base[name] = vals
        best = [min(v[i] for v in base.values()) for i in range(len(NFE_TAB))]
        bestn = [min(base, key=lambda k: base[k][i]) for i in range(len(NFE_TAB))]
        print(f"  -- {ds}: best={[f'{bestn[i]}:{best[i]:.5f}' for i in range(3)]}")
        for n_startup, max_order in ((0, 4), (1, 4), (2, 4), (3, 4), (1, 3), (1, 2)):
            s = DiffuFuseSampler(cfg=cfg, rho=RHO, n_startup=n_startup, max_order=max_order)
            vals = []
            for nfe in NFE_TAB:
                model.reset_nfe()
                x, _ = s.sample(model, n, int(nfe), get_rng(INIT_SEED))
                vals.append(relerr(x))
            r = [vals[i] / best[i] for i in range(len(NFE_TAB))]
            print(
                f"     start={n_startup} ord={max_order} "
                f"err=[{', '.join(f'{v:.5f}' for v in vals)}] "
                f"ratio=[{', '.join(f'{v:.3f}' for v in r)}] worst={max(r):.3f}"
            )


if __name__ == "__main__":
    main()
