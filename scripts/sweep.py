"""Hyper-parameter sweep for the flagship, scored by trajectory error.

Objective: relative RMS error against a dense (NFE=4000 Heun) reference solution
started from the *same* initial noise.  This has ~1000x lower variance than
sample-based W2^2, so it can resolve differences the benchmark cannot.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from diffuforge.core.config import DiffusionConfig
from diffuforge.core.seed import get_rng, set_all
from diffuforge.data.densities import get_density
from diffuforge.samplers.diffufuse import DiffuFuseSampler
from diffuforge.samplers.ode import HeunSampler
from diffuforge.samplers.registry import get_sampler
from diffuforge.score.train import train_denoiser

NFE_REF = 4000
INIT_SEED = 12345
DATASETS = ("gmm8", "rings", "moons", "spiral")
NFE_TAB = (10, 20)
BASINES = ("dpm2m", "dpm3m", "dpm4m", "euler", "heun", "dpm2s")


def main() -> None:
    cfg = DiffusionConfig()
    n = 128
    ref_err: dict[str, dict[tuple[str, int], float]] = {}

    for ds in DATASETS:
        spec = get_density(ds)
        set_all(7)
        model, _ = train_denoiser(spec, 7, cfg)
        x_ref, _ = HeunSampler(rho=cfg.rho_schedule).sample(model, n, NFE_REF, get_rng(INIT_SEED))
        scale = float(np.sqrt(np.mean(x_ref * x_ref)))

        def relerr(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean((x - x_ref) ** 2))) / scale

        base: dict[tuple[str, int], float] = {}
        for name in BASINES:
            for nfe in NFE_TAB:
                model.reset_nfe()
                x, _ = get_sampler(name, cfg).sample(model, n, int(nfe), get_rng(INIT_SEED))
                base[(name, nfe)] = relerr(x)
        ref_err[ds] = base
        print(f"\n=== {ds}  baselines (relerr)")
        for nfe in NFE_TAB:
            row = {k[0]: v for k, v in base.items() if k[1] == nfe}
            best = min(row, key=row.get)
            print(
                f"  NFE={nfe}: "
                + "  ".join(f"{k}={v:.5f}" for k, v in sorted(row.items()))
                + f"   | best={best}"
            )

        # ---- flagship sweep ------------------------------------------------
        print("  --- flagship sweep (n_startup, max_order, rho)")
        results = []
        for n_startup, max_order, rho in itertools.product(
            (0, 1, 2, 3), (2, 3, 4), (3.0, 5.0, 7.0, 9.0)
        ):
            model.reset_nfe()
            s = DiffuFuseSampler(cfg=cfg, rho=rho, n_startup=n_startup, max_order=max_order)
            vals = []
            ok = True
            for nfe in NFE_TAB:
                try:
                    x, _ = s.sample(model, n, int(nfe), get_rng(INIT_SEED))
                except Exception as exc:
                    ok = False
                    print(f"    FAIL {n_startup},{max_order},{rho}: {exc}")
                    break
                vals.append(relerr(x))
            if not ok:
                continue
            # ratio vs best baseline at each NFE (lower is better)
            ratio = max(
                vals[i] / min(v for k, v in base.items() if k[1] == NFE_TAB[i])
                for i in range(len(NFE_TAB))
            )
            results.append((ratio, n_startup, max_order, rho, vals))
        results.sort()
        for ratio, ns, mo, rho, vals in results[:10]:
            print(
                f"    n_start={ns} order={mo} rho={rho:<4.1f} "
                f"vals=[{', '.join(f'{v:.5f}' for v in vals)}] worst_ratio={ratio:.3f}"
            )


if __name__ == "__main__":
    main()
