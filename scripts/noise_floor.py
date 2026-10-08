"""Is the W2^2 metric saturated by the reference set's own sampling noise?

Compare every sampler's W2^2 against the *noise floor*: W2^2 between two
independent i.i.d. draws from the true density (same sizes as the benchmark).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from diffuforge.core.config import DiffusionConfig
from diffuforge.core.seed import get_rng, set_all
from diffuforge.data.densities import get_density
from diffuforge.eval.metrics import mmd2, nn_tst, w2_sq
from diffuforge.samplers.diffufuse import DiffuFuseSampler
from diffuforge.samplers.dpm import DPM2MSampler, DPM3MSampler, DPM4MSampler
from diffuforge.samplers.ode import EulerSampler, HeunSampler
from diffuforge.score.train import train_denoiser

DATASETS = ("gmm8", "rings", "moons", "spiral")
NFE_TAB = (10, 20)
NAMES = ("diffufuse", "dpm4m", "dpm3m", "dpm2m", "euler", "heun")


def main() -> None:
    for ot in (160, 512, 1024):
        cfg = DiffusionConfig(eval_ot_size=ot, n_gen=ot, n_eval_ref=ot, rho_schedule=3.0)
        print(f"\n############ n_gen=n_ref=ot={ot}")
        for ds in DATASETS:
            spec = get_density(ds)
            set_all(7)
            model, _ = train_denoiser(spec, 7, cfg)
            rho = 3.0
            ref = spec.sample(ot, get_rng(555))
            ref2 = spec.sample(ot, get_rng(556))
            floor_w = w2_sq(ref, ref2, ot)
            floor_m = mmd2(ref, ref2)
            floor_n = nn_tst(ref, ref2)
            print(
                f"\n  === {ds}   FLOOR  w2^2={floor_w:.4f}  mmd2={floor_m:.5f}  "
                f"nn_tst={floor_n:.4f}"
            )
            for name in NAMES:
                cls = {
                    "diffufuse": lambda: DiffuFuseSampler(
                        cfg=cfg, rho=rho, n_startup=1, max_order=4
                    ),
                    "dpm4m": lambda: DPM4MSampler(rho=rho),
                    "dpm3m": lambda: DPM3MSampler(rho=rho),
                    "dpm2m": lambda: DPM2MSampler(rho=rho),
                    "euler": lambda: EulerSampler(rho=rho),
                    "heun": lambda: HeunSampler(rho=rho),
                }[name]
                line = f"    {name:<10s} "
                for nfe in NFE_TAB:
                    model.reset_nfe()
                    gen, _ = cls().sample(model, ot, int(nfe), get_rng(777))
                    line += (
                        f"| N{nfe} w2={w2_sq(ref, gen, ot):7.4f} "
                        f"mmd={mmd2(ref, gen):.5f} nn={nn_tst(ref, gen):.3f} "
                    )
                print(line)


if __name__ == "__main__":
    main()
