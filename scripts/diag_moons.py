"""Diagnostic: sampler behaviour per (dataset, seed) + high-NFE convergence limit.

Not part of the deliverable; used to locate the flagship's instability.
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
from diffuforge.pipeline.pipeline import _ref_seed, _sample_seed
from diffuforge.samplers.registry import all_sampler_names, get_sampler
from diffuforge.score.train import train_denoiser

DATASETS = ("gmm8", "rings", "moons", "spiral")
NAMES = list(all_sampler_names())


def main() -> None:
    cfg = DiffusionConfig()
    nfetab = (10, 20, 200)
    for ds in DATASETS:
        spec = get_density(ds)
        table: dict[str, dict[int, list[float]]] = {n: {} for n in NAMES}
        for seed in (7, 17, 29):
            set_all(seed)
            model, _tres = train_denoiser(spec, int(seed), cfg)
            ref = spec.sample(int(cfg.n_eval_ref), get_rng(_ref_seed(ds, int(seed))))
            for name in NAMES:
                for nfe in nfetab:
                    model.reset_nfe()
                    rng = get_rng(_sample_seed(ds, int(seed), name, int(nfe)))
                    s = get_sampler(name, cfg)
                    x, _used = s.sample(model, int(cfg.n_gen), int(nfe), rng)
                    table[name].setdefault(nfe, []).append(w2_sq(ref, x, int(cfg.eval_ot_size)))
        print(f"\n=== {ds}")
        for name in NAMES:
            line = f"  {name:<10s}"
            for nfe in nfetab:
                v = table[name].get(nfe, [])
                line += f" | N{nfe:<3d} mean={np.mean(v):7.4f} [{', '.join(f'{q:.3f}' for q in v)}]"
            print(line)


if __name__ == "__main__":
    main()
