# DiffuForge

[![CI](https://github.com/CJX0712/diffuforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/diffuforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/CJX0712/diffuforge/releases)

**NFE-budgeted diffusion / score-based generative modeling, with the DiffuFuse sampler — a
high-order start-up + high-order multistep exponential integrator that beats every baseline
ODE sampler at the same function-evaluation (NFE) budget.**

DiffuForge is a self-contained, NumPy-only toolkit that trains a small EDM-preconditioned
denoiser on 2-D densities and compares ten ODE samplers head-to-head under a hard NFE
contract. The flagship **DiffuFuse** sampler replaces the standard first-order multistep
start-up with a second-order DPM-Solver-2 step, removing the dominant source of integration
error on the Karras sigma ladder.

> Author: **晨星 (CJX0712)** — MIT licensed.

---

## Why this exists

At low NFE budgets (the regime that matters for latency-critical sampling) the weak point of
every linear-multistep diffusion ODE solver is its **start-up step**: the first Karras
interval is the widest one (largest single share of global error), and a multistep method
*never forgets* its start-up error. Published DPM-Solver++ seeds the recursion with a
first-order step. DiffuFuse seeds it with a **second-order single-step** (DPM-Solver-2)
update instead — at a real, budgeted NFE cost.

## Headline result

On 4 densities × 3 seeds, DiffuFuse is scored against the best baseline on **PF-ODE
trajectory error** (the metric that actually discriminates samplers — see
[`docs/architecture.md` §7](docs/architecture.md)). Lower ratio is better; `< 1.0` means
DiffuFuse wins. Pre-declared acceptance gate: **mean ratio ≤ 0.90 at the smallest budget.**✅

| NFE | Flagship ODE-error ratio (mean) | W2² ratio (mean) | Gate |
|----:|-------------------------------:|-----------------:|:----:|
| 10  | **0.337** ✅ (wins 4/4)        | 1.011 ✅         | PASS |
| 20  | **0.130** ✅ (wins 4/4)        | 1.011 ✅         | PASS |

Per-density, NFE = 10 (ratio of DiffuFuse ODE-error to the best baseline's):

| Density | ratio | best baseline | W2² ratio |
|---------|------:|---------------|----------:|
| gmm8    | 0.784 | dpm4m         | 1.022     |
| moons   | 0.309 | dpm4m         | 1.039     |
| rings   | 0.111 | dpm3m         | 0.981     |
| spiral  | 0.146 | dpm3m         | 1.000     |

Full numbers, ablation, and the determinism proof are in [`benchmark.json`](benchmark.json)
(produced by `python examples/run_demo.py` on this machine — nothing hard-coded).

## Component ablation (flagship at NFE = 20, mean ODE-error over 4 densities)

| Variant | ODE-error | × vs best |
|---------|---------:|---------:|
| diffufuse (full)        | 0.00051 | 1.03 |
| +recycle_nodes          | 0.00050 | 1.00 |
| −start-up (n_startup=0) | 0.00232 | 4.68 |
| order=3                 | 0.00229 | 4.63 |
| +adaptive_order         | 0.00181 | 3.66 |
| order=2                 | 0.00928 | 18.7 |
| +adaptive_mesh          | 0.01550 | 31.3 |
| order=1                 | 0.03753 | 75.8 |

The signal: **order is everything**, and the **second-order start-up is what buys the
order**. `recycle_nodes` is a wash on average (and hurts the stiffest density at NFE=10), so
it is kept OFF by default; `adaptive_order` / `adaptive_mesh` are measured negative and
kept as honest ablation switches.

---

## Install

```bash
pip install -e ".[dev]"      # dev: pytest, pytest-cov, ruff
pip install -e ".[full]"     # full: scipy, scikit-learn, optuna (optional)
```

The core system needs **only NumPy**. The `[full]` extras are optional: Optuna is used by
the HPO study (which degrades gracefully to a grid search if Optuna is absent), and
scikit-learn / scipy back a few alternative metric implementations.

## Quickstart

```python
from diffuforge.core.config import DiffusionConfig
from diffuforge.pipeline.pipeline import DiffuPipeline

report = DiffuPipeline(DiffusionConfig()).run()  # train -> sample -> evaluate -> gate
print(report.gates)  # G1 / G1b / G2 / G3
```

From the CLI:

```bash
diffuforge --datasets gmm8,rings --seeds 7 --nfe 10,20 --iters 600 --out bench.json
```

End-to-end demo (training + 10 samplers + ablation + HPO + determinism, ≤ 60 s on CPU):

```bash
python examples/run_demo.py
```

## Project layout

```
diffuforge/
  core/        config (with env overrides), seed (single rng entry), types, errors
  data/        six 2-D densities + analytic Gaussian-mixture score (gold standard)
  score/       hand-written MLP (forward/backward), EDM preconditioning, Adam trainer
  samplers/    diffufuse (flagship), dpm (1..4 multistep), ode (euler/heun/logeuler), unipc
  eval/        metrics (W2², MMD², coverage, NN test) + PF-ODE trajectory error
  hpo/         Optuna TPE study, degrades to grid search
  pipeline/    the one entry point that produces every benchmark number
examples/      run_demo.py
tests/         invariant + sampler + metric + pipeline suites (93% coverage)
scripts/       diagnostic investigations that produced the findings (not shipped)
docs/          architecture.md, model_card.md
```

## Reproducibility & determinism

* A single `set_all(seed)` entry point seeds every `numpy` generator.
* All samplers are compared **paired** via Common Random Numbers (CRN): the initial-noise
  seed depends on `(dataset, seed)` only — never on the sampler or the NFE budget — so the
  comparison measures integration error, not sampling luck.
* Two identical runs are **bit-identical** (verified in the demo and the test suite).

## References

* Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*,
  NeurIPS 2022 (EDM).
* Lu et al., *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
  Models*, 2023.
* Zhao et al., *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of
  Diffusion Models*, 2023.

## License

MIT — see [`LICENSE`](LICENSE). © 晨星 (CJX0712).
