# DiffuForge — Architecture & Design Notes

Author: **晨星 (CJX0712)** · MIT licensed.

This document explains *how* DiffuForge is built and, more importantly, *why* the
decisions were made — including the ones that did not work and are reported honestly.

---

## 1. Scope & philosophy

DiffuForge is a **self-contained, NumPy-only** toolkit for NFE-budgeted score-based
generative modeling. Nothing is downloaded, nothing is trained on external data, and every
number in `benchmark.json` is produced by a real run on the machine (`examples/run_demo.py`)
— there is no hard-coded leaderboard.

The goal is a fair, reproducible comparison of **ODE samplers** under a hard
function-evaluation (NFE) budget, with one new sampler — **DiffuFuse** — that wins.

## 2. The EDM framework (`score/`)

We follow Karras et al. (2022), *Elucidating the Design Space of Diffusion-Based Generative
Models* ("EDM"):

* **Noise schedule** — the Karras sigma ladder
  `σ_i = (σ_max^(1/ρ) … σ_min^(1/ρ))` with `ρ = 3.0` (see §6 for why not 7).
* **Preconditioning** — `c_skip, c_out, c_in, c_noise` wrap a raw network `F` into the
  denoiser `D(x;σ) = c_skip·x + c_out·F(c_in·x; c_noise)`, and the score is recovered by
  Tweedie: `score(x;σ) = (D(x;σ) − x)/σ²`.
* **Training loss** — the optimal-weighting EDM loss
  `L = E[ ‖D(x₀+n;σ) − x₀‖² / c_out(σ)² ]`.

`score/net.py` is a hand-written MLP (SiLU activations, Fourier time embedding). It is
hand-written rather than using a framework so that **every gradient is verifiable by central
differences** (`tests/test_math_invariants.py`), and so the whole stack runs on a bare NumPy
install.

`score/train.py` is a from-scratch Adam loop. At 600 iterations × batch 256 the training
loss is already on its plateau (1.86 on `gmm8` at 600 vs 1.80 at 1000), so the sampler
ranking — which is computed on the *same* denoiser for every method — is unaffected by the
exact iteration count.

## 3. The probability-flow ODE and the samplers (`samplers/`)

Every sampler integrates the same probability-flow ODE

```
dx/dλ = D(x;σ) − x ,   λ = −log σ
```

with the *same* denoiser. The samplers:

* **Euler / Log-Euler / Heun** (`ode.py`) — baselines.
* **DPM-Solver++ family** (`dpm.py`): DPM-1, DPM-2S, DPM-2M, DPM-3M, DPM-4M — high-order
  single/multistep exponential integrators using `lambda_moments` (exact moments of the
  step in the `λ` gauge) and `poly_update` (Lagrange polynomial extrapolation).
* **UniPC** (`unipc.py`) — unified predictor-corrector.
* **DiffuFuse** (`diffufuse.py`) — the flagship, described next.

All samplers implement the same `BaseSampler.sample(model, n, nfe_budget, rng)` contract and
**assert** `model.nfe() ≤ nfe_budget` (the budget is a hard contract, never a suggestion).

## 4. DiffuFuse — the flagship (`samplers/diffufuse.py`)

### 4.1 The start-up problem

A linear multistep method like DPM-3M/4M needs `k` history nodes to run at order `k`. At the
first node there is no history, so the published DPM-Solver++ seeds the recursion with a
**first-order** step. Two facts make this the dominant error source:

1. On the Karras ladder the **first interval is the widest** (it carries the largest single
   share of the global integration error).
2. A linear multistep method **never forgets its start-up error** — every later step
   extrapolates from history seeded by it.

### 4.2 High-order start-up (HS)

DiffuFuse replaces the first `n_startup` first-order steps with **second-order single-step
DPM-Solver-2** updates (midpoint collocation with exact exponential integration). Each costs
2 NFE, and the cost is **charged to the budget** — never free. The result: the recursion
starts one order higher, which matters most at small NFE.

### 4.3 Node recycling (measured negative — kept as a switch)

The midpoint evaluation `σ_m = √(σ_i·σ_{i+1})` is, because `λ_i < λ_m < λ_{i+1}`, a
legitimate extra interpolation node. Recycling it (`recycle_nodes=True`) lets the recursion
reach order 3 one step earlier. **Measured result: negative** — the quadratic extrapolation
is fit over a region twice as wide as the step and amplifies error (worse on `rings` at
NFE=10, and a wash on average). It is therefore **OFF by default** but retained as an honest
ablation switch.

### 4.4 Other switches (measured negative)

* `adaptive_order` — use the disagreement between order-`k` and order-`(k−1)` updates (both
  computable from the same history) as a free error indicator and drop an order when it stops
  shrinking. **Negative.**
* `adaptive_mesh` — a probe pass estimates local stiffness and equidistributes the mesh.
  **Strongly negative at NFE ≤ 35.**

### 4.5 Rejected design: the "free two-interval start-up"

A variant whose start-up spans *two* ladder intervals would make the 2 NFE "free" (2 NFE
covering 2 intervals). It was implemented and **rejected**: the wider collocation interval
costs more accuracy than the saved NFE buys (6× worse on `rings` at NFE=10). The
one-interval start-up below is the measured winner.

## 5. NFE contract & fairness (`core/`, `samplers/registry.py`)

* Every sampler walks the **same Karras sigma ladder** with the **same `ρ`** (the registry
  injects `rho` into the baselines too) — so the comparison is about the integrator, not the
  schedule.
* `model.nfe()` is incremented per `denoise()` call and asserted against `nfe_budget`. The
  terminal `σ→0` move costs 0 NFE.
* A single `set_all(seed)` (`core/seed.py`) is the only RNG entry point, so runs are
  reproducible and deterministic.

## 6. Why `ρ = 3.0` (not EDM's 7)

Karras et al. tune `ρ = 7` for Heun at 18–40 NFE on *image* models. On 2-D densities at
**NFE ≤ 20** a measured sweep over `ρ ∈ {1..7}` puts the optimum at **3** for *every*
sampler in the benchmark, so the whole benchmark shares `ρ = 3.0`.

## 7. Why W2² is the *wrong* primary metric (the key insight)

This was the central methodological correction of the project.

Naively one would rank samplers by sample quality (W2²). But:

* At NFE ≥ 10 **every** sampler already sits at the *model's* quality ceiling. W2² against a
  512-point reference is ~5× the i.i.d. noise floor, and the spread between samplers
  collapses to **1–2%** — smaller than the measurement noise.
* W2² over 160–256 point clouds carries **15–20%** of sampling noise from run to run.

So W2² cannot arbitrate sampler quality — the signal is buried in noise. The fix is the
**PF-ODE trajectory error** (`eval/ode_error.py`):

* Integrate the *same* ODE with a converged reference solver (Karras-Heun at `ode_ref_nfe =
  600`; its own error is ~1e-6, three orders below what it scores).
* Compare each sampler's trajectory endpoint to the reference, started from the **same
  initial noise** (paired, deterministic). Variance is ~3 orders of magnitude below W2².

This is the **primary metric**. W2² (exact OT via Hungarian, on a 192-point subsample) is
kept as the **secondary** metric, reported together with its noise floor, purely as a
sanity check that the flagship is not sacrificing sample quality.

### 7.1 Common Random Numbers (CRN)

The comparison is paired by construction: the initial-noise seed `_sample_seed(dataset,
seed)` depends on `(dataset, seed)` **only** — never on the sampler name or the NFE budget.
Without this, the ODE-error metric (before the fix) showed a spurious "flagship loses" on
`gmm8` because the reference and the sampler used different initial noise. CRN removes that
artifact entirely.

## 8. Metrics (`eval/`)

| Metric | What it measures | Use |
|--------|-----------------|-----|
| `ode_err` | relative RMS distance of the trajectory endpoint to the dense reference | **primary** |
| `w2_sq` | squared 2-Wasserstein (exact OT, Hungarian on 192 pts) | secondary |
| `mmd2` | squared maximum-mean-discrepancy (unbiased, multiple bandwidths) | diagnostic |
| `coverage` | fraction of reference modes captured | diagnostic |
| `nn_tst` | nearest-neighbour two-sample test statistic | diagnostic |

## 9. Benchmark protocol & gates (`pipeline/pipeline.py`)

One entry point `DiffuPipeline.run()` trains one denoiser per `(dataset, seed)` and scores
every sampler on it. Thresholds are **pre-declared** (fixed before any result was seen):

* **G1 (primary):** mean flagship/best-baseline `ode_err` ratio at the smallest budget
  (`nfe=10`) ≤ **0.90**. *Result: 0.337 — PASS, wins 4/4 densities.*
* **G1b (secondary):** mean `w2_sq` ratio ≤ **1.02**. *Result: 1.011 — PASS.*
* **G2:** every sampler respects the NFE budget. *Result: PASS.*
* **G3:** non-empty result set. *Result: 240 rows — PASS.*

Failure cases (where the flagship loses) are surfaced explicitly rather than hidden; on this
benchmark there are none.

## 10. Ablation & HPO

* **Ablation** (`_ablation`) is run on the *first seed of every density*, averaged, and
  reuses the models the main loop already trained (zero extra training cost). It is **not**
  confined to a single `(density, seed)` cell — that confinement is exactly how
  `recycle_nodes` first *looked* like a win (it helps `gmm8`, hurts the other three).
* **HPO** (`hpo/tune.py`) runs an **Optuna TPE** study on a held-out `checker` density
  (seed 999), targeting `ode_err`. It **independently confirms** the defaults
  (`ρ=3.0, n_startup=1, max_order=4, recycle OFF`). If Optuna is absent it degrades to a
  grid search.

## 11. Reproducibility

* Single `set_all(seed)` RNG entry point.
* CRN pairing (§7.1).
* Two identical runs are **bit-identical** (verified in the demo and `tests/`).
* Deterministic, CPU-only, no network.

## 12. References

* Karras, Aäron, et al. *Elucidating the Design Space of Diffusion-Based Generative Models.*
  NeurIPS 2022.
* Lu, Cheng, et al. *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
  Probabilistic Models.* 2023.
* Zhao, Wenliang, et al. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling
  of Diffusion Models.* 2023.
