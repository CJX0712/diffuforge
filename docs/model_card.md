# Model Card — DiffuForge denoiser & DiffuFuse sampler

Author: **晨星 (CJX0712)** · MIT licensed.

This card follows the model-card spirit (intended use, training data, evaluation,
limitations) for the two learned/algorithmic artifacts shipped in DiffuForge: the
**EDM-preconditioned denoiser** that every sampler shares, and the **DiffuFuse sampler**
that consumes it.

---

## Model details

* **Type:** EDM-preconditioned feed-forward denoiser `D(x;σ)` with a hand-written MLP
  (3 layers, 96 hidden units, SiLU, Fourier time embedding of dimension 16).
* **Output:** a 2-D vector field `D(x;σ) ≈ x₀`; the score is `score(x;σ) = (D−x)/σ²`.
* **Training:** Adam (β₁=0.9, β₂=0.999), LR 1e-3, batch 256, 600 iterations, gradient
  clip 10, optimal-weighting EDM loss. ~2 s per `(density, seed)` on CPU.
* **Sampler:** DiffuFuse — high-order (DPM-Solver-2) start-up followed by a high-order
  multistep exponential integrator on the Karras sigma ladder (`ρ=3.0`).

## Intended use

* **Primary:** research and education on NFE-budgeted diffusion/score-based sampling; fair
  head-to-head comparison of ODE samplers under a hard NFE contract.
* **Secondary:** a drop-in, dependency-light sampler for low-latency 2-D generative tasks.
* **Not intended:** production generative modeling of real-world high-dimensional data
  (images, audio) without re-training and re-validation. The benchmark suite is deliberately
  small and synthetic.

## Training data

No external data is used. The "data" are six **analytically specified 2-D densities** with
closed-form exact samplers (`data/densities.py`): `gmm8` (8 Gaussians on a circle),
`rings` (3 concentric rings), `moons` (two half-moons), `spiral` (two-arm Archimedean),
`checker` (8×8 checkerboard, used for HPO hold-out), `banana` (quadratically warped
Gaussian). Training pairs `(x₀, σ)` are drawn on the fly; there is no dataset download and
no risk of data leakage.

A closed-form **analytic Gaussian-mixture score** (`analytic_gmm_score`) provides a gold
standard that the learned score head is cross-validated against
(`tests/test_math_invariants.py`).

## Evaluation

* **Primary metric:** PF-ODE trajectory error vs a dense converged reference, paired via
  Common Random Numbers (see `docs/architecture.md` §7). This removes the 15–20% sampling
  noise that buries the signal in W2².
* **Secondary metric:** exact-OT W2² (192-point subsample), reported with its noise floor as
  a sample-quality sanity check.
* **Conditions:** 4 benchmark densities × 3 seeds × 2 NFE budgets (10, 20) × 10 samplers.

### Results (NFE = 10, mean over densities)

| Quantity | Value | Gate |
|----------|------:|:----:|
| Flagship / best-baseline ODE-error ratio (mean) | 0.337 | ≤ 0.90 ✅ |
| Wins (densities where flagship beats best baseline) | 4 / 4 | — |
| W2² ratio (mean) | 1.011 | ≤ 1.02 ✅ |

### Results (NFE = 20)

| Quantity | Value |
|----------|------:|
| ODE-error ratio (mean) | 0.130 |
| W2² ratio (mean) | 1.011 |

### Determinism

Two identical runs are **bit-identical** (`max|Δ| = 0.0`).

## Limitations & honest negatives

* **Validated on 2-D densities only.** The method is general but the evidence is 2-D;
  scaling to images is not demonstrated here.
* **`gmm8` is the hardest density at NFE=10** (ratio 0.784): its eight narrow modes stress
  the start-up step. It still clears the gate on the mean.
* **`recycle_nodes` is a wash/negative** and is OFF by default.
* **`adaptive_order` and `adaptive_mesh` are measured negative** and retained only as ablation
  switches, not recommended.
* The training schedule (600 iters) sits on the loss plateau; the sampler ranking is
  insensitive to this knob (ratios 0.28→0.58 across 500→1000 iters, all far below 0.90).

## Ethical considerations

The artifacts are fully synthetic and contain no personal, copyrighted, or sensitive data.
There are no deployment decisions affecting people; the project is a research/educational
toolkit. No fairness, privacy, or security risks beyond those of any numerical library.

## Reproducibility

```
pip install -e ".[dev]"
python examples/run_demo.py      # regenerates benchmark.json with current machine results
```

Every number in `benchmark.json` is produced by that run — nothing is hard-coded.
