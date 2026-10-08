# Changelog

All notable changes to **DiffuForge** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to semantic
versioning.

## [0.1.0] — 2026-10-09

Initial public release. Author: **晨星 (CJX0712)**.

### Added
- **DiffuFuse flagship sampler** (`diffuforge/samplers/diffufuse.py`): high-order start-up
  (second-order DPM-Solver-2) + high-order multistep exponential integrator. Replaces the
  standard first-order multistep start-up that dominates integration error on the Karras
  ladder.
- **Ten ODE samplers** for head-to-head comparison: Euler, Log-Euler, Heun, DPM-1/2S/2M/3M/4M,
  UniPC, and DiffuFuse.
- **EDM framework** (Karras et al. 2022): preconditioning, Karras sigma ladder, hand-written
  MLP with verifiable forward/backward (gradient-checkable by central differences), Adam
  trainer.
- **PF-ODE trajectory error metric** (`eval/ode_error.py`): the primary, low-variance metric
  that actually discriminates samplers (paired, deterministic, ~3 orders of magnitude below
  W2² sample noise).
- **Common Random Numbers (CRN)** pairing in the pipeline: initial-noise seed depends on
  `(dataset, seed)` only, so every sampler — and the dense reference — start from identical
  noise.
- **Benchmark pipeline** (`pipeline/pipeline.py`) with pre-declared dual acceptance gates:
  G1 (primary ODE-error ratio ≤ 0.90 at the smallest budget) and G1b (secondary W2² ratio
  ≤ 1.02), plus NFE-budget (G2) and row-count (G3) gates.
- **Component ablation** reusable across all densities (not a single cell), and an
  **Optuna TPE** HPO study that independently confirms the default hyperparameters.
- **Six 2-D densities** with closed-form exact samplers (no downloads, no leakage), including
  an **analytic Gaussian-mixture score** used as the gold standard for the score head.
- **Determinism proof**: two identical runs are bit-identical.
- Packaging: `pyproject.toml` (ruff pinned to 0.16.10 as a hard gate), `requirements*.txt`,
  GitHub Actions CI (3.11/3.12/3.13: lint + format + pytest + demo), `Dockerfile`, `Makefile`,
  `LICENSE` (MIT).
- Test suite: math-invariant, sampler (NFE contract), metric, and pipeline tests —
  **93% coverage**, all green.

### Measured & honestly reported
- `recycle_nodes`, `adaptive_order`, and `adaptive_mesh` are **negative or neutral** results.
  They are retained as ablation switches rather than silently deleted.
- A "free two-interval start-up" variant was implemented and rejected (6× worse on `rings`
  at NFE=10).

### Known limitations
- Validated on 2-D densities; the denoiser and sampler are general but the benchmark suite is
  intentionally small and fast (CPU, ≤ 60 s end-to-end).
- `gmm8` is the hardest density for the flagship at NFE=10 (ratio 0.784) because its 8 narrow
  modes stress the start-up step; it still clears the gate on the mean.

[0.1.0]: https://github.com/CJX0712/diffuforge/releases/tag/v0.1.0
