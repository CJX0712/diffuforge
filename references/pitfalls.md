# Pitfalls & Hard-Won Fixes

A symptom → root-cause → fix log of the traps hit while building DiffuForge.
Honest record of what did **not** work, so it is not rediscovered.

---

## P1 — W2² cannot arbitrate sampler quality (the central trap)

* **Symptom:** Ranking samplers by W2² produced noisy, non-reproducible, and
  non-discriminating results. At NFE ≥ 10 every sampler sits at the *model's* quality
  ceiling; W2² vs a 512-point reference is ~5× the i.i.d. noise floor and the inter-sampler
  spread collapses to 1–2%. W2² over 160–256 clouds carries 15–20% sampling noise.
* **Root cause:** W2² measures *sample quality*, which is already saturated at these NFEs.
  The thing that actually differs between samplers is *ODE-integration accuracy*, which W2²
  cannot see through the sampling noise.
* **Fix:** Switched the **primary metric** to **PF-ODE trajectory error** (`eval/ode_error.py`):
  a converged Karras-Heun reference (error ~1e-6) vs each sampler's endpoint, paired via CRN.
  Variance ~3 orders of magnitude below W2². W2² kept only as a secondary sanity check (with
  its noise floor reported).

---

## P2 — ODE-error metric showed a false "flagship loses" (CRN bug)

* **Symptom:** The ODE-error metric reported `ode_err ≈ 1.4` (essentially "no agreement")
  and the flagship looked worse than baselines.
* **Root cause:** The reference solution and the sampled cloud were started from *different*
  initial noise, because the CRN seed depended on the NFE budget. The comparison was no
  longer paired — it measured sampling luck, not integration error.
* **Fix:** `_sample_seed(dataset, seed)` now depends on `(dataset, seed)` **only** — never on
  the sampler name or the NFE budget (see `pipeline/pipeline.py`). The reference and every
  sampler start from identical noise.

---

## P3 — "Free two-interval start-up" looked clever, was worse

* **Symptom:** A variant whose DPM-Solver-2 start-up spans *two* Karras intervals makes the
  2 NFE effectively "free" (2 NFE covering 2 intervals).
* **Root cause / Fix:** Implemented and **rejected** — the wider collocation interval costs
  more accuracy than the saved NFE buys (6× worse on `rings` at NFE=10). The one-interval
  start-up is the measured winner.

---

## P4 — `recycle_nodes` looked like a win on one density

* **Symptom:** An ablation confined to a single `(gmm8, seed=7)` cell showed
  `recycle_nodes` as the best variant.
* **Root cause:** A single density/seed cell is not representative. `recycle_nodes` helps
  `gmm8` but **hurts** the other three densities (the quadratic extrapolation over a
  fit-region twice the step width amplifies error).
* **Fix:** The ablation now runs on the *first seed of every density*, averaged, reusing the
  already-trained models. `recycle_nodes` is a wash/negative on average and is **OFF by
  default**, kept only as an honest ablation switch.

---

## P5 — `rho = 7` (EDM's image default) is sub-optimal on 2-D densities

* **Symptom:** Following Karras et al.'s `ρ = 7` gave needlessly high error at NFE ≤ 20.
* **Root cause:** `ρ = 7` is tuned for Heun at 18–40 NFE on *image* models.
* **Fix:** A measured sweep over `ρ ∈ {1..7}` on 2-D densities puts the optimum at **3** for
  every sampler; the whole benchmark shares `ρ = 3.0`.

---

## P6 — Demo overran the wall-clock budget on the first pass

* **Symptom:** The end-to-end demo exceeded the 60 s budget after adding the HPO study and
  determinism re-run.
* **Root cause:** Training schedule (1000 → 700 iters) plus extra pipeline stages.
* **Fix:** Reduced training to 600 iterations (loss already on plateau at 700; ranking
  insensitive). Demo now ~51 s on a quiet CPU; the guard is set to 120 s to absorb CI-runner
  variance (the gate is about correctness, not raw latency).

---

## P7 — `adaptive_order` / `adaptive_mesh` are measured negatives

* **Symptom:** Intuitively appealing "smart" switches.
* **Root cause / Fix:** Both measured worse than the fixed-order scheme on this benchmark
  (`adaptive_mesh` strongly negative at NFE ≤ 35). Retained as ablation switches, not
  recommended defaults — reported honestly rather than deleted.

---

## P8 — Demo determinism re-run inflated the W2² ratio above the gate once

* **Symptom:** A one-off run nudged the secondary W2² ratio near the 1.02 threshold.
* **Root cause:** Secondary metric sits close to the noise floor by construction (P1). The
  *primary* ODE-error gate is the real discriminator; the secondary is a sanity check.
* **Fix:** Keep the secondary gate (1.02) as a guard against quality regressions, but rely on
  the primary ODE-error gate for the headline claim. The primary margin (0.337) is huge.
