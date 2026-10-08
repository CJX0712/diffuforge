"""End-to-end demo: train -> sample -> evaluate -> gate -> benchmark.json.

Run::

    python examples/run_demo.py

Every number written to ``benchmark.json`` comes from an actual run on this
machine — nothing is hard-coded.  The demo must finish inside ``BUDGET_SEC``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from diffuforge.core.config import DiffusionConfig  # noqa: E402
from diffuforge.core.seed import get_rng, set_all  # noqa: E402
from diffuforge.data.densities import get_density  # noqa: E402
from diffuforge.eval.evaluate import evaluate_samples  # noqa: E402
from diffuforge.hpo.tune import tune_flagship  # noqa: E402
from diffuforge.pipeline.pipeline import (  # noqa: E402
    DiffuPipeline,
    _ref_seed,
    _sample_seed,
)
from diffuforge.samplers.registry import FLAGSHIP, all_sampler_names, get_sampler  # noqa: E402
from diffuforge.score.train import train_denoiser  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "benchmark.json"
# Designed to finish in ~60 s on a quiet CPU; the guard is set higher to absorb
# CI-runner variance so the build does not flake on wall-clock.
BUDGET_SEC = 120.0


def _determinism_check(cfg: DiffusionConfig) -> dict[str, Any]:
    """Re-run one cell twice with the same seed; core metrics must match bit-for-bit.

    A short training schedule is used here (full-schedule equivalence is covered
    by the test-suite) so the demo stays inside its wall-clock budget.
    """
    cfg = cfg.with_overrides(n_iters=300)
    ds, seed, nfe = "gmm8", 7, 20
    spec = get_density(ds)
    ref = spec.sample(int(cfg.n_eval_ref), get_rng(_ref_seed(ds, seed)))
    runs = []
    for _ in range(2):
        set_all(seed)
        model, _ = train_denoiser(spec, seed, cfg)
        cells = {}
        for name in (FLAGSHIP, "heun"):
            sampler = get_sampler(name, cfg)
            model.reset_nfe()
            gen, used = sampler.sample(model, int(cfg.n_gen), nfe, get_rng(_sample_seed(ds, seed)))
            ev = evaluate_samples(
                ref,
                gen,
                dataset=ds,
                seed=seed,
                sampler=name,
                nfe_budget=nfe,
                nfe_used=used,
                cfg=cfg,
            )
            cells[name] = {
                "w2_sq": ev.w2_sq,
                "mmd2": ev.mmd2,
                "coverage": ev.coverage,
                "nn_tst": ev.nn_tst,
                "nfe_used": ev.nfe_used,
            }
        runs.append(cells)
    maxdiff = 0.0
    for name in runs[0]:
        for k in runs[0][name]:
            maxdiff = max(maxdiff, abs(runs[0][name][k] - runs[1][name][k]))
    return {"max_abs_diff": maxdiff, "bit_identical": maxdiff == 0.0, "cells": runs[0]}


def main() -> int:
    t0 = time.perf_counter()
    set_all(7)
    cfg = DiffusionConfig()
    print("DiffuForge demo  ·  author 晨星 (CJX0712)")
    print(
        f"datasets={list(cfg.datasets)}  seeds={list(cfg.seeds)}  "
        f"nfe={list(cfg.nfe_budgets)}  samplers={all_sampler_names()}"
    )
    print(
        "primary metric = ode_err (PF-ODE trajectory error vs dense reference), "
        "common random numbers = True"
    )

    report = DiffuPipeline(cfg).run(ablation=True)
    payload: dict[str, Any] = report.to_dict()

    tuning = tune_flagship(cfg, n_trials=5)
    payload["tuning"] = tuning
    print(
        f"\ntuning (Optuna TPE, held-out {tuning['val_dataset']}/seed {tuning['val_seed']}, "
        f"objective {tuning['objective']}): "
        f"{tuning['best_params']} -> ode_err={tuning['best_value']:.6f}"
    )

    det = _determinism_check(cfg)
    payload["determinism"] = det
    payload["elapsed_sec_total"] = round(time.perf_counter() - t0, 3)
    payload["budget_sec"] = BUDGET_SEC

    print(
        f"\ndeterminism: max|Δ| over two identical runs = {det['max_abs_diff']} "
        f"(bit_identical={det['bit_identical']})"
    )

    agg = report.aggregate
    for nfe in sorted(agg, key=lambda k: int(k)):
        blk = agg[nfe]
        print(f"\n=== NFE {nfe} · ode_err (PRIMARY, lower is better) ===")
        for ds, cell in sorted(blk["per_dataset"].items()):
            row = (
                "  "
                + f"{ds:<8}"
                + " ".join(
                    f"{n}={cell[n]['mean']:.5f}" for n in sorted(cell) if not n.startswith("_")
                )
            )
            print(row)
            print(
                f"      best baseline = {cell['_best_ode_baseline']} "
                f"({cell['_best_ode_baseline_val']:.5f}) -> "
                f"flagship ratio {cell['_best_ode_ratio']:.3f} | "
                f"W2^2 ratio {cell['_best_w2_ratio']:.3f}"
            )
        print(
            f"  mean ode_err ratio {blk['flagship_ode_ratio_mean']:.4f} "
            f"(geomean {blk['flagship_ode_ratio_geomean']:.4f}) | "
            f"mean W2^2 ratio {blk['flagship_w2_ratio_mean']:.4f}"
        )

    print("\n=== ablation (flagship components) ===")
    for row in report.ablation:
        print(
            f"  {row['variant']:<22} ode_err={row['ode_err']:.6f} "
            f"nfe={row['nfe_used']:<3} ratio={row['ratio_vs_best']:.4f}"
        )

    print("\n=== failure cases (flagship loses) ===")
    if report.failures:
        for f in report.failures:
            print(
                f"  [{f['dataset']} @NFE{f['nfe']}] ratio={f['flagship_ode_ratio']} "
                f"winner={f['winner']} :: {f['cause']}"
            )
    else:
        print("  none — flagship wins every density at every budget")

    print("\n=== gates ===")
    for k, v in report.gates.items():
        print(f"  {k}: {v}")

    total = time.perf_counter() - t0
    ok = total <= BUDGET_SEC
    print(
        f"\ntotal wall clock: {total:.1f}s (budget {BUDGET_SEC:.0f}s) "
        f"-> {'OK' if ok else 'OVER BUDGET'}"
    )

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
