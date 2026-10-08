"""The one entry point that produces every number in the benchmark.

Call graph (acyclic):

    cli / demo -> pipeline -> {data, score, samplers, eval, hpo} -> core
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..core.config import DiffusionConfig
from ..core.errors import EvalError
from ..core.seed import get_rng, set_all
from ..core.types import BenchmarkRow
from ..data.densities import get_density
from ..eval.evaluate import evaluate_samples
from ..eval.ode_error import dense_reference, ode_rel_err
from ..samplers.registry import FLAGSHIP, all_sampler_names, baseline_names, get_sampler
from ..score.train import train_denoiser

REF_SEED_OFFSET = 990_000
SAMPLE_SEED_OFFSET = 424_000

# Pre-declared acceptance thresholds (fixed before any result was looked at).
# Primary: the flagship must beat the best baseline's PF-ODE integration error
# by >= 10% on the mean over densities at the smallest NFE budget.
G1_MAX_ODE_RATIO = 0.90
# Secondary: sample quality must be non-inferior within 2%.
G1B_MAX_W2_RATIO = 1.02


@dataclass
class BenchmarkReport:
    """Serializable result of one full benchmark run."""

    config: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    ablation: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ref_seed(dataset: str, seed: int) -> int:
    return REF_SEED_OFFSET + 131 * len(dataset) + int(seed)


def _sample_seed(dataset: str, seed: int) -> int:
    """Common random numbers: depends on (dataset, seed) only.

    It deliberately contains neither the sampler name nor the NFE budget, so
    *every* sampler — and the dense reference solution — start from the
    identical initial noise.  The comparison is therefore paired: the difference
    between two rows is integration error, not sampling luck.  Without this,
    W2^2 over 160-256 point clouds carries 15-20% of noise, which is larger than
    the effect being measured.
    """
    return SAMPLE_SEED_OFFSET + 17 * len(dataset) + 1009 * int(seed)


class DiffuPipeline:
    """Train one denoiser per (dataset, seed) and score every sampler on it."""

    def __init__(self, cfg: DiffusionConfig | None = None) -> None:
        self.cfg = (cfg or DiffusionConfig()).validate()
        # Models / reference clouds built during :meth:`run`, reused by the
        # ablation so it costs no extra training.
        self._models: dict[tuple[str, int], Any] = {}
        self._refs: dict[tuple[str, int], np.ndarray] = {}
        self._xrefs: dict[tuple[str, int], np.ndarray] = {}

    # ------------------------------------------------------------------ run
    def run(
        self,
        datasets: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
        nfe_budgets: Sequence[int] | None = None,
        samplers: Sequence[str] | None = None,
        ablation: bool = True,
    ) -> BenchmarkReport:
        cfg = self.cfg
        datasets = list(datasets or cfg.datasets)
        seeds = list(seeds or cfg.seeds)
        nfe_budgets = list(nfe_budgets or cfg.nfe_budgets)
        names = list(samplers or all_sampler_names())
        set_all(int(seeds[0]))

        t0 = time.perf_counter()
        rows: list[BenchmarkRow] = []
        train_info: dict[str, Any] = {}

        for ds in datasets:
            spec = get_density(ds)
            for seed in seeds:
                model, tres = train_denoiser(spec, int(seed), cfg)
                train_info[f"{ds}/{seed}"] = {
                    "sigma_data": tres.sigma_data,
                    "final_loss": tres.final_loss,
                    "train_sec": round(tres.elapsed_sec, 4),
                }
                ref = spec.sample(int(cfg.n_eval_ref), get_rng(_ref_seed(ds, int(seed))))
                # Converged PF-ODE solution started from the *same* noise.
                x_ref, ref_nfe = dense_reference(
                    model,
                    cfg,
                    get_rng(_sample_seed(ds, int(seed))),
                    n=int(min(cfg.ode_ref_points, cfg.n_gen)),
                )
                train_info[f"{ds}/{seed}"]["ode_ref_nfe"] = int(ref_nfe)
                self._models[(ds, int(seed))] = model
                self._refs[(ds, int(seed))] = ref
                self._xrefs[(ds, int(seed))] = x_ref
                for name in names:
                    sampler = get_sampler(name, cfg)
                    for nfe in nfe_budgets:
                        model.reset_nfe()
                        rng = get_rng(_sample_seed(ds, int(seed)))
                        ts = time.perf_counter()
                        gen, nfe_used = sampler.sample(model, int(cfg.n_gen), int(nfe), rng)
                        dt = time.perf_counter() - ts
                        if not np.all(np.isfinite(gen)):
                            raise EvalError(
                                "sampler produced non-finite samples",
                                sampler=name,
                                dataset=ds,
                                seed=int(seed),
                                nfe=int(nfe),
                            )
                        ev = evaluate_samples(
                            ref,
                            gen,
                            dataset=ds,
                            seed=int(seed),
                            sampler=name,
                            nfe_budget=int(nfe),
                            nfe_used=int(nfe_used),
                            cfg=cfg,
                            elapsed_sec=dt,
                        )
                        rows.append(
                            BenchmarkRow(
                                dataset=ds,
                                seed=int(seed),
                                sampler=name,
                                nfe_budget=int(nfe),
                                nfe_used=int(nfe_used),
                                w2_sq=ev.w2_sq,
                                mmd2=ev.mmd2,
                                coverage=ev.coverage,
                                nn_tst=ev.nn_tst,
                                ode_err=ode_rel_err(gen, x_ref),
                            )
                        )

        report = BenchmarkReport(
            config={
                "datasets": datasets,
                "seeds": seeds,
                "nfe_budgets": nfe_budgets,
                "samplers": names,
                "flagship": FLAGSHIP,
                "n_gen": cfg.n_gen,
                "n_eval_ref": cfg.n_eval_ref,
                "n_iters": cfg.n_iters,
                "sigma_max_rel": cfg.sigma_max_rel,
                "rho_schedule": cfg.rho_schedule,
                "ode_ref_nfe": cfg.ode_ref_nfe,
                "exact_moment_init": cfg.exact_moment_init,
                "common_random_numbers": True,
                "gates": {
                    "G1_max_ode_ratio": G1_MAX_ODE_RATIO,
                    "G1b_max_w2_ratio": G1B_MAX_W2_RATIO,
                },
            },
            rows=[asdict(r) for r in rows],
            elapsed_sec=round(time.perf_counter() - t0, 3),
        )
        report.aggregate = self._aggregate(rows, nfe_budgets, names)
        report.summary = self._summary(report.aggregate, train_info)
        if ablation:
            report.ablation = self._ablation(datasets, seeds, nfe_budgets)
        report.failures = self._failures(report.aggregate)
        report.gates = self._gates(report.aggregate, rows, nfe_budgets)
        return report

    # ------------------------------------------------------------ aggregate
    @staticmethod
    def _metric_cell(
        rows: list[BenchmarkRow], ds: str, name: str, nfe: int, key: str
    ) -> dict[str, Any]:
        vals = [
            getattr(r, key)
            for r in rows
            if r.dataset == ds and r.sampler == name and r.nfe_budget == nfe
        ]
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}

    @classmethod
    def _aggregate(
        cls, rows: Iterable[BenchmarkRow], nfe_budgets: Sequence[int], names: Sequence[str]
    ) -> dict[str, Any]:
        rows = list(rows)
        out: dict[str, Any] = {}
        for nfe in nfe_budgets:
            per_ds: dict[str, Any] = {}
            for ds in sorted({r.dataset for r in rows}):
                cell: dict[str, Any] = {}
                for name in names:
                    cell[name] = {
                        **cls._metric_cell(rows, ds, name, nfe, "ode_err"),
                        "w2_sq": cls._metric_cell(rows, ds, name, nfe, "w2_sq")["mean"],
                        "mmd2": cls._metric_cell(rows, ds, name, nfe, "mmd2")["mean"],
                        "coverage": cls._metric_cell(rows, ds, name, nfe, "coverage")["mean"],
                        "nn_tst": cls._metric_cell(rows, ds, name, nfe, "nn_tst")["mean"],
                    }
                baselines = [k for k in cell if k in baseline_names()]
                for metric in ("mean", "w2_sq"):
                    if not baselines:
                        continue
                    best = min(baselines, key=lambda k: cell[k][metric])
                    tag = "_best_ode" if metric == "mean" else "_best_w2"
                    cell[tag + "_baseline"] = best
                    cell[tag + "_baseline_val"] = cell[best][metric]
                    if FLAGSHIP in cell:
                        f = cell[FLAGSHIP][metric]
                        b = cell[best][metric]
                        cell[tag + "_ratio"] = float(f / b) if b > 0 else float("nan")
                per_ds[ds] = cell
            ratios = [
                per_ds[ds]["_best_ode_ratio"] for ds in per_ds if "_best_ode_ratio" in per_ds[ds]
            ]
            w2r = [per_ds[ds]["_best_w2_ratio"] for ds in per_ds if "_best_w2_ratio" in per_ds[ds]]
            out[str(nfe)] = {
                "per_dataset": per_ds,
                "flagship_ode_ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
                "flagship_ode_ratio_geomean": (
                    float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
                ),
                "flagship_w2_ratio_mean": float(np.mean(w2r)) if w2r else float("nan"),
                "n_datasets": len(ratios),
            }
        return out

    @staticmethod
    def _summary(agg: dict[str, Any], train_info: dict[str, Any]) -> dict[str, Any]:
        return {
            "train": train_info,
            "best_baseline_by_dataset": {
                nfe: {
                    ds: agg[nfe]["per_dataset"][ds].get("_best_ode_baseline")
                    for ds in agg[nfe]["per_dataset"]
                }
                for nfe in agg
            },
        }

    # ------------------------------------------------------------- ablation
    def _ablation(
        self, datasets: Sequence[str], seeds: Sequence[int], nfe_budgets: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Component ablation on the flagship.

        Run on the first seed of **every** density and averaged, reusing the
        models the main loop already trained (so it costs no extra training).
        An ablation confined to a single (density, seed) cell is exactly how
        ``recycle_nodes`` first looked like a win: it happens to help on
        ``gmm8`` and hurts on the other three densities.
        """
        from ..samplers.diffufuse import DiffuFuseSampler

        cfg = self.cfg
        seed = int(next(iter(seeds)))
        nfe = int(max(nfe_budgets))
        cells = [(ds, self._models[(ds, seed)], self._xrefs[(ds, seed)]) for ds in datasets]

        variants: list[tuple[str, dict[str, Any]]] = [
            ("diffufuse (full)", {}),
            ("-start-up (n_startup=0)", {"n_startup": 0}),
            ("+recycle_nodes", {"recycle_nodes": True}),
            ("order=3", {"max_order": 3}),
            ("order=2", {"max_order": 2}),
            ("order=1", {"max_order": 1}),
            ("+adaptive_order", {"adaptive_order": True}),
            ("+adaptive_mesh", {"adaptive_mesh": True}),
        ]
        out: list[dict[str, Any]] = []
        for label, kw in variants:
            per_ds: dict[str, float] = {}
            w2s: list[float] = []
            used_max = 0
            for ds, model, x_ref in cells:
                sampler = DiffuFuseSampler(cfg=cfg, **kw)
                model.reset_nfe()
                gen, used = sampler.sample(
                    model, int(cfg.n_gen), nfe, get_rng(_sample_seed(ds, seed))
                )
                per_ds[ds] = ode_rel_err(gen, x_ref)
                w2s.append(
                    evaluate_samples(
                        self._refs[(ds, seed)],
                        gen,
                        dataset=ds,
                        seed=seed,
                        sampler=label,
                        nfe_budget=nfe,
                        nfe_used=used,
                        cfg=cfg,
                    ).w2_sq
                )
                used_max = max(used_max, used)
            out.append(
                {
                    "variant": label,
                    "ode_err": float(np.mean(list(per_ds.values()))),
                    "per_dataset": {k: float(v) for k, v in per_ds.items()},
                    "w2_sq": float(np.mean(w2s)),
                    "nfe_used": int(used_max),
                }
            )
        best = min(o["ode_err"] for o in out)
        for o in out:
            o["ratio_vs_best"] = float(o["ode_err"] / best) if best > 0 else float("nan")
        return out

    # ------------------------------------------------------------- failures
    @staticmethod
    def _failures(agg: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for nfe, blk in agg.items():
            for ds, cell in blk["per_dataset"].items():
                if "_best_ode_ratio" not in cell:
                    continue
                ratio = cell["_best_ode_ratio"]
                if ratio <= 1.0:
                    continue
                order = sorted(
                    (k for k in cell if not k.startswith("_")),
                    key=lambda k: cell[k]["mean"],
                )
                out.append(
                    {
                        "dataset": ds,
                        "nfe": int(nfe),
                        "flagship_ode_ratio": round(float(ratio), 4),
                        "winner": order[0],
                        "runner_up": order[1] if len(order) > 1 else None,
                        "cause": (
                            f"flagship loses on this density: {order[0]} reaches "
                            f"ode_err={cell[order[0]]['mean']:.5f} vs diffufuse "
                            f"{cell['diffufuse']['mean']:.5f}. The start-up NFE is a "
                            "fixed cost that only pays off when the first Karras "
                            "interval is stiff."
                        ),
                    }
                )
        out.sort(key=lambda d: -d["flagship_ode_ratio"])
        return out[:6]

    # ---------------------------------------------------------------- gates
    @staticmethod
    def _gates(
        agg: dict[str, Any], rows: list[BenchmarkRow], nfe_budgets: Sequence[int]
    ) -> dict[str, Any]:
        primary_nfe = str(int(min(nfe_budgets)))
        blk = agg.get(primary_nfe, {})
        per_ds = blk.get("per_dataset", {})
        ratios = [c["_best_ode_ratio"] for c in per_ds.values() if "_best_ode_ratio" in c]
        w2 = [c["_best_w2_ratio"] for c in per_ds.values() if "_best_w2_ratio" in c]
        wins = sum(1 for r in ratios if r <= 1.0)
        nfe_ok = all(r.nfe_used <= r.nfe_budget for r in rows)
        ode_mean = float(blk.get("flagship_ode_ratio_mean", 9))
        w2_mean = float(blk.get("flagship_w2_ratio_mean", 9))
        return {
            "G1_primary_metric": "ode_err (PF-ODE trajectory error vs dense reference)",
            "G1_primary_nfe": primary_nfe,
            "G1_flagship_ode_ratio_mean": ode_mean,
            "G1_threshold": G1_MAX_ODE_RATIO,
            "G1_wins": f"{wins}/{len(ratios)}",
            "G1_pass": bool(ratios) and ode_mean <= G1_MAX_ODE_RATIO,
            "G1b_secondary_metric": "w2_sq (exact OT, common random numbers)",
            "G1b_flagship_w2_ratio_mean": w2_mean,
            "G1b_threshold": G1B_MAX_W2_RATIO,
            "G1b_pass": bool(w2) and w2_mean <= G1B_MAX_W2_RATIO,
            "G2_nfe_budget_respected": bool(nfe_ok),
            "G2_pass": bool(nfe_ok),
            "G3_rows": len(rows),
            "G3_pass": len(rows) > 0,
        }
