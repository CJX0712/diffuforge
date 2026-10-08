"""Command line interface for DiffuForge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .core.config import config_from_env
from .core.errors import DiffuError
from .core.seed import set_all
from .data.densities import list_densities
from .pipeline.pipeline import DiffuPipeline
from .samplers.registry import FLAGSHIP, all_sampler_names


def _banner(text: str) -> None:
    print(text, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diffuforge",
        description="DiffuForge: diffusion / score-based modeling with NFE-budgeted samplers",
    )
    p.add_argument("--datasets", default="", help="comma separated density names")
    p.add_argument("--seeds", default="", help="comma separated seeds")
    p.add_argument("--nfe", default="", help="comma separated NFE budgets")
    p.add_argument("--samplers", default="", help="comma separated sampler names")
    p.add_argument("--iters", type=int, default=0, help="training iterations (0 = config default)")
    p.add_argument("--n-gen", type=int, default=0, help="samples drawn per cell")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--tune", type=int, default=0, help="run Optuna tuning with N trials")
    p.add_argument("--out", default="benchmark.json", help="output JSON path")
    p.add_argument("--list", action="store_true", help="list densities and samplers, then exit")
    return p


def _split(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list:
            _banner("densities : " + ", ".join(list_densities()))
            _banner("samplers  : " + ", ".join(all_sampler_names()))
            _banner(f"flagship  : {FLAGSHIP}")
            return 0

        cfg = config_from_env()
        if args.iters:
            cfg = cfg.with_overrides(n_iters=int(args.iters))
        if args.n_gen:
            cfg = cfg.with_overrides(n_gen=int(args.n_gen))
        set_all(int(args.seed))

        report = DiffuPipeline(cfg).run(
            datasets=_split(args.datasets) or None,
            seeds=[int(s) for s in _split(args.seeds)] or None,
            nfe_budgets=[int(b) for b in _split(args.nfe)] or None,
            samplers=_split(args.samplers) or None,
            ablation=not args.no_ablation,
        )

        payload: dict[str, Any] = report.to_dict()
        if args.tune:
            from .hpo.tune import tune_flagship

            payload["tuning"] = tune_flagship(cfg, n_trials=int(args.tune))

        _render(report)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        _banner(f"\nwrote {args.out} ({report.elapsed_sec:.1f}s)")
        return 0
    except DiffuError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _render(report: Any) -> None:
    agg = report.aggregate
    for nfe in sorted(agg, key=lambda k: int(k)):
        blk = agg[nfe]
        per = blk["per_dataset"]
        names = sorted({n for c in per.values() for n in c if not n.startswith("_")})
        for metric, width, fmt in (("mean", 10, ".5f"), ("w2_sq", 9, ".4f")):
            label = "ode_err (PRIMARY, lower is better)" if metric == "mean" else "W2^2 (secondary)"
            _banner(f"\n=== NFE budget {nfe} — {label} ===")
            _banner(
                f"{'dataset':<10}"
                + "".join(f"{n[: width - 1]:>{width}}" for n in names)
                + f"{'best':>10}"
            )
            for ds in sorted(per):
                cell = per[ds]
                line = f"{ds:<10}"
                for n in names:
                    v = cell.get(n, {}).get(metric)
                    line += format(v, f"{width}{fmt}") if v is not None else f"{'-':>{width}}"
                tag = "_best_ode_baseline" if metric == "mean" else "_best_w2_baseline"
                line += f"{cell.get(tag, '-'):>10}"
                _banner(line)
        _banner(
            f"flagship vs best baseline: ode_err ratio mean "
            f"{blk['flagship_ode_ratio_mean']:.4f} (geomean "
            f"{blk['flagship_ode_ratio_geomean']:.4f}), W2^2 ratio mean "
            f"{blk['flagship_w2_ratio_mean']:.4f}, over {blk['n_datasets']} densities"
        )
    _banner("\n=== ablation (flagship components) ===")
    for row in report.ablation:
        _banner(
            f"  {row['variant']:<22} ode_err={row['ode_err']:.6f}  "
            f"W2^2={row['w2_sq']:.5f}  nfe={row['nfe_used']:<3} "
            f"ratio={row['ratio_vs_best']:.4f}"
        )
    _banner("\n=== gates ===")
    for k, v in report.gates.items():
        _banner(f"  {k}: {v}")
    if report.failures:
        _banner("\n=== failure cases ===")
        for f in report.failures:
            _banner(
                f"  [{f['dataset']} @NFE{f['nfe']}] ratio={f['flagship_ode_ratio']} :: {f['cause']}"
            )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
