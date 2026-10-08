"""Pipeline / determinism / CLI end-to-end contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from diffuforge.core.config import DiffusionConfig
from diffuforge.pipeline.pipeline import DiffuPipeline, _ref_seed, _sample_seed
from diffuforge.samplers.registry import FLAGSHIP, all_sampler_names, baseline_names


def test_report_shape(fast_cfg) -> None:
    rep = DiffuPipeline(fast_cfg).run(ablation=True)
    n_cells = len(fast_cfg.datasets) * len(fast_cfg.seeds) * len(all_sampler_names())
    assert len(rep.rows) == n_cells * len(fast_cfg.nfe_budgets)
    for row in rep.rows:
        assert row["nfe_used"] <= row["nfe_budget"]
        assert np.isfinite(row["ode_err"]) and row["ode_err"] >= 0.0
        assert np.isfinite(row["w2_sq"])
    assert set(rep.gates) >= {"G1_pass", "G1b_pass", "G2_pass", "G3_pass"}
    assert rep.gates["G2_pass"] is True


def test_every_row_carries_the_primary_metric(fast_cfg) -> None:
    rep = DiffuPipeline(fast_cfg).run(ablation=False)
    names = {r["sampler"] for r in rep.rows}
    assert names == set(all_sampler_names())
    assert all("ode_err" in r for r in rep.rows)


def test_common_random_numbers_are_sampler_independent() -> None:
    """CRN: the initial-noise seed must not depend on the sampler or the NFE."""
    assert _sample_seed("gmm8", 7) == _sample_seed("gmm8", 7)
    for _ in all_sampler_names():
        assert _sample_seed("gmm8", 7) == _sample_seed("gmm8", 7)
    assert _sample_seed("gmm8", 7) != _sample_seed("rings", 7)
    assert _sample_seed("gmm8", 7) != _sample_seed("gmm8", 17)
    assert _sample_seed("gmm8", 7) != _ref_seed("gmm8", 7)


def test_crn_makes_the_comparison_paired(fast_cfg, trained) -> None:
    """Two samplers started from the same seed must track each other closely."""
    from diffuforge.samplers.registry import get_sampler

    model, _ = trained[("gmm8", 7)]
    rng_seed = _sample_seed("gmm8", 7)
    from diffuforge.core.seed import get_rng

    a, _ = get_sampler(FLAGSHIP, fast_cfg).sample(model, 32, 20, get_rng(rng_seed))
    b, _ = get_sampler("dpm3m", fast_cfg).sample(model, 32, 20, get_rng(rng_seed))
    # Paired clouds: the displacement between them is small compared with the
    # scale of the cloud itself, whereas two independent draws are not.
    scale = float(np.sqrt(np.mean(a * a)))
    assert float(np.sqrt(np.mean((a - b) ** 2))) / scale < 0.25


def test_two_identical_runs_are_bit_identical(fast_cfg) -> None:
    cfg = fast_cfg.with_overrides(n_iters=40)
    a = DiffuPipeline(cfg).run(ablation=False)
    b = DiffuPipeline(cfg).run(ablation=False)
    for ra, rb in zip(a.rows, b.rows, strict=True):
        assert ra["ode_err"] == rb["ode_err"]
        assert ra["w2_sq"] == rb["w2_sq"]
        assert ra["mmd2"] == rb["mmd2"]
        assert ra["coverage"] == rb["coverage"]
        assert ra["nn_tst"] == rb["nn_tst"]


def test_aggregate_exposes_both_ratios(fast_cfg) -> None:
    rep = DiffuPipeline(fast_cfg).run(ablation=False)
    blk = rep.aggregate[str(int(min(fast_cfg.nfe_budgets)))]
    assert "flagship_ode_ratio_mean" in blk
    assert "flagship_w2_ratio_mean" in blk
    for cell in blk["per_dataset"].values():
        assert cell["_best_ode_baseline"] in baseline_names()
        assert "_best_ode_ratio" in cell


def test_ablation_covers_every_density_and_keeps_nfe(fast_cfg) -> None:
    rep = DiffuPipeline(fast_cfg).run(ablation=True)
    labels = [r["variant"] for r in rep.ablation]
    assert len(labels) >= 5
    for row in rep.ablation:
        assert row["nfe_used"] <= int(max(fast_cfg.nfe_budgets))
        assert set(row["per_dataset"]) == set(fast_cfg.datasets)
        assert row["ratio_vs_best"] >= 1.0 - 1e-9


def test_cli_list_and_smoke(tmp_path, fast_cfg) -> None:
    from diffuforge.cli import main

    assert main(["--list"]) == 0
    out = tmp_path / "b.json"
    rc = main(
        [
            "--datasets",
            "gmm8,rings",
            "--seeds",
            "7",
            "--nfe",
            "10",
            "--iters",
            "40",
            "--n-gen",
            "32",
            "--no-ablation",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rows"]
    assert payload["gates"]["G2_pass"] is True


def test_config_env_override() -> None:
    from diffuforge.core.config import config_from_env

    cfg = config_from_env(base=DiffusionConfig(), env={"DIFFUFORGE_RHO_SCHEDULE": "5.5"})
    assert cfg.rho_schedule == pytest.approx(5.5)
    cfg = config_from_env(base=DiffusionConfig(), env={"DIFFUFORGE_SEEDS": "1,2,3"})
    assert cfg.seeds == (1.0, 2.0, 3.0) or cfg.seeds == (1, 2, 3)
    cfg = config_from_env(base=DiffusionConfig(), env={"DIFFUFORGE_NFE_BUDGETS": "8,16"})
    assert tuple(int(x) for x in cfg.nfe_budgets) == (8, 16)


def test_config_validation_rejects_bad_values() -> None:
    from diffuforge.core.errors import ConfigError

    with pytest.raises(ConfigError):
        DiffusionConfig(rho_schedule=0.0).validate()
    with pytest.raises(ConfigError):
        DiffusionConfig(sigma_min_rel=0.0).validate()
    with pytest.raises(ConfigError):
        DiffusionConfig(n_layers=1).validate()
    with pytest.raises(ConfigError):
        DiffusionConfig().with_overrides(no_such_field=1)
