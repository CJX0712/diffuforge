"""Synthetic 2-D target densities with exact samplers (no downloads)."""

from __future__ import annotations

from .densities import (
    DENSITIES,
    analytic_gmm_logpdf,
    analytic_gmm_score,
    get_density,
    list_densities,
)

__all__ = [
    "DENSITIES",
    "analytic_gmm_logpdf",
    "analytic_gmm_score",
    "get_density",
    "list_densities",
]
