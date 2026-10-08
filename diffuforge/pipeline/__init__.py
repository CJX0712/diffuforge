"""End-to-end pipeline: train -> sample -> evaluate -> aggregate."""

from __future__ import annotations

from .pipeline import BenchmarkReport, DiffuPipeline

__all__ = ["BenchmarkReport", "DiffuPipeline"]
