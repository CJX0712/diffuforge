"""Hyper-parameter search for the flagship (Optuna TPE, offline & deterministic)."""

from __future__ import annotations

from .tune import tune_flagship

__all__ = ["tune_flagship"]
