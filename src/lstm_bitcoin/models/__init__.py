"""Forecasting models used by the reference experiment."""

from .baselines import HistoricalMean, RidgeReturn, ZeroReturn

__all__ = ["HistoricalMean", "RidgeReturn", "ZeroReturn"]
