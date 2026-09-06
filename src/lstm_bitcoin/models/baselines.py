"""Transparent zero, historical-mean, and ridge baselines."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .base import SequenceStandardizer


def _validate_fit_arrays(
    features: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
) -> None:
    if features.ndim != 3 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("features and targets have incompatible shapes")
    if len(features) == 0 or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("training arrays must be non-empty and finite")


class ZeroReturn:
    """Random-walk baseline: tomorrow's log return is zero."""

    def fit(
        self, features: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
    ) -> ZeroReturn:
        _validate_fit_arrays(features, targets)
        return self

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if features.ndim != 3:
            raise ValueError("features must be three-dimensional")
        return np.zeros(len(features), dtype=np.float64)


class HistoricalMean:
    """Constant drift baseline fitted only on earlier training returns."""

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(
        self, features: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
    ) -> HistoricalMean:
        _validate_fit_arrays(features, targets)
        self._mean = float(np.mean(targets))
        return self

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if features.ndim != 3:
            raise ValueError("features must be three-dimensional")
        if self._mean is None:
            raise RuntimeError("HistoricalMean must be fitted before prediction")
        return np.full(len(features), self._mean, dtype=np.float64)


class RidgeReturn:
    """Closed-form ridge regression over the same lagged feature window."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        if alpha < 0:
            raise ValueError("alpha must not be negative")
        self.alpha = alpha
        self._scaler: SequenceStandardizer | None = None
        self._coefficient: npt.NDArray[np.float64] | None = None
        self._intercept: float | None = None

    def fit(
        self, features: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
    ) -> RidgeReturn:
        _validate_fit_arrays(features, targets)
        self._scaler = SequenceStandardizer.fit(features)
        design = self._scaler.transform(features).reshape(len(features), -1)
        feature_mean = np.mean(design, axis=0)
        target_mean = float(np.mean(targets))
        centered_design = design - feature_mean
        centered_target = targets - target_mean
        if self.alpha == 0:
            self._coefficient = np.linalg.lstsq(centered_design, centered_target, rcond=None)[0]
        else:
            gram = centered_design.T @ centered_design
            penalty = np.eye(gram.shape[0], dtype=np.float64) * self.alpha
            self._coefficient = np.linalg.solve(gram + penalty, centered_design.T @ centered_target)
        self._intercept = target_mean - float(feature_mean @ self._coefficient)
        return self

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._scaler is None or self._coefficient is None or self._intercept is None:
            raise RuntimeError("RidgeReturn must be fitted before prediction")
        design = self._scaler.transform(features).reshape(len(features), -1)
        return (design @ self._coefficient + self._intercept).astype(np.float64, copy=False)
