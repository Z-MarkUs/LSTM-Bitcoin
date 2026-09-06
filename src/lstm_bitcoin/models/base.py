"""Shared model protocols and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


class ForecastModel(Protocol):
    """Minimal fit/predict contract for return forecasters."""

    def fit(
        self, features: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
    ) -> ForecastModel: ...

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True)
class SequenceStandardizer:
    """Per-feature moments learned exclusively from a training sequence tensor."""

    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]

    @classmethod
    def fit(cls, features: npt.NDArray[np.float64]) -> SequenceStandardizer:
        if features.ndim != 3 or len(features) == 0:
            raise ValueError("features must be a non-empty three-dimensional array")
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        flattened = features.reshape(-1, features.shape[-1])
        mean = np.mean(flattened, axis=0)
        scale = np.std(flattened, axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean.astype(np.float64), scale=scale.astype(np.float64))

    def transform(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if features.ndim != 3 or features.shape[-1] != len(self.mean):
            raise ValueError("features do not match the fitted scaler")
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        return ((features - self.mean) / self.scale).astype(np.float64, copy=False)


@dataclass(frozen=True)
class TargetStandardizer:
    """Scalar target moments learned exclusively from training targets."""

    mean: float
    scale: float

    @classmethod
    def fit(cls, targets: npt.NDArray[np.float64]) -> TargetStandardizer:
        if targets.ndim != 1 or len(targets) == 0:
            raise ValueError("targets must be a non-empty one-dimensional array")
        if not np.isfinite(targets).all():
            raise ValueError("targets must contain only finite values")
        scale = float(np.std(targets, ddof=0))
        return cls(mean=float(np.mean(targets)), scale=scale if scale >= 1e-12 else 1.0)

    def transform(self, targets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if targets.ndim != 1 or not np.isfinite(targets).all():
            raise ValueError("targets must be a one-dimensional finite array")
        return ((targets - self.mean) / self.scale).astype(np.float64, copy=False)

    def inverse(self, targets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if targets.ndim != 1 or not np.isfinite(targets).all():
            raise ValueError("targets must be a one-dimensional finite array")
        return (targets * self.scale + self.mean).astype(np.float64, copy=False)
