"""Forecast accuracy and uncertainty metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ForecastMetrics:
    """Primary out-of-sample return-forecast metrics."""

    observations: int
    mae: float
    rmse: float
    mean_error: float
    directional_accuracy: float | None
    directional_coverage: float
    pearson_correlation: float | None
    mae_ratio_vs_zero: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _aligned(
    actual: npt.NDArray[np.float64], predicted: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("actual and predicted arrays must be aligned and one-dimensional")
    if len(actual) == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metric arrays must be non-empty and finite")
    return actual, predicted


def forecast_metrics(
    actual: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
    *,
    zero_mae: float | None = None,
) -> ForecastMetrics:
    """Compute accuracy metrics in unscaled log-return units."""

    actual, predicted = _aligned(actual, predicted)
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    actual_std = float(np.std(actual, ddof=0))
    predicted_std = float(np.std(predicted, ddof=0))
    correlation: float | None
    if actual_std < 1e-15 or predicted_std < 1e-15:
        correlation = None
    else:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    ratio = None if zero_mae is None or zero_mae <= 0 else mae / zero_mae
    directional = predicted != 0.0
    directional_coverage = float(np.mean(directional))
    directional_accuracy = (
        None
        if not np.any(directional)
        else float(np.mean(np.sign(actual[directional]) == np.sign(predicted[directional])))
    )
    return ForecastMetrics(
        observations=len(actual),
        mae=mae,
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        mean_error=float(np.mean(error)),
        directional_accuracy=directional_accuracy,
        directional_coverage=directional_coverage,
        pearson_correlation=correlation,
        mae_ratio_vs_zero=ratio,
    )


def block_bootstrap_mae_difference(
    actual: npt.NDArray[np.float64],
    candidate: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.float64],
    *,
    block_size: int = 30,
    resamples: int = 2_000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Return point estimate and 95% moving-block interval for candidate minus baseline MAE."""

    _aligned(actual, candidate)
    _aligned(actual, baseline)
    if block_size < 1 or block_size > len(actual) or resamples < 100:
        raise ValueError("invalid bootstrap block size or resample count")
    differences = np.abs(candidate - actual) - np.abs(baseline - actual)
    point = float(np.mean(differences))
    generator = np.random.default_rng(seed)
    starts = np.arange(0, len(actual) - block_size + 1)
    blocks_per_sample = int(np.ceil(len(actual) / block_size))
    estimates = np.empty(resamples, dtype=np.float64)
    offsets = np.arange(block_size)
    for index in range(resamples):
        selected = generator.choice(starts, size=blocks_per_sample, replace=True)
        sample_indices = (selected[:, None] + offsets).reshape(-1)[: len(actual)]
        estimates[index] = float(np.mean(differences[sample_indices]))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return point, float(lower), float(upper)
