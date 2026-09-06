"""Past-only feature engineering and supervised window construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .data import PriceSeries

FEATURE_NAMES = (
    "log_return_1d",
    "log_return_7d",
    "log_return_30d",
    "realized_volatility_7d",
    "realized_volatility_30d",
)


@dataclass(frozen=True)
class SupervisedData:
    """Sequences ending at an origin date and their later return targets."""

    features: npt.NDArray[np.float64]
    targets: npt.NDArray[np.float64]
    origin_dates: npt.NDArray[np.datetime64]
    target_dates: npt.NDArray[np.datetime64]
    origin_prices: npt.NDArray[np.float64]
    target_prices: npt.NDArray[np.float64]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [samples, lookback, feature]")
        samples = self.features.shape[0]
        arrays = (
            self.targets,
            self.origin_dates,
            self.target_dates,
            self.origin_prices,
            self.target_prices,
        )
        if any(array.ndim != 1 or len(array) != samples for array in arrays):
            raise ValueError("all supervised arrays must share the sample dimension")
        if self.features.shape[2] != len(self.feature_names):
            raise ValueError("feature names do not match the feature dimension")
        if np.any(self.target_dates <= self.origin_dates):
            raise ValueError("every target must occur strictly after its forecast origin")
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise ValueError("supervised arrays must contain only finite values")


def _rolling_std(values: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    output = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(window, len(values)):
        sample = values[index - window + 1 : index + 1]
        if np.isfinite(sample).all():
            output[index] = float(np.std(sample, ddof=0))
    return output


def daily_feature_matrix(close_usd: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Create features that use prices available no later than each row's date."""

    if close_usd.ndim != 1 or len(close_usd) < 32:
        raise ValueError("at least 32 one-dimensional prices are required")
    if not np.isfinite(close_usd).all() or np.any(close_usd <= 0):
        raise ValueError("prices must be finite and positive")

    log_price = np.log(close_usd)
    return_1d = np.full(len(close_usd), np.nan, dtype=np.float64)
    return_1d[1:] = np.diff(log_price)
    return_7d = np.full(len(close_usd), np.nan, dtype=np.float64)
    return_7d[7:] = log_price[7:] - log_price[:-7]
    return_30d = np.full(len(close_usd), np.nan, dtype=np.float64)
    return_30d[30:] = log_price[30:] - log_price[:-30]
    volatility_7d = _rolling_std(return_1d, 7)
    volatility_30d = _rolling_std(return_1d, 30)
    return np.column_stack((return_1d, return_7d, return_30d, volatility_7d, volatility_30d))


def build_supervised(
    series: PriceSeries,
    *,
    lookback: int,
    horizon: int = 1,
) -> SupervisedData:
    """Build past-only sequences for a future cumulative log-return target."""

    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    daily_features = daily_feature_matrix(series.close_usd)
    first_finite = int(np.flatnonzero(np.isfinite(daily_features).all(axis=1))[0])
    first_origin = first_finite + lookback - 1
    last_origin = len(series.close_usd) - horizon - 1
    if first_origin > last_origin:
        raise ValueError("price series is too short for the requested lookback and horizon")

    feature_windows: list[npt.NDArray[np.float64]] = []
    targets: list[float] = []
    origins: list[int] = []
    target_indices: list[int] = []
    log_price = np.log(series.close_usd)
    for origin in range(first_origin, last_origin + 1):
        start = origin - lookback + 1
        window = daily_features[start : origin + 1]
        if not np.isfinite(window).all():
            continue
        target = origin + horizon
        feature_windows.append(window)
        targets.append(float(log_price[target] - log_price[origin]))
        origins.append(origin)
        target_indices.append(target)

    return SupervisedData(
        features=np.stack(feature_windows).astype(np.float64, copy=False),
        targets=np.asarray(targets, dtype=np.float64),
        origin_dates=series.dates[origins],
        target_dates=series.dates[target_indices],
        origin_prices=series.close_usd[origins],
        target_prices=series.close_usd[target_indices],
        feature_names=FEATURE_NAMES,
    )
