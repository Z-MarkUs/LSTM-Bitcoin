from __future__ import annotations

import numpy as np
import pytest

from lstm_bitcoin.metrics import block_bootstrap_mae_difference, forecast_metrics


def test_forecast_metrics_exact_values_and_serialization() -> None:
    actual = np.asarray([-1.0, 1.0, 2.0, -2.0])
    predicted = np.asarray([-0.5, 0.5, 1.0, -1.0])
    metrics = forecast_metrics(actual, predicted, zero_mae=1.5)

    assert metrics.observations == 4
    assert metrics.mae == pytest.approx(0.75)
    assert metrics.rmse == pytest.approx(np.sqrt(0.625))
    assert metrics.mean_error == pytest.approx(0.0)
    assert metrics.directional_accuracy == 1.0
    assert metrics.directional_coverage == 1.0
    assert metrics.pearson_correlation == pytest.approx(1.0)
    assert metrics.mae_ratio_vs_zero == pytest.approx(0.5)
    assert metrics.as_dict()["observations"] == 4


def test_forecast_metrics_null_correlation_and_ratio() -> None:
    actual = np.ones(3)
    predicted = np.zeros(3)
    metrics = forecast_metrics(actual, predicted, zero_mae=0)
    assert metrics.pearson_correlation is None
    assert metrics.mae_ratio_vs_zero is None
    assert metrics.directional_accuracy is None
    assert metrics.directional_coverage == 0.0


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        (np.ones((2, 1)), np.ones(2)),
        (np.ones(2), np.ones((2, 1))),
        (np.ones(2), np.ones(3)),
        (np.asarray([]), np.asarray([])),
        (np.asarray([1.0, np.nan]), np.ones(2)),
        (np.ones(2), np.asarray([1.0, np.inf])),
    ],
)
def test_forecast_metrics_rejects_invalid_arrays(actual: np.ndarray, predicted: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"aligned|non-empty and finite"):
        forecast_metrics(actual, predicted)


def test_block_bootstrap_is_deterministic_and_sign_is_candidate_minus_baseline() -> None:
    actual = np.linspace(-0.04, 0.04, 120)
    candidate = actual + 0.001
    baseline = actual + 0.01

    first = block_bootstrap_mae_difference(
        actual, candidate, baseline, block_size=10, resamples=200, seed=11
    )
    second = block_bootstrap_mae_difference(
        actual, candidate, baseline, block_size=10, resamples=200, seed=11
    )
    assert first == second
    assert first[0] == pytest.approx(-0.009)
    assert first[1] <= first[0] <= first[2]
    assert first[2] < 0


def test_block_bootstrap_whole_series_block_is_degenerate() -> None:
    actual = np.asarray([1.0, 2.0, 3.0])
    candidate = actual.copy()
    baseline = np.zeros(3)
    point, lower, upper = block_bootstrap_mae_difference(
        actual, candidate, baseline, block_size=3, resamples=100
    )
    assert lower == pytest.approx(point)
    assert upper == pytest.approx(point)


@pytest.mark.parametrize(
    ("block_size", "resamples"),
    [(0, 100), (4, 100), (1, 99)],
)
def test_block_bootstrap_validates_parameters(block_size: int, resamples: int) -> None:
    actual = np.ones(3)
    with pytest.raises(ValueError, match="invalid bootstrap"):
        block_bootstrap_mae_difference(
            actual,
            actual,
            actual,
            block_size=block_size,
            resamples=resamples,
        )


def test_block_bootstrap_validates_baseline_alignment() -> None:
    with pytest.raises(ValueError, match="aligned"):
        block_bootstrap_mae_difference(
            np.ones(5), np.ones(5), np.ones(4), block_size=1, resamples=100
        )
