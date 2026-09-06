from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from conftest import deterministic_price_series

from lstm_bitcoin.data import PriceSeries
from lstm_bitcoin.features import (
    FEATURE_NAMES,
    SupervisedData,
    build_supervised,
    daily_feature_matrix,
)


def test_daily_feature_matrix_matches_hand_calculated_returns() -> None:
    prices = np.exp(np.arange(80, dtype=np.float64) * 0.01)
    features = daily_feature_matrix(prices)

    assert features.shape == (80, len(FEATURE_NAMES))
    np.testing.assert_allclose(features[30:, 0], 0.01, atol=1e-14)
    np.testing.assert_allclose(features[30:, 1], 0.07, atol=1e-14)
    np.testing.assert_allclose(features[30:, 2], 0.30, atol=1e-14)
    np.testing.assert_allclose(features[30:, 3:], 0.0, atol=1e-14)
    assert np.isnan(features[:30]).any()


@pytest.mark.parametrize(
    "prices",
    [
        np.ones((40, 1)),
        np.ones(31),
        np.r_[np.ones(39), np.nan],
        np.r_[np.ones(39), 0.0],
    ],
)
def test_daily_feature_matrix_rejects_invalid_prices(prices: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"prices|required"):
        daily_feature_matrix(prices)


def test_build_supervised_alignment_and_exact_target(price_series: PriceSeries) -> None:
    supervised = build_supervised(price_series, lookback=12, horizon=3)

    assert supervised.features.shape[1:] == (12, len(FEATURE_NAMES))
    assert supervised.feature_names == FEATURE_NAMES
    assert np.all(supervised.target_dates > supervised.origin_dates)
    assert np.all(supervised.target_dates - supervised.origin_dates == np.timedelta64(3, "D"))
    np.testing.assert_allclose(
        supervised.targets,
        np.log(supervised.target_prices / supervised.origin_prices),
        rtol=0,
        atol=1e-15,
    )


def test_future_mutation_cannot_change_earlier_features_or_targets() -> None:
    original = deterministic_price_series(start=date(2018, 1, 1), days=500)
    changed_prices = original.close_usd.copy()
    mutation_date = np.datetime64("2019-01-01")
    changed_prices[original.dates >= mutation_date] *= np.linspace(
        2.0, 5.0, np.count_nonzero(original.dates >= mutation_date)
    )
    changed = PriceSeries(dates=original.dates.copy(), close_usd=changed_prices)

    before = build_supervised(original, lookback=20, horizon=1)
    after = build_supervised(changed, lookback=20, horizon=1)
    safe = before.target_dates < mutation_date
    np.testing.assert_array_equal(before.origin_dates[safe], after.origin_dates[safe])
    np.testing.assert_array_equal(before.target_dates[safe], after.target_dates[safe])
    np.testing.assert_allclose(before.features[safe], after.features[safe], rtol=0, atol=0)
    np.testing.assert_allclose(before.targets[safe], after.targets[safe], rtol=0, atol=0)


@pytest.mark.parametrize(("lookback", "horizon"), [(1, 1), (2, 0)])
def test_build_supervised_validates_window_arguments(
    price_series: PriceSeries, lookback: int, horizon: int
) -> None:
    with pytest.raises(ValueError):
        build_supervised(price_series, lookback=lookback, horizon=horizon)


def test_build_supervised_rejects_insufficient_history() -> None:
    short = deterministic_price_series(days=40)
    with pytest.raises(ValueError, match="too short"):
        build_supervised(short, lookback=20, horizon=5)


def test_supervised_data_checks_shapes_dates_names_and_finiteness() -> None:
    valid = {
        "features": np.ones((2, 3, 1)),
        "targets": np.ones(2),
        "origin_dates": np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        "target_dates": np.asarray(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        "origin_prices": np.asarray([100.0, 101.0]),
        "target_prices": np.asarray([101.0, 102.0]),
        "feature_names": ("feature",),
    }
    assert SupervisedData(**valid).features.shape == (2, 3, 1)  # type: ignore[arg-type]

    for key, replacement, message in [
        ("features", np.ones((2, 3)), "shape"),
        ("targets", np.ones((2, 1)), "sample dimension"),
        ("feature_names", ("one", "two"), "feature names"),
        (
            "target_dates",
            np.asarray(["2020-01-01", "2020-01-03"], dtype="datetime64[D]"),
            "strictly after",
        ),
        ("features", np.full((2, 3, 1), np.nan), "finite"),
        ("targets", np.asarray([1.0, np.inf]), "finite"),
    ]:
        arguments = dict(valid)
        arguments[key] = replacement
        with pytest.raises(ValueError, match=message):
            SupervisedData(**arguments)  # type: ignore[arg-type]

    arguments = dict(valid)
    arguments["features"] = np.asarray(1.0)
    with pytest.raises(ValueError, match="shape"):
        SupervisedData(**arguments)  # type: ignore[arg-type]
