from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from lstm_bitcoin.data import PriceSeries
from lstm_bitcoin.features import build_supervised
from lstm_bitcoin.splits import Fold, annual_expanding_splits


def test_annual_splits_are_expanding_chronological_and_unique(
    price_series: PriceSeries,
) -> None:
    supervised = build_supervised(price_series, lookback=10)
    folds = annual_expanding_splits(
        supervised.target_dates,
        first_test_year=2018,
        last_test_year=2020,
        validation_years=1,
    )

    assert [fold.number for fold in folds] == [1, 2, 3]
    assert [fold.test_year for fold in folds] == [2018, 2019, 2020]
    all_tests = np.concatenate([fold.test for fold in folds])
    assert len(np.unique(all_tests)) == len(all_tests)
    for previous, fold in pairwise(folds):
        assert len(fold.train) > len(previous.train)
        assert fold.train[-1] < fold.validation[0] < fold.test[0]
        assert supervised.target_dates[fold.test[0]].astype("datetime64[Y]") == np.datetime64(
            f"{fold.test_year}", "Y"
        )


@pytest.mark.parametrize(
    "dates",
    [
        np.arange(9).astype("timedelta64[D]") + np.datetime64("2020-01-01"),
        np.asarray(["2020-01-01"] * 10, dtype="datetime64[D]"),
        np.asarray(
            [
                "2020-01-01",
                "2020-01-03",
                "2020-01-02",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
            ],
            dtype="datetime64[D]",
        ),
    ],
)
def test_annual_splits_validate_dates(dates: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"ten|unique"):
        annual_expanding_splits(dates, first_test_year=2020, last_test_year=2020)


@pytest.mark.parametrize(
    ("first", "last", "validation"),
    [(2021, 2020, 1), (2020, 2020, 0)],
)
def test_annual_splits_validate_year_arguments(first: int, last: int, validation: int) -> None:
    dates = np.arange(365 * 4).astype("timedelta64[D]") + np.datetime64("2018-01-01")
    with pytest.raises(ValueError, match="invalid test-year"):
        annual_expanding_splits(
            dates,
            first_test_year=first,
            last_test_year=last,
            validation_years=validation,
        )


def test_annual_splits_fail_when_requested_partition_is_absent() -> None:
    dates = np.arange(365).astype("timedelta64[D]") + np.datetime64("2020-01-01")
    with pytest.raises(ValueError, match="non-empty"):
        annual_expanding_splits(dates, first_test_year=2020, last_test_year=2021)


def test_fold_rejects_bad_numbers_shapes_order_overlap_and_internal_sort() -> None:
    valid = {
        "number": 1,
        "test_year": 2020,
        "train": np.asarray([0, 1], dtype=np.int64),
        "validation": np.asarray([2], dtype=np.int64),
        "test": np.asarray([3], dtype=np.int64),
    }
    assert Fold(**valid).number == 1  # type: ignore[arg-type]

    cases = [
        ({"number": 0}, "start at one"),
        ({"train": np.asarray([], dtype=np.int64)}, "non-empty"),
        ({"train": np.asarray([[0, 1]], dtype=np.int64)}, "one-dimensional"),
        ({"validation": np.asarray([0], dtype=np.int64)}, "chronological|overlap"),
        ({"test": np.asarray([2], dtype=np.int64)}, "chronological|overlap"),
        ({"train": np.asarray([1, 0], dtype=np.int64)}, "strictly increasing"),
        ({"train": np.asarray([0, 4, 1], dtype=np.int64)}, "chronological"),
    ]
    for changes, message in cases:
        values = dict(valid)
        values.update(changes)
        with pytest.raises(ValueError, match=message):
            Fold(**values)  # type: ignore[arg-type]
