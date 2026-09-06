"""Calendar-defined expanding-window splits for crypto time series."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Fold:
    """Earlier fit data, a later validation block, and one locked test year."""

    number: int
    test_year: int
    train: npt.NDArray[np.int64]
    validation: npt.NDArray[np.int64]
    test: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("fold numbers start at one")
        arrays = (self.train, self.validation, self.test)
        if any(index.ndim != 1 or len(index) == 0 for index in arrays):
            raise ValueError("fold partitions must be non-empty one-dimensional arrays")
        if any(np.any(index[1:] <= index[:-1]) for index in arrays):
            raise ValueError(
                "fold partitions must be strictly increasing for chronological evaluation"
            )
        if not int(self.train[-1]) < int(self.validation[0]) < int(self.test[0]):
            raise ValueError("fold partitions must be strictly chronological")
        combined = np.concatenate(arrays)
        if len(np.unique(combined)) != len(combined):
            raise ValueError("fold partitions must not overlap")


def _calendar_years(dates: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int64]:
    return dates.astype("datetime64[Y]").astype(np.int64) + 1970


def annual_expanding_splits(
    target_dates: npt.NDArray[np.datetime64],
    *,
    first_test_year: int,
    last_test_year: int,
    validation_years: int = 1,
) -> tuple[Fold, ...]:
    """Create expanding folds with explicit calendar validation and test periods."""

    if target_dates.ndim != 1 or len(target_dates) < 10:
        raise ValueError("target_dates must contain at least ten observations")
    if np.any(target_dates[1:] <= target_dates[:-1]):
        raise ValueError("target_dates must be unique and strictly increasing")
    if first_test_year > last_test_year or validation_years < 1:
        raise ValueError("invalid test-year range or validation_years")

    years = _calendar_years(target_dates)
    result: list[Fold] = []
    for number, test_year in enumerate(range(first_test_year, last_test_year + 1), start=1):
        validation_start = test_year - validation_years
        train = np.flatnonzero(years < validation_start).astype(np.int64)
        validation = np.flatnonzero((years >= validation_start) & (years < test_year)).astype(
            np.int64
        )
        test = np.flatnonzero(years == test_year).astype(np.int64)
        result.append(
            Fold(
                number=number,
                test_year=test_year,
                train=train,
                validation=validation,
                test=test,
            )
        )
    for previous, current in pairwise(result):
        if int(previous.test[-1]) >= int(current.test[0]):
            raise ValueError("outer test folds must be unique and chronological")
    return tuple(result)
