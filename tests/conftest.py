from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from lstm_bitcoin.data import PriceSeries


def deterministic_price_series(
    *,
    start: date = date(2012, 1, 1),
    days: int = 365 * 10 + 3,
) -> PriceSeries:
    """Return a positive, non-trivial daily series without random state."""

    offsets = np.arange(days, dtype=np.float64)
    log_prices = (
        np.log(100.0)
        + 0.0007 * offsets
        + 0.025 * np.sin(offsets / 11.0)
        + 0.009 * np.cos(offsets / 37.0)
    )
    dates = np.asarray(
        [(start + timedelta(days=index)).isoformat() for index in range(days)],
        dtype="datetime64[D]",
    )
    return PriceSeries(dates=dates, close_usd=np.exp(log_prices).astype(np.float64))


@pytest.fixture
def price_series() -> PriceSeries:
    return deterministic_price_series()


@pytest.fixture
def write_price_csv(tmp_path: Path):
    def _write(
        rows: list[tuple[str, str]],
        *,
        header: tuple[str, str] = ("date", "close_usd"),
        name: str = "prices.csv",
    ) -> Path:
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        return path

    return _write
