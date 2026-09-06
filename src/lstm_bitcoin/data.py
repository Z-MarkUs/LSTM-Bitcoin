"""Frozen reference-data download, validation, and loading."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import numpy.typing as npt

SOURCE_REPOSITORY = "https://github.com/coinmetrics/data"
SOURCE_COMMIT = "f1a36afb962731c387bb03982758ab0103063da5"
SOURCE_PATH = "csv/btc.csv"
SOURCE_URL = f"https://raw.githubusercontent.com/coinmetrics/data/{SOURCE_COMMIT}/{SOURCE_PATH}"
SOURCE_SHA256 = "06495ff8e643432e6948b7b4686ce44fc106217287dabdc1b38351d9ddec46c3"
SOURCE_LICENSE = "CC BY-NC 4.0"
SOURCE_ATTRIBUTION = "Coin Metrics Community Data, BTC daily PriceUSD"
SOURCE_TIMEZONE = "UTC"
SOURCE_TIMESTAMP_CONVENTION = (
    "PriceUSD daily close at the 00:00 UTC cutoff, labeled by Coin Metrics using "
    "the beginning-of-interval date convention"
)
ALLOWED_SOURCE_HOST = "raw.githubusercontent.com"
MAX_SOURCE_BYTES = 5_000_000
REFERENCE_START_DATE = date(2013, 1, 1)
REFERENCE_END_DATE = date(2025, 12, 31)


@dataclass(frozen=True)
class PriceSeries:
    """Validated daily closing-price series."""

    dates: npt.NDArray[np.datetime64]
    close_usd: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.dates.ndim != 1 or self.close_usd.ndim != 1:
            raise ValueError("price arrays must be one-dimensional")
        if len(self.dates) != len(self.close_usd):
            raise ValueError("dates and prices must have equal length")
        if len(self.dates) < 2:
            raise ValueError("at least two daily observations are required")
        if not np.isfinite(self.close_usd).all() or np.any(self.close_usd <= 0):
            raise ValueError("prices must be finite and positive")
        if np.any(np.diff(self.dates.astype("datetime64[D]")) != np.timedelta64(1, "D")):
            raise ValueError("dates must be unique, strictly increasing, and exactly daily")


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _download_source() -> bytes:
    parsed = urlparse(SOURCE_URL)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_SOURCE_HOST:
        raise RuntimeError("reference source URL is outside the allowlisted HTTPS host")

    request = Request(SOURCE_URL, headers={"User-Agent": "LSTM-Bitcoin/1.0 data-fetch"})
    # The request URL is a module constant pinned to HTTPS; redirects are rejected
    # unless they remain on the allowlisted raw-content host.
    with urlopen(request, timeout=60) as response:  # nosec B310
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != ALLOWED_SOURCE_HOST:
            raise RuntimeError("reference source redirected outside the allowlisted HTTPS host")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_SOURCE_BYTES:
            raise RuntimeError("reference source exceeds the maximum permitted download size")
        stream = cast(BinaryIO, response)
        payload = stream.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise RuntimeError("reference source exceeds the maximum permitted download size")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"reference source SHA-256 mismatch: expected {SOURCE_SHA256}, got {digest}"
        )
    return payload


def _extract_price_rows(payload: bytes) -> tuple[list[tuple[date, float]], int]:
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    required = {"time", "PriceUSD"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("Coin Metrics source is missing required time and PriceUSD columns")

    rows: list[tuple[date, float]] = []
    skipped = 0
    for row in reader:
        raw_price = row.get("PriceUSD", "").strip()
        if not raw_price:
            skipped += 1
            continue
        observation_date = date.fromisoformat(row["time"])
        price = float(raw_price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"invalid PriceUSD value on {observation_date.isoformat()}")
        if REFERENCE_START_DATE <= observation_date <= REFERENCE_END_DATE:
            rows.append((observation_date, price))
    if not rows:
        raise ValueError("Coin Metrics source contains no usable PriceUSD observations")
    _validate_rows(rows)
    return rows, skipped


def _validate_rows(rows: list[tuple[date, float]]) -> None:
    for previous, current in pairwise(rows):
        if current[0] != previous[0] + timedelta(days=1):
            raise ValueError(
                "reference prices must be unique, strictly ordered, and complete at daily grain"
            )


def _csv_payload(rows: list[tuple[date, float]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["date", "close_usd"])
    for observation_date, price in rows:
        writer.writerow([observation_date.isoformat(), format(price, ".12g")])
    return output.getvalue().encode("utf-8")


def fetch_reference_data(output: Path, manifest: Path, *, force: bool = False) -> dict[str, Any]:
    """Download the pinned source, verify it, and write a minimal frozen snapshot."""

    if output.resolve() == manifest.resolve():
        raise ValueError("reference output and manifest paths must be different")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not force:
        raise FileExistsError("reference output already exists; pass --force to replace it")
    for path in existing:
        if not path.is_file():
            raise FileExistsError(f"refusing to replace non-file reference path: {path}")
    source_payload = _download_source()
    rows, skipped = _extract_price_rows(source_payload)
    processed_payload = _csv_payload(rows)
    processed_sha256 = hashlib.sha256(processed_payload).hexdigest()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "path": SOURCE_PATH,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "license": SOURCE_LICENSE,
            "attribution": SOURCE_ATTRIBUTION,
            "timezone": SOURCE_TIMEZONE,
            "timestamp_convention": SOURCE_TIMESTAMP_CONVENTION,
        },
        "processed": {
            "path": output.as_posix(),
            "sha256": processed_sha256,
            "selected_columns": ["time", "PriceUSD"],
            "output_columns": ["date", "close_usd"],
            "rows": len(rows),
            "start_date": rows[0][0].isoformat(),
            "end_date": rows[-1][0].isoformat(),
            "source_rows_skipped_for_missing_price": skipped,
            "missing_value_policy": (
                "skip source rows with missing PriceUSD before filtering; "
                "require the retained reference period to be gap-free"
            ),
            "date_filter": {
                "start_inclusive": REFERENCE_START_DATE.isoformat(),
                "end_inclusive": REFERENCE_END_DATE.isoformat(),
                "reason": "freeze complete calendar years through 2025 for time-ordered evaluation",
            },
        },
    }
    _atomic_write(output, processed_payload)
    _atomic_write(
        manifest,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return metadata


def load_price_series(path: Path) -> PriceSeries:
    """Load a deterministic two-column daily price CSV with strict validation."""

    observations: list[tuple[date, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["date", "close_usd"]:
            raise ValueError("price CSV header must be exactly: date,close_usd")
        for row_number, row in enumerate(reader, start=2):
            try:
                observation_date = date.fromisoformat(row["date"])
                price = float(row["close_usd"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid price row {row_number}") from exc
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"close_usd must be finite and positive at row {row_number}")
            observations.append((observation_date, price))
    _validate_rows(observations)
    return PriceSeries(
        dates=np.asarray([item[0].isoformat() for item in observations], dtype="datetime64[D]"),
        close_usd=np.asarray([item[1] for item in observations], dtype=np.float64),
    )


def validate_reference_data(path: Path, manifest: Path) -> dict[str, Any]:
    """Validate snapshot integrity, schema, grain, and manifest agreement."""

    with manifest.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("unsupported source manifest schema")
    processed = metadata.get("processed")
    source = metadata.get("source")
    if not isinstance(processed, dict) or not isinstance(source, dict):
        raise ValueError("source manifest is missing source or processed metadata")
    expected_sha = processed.get("sha256")
    if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
        raise ValueError("processed data SHA-256 does not match the source manifest")
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "path": SOURCE_PATH,
        "url": SOURCE_URL,
        "sha256": SOURCE_SHA256,
        "license": SOURCE_LICENSE,
        "attribution": SOURCE_ATTRIBUTION,
        "timezone": SOURCE_TIMEZONE,
        "timestamp_convention": SOURCE_TIMESTAMP_CONVENTION,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise ValueError("source manifest does not identify the pinned reference source")

    series = load_price_series(path)
    actual = {
        "rows": len(series.dates),
        "start_date": str(series.dates[0]),
        "end_date": str(series.dates[-1]),
    }
    for key, value in actual.items():
        if processed.get(key) != value:
            raise ValueError(f"processed data {key} disagrees with the source manifest")
    return metadata
