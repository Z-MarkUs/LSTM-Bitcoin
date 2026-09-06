from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lstm_bitcoin import data
from lstm_bitcoin.data import PriceSeries, fetch_reference_data, load_price_series


def source_csv(rows: list[tuple[str, str]]) -> bytes:
    body = "time,PriceUSD,ignored\n" + "".join(f"{day},{price},x\n" for day, price in rows)
    return body.encode()


def test_price_series_validates_public_invariants() -> None:
    dates = np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    values = np.asarray([100.0, 101.0])
    assert len(PriceSeries(dates, values).dates) == 2

    with pytest.raises(ValueError, match="one-dimensional"):
        PriceSeries(dates[:, None], values)
    with pytest.raises(ValueError, match="equal length"):
        PriceSeries(dates, values[:1])
    with pytest.raises(ValueError, match="at least two"):
        PriceSeries(dates[:1], values[:1])
    with pytest.raises(ValueError, match="daily"):
        PriceSeries(np.asarray(["2020-01-01", "2020-01-03"], dtype="datetime64[D]"), values)
    with pytest.raises(ValueError, match="positive"):
        PriceSeries(dates, np.asarray([100.0, np.nan]))


def test_load_price_series_round_trip_and_hash(write_price_csv: Any) -> None:
    path = write_price_csv([("2020-01-01", "100"), ("2020-01-02", "101.5"), ("2020-01-03", "99")])
    loaded = load_price_series(path)

    np.testing.assert_array_equal(
        loaded.dates,
        np.asarray(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
    )
    np.testing.assert_allclose(loaded.close_usd, [100, 101.5, 99])
    assert data.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("bad", "100"), ("2020-01-02", "101")], "invalid price row 2"),
        ([("2020-01-01", "x"), ("2020-01-02", "101")], "invalid price row 2"),
        ([("2020-01-01", "0"), ("2020-01-02", "101")], "finite and positive"),
        ([("2020-01-01", "nan"), ("2020-01-02", "101")], "finite and positive"),
        ([("2020-01-01", "100"), ("2020-01-03", "101")], "strictly ordered"),
        ([("2020-01-02", "100"), ("2020-01-01", "101")], "strictly ordered"),
        ([("2020-01-01", "100"), ("2020-01-01", "101")], "strictly ordered"),
    ],
)
def test_load_price_series_rejects_bad_rows(
    write_price_csv: Any, rows: list[tuple[str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_price_series(write_price_csv(rows))


def test_load_price_series_requires_exact_header(write_price_csv: Any) -> None:
    path = write_price_csv(
        [("2020-01-01", "100"), ("2020-01-02", "101")],
        header=("close_usd", "date"),
    )
    with pytest.raises(ValueError, match="header must be exactly"):
        load_price_series(path)


def test_extract_rows_filters_snapshot_and_counts_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "REFERENCE_START_DATE", date(2020, 1, 1))
    monkeypatch.setattr(data, "REFERENCE_END_DATE", date(2020, 1, 2))
    payload = source_csv(
        [
            ("2019-12-31", "99"),
            ("2020-01-01", "100"),
            ("2020-01-02", "101"),
            ("2020-01-03", ""),
        ]
    )

    rows, skipped = data._extract_price_rows(payload)
    assert rows == [(date(2020, 1, 1), 100.0), (date(2020, 1, 2), 101.0)]
    assert skipped == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"time,value\n2020-01-01,1\n", "missing required"),
        (b"time,PriceUSD\n2020-01-01,nan\n", "invalid PriceUSD"),
        (b"time,PriceUSD\n2020-01-01,0\n", "invalid PriceUSD"),
        (b"time,PriceUSD\n2010-01-01,1\n", "no usable"),
    ],
)
def test_extract_rows_rejects_invalid_source(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        data._extract_price_rows(payload)


def test_fetch_reference_data_uses_verified_source_and_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = source_csv([("2020-01-01", "100"), ("2020-01-02", "101")])
    monkeypatch.setattr(data, "REFERENCE_START_DATE", date(2020, 1, 1))
    monkeypatch.setattr(data, "REFERENCE_END_DATE", date(2020, 1, 2))
    monkeypatch.setattr(data, "_download_source", lambda: payload)
    output = tmp_path / "nested" / "prices.csv"
    manifest = tmp_path / "nested" / "manifest.json"

    metadata = fetch_reference_data(output, manifest)
    assert metadata["processed"]["rows"] == 2
    assert metadata["source"]["commit"] == data.SOURCE_COMMIT
    assert load_price_series(output).close_usd.tolist() == [100.0, 101.0]
    assert json.loads(manifest.read_text(encoding="utf-8")) == metadata

    with pytest.raises(FileExistsError, match="--force"):
        fetch_reference_data(output, manifest)
    overwritten = fetch_reference_data(output, manifest, force=True)
    assert overwritten["processed"]["sha256"] == data.sha256_file(output)


def test_fetch_reference_data_preflights_distinct_file_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    with pytest.raises(ValueError, match="must be different"):
        fetch_reference_data(shared, shared)

    output = tmp_path / "prices.csv"
    output.write_text("valuable prior snapshot\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.mkdir()
    monkeypatch.setattr(
        data,
        "_download_source",
        lambda: pytest.fail("preflight must happen before the network request"),
    )
    with pytest.raises(FileExistsError, match="non-file"):
        fetch_reference_data(output, manifest, force=True)
    assert output.read_text(encoding="utf-8") == "valuable prior snapshot\n"


def manifest_for(path: Path) -> dict[str, Any]:
    series = load_price_series(path)
    return {
        "schema_version": 1,
        "source": {
            "repository": data.SOURCE_REPOSITORY,
            "commit": data.SOURCE_COMMIT,
            "path": data.SOURCE_PATH,
            "url": data.SOURCE_URL,
            "sha256": data.SOURCE_SHA256,
            "license": data.SOURCE_LICENSE,
            "attribution": data.SOURCE_ATTRIBUTION,
            "timezone": data.SOURCE_TIMEZONE,
            "timestamp_convention": data.SOURCE_TIMESTAMP_CONVENTION,
        },
        "processed": {
            "path": path.as_posix(),
            "sha256": data.sha256_file(path),
            "selected_columns": ["time", "PriceUSD"],
            "output_columns": ["date", "close_usd"],
            "rows": len(series.dates),
            "start_date": str(series.dates[0]),
            "end_date": str(series.dates[-1]),
        },
    }


def test_validate_reference_data_checks_integrity_and_full_provenance(
    write_price_csv: Any, tmp_path: Path
) -> None:
    path = write_price_csv([("2020-01-01", "100"), ("2020-01-02", "101")])
    manifest = tmp_path / "manifest.json"
    metadata = manifest_for(path)
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    assert data.validate_reference_data(path, manifest) == metadata

    for key in (
        "repository",
        "path",
        "url",
        "license",
        "attribution",
        "timezone",
        "timestamp_convention",
    ):
        changed = json.loads(json.dumps(metadata))
        changed["source"][key] = "tampered"
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="pinned reference source"):
            data.validate_reference_data(path, manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc.update(schema_version=2), "unsupported"),
        (lambda doc: doc.pop("processed"), "missing source or processed"),
        (lambda doc: doc["processed"].update(sha256="0" * 64), "SHA-256"),
        (lambda doc: doc["source"].update(commit="bad"), "pinned reference"),
        (lambda doc: doc["processed"].update(rows=999), "rows disagrees"),
        (lambda doc: doc["processed"].update(start_date="1999-01-01"), "start_date disagrees"),
        (lambda doc: doc["processed"].update(end_date="1999-01-02"), "end_date disagrees"),
    ],
)
def test_validate_reference_data_rejects_bad_manifest(
    write_price_csv: Any,
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    path = write_price_csv([("2020-01-01", "100"), ("2020-01-02", "101")])
    metadata = manifest_for(path)
    mutate(metadata)
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        data.validate_reference_data(path, manifest)


class FakeResponse(AbstractContextManager["FakeResponse"]):
    def __init__(self, payload: bytes, *, url: str, content_length: str | None = None) -> None:
        self.payload = payload
        self.url = url
        self.headers = {} if content_length is None else {"Content-Length": content_length}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, amount: int) -> bytes:
        return self.payload[:amount]


def test_download_source_validates_redirect_size_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"abc"
    monkeypatch.setattr(data, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        data,
        "urlopen",
        lambda request, timeout: FakeResponse(payload, url=data.SOURCE_URL),
    )
    assert data._download_source() == payload

    monkeypatch.setattr(
        data,
        "urlopen",
        lambda request, timeout: FakeResponse(payload, url="https://example.com/file"),
    )
    with pytest.raises(RuntimeError, match="redirected outside"):
        data._download_source()

    monkeypatch.setattr(
        data,
        "urlopen",
        lambda request, timeout: FakeResponse(
            payload, url=data.SOURCE_URL, content_length=str(data.MAX_SOURCE_BYTES + 1)
        ),
    )
    with pytest.raises(RuntimeError, match="maximum permitted"):
        data._download_source()

    oversized = b"x" * (data.MAX_SOURCE_BYTES + 1)
    monkeypatch.setattr(
        data,
        "urlopen",
        lambda request, timeout: FakeResponse(oversized, url=data.SOURCE_URL),
    )
    with pytest.raises(RuntimeError, match="maximum permitted"):
        data._download_source()

    monkeypatch.setattr(
        data,
        "urlopen",
        lambda request, timeout: FakeResponse(b"wrong", url=data.SOURCE_URL),
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        data._download_source()


def test_download_source_rejects_non_allowlisted_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "SOURCE_URL", "http://example.com/source.csv")
    with pytest.raises(RuntimeError, match="outside the allowlisted"):
        data._download_source()


def test_atomic_write_cleans_temporary_file_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "destination.bin"

    def fail_replace(source: str, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(data.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        data._atomic_write(destination, b"payload")
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
