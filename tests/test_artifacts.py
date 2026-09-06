from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from lstm_bitcoin import artifacts


def test_stable_json_csv_and_hash_serialization(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "document.json"
    artifacts.write_json(json_path, {"z": 1, "a": {"value": 2}})
    assert json_path.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "value": 2\n  },\n  "z": 1\n}\n'
    )
    assert artifacts.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert artifacts.sha256_path(json_path) == hashlib.sha256(json_path.read_bytes()).hexdigest()

    csv_path = tmp_path / "rows.csv"
    artifacts.write_csv(
        csv_path,
        fieldnames=("name", "value", "missing"),
        rows=[{"name": "alpha", "value": 1.23456789012345, "extra": "ignored"}],
    )
    assert list(csv.DictReader(csv_path.open(encoding="utf-8"))) == [
        {"name": "alpha", "value": "1.23456789012", "missing": ""}
    ]


def test_prepare_output_directory_preserves_unknown_files(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    known = output / "metrics.json"
    unknown = output / "notes.txt"
    known.write_text("old", encoding="utf-8")
    unknown.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --force"):
        artifacts.prepare_output_directory(output, force=False, known_files=("metrics.json",))
    artifacts.prepare_output_directory(output, force=True, known_files=("metrics.json",))
    assert known.read_text(encoding="utf-8") == "old"
    assert unknown.read_text(encoding="utf-8") == "keep"

    file_path = tmp_path / "not-directory"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not a directory"):
        artifacts.prepare_output_directory(file_path, force=False, known_files=())


def test_prepare_output_refuses_to_replace_known_directory(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "metrics.json").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="non-file"):
        artifacts.prepare_output_directory(output, force=True, known_files=("metrics.json",))


def test_prepare_output_preflights_every_target_before_removing_anything(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    existing_file = output / "metrics.json"
    existing_file.write_text("valuable prior run", encoding="utf-8")
    (output / "predictions.csv").mkdir()

    with pytest.raises(FileExistsError, match="non-file"):
        artifacts.prepare_output_directory(
            output,
            force=True,
            known_files=("metrics.json", "predictions.csv"),
        )

    assert existing_file.read_text(encoding="utf-8") == "valuable prior run"


def test_checksum_round_trip_is_sorted_and_complete(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    checksum = artifacts.write_checksums(tmp_path, ("b.txt", "a.txt"))

    lines = checksum.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("  a.txt")
    assert lines[1].endswith("  b.txt")
    assert artifacts.verify_checksums(tmp_path) == ("a.txt", "b.txt")


def test_write_checksums_rejects_unsafe_and_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="simple filenames"):
        artifacts.write_checksums(tmp_path, ("../outside",))
    with pytest.raises(FileNotFoundError):
        artifacts.write_checksums(tmp_path, ("missing",))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not-a-checksum\n", "invalid SHA256SUMS line"),
        (f"{'g' * 64}  file.txt\n", "invalid SHA-256"),
        (f"{'0' * 64}  ../file.txt\n", "unsafe or duplicate"),
        (f"{'0' * 64}  file.txt\n", "verification failed"),
        ("\n", "no entries"),
    ],
)
def test_verify_checksums_rejects_invalid_manifests(
    tmp_path: Path, line: str, message: str
) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        artifacts.verify_checksums(tmp_path)


def test_verify_checksums_rejects_duplicate_entry(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")
    line = f"{artifacts.sha256_path(path)}  file.txt\n"
    (tmp_path / "SHA256SUMS").write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        artifacts.verify_checksums(tmp_path)


def test_atomic_write_cleans_up_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact"

    def fail_replace(source: str, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        artifacts.atomic_write(destination, b"payload")
    assert list(tmp_path.iterdir()) == []
