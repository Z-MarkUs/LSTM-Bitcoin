"""Deterministic artifact serialization and checksum verification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    """Replace one file atomically without exposing a partially written artifact."""

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


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    atomic_write(path, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _cell(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write a stable UTF-8 CSV with bounded floating-point precision."""

    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _cell(row.get(name, "")) for name in fieldnames})
    atomic_write(path, output.getvalue().encode("utf-8"))


def prepare_output_directory(path: Path, *, force: bool, known_files: Sequence[str]) -> None:
    """Create and preflight a directory without deleting prior generated evidence."""

    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in known_files if (path / name).exists()]
    if existing and not force:
        names = ", ".join(item.name for item in existing)
        raise FileExistsError(f"generated artifacts already exist ({names}); pass --force")
    for item in existing:
        if not item.is_file():
            raise FileExistsError(f"refusing to replace non-file artifact path: {item}")


def write_checksums(directory: Path, filenames: Sequence[str]) -> Path:
    """Write GNU-compatible SHA256SUMS for reviewed files in one directory."""

    lines: list[str] = []
    for filename in sorted(filenames):
        if Path(filename).name != filename:
            raise ValueError("checksum entries must be simple filenames")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{sha256_path(path)}  {filename}")
    output = directory / "SHA256SUMS"
    atomic_write(output, ("\n".join(lines) + "\n").encode("utf-8"))
    return output


def verify_checksums(directory: Path) -> tuple[str, ...]:
    """Verify every SHA256SUMS entry and reject unsafe or duplicate paths."""

    checksum_file = directory / "SHA256SUMS"
    seen: set[str] = set()
    verified: list[str] = []
    with checksum_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                expected, filename = line.split("  ", maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"invalid SHA256SUMS line {line_number}") from exc
            if len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise ValueError(f"invalid SHA-256 on line {line_number}")
            if Path(filename).name != filename or filename in seen:
                raise ValueError(f"unsafe or duplicate checksum path on line {line_number}")
            path = directory / filename
            if not path.is_file() or sha256_path(path) != expected:
                raise ValueError(f"checksum verification failed for {filename}")
            seen.add(filename)
            verified.append(filename)
    if not verified:
        raise ValueError("SHA256SUMS contains no entries")
    return tuple(verified)
