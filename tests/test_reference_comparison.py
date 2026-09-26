from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
compare = runpy.run_path(str(ROOT / "scripts/compare_reference.py"))["compare_evidence"]


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[Path, Path]:
    expected = ROOT / "evaluation/results/reference-v1"
    actual = tmp_path / "reproduced"
    shutil.copytree(expected, actual)
    return expected, actual


def edit_manifest(actual: Path, field: str, value: Any) -> None:
    path = actual / "manifest.json"
    manifest = json.loads(path.read_text())
    section, key = field.split(".")
    manifest[section][key] = value
    path.write_text(json.dumps(manifest))


def test_identical_evidence_passes(evidence: tuple[Path, Path]) -> None:
    compare(*evidence)


def test_dependency_drift_requires_explicit_opt_in(evidence: tuple[Path, Path]) -> None:
    expected, actual = evidence
    dependencies = json.loads((actual / "manifest.json").read_text())["environment"]["dependencies"]
    dependencies["matplotlib"] = "3.11.2"
    edit_manifest(actual, "environment.dependencies", dependencies)
    with pytest.raises(AssertionError, match="dependency versions differ"):
        compare(expected, actual)
    compare(expected, actual, allow_dependency_changes=True)


def test_dependency_opt_in_still_rejects_prediction_drift(evidence: tuple[Path, Path]) -> None:
    expected, actual = evidence
    path = actual / "predictions.csv"
    lines = path.read_text().splitlines()
    columns = lines[1].split(",")
    for index, value in enumerate(columns):
        try:
            number = float(value)
        except ValueError:
            continue
        columns[index] = str(number + 100)
        break
    else:
        pytest.fail("fixture has no numeric prediction field")
    lines[1] = ",".join(columns)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError, match=r"predictions\.csv"):
        compare(expected, actual, allow_dependency_changes=True)


def test_dependency_opt_in_still_rejects_dirty_run(evidence: tuple[Path, Path]) -> None:
    expected, actual = evidence
    edit_manifest(actual, "code.dirty", True)
    with pytest.raises(AssertionError):
        compare(expected, actual, allow_dependency_changes=True)


def test_dependency_opt_in_still_rejects_changed_inputs(evidence: tuple[Path, Path]) -> None:
    expected, actual = evidence
    edit_manifest(actual, "inputs.config_sha256", "0" * 64)
    with pytest.raises(AssertionError, match="config_sha256"):
        compare(expected, actual, allow_dependency_changes=True)


def test_missing_artifact_is_rejected(evidence: tuple[Path, Path]) -> None:
    expected, actual = evidence
    (actual / "training.csv").unlink()
    with pytest.raises(AssertionError, match="artifact set differs"):
        compare(expected, actual, allow_dependency_changes=True)
