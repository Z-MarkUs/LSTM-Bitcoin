"""Compare reproduced evidence without rewriting the published reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
import pathlib
from typing import Any


def compare_evidence(
    expected_root: pathlib.Path,
    actual_root: pathlib.Path,
    *,
    allow_dependency_changes: bool = False,
) -> None:
    # The lock fixes package versions, but CPU kernels can differ slightly between
    # Windows (where the checked evidence was produced) and this Linux runner.
    relative_tolerance = 1e-5
    absolute_tolerance = 1e-7
    required = {
        "fold_metrics.csv",
        "folds.csv",
        "manifest.json",
        "metrics.json",
        "predictions.csv",
        "seed_metrics.csv",
        "seed_predictions.csv",
        "signal_metrics.csv",
        "signal_simulation.csv",
        "training.csv",
        "SHA256SUMS",
    }

    for root in (expected_root, actual_root):
        files = {path.name for path in root.iterdir() if path.is_file()}
        assert files == required, (
            f"{root} artifact set differs; missing={sorted(required - files)}, "
            f"unexpected={sorted(files - required)}"
        )

    def compare_json(expected: Any, actual: Any, path: str = "$") -> None:
        if isinstance(expected, dict):
            assert isinstance(actual, dict), f"{path}: expected an object"
            assert expected.keys() == actual.keys(), f"{path}: JSON keys differ"
            for key in expected:
                compare_json(expected[key], actual[key], f"{path}.{key}")
            return
        if isinstance(expected, list):
            assert isinstance(actual, list), f"{path}: expected a list"
            assert len(expected) == len(actual), f"{path}: list length differs"
            for index, (expected_item, actual_item) in enumerate(
                zip(expected, actual, strict=True)
            ):
                compare_json(expected_item, actual_item, f"{path}[{index}]")
            return
        if (
            isinstance(expected, numbers.Real)
            and not isinstance(expected, bool)
            and isinstance(actual, numbers.Real)
            and not isinstance(actual, bool)
        ):
            assert math.isclose(
                float(expected),
                float(actual),
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ), f"{path}: {expected!r} != {actual!r}"
            return
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"

    def compare_csv(name: str) -> None:
        with expected_root.joinpath(name).open(encoding="utf-8", newline="") as expected_file:
            expected_reader = csv.DictReader(expected_file)
            expected_fields = expected_reader.fieldnames
            expected_rows = list(expected_reader)
        with actual_root.joinpath(name).open(encoding="utf-8", newline="") as actual_file:
            actual_reader = csv.DictReader(actual_file)
            actual_fields = actual_reader.fieldnames
            actual_rows = list(actual_reader)

        assert expected_fields == actual_fields, f"{name}: headers differ"
        assert len(expected_rows) == len(actual_rows), f"{name}: row count differs"
        for row_index, (expected_row, actual_row) in enumerate(
            zip(expected_rows, actual_rows, strict=True), start=2
        ):
            assert expected_row.keys() == actual_row.keys(), f"{name}:{row_index}: keys differ"
            for field in expected_row:
                expected_value = expected_row[field]
                actual_value = actual_row[field]
                assert expected_value is not None and actual_value is not None
                try:
                    expected_number = float(expected_value)
                    actual_number = float(actual_value)
                except ValueError:
                    assert expected_value == actual_value, (
                        f"{name}:{row_index}:{field}: {expected_value!r} != {actual_value!r}"
                    )
                else:
                    assert math.isfinite(expected_number) and math.isfinite(actual_number)
                    assert math.isclose(
                        expected_number,
                        actual_number,
                        rel_tol=relative_tolerance,
                        abs_tol=absolute_tolerance,
                    ), f"{name}:{row_index}:{field}: {expected_value!r} != {actual_value!r}"

    expected_metrics = json.loads(
        expected_root.joinpath("metrics.json").read_text(encoding="utf-8")
    )
    actual_metrics = json.loads(actual_root.joinpath("metrics.json").read_text(encoding="utf-8"))
    compare_json(expected_metrics, actual_metrics)

    for name in (
        "fold_metrics.csv",
        "folds.csv",
        "predictions.csv",
        "seed_metrics.csv",
        "seed_predictions.csv",
        "signal_metrics.csv",
        "signal_simulation.csv",
        "training.csv",
    ):
        compare_csv(name)

    expected_manifest = json.loads(
        expected_root.joinpath("manifest.json").read_text(encoding="utf-8")
    )
    actual_manifest = json.loads(actual_root.joinpath("manifest.json").read_text(encoding="utf-8"))
    for field in ("schema_version", "experiment", "features", "folds", "inputs", "limitations"):
        compare_json(expected_manifest[field], actual_manifest[field], f"$.manifest.{field}")
    assert (
        expected_manifest["code"]["package_version"] == actual_manifest["code"]["package_version"]
    )
    assert actual_manifest["code"]["dirty"] is False
    assert len(actual_manifest["code"]["revision"]) == 40
    expected_dependencies = expected_manifest["environment"]["dependencies"]
    actual_dependencies = actual_manifest["environment"]["dependencies"]
    assert expected_dependencies.keys() == actual_dependencies.keys(), "dependency names differ"
    if not allow_dependency_changes:
        assert expected_dependencies == actual_dependencies, "dependency versions differ"
    else:
        assert all(isinstance(value, str) and value for value in actual_dependencies.values())
        print(f"Reference dependencies: {expected_dependencies}")
        print(f"Reproduced dependencies: {actual_dependencies}")
    assert (
        expected_manifest["environment"]["implementation"]
        == actual_manifest["environment"]["implementation"]
    )
    assert actual_manifest["environment"]["python"].split(".")[:2] == ["3", "12"]
    assert set(actual_manifest["artifacts"]) == required - {"manifest.json", "SHA256SUMS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected", type=pathlib.Path, default=pathlib.Path("evaluation/results/reference-v1")
    )
    parser.add_argument("--actual", type=pathlib.Path, default=pathlib.Path("build/reference-v1"))
    parser.add_argument("--allow-dependency-changes", action="store_true")
    args = parser.parse_args()
    compare_evidence(
        args.expected, args.actual, allow_dependency_changes=args.allow_dependency_changes
    )


if __name__ == "__main__":
    main()
