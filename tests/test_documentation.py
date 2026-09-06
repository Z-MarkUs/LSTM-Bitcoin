from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lstm_bitcoin.artifacts import sha256_path

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]


def test_readme_headline_is_generated_result_consistent() -> None:
    metrics = json.loads(
        (ROOT / "evaluation/results/reference-v1/metrics.json").read_text(encoding="utf-8")
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    zero_mae = float(metrics["models"]["zero_return"]["mae"])
    lstm_mae = float(metrics["models"]["lstm"]["mae"])
    lower, upper = metrics["comparison"]["block_bootstrap_95pct_interval"]

    assert f"{lstm_mae:.6f}" in readme
    assert f"{zero_mae:.6f}" in readme
    assert f"{float(lower):.6f}" in readme
    assert f"{float(upper):.6f}" in readme
    if metrics["comparison"]["beats_best_baseline"]:
        assert "no demonstrated improvement" not in readme
    else:
        assert "no demonstrated improvement" in readme


def test_reference_manifest_matches_tracked_inputs() -> None:
    run_manifest = json.loads(
        (ROOT / "evaluation/results/reference-v1/manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads((ROOT / "data/source-manifest.json").read_text(encoding="utf-8"))
    inputs = run_manifest["inputs"]
    code = run_manifest["code"]

    assert re.fullmatch(r"[0-9a-f]{40}", code["revision"])
    assert code["dirty"] is False
    assert inputs["config_sha256"] == sha256_path(ROOT / "configs/reference.toml")
    assert inputs["data_sha256"] == sha256_path(ROOT / "data/btc_usd_daily.csv")
    assert inputs["source_manifest_sha256"] == sha256_path(ROOT / "data/source-manifest.json")
    assert inputs["source"] == source_manifest["source"]
    assert inputs["processed"] == source_manifest["processed"]


def test_documentation_relative_links_resolve() -> None:
    markdown_files = [
        ROOT / "README.md",
        ROOT / "DATA_LICENSE.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        ROOT / "RELEASING.md",
        ROOT / "CHANGELOG.md",
        ROOT / "data/README.md",
        ROOT / "evaluation/README.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    pattern = re.compile(r"\[[^]]+]\(([^)]+)\)")
    broken: list[str] = []
    for document in markdown_files:
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            if not (document.parent / target).resolve().exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, "broken relative documentation links:\n" + "\n".join(broken)


def test_published_notebook_is_executed_without_errors() -> None:
    notebook = nbformat.read(ROOT / "notebooks/reference_evaluation.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not [
        output for cell in code_cells for output in cell.outputs if output.output_type == "error"
    ]
    assert (
        sum(
            output.output_type == "display_data" and "image/svg+xml" in output.get("data", {})
            for cell in code_cells
            for output in cell.outputs
        )
        == 4
    )
    serialized = json.dumps(notebook, ensure_ascii=False)
    assert sha256_path(ROOT / "data/source-manifest.json") in serialized
    assert sha256_path(ROOT / "evaluation/results/reference-v1/manifest.json") in serialized
