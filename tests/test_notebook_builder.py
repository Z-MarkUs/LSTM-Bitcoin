from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_notebook.py"


def load_builder() -> ModuleType:
    specification = importlib.util.spec_from_file_location("build_notebook", BUILDER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load notebook builder")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_notebook_builder_enforces_read_only_evidence_contract(tmp_path: Path) -> None:
    builder = load_builder()
    notebook = builder.build_notebook()
    output = tmp_path / "reference_evaluation.ipynb"
    builder.write_notebook(notebook, output)

    builder.check_notebook(output)
    headings = builder._section_headings(notebook)
    assert headings == (
        "## tl;dr",
        "## Context & Methods",
        "## Data",
        "## Results",
        "## Takeaways",
    )
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )

    source = "\n".join(cell.source for cell in notebook.cells)
    assert "verify_checksums" in source
    assert "validate_reference_data" in source
    assert "does not retrain" in source
    assert "run_experiment" not in source
    assert "TorchLSTM" not in source


def test_notebook_source_check_allows_execution_outputs(tmp_path: Path) -> None:
    builder = load_builder()
    notebook = builder.build_notebook()
    first_code = next(cell for cell in notebook.cells if cell.cell_type == "code")
    first_code.execution_count = 1
    first_code.outputs = [nbformat.v4.new_output("stream", name="stdout", text="verified\n")]
    output = tmp_path / "reference_evaluation.ipynb"
    builder.write_notebook(notebook, output)

    builder.check_notebook(output)
