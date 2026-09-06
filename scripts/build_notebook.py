#!/usr/bin/env python3
"""Build and optionally execute the reference-evaluation notebook.

The notebook is intentionally a read-only view over reviewed artifacts. It verifies
their checksums and provenance, but it never imports or runs the training pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from typing import Any

import nbformat
from nbformat.notebooknode import NotebookNode

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "notebooks" / "reference_evaluation.ipynb"
SECTION_HEADINGS = (
    "## tl;dr",
    "## Context & Methods",
    "## Data",
    "## Results",
    "## Takeaways",
)


def _markdown(cell_id: str, source: str) -> NotebookNode:
    cell = nbformat.v4.new_markdown_cell(dedent(source).strip())
    cell["id"] = cell_id
    return cell


def _code(cell_id: str, source: str) -> NotebookNode:
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    cell["id"] = cell_id
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def build_notebook() -> NotebookNode:
    """Return the deterministic, unexecuted reference notebook."""

    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        _markdown(
            "title",
            """
            # LSTM-Bitcoin reference evaluation

            This is a read-only evidence notebook for the checked-in `reference-v1`
            run. It verifies and summarizes released artifacts; **it does not retrain,
            tune, or select a model**. Rebuild the evidence with the documented CLI,
            review it, and only then execute this notebook.
            """,
        ),
        _markdown(
            "tldr-heading",
            """
            ## tl;dr

            The next cell first checks the pinned data, the complete result bundle,
            the four published figures, and their recorded SHA-256 digests. Its summary
            is generated from those verified files so the notebook cannot silently
            preserve a stale performance claim.
            """,
        ),
        _code(
            "load-and-verify",
            r'''
            import csv
            import json
            from html import escape
            from pathlib import Path

            from IPython.display import HTML, SVG, Markdown, display

            from lstm_bitcoin.artifacts import sha256_path, verify_checksums
            from lstm_bitcoin.data import validate_reference_data


            def find_repository_root(start: Path) -> Path:
                """Find the checkout without assuming the notebook server's directory."""
                for candidate in (start.resolve(), *start.resolve().parents):
                    if (candidate / "pyproject.toml").is_file() and (
                        candidate / "src" / "lstm_bitcoin"
                    ).is_dir():
                        return candidate
                raise FileNotFoundError("run this notebook from inside the LSTM-Bitcoin checkout")


            ROOT = find_repository_root(Path.cwd())
            DATA_PATH = ROOT / "data" / "btc_usd_daily.csv"
            SOURCE_MANIFEST_PATH = ROOT / "data" / "source-manifest.json"
            CONFIG_PATH = ROOT / "configs" / "reference.toml"
            RUN_DIRECTORY = ROOT / "evaluation" / "results" / "reference-v1"
            FIGURE_DIRECTORY = ROOT / "docs" / "assets"

            REQUIRED_RUN_FILES = {
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
            }
            REQUIRED_FIGURES = {
                "forecast_mae.svg",
                "forecast_scatter.svg",
                "signal_equity.svg",
                "walk_forward_folds.svg",
            }


            def require(condition: bool, message: str) -> None:
                if not condition:
                    raise ValueError(message)


            def read_json(path: Path) -> dict[str, object]:
                with path.open("r", encoding="utf-8") as handle:
                    document = json.load(handle)
                require(isinstance(document, dict), f"expected a JSON object: {path}")
                return document


            def read_rows(path: Path) -> list[dict[str, str]]:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    return list(csv.DictReader(handle))


            def format_value(value: object) -> str:
                if value is None:
                    return "N/A"
                if isinstance(value, bool):
                    return "yes" if value else "no"
                if isinstance(value, float):
                    return f"{value:.6g}"
                if isinstance(value, (list, tuple)):
                    return ", ".join(format_value(item) for item in value)
                return str(value)


            def table_cell(row: dict[str, object], column: str) -> str:
                return f"<td>{escape(format_value(row.get(column, '')))}</td>"


            def table_row(row: dict[str, object], columns: tuple[str, ...]) -> str:
                cells = "".join(table_cell(row, column) for column in columns)
                return f"<tr>{cells}</tr>"


            def bounded_table(
                rows: list[dict[str, object]],
                columns: tuple[str, ...],
                *,
                max_rows: int,
            ) -> HTML:
                """Render a deliberately bounded HTML table without a pandas dependency."""
                require(max_rows > 0, "max_rows must be positive")
                shown = rows[:max_rows]
                header = "".join(f"<th>{escape(column)}</th>" for column in columns)
                body = "".join(table_row(row, columns) for row in shown)
                note = (
                    f"<p><em>Showing {len(shown)} of {len(rows)} rows.</em></p>"
                    if len(rows) > len(shown)
                    else f"<p><em>{len(shown)} rows.</em></p>"
                )
                return HTML(
                    "<div style='overflow-x:auto'>"
                    "<table>"
                    f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody>"
                    "</table>"
                    f"{note}</div>"
                )


            evidence_paths = (
                DATA_PATH,
                SOURCE_MANIFEST_PATH,
                CONFIG_PATH,
                RUN_DIRECTORY,
                FIGURE_DIRECTORY,
            )
            missing = [path for path in evidence_paths if not path.exists()]
            if missing:
                rendered = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
                raise FileNotFoundError(
                    "The reviewed reference evidence is not complete. Generate and review "
                    f"it before executing this notebook:\n{rendered}"
                )

            validate_reference_data(DATA_PATH, SOURCE_MANIFEST_PATH)
            verified_run_files = set(verify_checksums(RUN_DIRECTORY))
            verified_figures = set(verify_checksums(FIGURE_DIRECTORY))
            require(
                verified_run_files == REQUIRED_RUN_FILES,
                "reference-v1/SHA256SUMS does not enumerate the exact released result set",
            )
            require(
                verified_figures == REQUIRED_FIGURES,
                "docs/assets/SHA256SUMS does not enumerate the exact four released figures",
            )

            source_manifest = read_json(SOURCE_MANIFEST_PATH)
            run_manifest = read_json(RUN_DIRECTORY / "manifest.json")
            metrics = read_json(RUN_DIRECTORY / "metrics.json")
            prices = read_rows(DATA_PATH)
            folds = read_rows(RUN_DIRECTORY / "folds.csv")
            fold_metrics = read_rows(RUN_DIRECTORY / "fold_metrics.csv")
            predictions = read_rows(RUN_DIRECTORY / "predictions.csv")
            seed_metrics = read_rows(RUN_DIRECTORY / "seed_metrics.csv")
            signal_metrics = read_rows(RUN_DIRECTORY / "signal_metrics.csv")

            inputs = run_manifest["inputs"]
            require(isinstance(inputs, dict), "run manifest inputs must be an object")
            require(inputs["data_sha256"] == sha256_path(DATA_PATH), "run/data digest mismatch")
            require(
                inputs["source_manifest_sha256"] == sha256_path(SOURCE_MANIFEST_PATH),
                "run/source-manifest digest mismatch",
            )
            require(
                inputs["config_sha256"] == sha256_path(CONFIG_PATH),
                "run/config digest mismatch",
            )

            recorded_artifacts = run_manifest["artifacts"]
            require(isinstance(recorded_artifacts, dict), "run artifact index must be an object")
            require(
                set(recorded_artifacts) == REQUIRED_RUN_FILES - {"manifest.json"},
                "manifest artifact index is incomplete or contains an unexpected file",
            )
            for filename, expected_digest in recorded_artifacts.items():
                require(
                    sha256_path(RUN_DIRECTORY / filename) == expected_digest,
                    f"manifest digest mismatch for {filename}",
                )

            processed = source_manifest["processed"]
            require(isinstance(processed, dict), "processed data manifest must be an object")
            require(len(prices) == processed["rows"], "processed data row count mismatch")
            require(prices[0]["date"] == processed["start_date"], "processed start date mismatch")
            require(prices[-1]["date"] == processed["end_date"], "processed end date mismatch")

            period = metrics["out_of_sample_period"]
            require(isinstance(period, dict), "out-of-sample period must be an object")
            models = metrics["models"]
            require(isinstance(models, dict) and models, "metrics must contain released models")
            expected_observations = int(period["observations"])
            for model_name in models:
                model_rows = [row for row in predictions if row["model"] == model_name]
                target_dates = [row["target_date"] for row in model_rows]
                require(
                    len(model_rows) == expected_observations == len(set(target_dates)),
                    f"{model_name} predictions are incomplete or duplicate a target date",
                )

            comparison = metrics["comparison"]
            require(isinstance(comparison, dict), "comparison must be an object")
            interval = comparison.get("block_bootstrap_95pct_interval")
            require(isinstance(interval, list) and len(interval) == 2, "missing bootstrap interval")
            difference = float(comparison["lstm_minus_best_baseline_mae"])
            interval_text = f"[{float(interval[0]):.6g}, {float(interval[1]):.6g}]"
            direction = "lower" if difference < 0 else "higher"
            verdict = "did" if comparison["beats_best_baseline"] else "did not"
            source_manifest_digest = sha256_path(SOURCE_MANIFEST_PATH)
            run_manifest_digest = sha256_path(RUN_DIRECTORY / "manifest.json")
            display(
                Markdown(
                    f"**Integrity checks passed.** The released LSTM ensemble {verdict} beat the "
                    f"best MAE baseline (`{comparison['best_baseline']}`): its MAE was "
                    f"{abs(difference):.6g} {direction}. The 95% moving-block bootstrap interval "
                    f"for LSTM minus baseline MAE is {interval_text}. These are historical "
                    "out-of-sample results, not evidence of future profitability. "
                    f"Evidence fingerprints: source manifest `{source_manifest_digest}`; "
                    f"run manifest `{run_manifest_digest}`."
                )
            )
            display(
                bounded_table(
                    [
                        {
                            "verified component": "processed daily data",
                            "files": 1,
                            "status": "pinned source and digest verified",
                        },
                        {
                            "verified component": "reference result bundle",
                            "files": len(verified_run_files),
                            "status": "all SHA-256 entries verified",
                        },
                        {
                            "verified component": "published evidence figures",
                            "files": len(verified_figures),
                            "status": "all SHA-256 entries verified",
                        },
                    ],
                    ("verified component", "files", "status"),
                    max_rows=3,
                )
            )
            ''',
        ),
        _markdown(
            "context-heading",
            """
            ## Context & Methods

            The study forecasts the next close-to-close daily Bitcoin log return from a
            30-day window of price-derived features available by the forecast origin.
            Calendar-based expanding folds keep training, validation, and test periods
            strictly ordered. The released LSTM is an ensemble of three deterministic CPU
            seeds and is compared with zero-return, historical-mean, and ridge baselines.

            Forecast MAE is the primary evidence. A long/flat signal simulation is shown
            only as secondary, idealized sensitivity analysis with explicit turnover costs;
            it omits market impact, taxes, custody, and outages. No cell below fits a model.
            """,
        ),
        _code(
            "method-summary",
            r"""
            experiment = run_manifest["experiment"]
            experiment_settings = experiment["experiment"]
            lstm_settings = experiment["lstm"]
            feature_details = run_manifest["features"]
            layer_count = lstm_settings["layers"]
            hidden_size = lstm_settings["hidden_size"]
            seed_values = experiment_settings["seeds"]
            design_rows = [
                {"item": "target", "released setting": metrics["forecast_target"]},
                {
                    "item": "data window",
                    "released setting": (
                        f"{experiment_settings['start_date']} to {experiment_settings['end_date']}"
                    ),
                },
                {
                    "item": "lookback / horizon",
                    "released setting": (
                        f"{experiment_settings['lookback']} / {experiment_settings['horizon']} days"
                    ),
                },
                {"item": "features", "released setting": feature_details["names"]},
                {
                    "item": "test folds",
                    "released setting": (
                        f"annual {experiment_settings['first_test_year']}-"
                        f"{experiment_settings['last_test_year']}"
                    ),
                },
                {
                    "item": "validation",
                    "released setting": (
                        f"{experiment_settings['validation_years']} calendar year before each test"
                    ),
                },
                {
                    "item": "LSTM",
                    "released setting": (
                        f"{layer_count} layer, hidden size {hidden_size}, seeds {seed_values}"
                    ),
                },
                {
                    "item": "primary transaction cost",
                    "released setting": (
                        f"{experiment_settings['transaction_cost_bps']:g} bps per one-way turnover"
                    ),
                },
            ]
            display(bounded_table(design_rows, ("item", "released setting"), max_rows=10))
            """,
        ),
        _markdown(
            "data-heading",
            """
            ## Data

            The maintained analysis uses only the Coin Metrics Community Data daily
            `PriceUSD` series. The source commit, raw file digest, transformation, output
            digest, coverage, attribution, and separate CC BY-NC 4.0 terms are pinned in
            `data/source-manifest.json`. No legacy macro, ETF, search-trend, or proprietary
            vendor columns enter this study.
            """,
        ),
        _code(
            "data-provenance",
            r"""
            source = source_manifest["source"]
            provenance_rows = [
                {"field": "attribution", "value": source["attribution"]},
                {"field": "license", "value": source["license"]},
                {"field": "timezone", "value": source["timezone"]},
                {"field": "timestamp convention", "value": source["timestamp_convention"]},
                {"field": "source repository", "value": source["repository"]},
                {"field": "pinned commit", "value": source["commit"]},
                {"field": "raw SHA-256", "value": source["sha256"]},
                {"field": "processed SHA-256", "value": processed["sha256"]},
                {"field": "processed rows", "value": processed["rows"]},
                {
                    "field": "processed range",
                    "value": f"{processed['start_date']} to {processed['end_date']}",
                },
            ]
            display(bounded_table(provenance_rows, ("field", "value"), max_rows=10))

            price_preview = prices[:3] + prices[-3:]
            display(Markdown("### Bounded processed-data preview"))
            display(bounded_table(price_preview, ("date", "close_usd"), max_rows=6))
            """,
        ),
        _code(
            "fold-boundaries",
            r"""
            display(Markdown("### Released walk-forward boundaries"))
            display(
                bounded_table(
                    folds,
                    (
                        "test_year",
                        "train_start",
                        "train_end",
                        "validation_start",
                        "validation_end",
                        "test_start",
                        "test_end",
                        "test_samples",
                    ),
                    max_rows=12,
                )
            )
            """,
        ),
        _markdown(
            "results-heading",
            """
            ## Results

            Lower MAE and RMSE are better. Directional accuracy is descriptive, and the
            MAE ratio compares each model with the zero-return baseline. The final 2025
            test fold and seed-level results are shown separately to prevent the aggregate
            from hiding instability. Tables intentionally cap their row counts.
            """,
        ),
        _code(
            "forecast-results",
            r"""
            forecast_rows = []
            for model_name, values in models.items():
                forecast_rows.append(
                    {
                        "model": model_name,
                        "observations": values["observations"],
                        "MAE": float(values["mae"]),
                        "RMSE": float(values["rmse"]),
                        "directional accuracy": values["directional_accuracy"],
                        "directional coverage": float(values["directional_coverage"]),
                        "MAE ratio vs zero": float(values["mae_ratio_vs_zero"]),
                    }
                )
            forecast_rows.sort(key=lambda row: row["MAE"])
            display(Markdown("### Combined out-of-sample forecast metrics"))
            display(
                bounded_table(
                    forecast_rows,
                    (
                        "model",
                        "observations",
                        "MAE",
                        "RMSE",
                        "directional accuracy",
                        "directional coverage",
                        "MAE ratio vs zero",
                    ),
                    max_rows=8,
                )
            )

            comparison_rows = []
            for key, value in comparison.items():
                comparison_rows.append({"comparison field": key, "released value": value})
            display(Markdown("### Baseline comparison and uncertainty"))
            display(
                bounded_table(
                    comparison_rows,
                    ("comparison field", "released value"),
                    max_rows=10,
                )
            )

            lstm_fold_rows = [row for row in fold_metrics if row["model"] == "lstm"]
            display(Markdown("### LSTM performance by annual test fold"))
            display(
                bounded_table(
                    lstm_fold_rows,
                    (
                        "test_year",
                        "observations",
                        "mae",
                        "rmse",
                        "directional_accuracy",
                        "directional_coverage",
                        "mae_ratio_vs_zero",
                    ),
                    max_rows=12,
                )
            )
            """,
        ),
        _code(
            "final-year-and-seeds",
            r"""
            final_year = metrics["final_year"]
            final_year_rows = [
                {
                    "year": final_year["year"],
                    "model": model_name,
                    "MAE": float(values["mae"]),
                    "RMSE": float(values["rmse"]),
                    "directional accuracy": values["directional_accuracy"],
                    "directional coverage": float(values["directional_coverage"]),
                }
                for model_name, values in final_year["models"].items()
            ]
            final_year_rows.sort(key=lambda row: row["MAE"])
            display(Markdown(f"### Final {final_year['year']} test fold"))
            display(
                bounded_table(
                    final_year_rows,
                    (
                        "year",
                        "model",
                        "MAE",
                        "RMSE",
                        "directional accuracy",
                        "directional coverage",
                    ),
                    max_rows=8,
                )
            )

            display(Markdown("### LSTM seed sensitivity"))
            display(
                bounded_table(
                    seed_metrics,
                    (
                        "seed",
                        "observations",
                        "mae",
                        "rmse",
                        "directional_accuracy",
                        "directional_coverage",
                    ),
                    max_rows=6,
                )
            )
            """,
        ),
        _code(
            "signal-sensitivity",
            r"""
            primary_cost = float(metrics["signal_simulation"]["primary_cost_bps"])
            primary_signal_rows = []
            for row in signal_metrics:
                if float(row["cost_bps"]) == primary_cost:
                    primary_signal_rows.append(row)
            display(Markdown("### Secondary signal simulation at the stated primary cost"))
            display(
                bounded_table(
                    primary_signal_rows,
                    (
                        "model",
                        "cost_bps",
                        "total_return",
                        "cagr",
                        "annualized_volatility",
                        "sharpe_zero_rate",
                        "max_drawdown",
                        "turnover",
                        "exposure",
                    ),
                    max_rows=8,
                )
            )
            display(
                Markdown(
                    "This simplified long/flat simulation is included to expose cost and "
                    "execution assumptions. It is not a backtest suitable for deployment and "
                    "is not an investment recommendation."
                )
            )

            sensitivity_rows = [row for row in signal_metrics if row["model"] in {"ridge", "lstm"}]
            display(Markdown("### Ridge and LSTM transaction-cost sensitivity"))
            display(
                bounded_table(
                    sensitivity_rows,
                    (
                        "model",
                        "cost_bps",
                        "total_return",
                        "cagr",
                        "max_drawdown",
                        "turnover",
                    ),
                    max_rows=10,
                )
            )
            """,
        ),
        _code(
            "evidence-figures",
            r"""
            figures = (
                ("forecast_mae.svg", "Forecast MAE"),
                ("forecast_scatter.svg", "Predicted versus realized returns"),
                ("signal_equity.svg", "Illustrative signal equity"),
                ("walk_forward_folds.svg", "Walk-forward fold design"),
            )
            for filename, title in figures:
                display(Markdown(f"### {title}"))
                display(SVG(filename=str(FIGURE_DIRECTORY / filename)))
            """,
        ),
        _markdown(
            "takeaways-heading",
            """
            ## Takeaways

            The value of this project is the auditable evaluation design: pinned data,
            origin-time features, chronological folds, strong simple baselines, seed
            sensitivity, checksum-bound evidence, and explicit limits. The final cell
            states the result from the verified bundle without turning historical error
            differences or an idealized simulation into a profitability claim.
            """,
        ),
        _code(
            "takeaway-verdict",
            r"""
            lower, upper = (float(value) for value in interval)
            crosses_zero = lower <= 0.0 <= upper
            baseline_name = comparison["best_baseline"]
            if comparison["beats_best_baseline"]:
                ranking_sentence = f"LSTM aggregate MAE is lower than `{baseline_name}`."
            else:
                ranking_sentence = f"LSTM does not beat `{baseline_name}` on aggregate MAE."
            uncertainty_sentence = (
                "The block-bootstrap interval crosses zero, so the observed MAE ordering is "
                "not cleanly separated under this uncertainty check."
                if crosses_zero
                else "The block-bootstrap interval does not cross zero for this frozen sample."
            )
            display(
                Markdown(
                    f"- {ranking_sentence}\n"
                    f"- {uncertainty_sentence}\n"
                    "- The 2025 test fold and three seed runs should be considered alongside the "
                    "aggregate, not treated as footnotes.\n"
                    "- The signal simulation is secondary and idealized; none of these outputs "
                    "establish future predictability or justify a live trading decision.\n"
                    "- Reproduction belongs in the CLI pipeline. This notebook remains a "
                    "checksum-verifying presentation layer and performs no retraining."
                )
            )
            """,
        ),
    ]
    nbformat.validate(notebook)
    _validate_notebook_contract(notebook)
    return notebook


def _section_headings(notebook: NotebookNode) -> tuple[str, ...]:
    headings: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        headings.extend(
            line.strip() for line in cell.source.splitlines() if line.strip().startswith("## ")
        )
    return tuple(headings)


def _validate_notebook_contract(notebook: NotebookNode) -> None:
    if _section_headings(notebook) != SECTION_HEADINGS:
        raise ValueError("notebook section order or names changed")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"<notebook:{cell.get('id', 'code')}>", "exec")
    source = "\n".join(cell.source for cell in notebook.cells)
    for filename in (
        "forecast_mae.svg",
        "forecast_scatter.svg",
        "signal_equity.svg",
        "walk_forward_folds.svg",
    ):
        if filename not in source:
            raise ValueError(f"notebook does not display {filename}")
    forbidden = ("run_experiment", "TorchLSTM", ".fit(")
    if any(token in source for token in forbidden):
        raise ValueError("reference notebook must not contain training code")


def _source_signature(notebook: NotebookNode) -> dict[str, Any]:
    return {
        "sections": _section_headings(notebook),
        "cells": [
            {
                "cell_type": cell.cell_type,
                "id": cell.get("id"),
                "source": cell.source,
            }
            for cell in notebook.cells
        ],
    }


def write_notebook(notebook: NotebookNode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)


def check_notebook(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = build_notebook()
    actual = nbformat.read(path, as_version=4)
    nbformat.validate(actual)
    _validate_notebook_contract(actual)
    if _source_signature(actual) != _source_signature(expected):
        raise ValueError(f"{path} was not generated from {Path(__file__).name}")


def execute_notebook(notebook: NotebookNode) -> NotebookNode:
    """Execute the read-only notebook after the reference evidence exists."""

    from nbclient import NotebookClient

    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=180,
        allow_errors=False,
        record_timing=False,
    )
    client.execute(cwd=str(REPOSITORY_ROOT))
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="validate generated source without writing"
    )
    mode.add_argument("--execute", action="store_true", help="execute against reviewed evidence")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.check:
        check_notebook(output)
        print(f"Notebook source is current: {output}")
        return 0

    notebook = build_notebook()
    if args.execute:
        notebook = execute_notebook(notebook)
    write_notebook(notebook, output)
    mode = "executed" if args.execute else "unexecuted"
    print(f"Wrote {mode} notebook: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
