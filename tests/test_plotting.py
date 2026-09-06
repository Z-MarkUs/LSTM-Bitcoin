from __future__ import annotations

from pathlib import Path

import pytest

from lstm_bitcoin.artifacts import verify_checksums, write_checksums, write_csv, write_json
from lstm_bitcoin.plotting import PLOT_FILES, _scatter_plot, generate_plots


def write_plot_inputs(run: Path) -> None:
    metrics = {
        "out_of_sample_period": {
            "start": "2020-01-01",
            "end": "2020-01-03",
            "observations": 3,
        },
        "models": {
            "zero_return": {"mae": 0.02},
            "historical_mean": {"mae": 0.019},
            "ridge": {"mae": 0.018},
            "lstm": {"mae": 0.017},
        },
        "signal_simulation": {"primary_cost_bps": 10},
    }
    predictions = []
    for model, values in {
        "ridge": [0.005, -0.008, 0.012],
        "lstm": [0.006, -0.007, 0.011],
    }.items():
        for day, actual, predicted in zip(
            ("2020-01-01", "2020-01-02", "2020-01-03"),
            (0.01, -0.02, 0.015),
            values,
            strict=True,
        ):
            predictions.append(
                {
                    "target_date": day,
                    "model": model,
                    "actual_log_return": actual,
                    "predicted_log_return": predicted,
                }
            )
    simulations = []
    for model, equities in {
        "buy_and_hold": [1.01, 0.99, 1.02],
        "ridge": [1.005, 1.004, 1.015],
        "lstm": [1.006, 1.005, 1.016],
    }.items():
        for day, equity in zip(("2020-01-01", "2020-01-02", "2020-01-03"), equities, strict=True):
            simulations.append({"target_date": day, "model": model, "equity": equity})
    folds = [
        {
            "train_start": "2017-01-01",
            "train_end": "2018-12-31",
            "validation_start": "2019-01-01",
            "validation_end": "2019-12-31",
            "test_start": "2020-01-01",
            "test_end": "2020-12-31",
            "test_year": 2020,
        }
    ]
    run.mkdir(parents=True)
    write_json(run / "metrics.json", metrics)
    write_csv(
        run / "predictions.csv",
        fieldnames=("target_date", "model", "actual_log_return", "predicted_log_return"),
        rows=predictions,
    )
    write_csv(
        run / "signal_simulation.csv",
        fieldnames=("target_date", "model", "equity"),
        rows=simulations,
    )
    write_csv(
        run / "folds.csv",
        fieldnames=(
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
            "test_year",
        ),
        rows=folds,
    )
    write_checksums(
        run,
        ("metrics.json", "predictions.csv", "signal_simulation.csv", "folds.csv"),
    )


def test_generate_plots_is_verified_complete_and_deterministic(tmp_path: Path) -> None:
    run = tmp_path / "run"
    output = tmp_path / "plots"
    write_plot_inputs(run)

    generated = generate_plots(run, output)
    assert {path.name for path in generated} == set(PLOT_FILES) - {"SHA256SUMS"}
    first_payloads = {path.name: path.read_bytes() for path in generated}
    assert set(verify_checksums(output)) == set(first_payloads)
    for payload in first_payloads.values():
        assert payload.startswith(b"<?xml")

    with pytest.raises(FileExistsError, match="--force"):
        generate_plots(run, output)
    regenerated = generate_plots(run, output, force=True)
    assert {path.name: path.read_bytes() for path in regenerated} == first_payloads


def test_scatter_plot_falls_back_to_ridge(tmp_path: Path) -> None:
    rows = [
        {
            "model": "ridge",
            "actual_log_return": "0.01",
            "predicted_log_return": "0.005",
        },
        {
            "model": "ridge",
            "actual_log_return": "-0.01",
            "predicted_log_return": "-0.005",
        },
    ]
    _scatter_plot(rows, tmp_path)
    assert (tmp_path / "forecast_scatter.svg").is_file()
