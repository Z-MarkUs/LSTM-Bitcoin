"""End-to-end expanding-window evaluation and artifact generation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess  # nosec B404
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from . import __version__
from .artifacts import (
    prepare_output_directory,
    sha256_path,
    write_checksums,
    write_csv,
    write_json,
)
from .backtest import Simulation, simulate_buy_and_hold, simulate_long_flat
from .config import ProjectConfig
from .data import load_price_series, validate_reference_data
from .features import SupervisedData, build_supervised
from .metrics import block_bootstrap_mae_difference, forecast_metrics
from .models.base import ForecastModel
from .models.baselines import HistoricalMean, RidgeReturn, ZeroReturn
from .models.lstm import TorchLSTM
from .splits import Fold, annual_expanding_splits

ModelSet = Literal["baselines", "all"]

RUN_FILES = (
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
)

PREDICTION_FIELDS = (
    "fold",
    "test_year",
    "origin_date",
    "target_date",
    "model",
    "actual_log_return",
    "predicted_log_return",
)

METRIC_FIELDS = (
    "fold",
    "test_year",
    "model",
    "seed",
    "observations",
    "mae",
    "rmse",
    "mean_error",
    "directional_accuracy",
    "directional_coverage",
    "pearson_correlation",
    "mae_ratio_vs_zero",
)


def _git_state(repository: Path, revision: str | None) -> dict[str, Any]:
    if revision is not None:
        return {"revision": revision, "dirty": None}
    git_executable = shutil.which("git")
    if git_executable is None:
        return {"revision": "unavailable", "dirty": None}
    try:
        # The resolved executable and every argument are fixed; shell expansion is disabled.
        head = subprocess.run(  # nosec B603
            [git_executable, "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(  # nosec B603
            [git_executable, "status", "--porcelain=v1"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"revision": head, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "dirty": None}


def _dependency_versions(include_lstm: bool) -> dict[str, str | None]:
    names = ["numpy", "matplotlib"]
    if include_lstm:
        names.append("torch")
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _dates(values: npt.NDArray[np.datetime64]) -> list[str]:
    return [str(value) for value in values]


def _model_predictions(
    data: SupervisedData,
    fold: Fold,
    config: ProjectConfig,
    *,
    include_lstm: bool,
) -> tuple[
    dict[str, npt.NDArray[np.float64]],
    dict[int, npt.NDArray[np.float64]],
    list[dict[str, object]],
]:
    prior = np.concatenate((fold.train, fold.validation))
    predictions: dict[str, npt.NDArray[np.float64]] = {}
    baseline_models: dict[str, ForecastModel] = {
        "zero_return": ZeroReturn(),
        "historical_mean": HistoricalMean(),
        "ridge": RidgeReturn(alpha=config.ridge.alpha),
    }
    for name, baseline_model in baseline_models.items():
        baseline_model.fit(data.features[prior], data.targets[prior])
        predictions[name] = baseline_model.predict(data.features[fold.test])

    seed_predictions: dict[int, npt.NDArray[np.float64]] = {}
    training_rows: list[dict[str, object]] = []
    if include_lstm:
        for seed in config.experiment.seeds:
            lstm_model = TorchLSTM(config.lstm, seed=seed)
            lstm_model.fit(
                data.features[fold.train],
                data.targets[fold.train],
                data.features[fold.validation],
                data.targets[fold.validation],
            )
            seed_predictions[seed] = lstm_model.predict(data.features[fold.test])
            if lstm_model.summary is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("LSTM completed without a training summary")
            training_rows.append(
                {
                    "fold": fold.number,
                    "test_year": fold.test_year,
                    "seed": seed,
                    **asdict(lstm_model.summary),
                }
            )
        predictions["lstm"] = np.mean(np.stack(list(seed_predictions.values())), axis=0)
    return predictions, seed_predictions, training_rows


def _metric_row(
    *,
    fold: int | str,
    test_year: int | str,
    model: str,
    seed: int | str,
    actual: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
    zero_mae: float,
) -> dict[str, object]:
    return {
        "fold": fold,
        "test_year": test_year,
        "model": model,
        "seed": seed,
        **forecast_metrics(actual, predicted, zero_mae=zero_mae).as_dict(),
    }


def _simulation_rows(
    data: SupervisedData,
    predictions_by_model: dict[str, npt.NDArray[np.float64]],
    test_indices: npt.NDArray[np.int64],
    fold_numbers: npt.NDArray[np.int64],
    test_years: npt.NDArray[np.int64],
    *,
    cost_bps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    actual = data.targets[test_indices]
    origins = data.origin_dates[test_indices]
    dates = data.target_dates[test_indices]
    if len(fold_numbers) != len(actual) or len(test_years) != len(actual):
        raise ValueError("simulation fold labels must align with test observations")
    simulation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    simulations: dict[str, Simulation] = {
        name: simulate_long_flat(actual, predicted, transaction_cost_bps=cost_bps)
        for name, predicted in predictions_by_model.items()
    }
    simulations["buy_and_hold"] = simulate_buy_and_hold(actual, transaction_cost_bps=cost_bps)
    for model_name, simulation in simulations.items():
        summary_rows.append(
            {
                "model": model_name,
                "cost_bps": cost_bps,
                **simulation.metrics.as_dict(),
            }
        )
        predicted = predictions_by_model.get(model_name)
        for index, target_date in enumerate(dates):
            simulation_rows.append(
                {
                    "fold": fold_numbers[index],
                    "test_year": test_years[index],
                    "origin_date": str(origins[index]),
                    "target_date": str(target_date),
                    "model": model_name,
                    "cost_bps": cost_bps,
                    "actual_log_return": actual[index],
                    "predicted_log_return": "" if predicted is None else predicted[index],
                    "position": simulation.positions[index],
                    "turnover": simulation.turnover[index],
                    "gross_simple_return": simulation.gross_simple_returns[index],
                    "cost": simulation.costs[index],
                    "net_simple_return": simulation.net_simple_returns[index],
                    "equity": simulation.equity[index],
                }
            )
    return simulation_rows, summary_rows


def run_experiment(
    *,
    config: ProjectConfig,
    config_path: Path,
    data_path: Path,
    source_manifest_path: Path,
    output: Path,
    models: ModelSet = "all",
    force: bool = False,
    revision: str | None = None,
) -> dict[str, Any]:
    """Run all calendar folds and write a self-checking result bundle."""

    if models not in {"baselines", "all"}:
        raise ValueError("models must be 'baselines' or 'all'")
    include_lstm = models == "all"
    validate_reference_data(data_path, source_manifest_path)
    series = load_price_series(data_path)
    expected_start = np.datetime64(config.experiment.start_date)
    expected_end = np.datetime64(config.experiment.end_date)
    if series.dates[0] != expected_start or series.dates[-1] != expected_end:
        raise ValueError("data date range does not match the experiment configuration")
    supervised = build_supervised(
        series,
        lookback=config.experiment.lookback,
        horizon=config.experiment.horizon,
    )
    folds = annual_expanding_splits(
        supervised.target_dates,
        first_test_year=config.experiment.first_test_year,
        last_test_year=config.experiment.last_test_year,
        validation_years=config.experiment.validation_years,
    )
    repository = config_path.resolve().parent.parent
    code_state = _git_state(repository, revision)
    prepare_output_directory(output, force=force, known_files=RUN_FILES)

    prediction_rows: list[dict[str, object]] = []
    seed_prediction_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    combined_predictions: dict[str, list[npt.NDArray[np.float64]]] = {}
    combined_seed_predictions: dict[int, list[npt.NDArray[np.float64]]] = {
        seed: [] for seed in config.experiment.seeds
    }
    combined_actual: list[npt.NDArray[np.float64]] = []
    combined_test: list[npt.NDArray[np.int64]] = []
    combined_fold_numbers: list[npt.NDArray[np.int64]] = []
    combined_test_years: list[npt.NDArray[np.int64]] = []
    folds_rows: list[dict[str, object]] = []

    for fold in folds:
        predictions, seed_predictions, diagnostics = _model_predictions(
            supervised,
            fold,
            config,
            include_lstm=include_lstm,
        )
        actual = supervised.targets[fold.test]
        zero_mae = float(np.mean(np.abs(actual)))
        origin_dates = _dates(supervised.origin_dates[fold.test])
        target_dates = _dates(supervised.target_dates[fold.test])
        for model_name, predicted in predictions.items():
            combined_predictions.setdefault(model_name, []).append(predicted)
            fold_metric_rows.append(
                _metric_row(
                    fold=fold.number,
                    test_year=fold.test_year,
                    model=model_name,
                    seed="",
                    actual=actual,
                    predicted=predicted,
                    zero_mae=zero_mae,
                )
            )
            for index in range(len(actual)):
                prediction_rows.append(
                    {
                        "fold": fold.number,
                        "test_year": fold.test_year,
                        "origin_date": origin_dates[index],
                        "target_date": target_dates[index],
                        "model": model_name,
                        "actual_log_return": actual[index],
                        "predicted_log_return": predicted[index],
                    }
                )
        for seed, predicted in seed_predictions.items():
            combined_seed_predictions[seed].append(predicted)
            for index in range(len(actual)):
                seed_prediction_rows.append(
                    {
                        "fold": fold.number,
                        "test_year": fold.test_year,
                        "origin_date": origin_dates[index],
                        "target_date": target_dates[index],
                        "model": "lstm",
                        "seed": seed,
                        "actual_log_return": actual[index],
                        "predicted_log_return": predicted[index],
                    }
                )
        training_rows.extend(diagnostics)
        combined_actual.append(actual)
        combined_test.append(fold.test)
        combined_fold_numbers.append(np.full(len(fold.test), fold.number, dtype=np.int64))
        combined_test_years.append(np.full(len(fold.test), fold.test_year, dtype=np.int64))
        folds_rows.append(
            {
                "fold": fold.number,
                "test_year": fold.test_year,
                "train_start": str(supervised.target_dates[fold.train[0]]),
                "train_end": str(supervised.target_dates[fold.train[-1]]),
                "train_samples": len(fold.train),
                "validation_start": str(supervised.target_dates[fold.validation[0]]),
                "validation_end": str(supervised.target_dates[fold.validation[-1]]),
                "validation_samples": len(fold.validation),
                "test_start": str(supervised.target_dates[fold.test[0]]),
                "test_end": str(supervised.target_dates[fold.test[-1]]),
                "test_samples": len(fold.test),
            }
        )

    actual_oos = np.concatenate(combined_actual)
    test_indices = np.concatenate(combined_test)
    fold_numbers = np.concatenate(combined_fold_numbers)
    test_years = np.concatenate(combined_test_years)
    if len(np.unique(test_indices)) != len(test_indices):
        raise RuntimeError("released out-of-sample target dates are not unique")
    prediction_vectors = {
        name: np.concatenate(parts) for name, parts in combined_predictions.items()
    }
    zero_mae = float(np.mean(np.abs(actual_oos)))
    overall_metrics = {
        name: forecast_metrics(actual_oos, predicted, zero_mae=zero_mae).as_dict()
        for name, predicted in prediction_vectors.items()
    }

    seed_metric_rows: list[dict[str, object]] = []
    if include_lstm:
        for seed in config.experiment.seeds:
            predicted = np.concatenate(combined_seed_predictions[seed])
            seed_metric_rows.append(
                _metric_row(
                    fold="all",
                    test_year=f"{folds[0].test_year}-{folds[-1].test_year}",
                    model="lstm",
                    seed=seed,
                    actual=actual_oos,
                    predicted=predicted,
                    zero_mae=zero_mae,
                )
            )

    baseline_names = ("zero_return", "historical_mean", "ridge")
    best_baseline = min(baseline_names, key=lambda name: cast(float, overall_metrics[name]["mae"]))
    comparison: dict[str, Any] = {"best_baseline": best_baseline}
    if include_lstm:
        point, lower, upper = block_bootstrap_mae_difference(
            actual_oos,
            prediction_vectors["lstm"],
            prediction_vectors[best_baseline],
        )
        comparison.update(
            {
                "lstm_minus_best_baseline_mae": point,
                "block_bootstrap_95pct_interval": [lower, upper],
                "block_size_days": 30,
                "bootstrap_resamples": 2_000,
                "beats_best_baseline": cast(float, overall_metrics["lstm"]["mae"])
                < cast(float, overall_metrics[best_baseline]["mae"]),
            }
        )

    primary_cost = config.experiment.transaction_cost_bps
    simulation_rows, primary_signal_metrics = _simulation_rows(
        supervised,
        prediction_vectors,
        test_indices,
        fold_numbers,
        test_years,
        cost_bps=primary_cost,
    )
    signal_metric_rows: list[dict[str, object]] = []
    for cost_bps in (0.0, 5.0, 10.0, 25.0, 50.0):
        _, summaries = _simulation_rows(
            supervised,
            prediction_vectors,
            test_indices,
            fold_numbers,
            test_years,
            cost_bps=cost_bps,
        )
        signal_metric_rows.extend(summaries)

    final_year = config.experiment.last_test_year
    final_indices = folds[-1].test
    final_actual = supervised.targets[final_indices]
    final_zero_mae = float(np.mean(np.abs(final_actual)))
    final_year_metrics = {
        name: forecast_metrics(
            final_actual,
            combined_predictions[name][-1],
            zero_mae=final_zero_mae,
        ).as_dict()
        for name in prediction_vectors
    }

    metrics_document: dict[str, Any] = {
        "schema_version": 1,
        "forecast_target": "next-day close-to-close log return",
        "out_of_sample_period": {
            "start": str(supervised.target_dates[test_indices[0]]),
            "end": str(supervised.target_dates[test_indices[-1]]),
            "observations": len(test_indices),
        },
        "models": overall_metrics,
        "comparison": comparison,
        "final_year": {"year": final_year, "models": final_year_metrics},
        "signal_simulation": {
            "role": "secondary illustrative analysis; not a trading recommendation",
            "primary_cost_bps": primary_cost,
            "position_rule": "long for positive origin-time forecast, otherwise flat",
            "execution": "position formed after origin close and applied only to the next close",
            "annualization_days": 365,
            "primary_metrics": primary_signal_metrics,
            "cost_sensitivity_bps": [0, 5, 10, 25, 50],
        },
    }

    write_csv(output / "predictions.csv", fieldnames=PREDICTION_FIELDS, rows=prediction_rows)
    write_csv(
        output / "seed_predictions.csv",
        fieldnames=(*PREDICTION_FIELDS, "seed"),
        rows=seed_prediction_rows,
    )
    write_csv(output / "fold_metrics.csv", fieldnames=METRIC_FIELDS, rows=fold_metric_rows)
    write_csv(output / "seed_metrics.csv", fieldnames=METRIC_FIELDS, rows=seed_metric_rows)
    write_csv(
        output / "folds.csv",
        fieldnames=(
            "fold",
            "test_year",
            "train_start",
            "train_end",
            "train_samples",
            "validation_start",
            "validation_end",
            "validation_samples",
            "test_start",
            "test_end",
            "test_samples",
        ),
        rows=folds_rows,
    )
    write_csv(
        output / "training.csv",
        fieldnames=(
            "fold",
            "test_year",
            "seed",
            "best_epoch",
            "epochs_ran",
            "best_validation_loss",
            "final_training_loss",
        ),
        rows=training_rows,
    )
    write_csv(
        output / "signal_metrics.csv",
        fieldnames=(
            "model",
            "cost_bps",
            "observations",
            "total_return",
            "cagr",
            "annualized_volatility",
            "sharpe_zero_rate",
            "max_drawdown",
            "turnover",
            "exposure",
            "trades",
        ),
        rows=signal_metric_rows,
    )
    write_csv(
        output / "signal_simulation.csv",
        fieldnames=(
            "fold",
            "test_year",
            "origin_date",
            "target_date",
            "model",
            "cost_bps",
            "actual_log_return",
            "predicted_log_return",
            "position",
            "turnover",
            "gross_simple_return",
            "cost",
            "net_simple_return",
            "equity",
        ),
        rows=simulation_rows,
    )
    write_json(output / "metrics.json", metrics_document)

    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    artifact_names = (
        "fold_metrics.csv",
        "folds.csv",
        "metrics.json",
        "predictions.csv",
        "seed_metrics.csv",
        "seed_predictions.csv",
        "signal_metrics.csv",
        "signal_simulation.csv",
        "training.csv",
    )
    artifact_hashes = {name: sha256_path(output / name) for name in artifact_names}
    command = [
        "lstm-bitcoin",
        "reproduce",
        "--config",
        config_path.as_posix(),
        "--data",
        data_path.as_posix(),
        "--manifest",
        source_manifest_path.as_posix(),
        "--output",
        output.as_posix(),
        "--models",
        models,
    ]
    if revision is not None:
        command.extend(("--revision", revision))
    manifest_document: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config.as_dict(),
        "command": command,
        "code": {"package_version": __version__, **code_state},
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": _dependency_versions(include_lstm),
        },
        "inputs": {
            "config_path": config_path.as_posix(),
            "config_sha256": sha256_path(config_path),
            "data_path": data_path.as_posix(),
            "data_sha256": sha256_path(data_path),
            "source_manifest_path": source_manifest_path.as_posix(),
            "source_manifest_sha256": sha256_path(source_manifest_path),
            "source": source_manifest.get("source"),
            "processed": source_manifest.get("processed"),
        },
        "features": {
            "names": list(supervised.feature_names),
            "policy": "every feature window ends at the origin; targets occur later",
        },
        "folds": folds_rows,
        "artifacts": artifact_hashes,
        "limitations": [
            "historical observational study; no evidence of future profitability",
            "single reference price series with idealized close-to-close execution",
            "signal simulation omits market impact, taxes, custody, and outages",
            "Coin Metrics data is separately licensed CC BY-NC 4.0",
        ],
    }
    write_json(output / "manifest.json", manifest_document)
    checksum_names = (*artifact_names, "manifest.json")
    write_checksums(output, checksum_names)
    return metrics_document
