from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import deterministic_price_series

from lstm_bitcoin import data as data_module
from lstm_bitcoin import experiment as experiment_module
from lstm_bitcoin.artifacts import verify_checksums
from lstm_bitcoin.config import load_config
from lstm_bitcoin.data import sha256_file
from lstm_bitcoin.experiment import RUN_FILES, run_experiment
from lstm_bitcoin.models.lstm import TrainingSummary


def write_experiment_inputs(tmp_path: Path) -> tuple[Path, Path]:
    series = deterministic_price_series(start=date(2015, 1, 1), days=365 * 6 + 2)
    data_path = tmp_path / "daily.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["date", "close_usd"])
        writer.writerows(zip(series.dates.astype(str), series.close_usd, strict=True))
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "repository": data_module.SOURCE_REPOSITORY,
                    "commit": data_module.SOURCE_COMMIT,
                    "path": data_module.SOURCE_PATH,
                    "url": data_module.SOURCE_URL,
                    "sha256": data_module.SOURCE_SHA256,
                    "license": data_module.SOURCE_LICENSE,
                    "attribution": data_module.SOURCE_ATTRIBUTION,
                    "timezone": data_module.SOURCE_TIMEZONE,
                    "timestamp_convention": data_module.SOURCE_TIMESTAMP_CONVENTION,
                },
                "processed": {
                    "sha256": sha256_file(data_path),
                    "rows": len(series.dates),
                    "start_date": str(series.dates[0]),
                    "end_date": str(series.dates[-1]),
                },
            }
        ),
        encoding="utf-8",
    )
    return data_path, manifest_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_baseline_pipeline_writes_reproducible_unique_oos_bundle(tmp_path: Path) -> None:
    config_path = Path("configs/ci.toml")
    config = load_config(config_path)
    data_path, source_manifest = write_experiment_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    metrics = run_experiment(
        config=config,
        config_path=config_path,
        data_path=data_path,
        source_manifest_path=source_manifest,
        output=first,
        models="baselines",
        revision="test-revision",
    )
    run_experiment(
        config=config,
        config_path=config_path,
        data_path=data_path,
        source_manifest_path=source_manifest,
        output=second,
        models="baselines",
        revision="test-revision",
    )

    assert set(verify_checksums(first)) == set(RUN_FILES) - {"SHA256SUMS"}
    assert metrics["final_year"]["year"] == 2019
    assert metrics["comparison"]["best_baseline"] in {
        "zero_return",
        "historical_mean",
        "ridge",
    }
    assert "lstm_minus_best_baseline_mae" not in metrics["comparison"]

    for filename in set(RUN_FILES) - {"manifest.json", "SHA256SUMS"}:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    predictions = read_csv(first / "predictions.csv")
    models = {row["model"] for row in predictions}
    assert models == {"zero_return", "historical_mean", "ridge"}
    for model in models:
        dates = [row["target_date"] for row in predictions if row["model"] == model]
        assert dates == sorted(dates)
        assert len(dates) == len(set(dates))

    folds = read_csv(first / "folds.csv")
    assert [row["test_year"] for row in folds] == ["2018", "2019"]
    for row in folds:
        assert row["train_end"] < row["validation_start"]
        assert row["validation_end"] < row["test_start"]
    simulation = read_csv(first / "signal_simulation.csv")
    assert simulation
    assert set(simulation[0]) == {
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
    }
    assert all(row["origin_date"] < row["target_date"] for row in simulation)
    assert {row["test_year"] for row in simulation} == {"2018", "2019"}
    assert {row["cost_bps"] for row in simulation} == {"10"}

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"][-4:] == [
        "--models",
        "baselines",
        "--revision",
        "test-revision",
    ]
    assert read_csv(first / "seed_predictions.csv") == []
    assert read_csv(first / "seed_metrics.csv") == []
    assert read_csv(first / "training.csv") == []


def test_all_model_pipeline_records_seed_level_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTorchLSTM:
        def __init__(self, settings: object, *, seed: int) -> None:
            self.seed = seed
            self.summary: TrainingSummary | None = None

        def fit(
            self,
            train_features: object,
            train_targets: object,
            validation_features: object,
            validation_targets: object,
        ) -> FakeTorchLSTM:
            self.summary = TrainingSummary(
                best_epoch=1,
                epochs_ran=2,
                best_validation_loss=0.25,
                final_training_loss=0.5,
            )
            return self

        def predict(self, features: Any) -> Any:
            import numpy as np

            return np.full(len(features), self.seed * 1e-6, dtype=np.float64)

    monkeypatch.setattr(experiment_module, "TorchLSTM", FakeTorchLSTM)
    config_path = Path("configs/ci.toml")
    data_path, source_manifest = write_experiment_inputs(tmp_path)
    output = tmp_path / "run"

    metrics = run_experiment(
        config=load_config(config_path),
        config_path=config_path,
        data_path=data_path,
        source_manifest_path=source_manifest,
        output=output,
        models="all",
        revision="test-revision",
    )

    assert set(metrics["models"]) == {"zero_return", "historical_mean", "ridge", "lstm"}
    assert set(metrics["comparison"]) == {
        "best_baseline",
        "lstm_minus_best_baseline_mae",
        "block_bootstrap_95pct_interval",
        "block_size_days",
        "bootstrap_resamples",
        "beats_best_baseline",
    }
    training = read_csv(output / "training.csv")
    assert [(row["fold"], row["seed"]) for row in training] == [("1", "7"), ("2", "7")]
    seed_metrics = read_csv(output / "seed_metrics.csv")
    assert [(row["model"], row["seed"]) for row in seed_metrics] == [("lstm", "7")]
    seed_predictions = read_csv(output / "seed_predictions.csv")
    assert seed_predictions
    assert {row["seed"] for row in seed_predictions} == {"7"}
    assert {row["model"] for row in seed_predictions} == {"lstm"}


def test_pipeline_validates_model_choice_data_range_and_existing_output(tmp_path: Path) -> None:
    config_path = Path("configs/ci.toml")
    config = load_config(config_path)
    data_path, source_manifest = write_experiment_inputs(tmp_path)
    common: dict[str, Any] = {
        "config": config,
        "config_path": config_path,
        "data_path": data_path,
        "source_manifest_path": source_manifest,
        "output": tmp_path / "run",
        "revision": "x",
    }

    with pytest.raises(ValueError, match="models must be"):
        run_experiment(**common, models="unknown")  # type: ignore[arg-type]

    bad_config = load_config(config_path)
    bad_config = type(bad_config)(
        experiment=type(bad_config.experiment)(
            **{**bad_config.experiment.__dict__, "end_date": "2020-12-30"}
        ),
        ridge=bad_config.ridge,
        lstm=bad_config.lstm,
    )
    with pytest.raises(ValueError, match="date range"):
        run_experiment(**{**common, "config": bad_config}, models="baselines")

    run_experiment(**common, models="baselines")
    with pytest.raises(FileExistsError, match="--force"):
        run_experiment(**common, models="baselines")
    run_experiment(**common, models="baselines", force=True)


def test_public_pipeline_rejects_unvalidated_source_manifest(tmp_path: Path) -> None:
    config_path = Path("configs/ci.toml")
    data_path, source_manifest = write_experiment_inputs(tmp_path)
    source_manifest.write_text(
        json.dumps({"schema_version": 1, "source": {"commit": "tampered"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"source manifest|provenance"):
        run_experiment(
            config=load_config(config_path),
            config_path=config_path,
            data_path=data_path,
            source_manifest_path=source_manifest,
            output=tmp_path / "run",
            models="baselines",
            revision="x",
        )


def test_force_preserves_prior_run_when_inputs_are_invalid(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    prior_metrics = output / "metrics.json"
    prior_metrics.write_text("valuable prior run\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        run_experiment(
            config=load_config(Path("configs/ci.toml")),
            config_path=Path("configs/ci.toml"),
            data_path=tmp_path / "missing.csv",
            source_manifest_path=tmp_path / "missing.json",
            output=output,
            models="baselines",
            force=True,
            revision="x",
        )

    assert prior_metrics.read_text(encoding="utf-8") == "valuable prior run\n"


def test_git_state_and_dependency_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lstm_bitcoin import experiment

    assert experiment._git_state(tmp_path, "fixed") == {"revision": "fixed", "dirty": None}
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no git")),
    )
    assert experiment._git_state(tmp_path, None) == {"revision": "unavailable", "dirty": None}
    versions = experiment._dependency_versions(include_lstm=False)
    assert versions["numpy"] is not None
    assert "torch" not in versions
