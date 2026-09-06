from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lstm_bitcoin.config import (
    ExperimentSettings,
    LSTMSettings,
    ProjectConfig,
    RidgeSettings,
    load_config,
)


def valid_experiment(**overrides: object) -> ExperimentSettings:
    values: dict[str, object] = {
        "name": "test",
        "lookback": 30,
        "horizon": 1,
        "start_date": "2013-01-01",
        "end_date": "2025-12-31",
        "first_test_year": 2018,
        "last_test_year": 2025,
        "validation_years": 1,
        "transaction_cost_bps": 10.0,
        "seeds": (7, 17),
    }
    values.update(overrides)
    return ExperimentSettings(**values)  # type: ignore[arg-type]


def valid_lstm(**overrides: object) -> LSTMSettings:
    values: dict[str, object] = {
        "hidden_size": 8,
        "layers": 1,
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "patience": 2,
        "gradient_clip": 1.0,
    }
    values.update(overrides)
    return LSTMSettings(**values)  # type: ignore[arg-type]


def test_load_reference_config_and_serialize() -> None:
    config = load_config(Path("configs/reference.toml"))

    assert config.experiment.name == "reference-v1"
    assert config.experiment.seeds == (7, 17, 29)
    assert config.as_dict()["experiment"]["seeds"] == [7, 17, 29]
    assert config.as_dict()["lstm"]["hidden_size"] == 16


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "  "}, "must not be blank"),
        ({"lookback": 1}, "at least 2"),
        ({"horizon": 0}, "exactly 1"),
        ({"horizon": 2}, "exactly 1"),
        ({"start_date": "01-01-2013"}, "YYYY-MM-DD"),
        ({"start_date": "2025-01-01", "end_date": "2025-01-01"}, "must precede"),
        ({"first_test_year": 2026}, "must not exceed"),
        ({"last_test_year": 2026}, "within the end year"),
        ({"validation_years": 0}, "must be positive"),
        ({"first_test_year": 2014, "validation_years": 1}, "needs training years"),
        ({"transaction_cost_bps": -0.1}, "must not be negative"),
        ({"seeds": ()}, "at least one"),
        ({"seeds": (-1,)}, "negative"),
        ({"seeds": (7, 7)}, "unique"),
    ],
)
def test_experiment_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        valid_experiment(**overrides)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_experiment_rejects_nonfinite_transaction_cost(value: float) -> None:
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        valid_experiment(transaction_cost_bps=value)


def test_ridge_validation() -> None:
    assert RidgeSettings(alpha=0).alpha == 0
    with pytest.raises(ValueError, match="must not be negative"):
        RidgeSettings(alpha=-1)
    with pytest.raises(ValueError, match="finite"):
        RidgeSettings(alpha=float("nan"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"layers": 0}, "layers"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"patience": 0}, "patience"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"weight_decay": -1}, "weight_decay"),
        ({"gradient_clip": 0}, "gradient_clip"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"weight_decay": float("inf")}, "weight_decay"),
        ({"gradient_clip": float("nan")}, "gradient_clip"),
    ],
)
def test_lstm_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        valid_lstm(**overrides)


def test_project_config_serialization_is_detached() -> None:
    config = ProjectConfig(valid_experiment(), RidgeSettings(1.0), valid_lstm())
    result = config.as_dict()
    result["experiment"]["seeds"].append(99)
    assert config.experiment.seeds == (7, 17)


def test_load_config_reports_missing_section_and_key(tmp_path: Path) -> None:
    missing_section = tmp_path / "missing-section.toml"
    missing_section.write_text("[experiment]\nname='x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing \[ridge\] section"):
        load_config(missing_section)

    valid = load_config(Path("configs/ci.toml"))
    missing_key = tmp_path / "missing-key.toml"
    missing_key.write_text(
        """
[experiment]
name = "x"
lookback = 8
horizon = 1
start_date = "2015-01-01"
end_date = "2020-12-31"
first_test_year = 2018
last_test_year = 2019
validation_years = 1
transaction_cost_bps = 10
seeds = [7]

[ridge]
alpha = 1

[lstm]
hidden_size = 4
layers = 1
epochs = 2
batch_size = 8
learning_rate = 0.001
weight_decay = 0
patience = 2
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required configuration key: gradient_clip"):
        load_config(missing_key)

    assert replace(valid.ridge, alpha=3.0).alpha == 3.0
