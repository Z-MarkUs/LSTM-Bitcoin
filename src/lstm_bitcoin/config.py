"""Typed experiment configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSettings:
    """Outer walk-forward evaluation settings."""

    name: str
    lookback: int
    horizon: int
    start_date: str
    end_date: str
    first_test_year: int
    last_test_year: int
    validation_years: int
    transaction_cost_bps: float
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment.name must not be blank")
        if self.lookback < 2:
            raise ValueError("experiment.lookback must be at least 2")
        if self.horizon != 1:
            raise ValueError("experiment.horizon must be exactly 1 for next-day evaluation")
        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
        except ValueError as exc:
            raise ValueError("experiment dates must use YYYY-MM-DD") from exc
        if start >= end:
            raise ValueError("experiment.start_date must precede end_date")
        if self.first_test_year > self.last_test_year:
            raise ValueError("first_test_year must not exceed last_test_year")
        if not start.year < self.first_test_year <= self.last_test_year <= end.year:
            raise ValueError("test years must fall after the start and within the end year")
        if self.validation_years < 1:
            raise ValueError("validation_years must be positive")
        if self.first_test_year - self.validation_years <= start.year:
            raise ValueError("the first fold needs training years before validation")
        if not isfinite(self.transaction_cost_bps):
            raise ValueError("transaction_cost_bps must be finite")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must not be negative")
        if not self.seeds:
            raise ValueError("experiment.seeds must contain at least one seed")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("experiment.seeds must not contain negative values")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("experiment.seeds must be unique")


@dataclass(frozen=True)
class RidgeSettings:
    """Closed-form ridge baseline settings."""

    alpha: float

    def __post_init__(self) -> None:
        if not isfinite(self.alpha):
            raise ValueError("ridge.alpha must be finite")
        if self.alpha < 0:
            raise ValueError("ridge.alpha must not be negative")


@dataclass(frozen=True)
class LSTMSettings:
    """CPU PyTorch LSTM training settings."""

    hidden_size: int
    layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    gradient_clip: float

    def __post_init__(self) -> None:
        positive_ints = {
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"lstm.{name} must be positive")
        if not isfinite(self.learning_rate):
            raise ValueError("lstm.learning_rate must be finite")
        if self.learning_rate <= 0:
            raise ValueError("lstm.learning_rate must be positive")
        if not isfinite(self.weight_decay):
            raise ValueError("lstm.weight_decay must be finite")
        if self.weight_decay < 0:
            raise ValueError("lstm.weight_decay must not be negative")
        if not isfinite(self.gradient_clip):
            raise ValueError("lstm.gradient_clip must be finite")
        if self.gradient_clip <= 0:
            raise ValueError("lstm.gradient_clip must be positive")


@dataclass(frozen=True)
class ProjectConfig:
    """Complete reproducible experiment configuration."""

    experiment: ExperimentSettings
    ridge: RidgeSettings
    lstm: LSTMSettings

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "experiment": {
                **self.experiment.__dict__,
                "seeds": list(self.experiment.seeds),
            },
            "ridge": dict(self.ridge.__dict__),
            "lstm": dict(self.lstm.__dict__),
        }


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] section")
    return value


def load_config(path: Path) -> ProjectConfig:
    """Load and validate a project TOML configuration file."""

    with path.open("rb") as handle:
        document = tomllib.load(handle)

    experiment = _section(document, "experiment")
    ridge = _section(document, "ridge")
    lstm = _section(document, "lstm")
    try:
        return ProjectConfig(
            experiment=ExperimentSettings(
                name=str(experiment["name"]),
                lookback=int(experiment["lookback"]),
                horizon=int(experiment["horizon"]),
                start_date=str(experiment["start_date"]),
                end_date=str(experiment["end_date"]),
                first_test_year=int(experiment["first_test_year"]),
                last_test_year=int(experiment["last_test_year"]),
                validation_years=int(experiment["validation_years"]),
                transaction_cost_bps=float(experiment["transaction_cost_bps"]),
                seeds=tuple(int(seed) for seed in experiment["seeds"]),
            ),
            ridge=RidgeSettings(alpha=float(ridge["alpha"])),
            lstm=LSTMSettings(
                hidden_size=int(lstm["hidden_size"]),
                layers=int(lstm["layers"]),
                epochs=int(lstm["epochs"]),
                batch_size=int(lstm["batch_size"]),
                learning_rate=float(lstm["learning_rate"]),
                weight_decay=float(lstm["weight_decay"]),
                patience=int(lstm["patience"]),
                gradient_clip=float(lstm["gradient_clip"]),
            ),
        )
    except KeyError as exc:
        raise ValueError(f"missing required configuration key: {exc.args[0]}") from exc
