"""Reproducible, evidence-first SVGs for the checked reference run."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import Formatter, Locator

from .artifacts import prepare_output_directory, verify_checksums, write_checksums

PLOT_FILES = (
    "forecast_mae.svg",
    "forecast_scatter.svg",
    "signal_equity.svg",
    "walk_forward_folds.svg",
    "SHA256SUMS",
)

LABELS = {
    "zero_return": "Zero return",
    "historical_mean": "Historical mean",
    "ridge": "Ridge",
    "lstm": "LSTM ensemble",
    "buy_and_hold": "Buy and hold",
}
COLORS = {
    "zero_return": "#9CA3AF",
    "historical_mean": "#D1D5DB",
    "ridge": "#D97706",
    "lstm": "#2563EB",
    "buy_and_hold": "#374151",
}

_DATE_TO_NUM = cast(Callable[[date], float], mdates.date2num)
_YEAR_LOCATOR = cast(Callable[[int], Locator], mdates.YearLocator)
_DATE_FORMATTER = cast(Callable[[str], Formatter], mdates.DateFormatter)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": "#374151",
            "axes.titlecolor": "#111827",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "grid.color": "#E5E7EB",
            "svg.fonttype": "none",
            "svg.hashsalt": "lstm-bitcoin-reference-v1",
            "text.color": "#111827",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )


def _save(figure: Any, path: Path) -> None:
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)


def _heading(axis: Any, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold", pad=28)
    axis.text(
        0,
        1.015,
        subtitle,
        transform=axis.transAxes,
        color="#4B5563",
        fontsize=9,
    )


def _mae_plot(metrics: dict[str, Any], output: Path) -> None:
    models = list(metrics["models"])
    values = [float(metrics["models"][name]["mae"]) for name in models]
    order = np.argsort(values)[::-1]
    ordered_models = [models[index] for index in order]
    ordered_values = [values[index] for index in order]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.barh(
        [LABELS[name] for name in ordered_models],
        ordered_values,
        color=[COLORS[name] for name in ordered_models],
        edgecolor="#374151",
        linewidth=0.7,
    )
    axis.bar_label(bars, labels=[f"{value:.5f}" for value in ordered_values], padding=5)
    period = metrics["out_of_sample_period"]
    _heading(
        axis,
        "Out-of-sample next-day return error",
        f"Mean absolute log-return error · {period['start']} to {period['end']} · lower is better",
    )
    axis.set_xlabel("MAE (log-return units)")
    axis.grid(axis="x", alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_xlim(left=0)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, output / "forecast_mae.svg")


def _scatter_plot(prediction_rows: list[dict[str, str]], output: Path) -> None:
    selected = [row for row in prediction_rows if row["model"] == "lstm"]
    if not selected:
        selected = [row for row in prediction_rows if row["model"] == "ridge"]
    actual = np.asarray([float(row["actual_log_return"]) for row in selected])
    predicted = np.asarray([float(row["predicted_log_return"]) for row in selected])
    extent = float(max(np.max(np.abs(actual)), np.max(np.abs(predicted))))
    figure, axis = plt.subplots(figsize=(6.6, 6.0))
    axis.scatter(
        actual,
        predicted,
        s=13,
        alpha=0.28,
        color="#2563EB",
        edgecolors="none",
        rasterized=False,
    )
    axis.plot([-extent, extent], [-extent, extent], color="#374151", linestyle="--", linewidth=1)
    axis.axhline(0, color="#9CA3AF", linewidth=0.8)
    axis.axvline(0, color="#9CA3AF", linewidth=0.8)
    _heading(
        axis,
        "Predicted versus realized next-day returns",
        f"{LABELS[selected[0]['model']]} · {len(selected):,} unique out-of-sample days",
    )
    axis.set_xlabel("Realized log return")
    axis.set_ylabel("Predicted log return")
    axis.grid(alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, output / "forecast_scatter.svg")


def _equity_plot(
    simulation_rows: list[dict[str, str]], metrics: dict[str, Any], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    models = [
        name
        for name in ("buy_and_hold", "ridge", "lstm")
        if any(row["model"] == name for row in simulation_rows)
    ]
    for model in models:
        selected = [row for row in simulation_rows if row["model"] == model]
        dates = np.asarray([row["target_date"] for row in selected], dtype="datetime64[D]")
        equity = np.asarray([float(row["equity"]) for row in selected])
        axis.plot(
            dates,
            equity,
            label=LABELS[model],
            color=COLORS[model],
            linewidth=2 if model == "lstm" else 1.5,
        )
    cost = metrics["signal_simulation"]["primary_cost_bps"]
    _heading(
        axis,
        "Illustrative signal simulation",
        (
            f"Growth of $1 · long/flat forecasts · {cost:g} bps per one-way turnover "
            "· not investment advice"
        ),
    )
    axis.set_ylabel("Growth of $1 (net of stated cost)")
    axis.grid(axis="y", alpha=0.7)
    axis.legend(frameon=False, ncol=len(models), loc="upper left")
    axis.text(
        0.99,
        0.02,
        "Historical mean overlaps buy-and-hold; zero-return stays at $1.",
        transform=axis.transAxes,
        ha="right",
        color="#4B5563",
        fontsize=8,
    )
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, output / "signal_equity.svg")


def _fold_plot(fold_rows: list[dict[str, str]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.4))
    for vertical, row in enumerate(fold_rows):
        train_start = _DATE_TO_NUM(date.fromisoformat(row["train_start"]))
        train_end = _DATE_TO_NUM(date.fromisoformat(row["train_end"]))
        validation_start = _DATE_TO_NUM(date.fromisoformat(row["validation_start"]))
        validation_end = _DATE_TO_NUM(date.fromisoformat(row["validation_end"]))
        test_start = _DATE_TO_NUM(date.fromisoformat(row["test_start"]))
        test_end = _DATE_TO_NUM(date.fromisoformat(row["test_end"]))
        axis.broken_barh(
            [(train_start, train_end - train_start + 1)],
            (vertical - 0.32, 0.64),
            facecolors="#CBD5E1",
            edgecolors="#64748B",
            linewidth=0.4,
        )
        axis.broken_barh(
            [(validation_start, validation_end - validation_start + 1)],
            (vertical - 0.32, 0.64),
            facecolors="#D97706",
            edgecolors="#92400E",
            hatch="///",
            linewidth=0.4,
        )
        axis.broken_barh(
            [(test_start, test_end - test_start + 1)],
            (vertical - 0.32, 0.64),
            facecolors="#2563EB",
            edgecolors="#1E3A8A",
            hatch="xx",
            linewidth=0.4,
        )
    axis.set_yticks(range(len(fold_rows)), [f"Test {row['test_year']}" for row in fold_rows])
    axis.xaxis_date()
    axis.xaxis.set_major_locator(_YEAR_LOCATOR(2))
    axis.xaxis.set_major_formatter(_DATE_FORMATTER("%Y"))
    _heading(
        axis,
        "Annual expanding-window evaluation",
        "Training expands through time; each orange validation year and blue test year stays later",
    )
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#CBD5E1", edgecolor="#64748B", label="Train"),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#D97706",
            edgecolor="#92400E",
            hatch="///",
            label="Validation",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#2563EB",
            edgecolor="#1E3A8A",
            hatch="xx",
            label="Test",
        ),
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    axis.grid(axis="x", alpha=0.6)
    axis.invert_yaxis()
    axis.spines[["top", "right", "left"]].set_visible(False)
    _save(figure, output / "walk_forward_folds.svg")


def generate_plots(run: Path, output: Path, *, force: bool = False) -> tuple[Path, ...]:
    """Verify a run and generate four deterministic SVG evidence views."""

    verify_checksums(run)
    prepare_output_directory(output, force=force, known_files=PLOT_FILES)
    with (run / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    prediction_rows = _rows(run / "predictions.csv")
    simulation_rows = _rows(run / "signal_simulation.csv")
    fold_rows = _rows(run / "folds.csv")
    _style()
    _mae_plot(metrics, output)
    _scatter_plot(prediction_rows, output)
    _equity_plot(simulation_rows, metrics, output)
    _fold_plot(fold_rows, output)
    names = tuple(name for name in PLOT_FILES if name != "SHA256SUMS")
    write_checksums(output, names)
    return tuple(output / name for name in names)
