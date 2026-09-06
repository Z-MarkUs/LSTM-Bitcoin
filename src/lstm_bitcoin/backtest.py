"""Explicitly idealized, cost-aware signal simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SimulationMetrics:
    """Secondary signal-simulation summary using 365-day annualization."""

    observations: int
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_zero_rate: float | None
    max_drawdown: float
    turnover: float
    exposure: float
    trades: int

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class Simulation:
    """Per-period long/flat positions and returns."""

    positions: npt.NDArray[np.float64]
    turnover: npt.NDArray[np.float64]
    gross_simple_returns: npt.NDArray[np.float64]
    costs: npt.NDArray[np.float64]
    net_simple_returns: npt.NDArray[np.float64]
    equity: npt.NDArray[np.float64]
    metrics: SimulationMetrics


def _metrics(
    net_returns: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    turnover: npt.NDArray[np.float64],
) -> SimulationMetrics:
    equity = np.cumprod(1.0 + net_returns)
    total_return = float(equity[-1] - 1.0)
    years = len(net_returns) / 365.0
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    volatility = float(np.std(net_returns, ddof=0) * np.sqrt(365.0))
    standard_deviation = float(np.std(net_returns, ddof=0))
    sharpe = (
        None
        if standard_deviation < 1e-15
        else float(np.mean(net_returns) / standard_deviation * np.sqrt(365.0))
    )
    equity_with_initial = np.concatenate((np.ones(1, dtype=np.float64), equity))
    running_maximum = np.maximum.accumulate(equity_with_initial)
    max_drawdown = float(np.min(equity_with_initial / running_maximum - 1.0))
    return SimulationMetrics(
        observations=len(net_returns),
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=volatility,
        sharpe_zero_rate=sharpe,
        max_drawdown=max_drawdown,
        turnover=float(np.sum(turnover)),
        exposure=float(np.mean(positions)),
        trades=int(np.count_nonzero(turnover)),
    )


def simulate_long_flat(
    actual_log_returns: npt.NDArray[np.float64],
    predicted_log_returns: npt.NDArray[np.float64],
    *,
    transaction_cost_bps: float,
) -> Simulation:
    """Apply a signal formed at each origin only to its subsequent target return."""

    if (
        actual_log_returns.ndim != 1
        or predicted_log_returns.ndim != 1
        or len(actual_log_returns) != len(predicted_log_returns)
        or len(actual_log_returns) == 0
    ):
        raise ValueError("actual and predicted returns must be non-empty aligned vectors")
    if not np.isfinite(transaction_cost_bps):
        raise ValueError("transaction_cost_bps must be finite")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must not be negative")
    if not np.isfinite(actual_log_returns).all() or not np.isfinite(predicted_log_returns).all():
        raise ValueError("simulation returns must be finite")

    positions = (predicted_log_returns > 0).astype(np.float64)
    previous = np.concatenate((np.zeros(1, dtype=np.float64), positions[:-1]))
    turnover = np.abs(positions - previous)
    with np.errstate(over="ignore", invalid="ignore"):
        realized_simple = np.expm1(actual_log_returns)
    gross = positions * realized_simple
    costs = turnover * (transaction_cost_bps / 10_000.0)
    net = gross - costs
    if not all(np.isfinite(values).all() for values in (realized_simple, gross, costs, net)):
        raise ValueError("simulation outputs must be finite")
    if np.any(net <= -1.0):
        raise ValueError("simulation produced a return at or below -100%")
    equity = np.cumprod(1.0 + net).astype(np.float64, copy=False)
    if not np.isfinite(equity).all():
        raise ValueError("simulation equity must be finite")
    return Simulation(
        positions=positions,
        turnover=turnover,
        gross_simple_returns=gross,
        costs=costs,
        net_simple_returns=net,
        equity=equity,
        metrics=_metrics(net, positions, turnover),
    )


def simulate_buy_and_hold(
    actual_log_returns: npt.NDArray[np.float64], *, transaction_cost_bps: float
) -> Simulation:
    """Long-only comparator charged one entry cost at the first forecast origin."""

    predictions = np.ones_like(actual_log_returns, dtype=np.float64)
    return simulate_long_flat(
        actual_log_returns,
        predictions,
        transaction_cost_bps=transaction_cost_bps,
    )
