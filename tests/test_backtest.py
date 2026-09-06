from __future__ import annotations

import math

import numpy as np
import pytest

from lstm_bitcoin.backtest import simulate_buy_and_hold, simulate_long_flat


def test_signal_timing_position_and_cost_are_exact() -> None:
    actual_simple = np.asarray([0.10, 0.20, -0.10, 0.05])
    actual_log = np.log1p(actual_simple)
    predictions = np.asarray([0.01, 0.02, -0.01, 0.01])

    simulation = simulate_long_flat(actual_log, predictions, transaction_cost_bps=10.0)

    np.testing.assert_array_equal(simulation.positions, [1, 1, 0, 1])
    np.testing.assert_array_equal(simulation.turnover, [1, 0, 1, 1])
    np.testing.assert_allclose(simulation.gross_simple_returns, [0.10, 0.20, 0.0, 0.05])
    np.testing.assert_allclose(simulation.costs, [0.001, 0.0, 0.001, 0.001])
    np.testing.assert_allclose(simulation.net_simple_returns, [0.099, 0.20, -0.001, 0.049])
    np.testing.assert_allclose(simulation.equity, np.cumprod([1.099, 1.20, 0.999, 1.049]))
    assert simulation.metrics.turnover == 3.0
    assert simulation.metrics.trades == 3
    assert simulation.metrics.exposure == 0.75
    assert simulation.metrics.observations == 4


def test_prediction_at_each_origin_applies_only_to_corresponding_future_return() -> None:
    actual = np.log1p(np.asarray([0.50, -0.50]))
    first_day_long = simulate_long_flat(actual, np.asarray([1.0, -1.0]), transaction_cost_bps=0)
    second_day_long = simulate_long_flat(actual, np.asarray([-1.0, 1.0]), transaction_cost_bps=0)
    np.testing.assert_allclose(first_day_long.net_simple_returns, [0.5, 0.0])
    np.testing.assert_allclose(second_day_long.net_simple_returns, [0.0, -0.5])


def test_flat_strategy_has_defined_zero_metrics() -> None:
    simulation = simulate_long_flat(
        np.asarray([0.2, -0.1, 0.05]),
        np.zeros(3),
        transaction_cost_bps=50,
    )

    np.testing.assert_array_equal(simulation.positions, 0)
    np.testing.assert_array_equal(simulation.net_simple_returns, 0)
    np.testing.assert_array_equal(simulation.equity, 1)
    assert simulation.metrics.total_return == 0
    assert simulation.metrics.cagr == 0
    assert simulation.metrics.annualized_volatility == 0
    assert simulation.metrics.sharpe_zero_rate is None
    assert simulation.metrics.max_drawdown == 0
    assert simulation.metrics.as_dict()["trades"] == 0


def test_crypto_annualization_uses_365_days() -> None:
    daily_simple_return = 0.001
    simulation = simulate_buy_and_hold(
        np.full(365, np.log1p(daily_simple_return)), transaction_cost_bps=0
    )
    expected = (1 + daily_simple_return) ** 365 - 1
    assert simulation.metrics.total_return == pytest.approx(expected)
    assert simulation.metrics.cagr == pytest.approx(expected)


def test_max_drawdown_includes_initial_capital() -> None:
    simulation = simulate_long_flat(
        np.asarray([math.log(0.5)]), np.asarray([1.0]), transaction_cost_bps=0
    )
    assert simulation.metrics.max_drawdown == pytest.approx(-0.5)


def test_buy_and_hold_is_always_long_and_charged_one_entry() -> None:
    returns = np.log1p(np.asarray([0.01, -0.01, 0.02]))
    simulation = simulate_buy_and_hold(returns, transaction_cost_bps=25)
    np.testing.assert_array_equal(simulation.positions, 1)
    np.testing.assert_array_equal(simulation.turnover, [1, 0, 0])
    np.testing.assert_allclose(simulation.costs, [0.0025, 0, 0])


@pytest.mark.parametrize(
    ("actual", "predicted", "cost", "message"),
    [
        (np.ones((2, 1)), np.ones(2), 0, "aligned vectors"),
        (np.ones(2), np.ones((2, 1)), 0, "aligned vectors"),
        (np.ones(2), np.ones(3), 0, "aligned vectors"),
        (np.asarray([]), np.asarray([]), 0, "aligned vectors"),
        (np.ones(2), np.ones(2), -1, "must not be negative"),
        (np.asarray([1.0, np.nan]), np.ones(2), 0, "must be finite"),
        (np.ones(2), np.asarray([1.0, np.inf]), 0, "must be finite"),
    ],
)
def test_simulation_validates_inputs(
    actual: np.ndarray, predicted: np.ndarray, cost: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_long_flat(actual, predicted, transaction_cost_bps=cost)


def test_simulation_rejects_cost_that_loses_more_than_capital() -> None:
    with pytest.raises(ValueError, match="below -100%"):
        simulate_long_flat(np.asarray([0.0]), np.asarray([1.0]), transaction_cost_bps=10_001)


def test_simulation_rejects_exponential_overflow() -> None:
    with pytest.raises(ValueError, match="finite"):
        simulate_long_flat(np.asarray([1_000.0]), np.asarray([1.0]), transaction_cost_bps=0)
