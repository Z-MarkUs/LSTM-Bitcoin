# Limitations

## Data

- The study uses one historical daily reference-price series. It does not model
  exchange dispersion, intraday liquidity, order books, outages, or executable quotes.
- Only price-derived features are used. Potentially relevant macroeconomic, on-chain,
  derivatives, sentiment, and market-microstructure variables are intentionally
  excluded because their point-in-time availability and redistribution rights require
  separate work.
- A pinned archive improves reproducibility but cannot prove that upstream historical
  observations were never revised before that commit.
- The sample covers 2013–2025. It contains a limited number of market regimes and
  cannot represent future structural change.
- The Coin Metrics subset and data-derived artifacts are CC BY-NC 4.0, not MIT, which
  limits reuse in commercial settings.

## Forecast study

- Eight yearly test folds provide a more defensible estimate than a random split, but
  adjacent daily errors remain dependent and the number of independent regimes is
  small.
- The 30-day block bootstrap is an approximation. Its interval depends on the chosen
  block length and does not correct every form of model-selection or regime risk.
- Architecture and hyperparameters reflect prior researcher judgment. The project
  does not claim an exhaustive or neutral search over all LSTM and baseline designs.
- Directional accuracy ignores return magnitude and is reported only among nonzero
  direction calls; directional coverage must be read beside it. The zero-return
  baseline therefore has zero coverage and undefined accuracy. Correlation may be
  unstable and is undefined for constant predictions.
- The seed ensemble reduces one source of randomness but does not measure all training
  uncertainty.
- A result in which the LSTM beats a baseline would not establish durable predictability.
  A result in which it loses would not establish that all neural time-series models
  must fail.

## Signal simulation

The long/flat calculation is an explanatory transformation of forecasts, not a
production backtest.

- It assumes the position can be formed immediately after observing the origin close
  at the documented 00:00 UTC cutoff and then earn the next close-to-close return.
- It uses a fixed one-way turnover cost, not observed spread, slippage, fees, or market
  impact.
- It omits latency, partial fills, minimum order sizes, venue limits, funding, borrow,
  taxes, custody, counterparty risk, operational outages, and cash yield.
- It permits only long or flat positions and applies no risk sizing, leverage,
  portfolio constraints, or execution policy.
- Buy-and-hold receives one initial entry cost; neither strategy is charged a final
  liquidation cost.
- Sharpe uses a zero risk-free rate and 365-day annualization. It is descriptive, not
  a forecast of risk-adjusted performance.

Consequently, simulated CAGR or equity is not evidence that a person could have
obtained the same result and is never presented as a profitability claim.

## Reproducibility and verification

- Deterministic execution is scoped to the locked CPU environment. Different hardware,
  numerical libraries, or dependency versions can change neural training.
- <code>lstm-bitcoin verify</code> checks files against the included checksum list. It
  does not independently validate economic assumptions, licensing, source-code
  correctness, or the truth of a result generated together with a new checksum file.
- Historical commits predate the rebuilt controls and are unsupported.
- The original repository's secret audit found no credential, but no scanner can prove
  the absence of every encoded or unreachable secret.

## Intended use

This repository is research and engineering evidence. It is not investment advice,
an offer, a recommendation, a price target, a live signal, or a production trading
system. No future return or profitability is claimed.
