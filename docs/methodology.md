# Methodology

## Research question

Does a compact LSTM improve next-day Bitcoin log-return forecasting error over simple
out-of-sample baselines under a strictly chronological protocol?

The primary question is forecast accuracy, not strategy profitability. The signal
simulation is secondary and illustrative.

## Data and target

The frozen input is Coin Metrics daily BTC <code>PriceUSD</code> from 2013-01-01
through 2025-12-31. The source revision and raw SHA-256 are fixed in
[data-provenance.md](data-provenance.md). The selected subset has 4,748 complete daily
observations.

[Coin Metrics documents](https://docs.coinmetrics.io/resources/faqs)
<code>PriceUSD</code> as a daily close at the 00:00 UTC cutoff and labels it with its
beginning-of-interval timestamp convention. This project keeps that provider date
label. “Next day” therefore means the return between successive 24-hour UTC-cutoff
observations, not an exchange close in a local timezone. A forecast is formed only
after the observation carrying the origin label is available.

For forecast origin \(t\) and configured horizon \(h\), the target is cumulative log
return:

\[
y_t = \log(P_{t+h}) - \log(P_t).
\]

The experiment configuration requires \(h=1\), matching the metric and simulation
contract. The lower-level feature builder remains generic for isolated tests. Every
corresponding target date must be strictly later than the origin date.

## Features and windows

Five deterministic features are calculated from prices available no later than each
row:

- one-day log return;
- seven-day log return;
- thirty-day log return;
- seven-day realized volatility of daily log returns;
- thirty-day realized volatility of daily log returns.

The reference run uses a 30-day sequence ending at the forecast origin. Rows lacking
the required warm-up history are excluded. No future price, target, decoder seed, or
revised external covariate enters a feature window.

## Walk-forward design

The reference configuration defines eight annual expanding folds:

| Fold | Training targets | Validation targets | Test targets |
| --- | --- | --- | --- |
| 1 | Before 2017 | 2017 | 2018 |
| 2 | Before 2018 | 2018 | 2019 |
| 3 | Before 2019 | 2019 | 2020 |
| 4 | Before 2020 | 2020 | 2021 |
| 5 | Before 2021 | 2021 | 2022 |
| 6 | Before 2022 | 2022 | 2023 |
| 7 | Before 2023 | 2023 | 2024 |
| 8 | Before 2024 | 2024 | 2025 |

Test target dates appear exactly once in the concatenated out-of-sample record.
The 2025 fold is also reported separately as the final-year result. Outer-test
performance must not select features, thresholds, hyperparameters, or seeds. This
project does not claim a preregistered holdout.

## Models

All models forecast the same target dates using the same feature windows.

- **Zero return:** predicts zero, equivalent to price persistence for a one-day return.
- **Historical mean:** predicts the mean return observed before the test year.
- **Ridge:** closed-form ridge regression over the flattened standardized window;
  reference \(\alpha=10\).
- **LSTM:** one PyTorch LSTM layer with 16 hidden units and a linear head.

For the fixed baselines, training and validation indices are combined because no
validation-driven choice is made. The LSTM fits on the earlier training block and uses
the following validation year only for early stopping. Its feature and target
standardizers are fitted on the training block.

The reference LSTM uses Smooth L1 loss, AdamW, a learning rate of 0.001, weight decay
of 0.0001, batch size 128, at most 25 epochs, patience 4, and gradient-norm clipping at
1.0. Seeds 7, 17, and 29 run independently on one CPU thread with deterministic
PyTorch algorithms. Their predictions are averaged to form the released LSTM
ensemble. Per-seed predictions and training diagnostics remain available for review.

## Forecast evaluation

The primary released metric is mean absolute error in unscaled log-return units.
Supporting metrics are:

- root mean squared error;
- mean signed error;
- directional accuracy among nonzero direction calls, paired with directional coverage;
- Pearson correlation, reported as null when undefined;
- MAE ratio relative to the zero-return baseline.

Overall metrics use the concatenated, unique 2018–2025 outer-test dates. Fold metrics
remain available to show regime variation.

The best baseline is selected only by its overall out-of-sample MAE for descriptive
comparison. The LSTM-minus-best-baseline MAE difference receives a fixed-seed,
2,000-resample moving-block bootstrap interval with 30-day blocks and seed 2026. A
negative difference favors the LSTM. <code>beats_best_baseline</code> records the
literal MAE comparison; it is not a statistical-significance claim.

## Secondary signal simulation

For a prediction made after origin \(t\), the position for the subsequent
\(t\)-to-\(t+1\) return is:

\[
z_t = \mathbf{1}[\hat{y}_t > 0].
\]

The strategy is long or flat; it is never short. Net simple return is:

\[
R^{net}_{t+1}
= z_t\left(e^{y_t}-1\right)
- c\left|z_t-z_{t-1}\right|,
\]

where \(c\) is one-way transaction cost in decimal units. The reference case uses
10 basis points and summary sensitivity is produced for 0, 5, 10, 25, and 50 basis
points. Buy-and-hold is charged one initial entry cost. The simulation does not add a
terminal liquidation cost.

Reported simulation fields include total return, true compounded CAGR, 365-day
annualized volatility, zero-rate Sharpe ratio, maximum drawdown, turnover, exposure,
and trade count.

This idealized calculation is not a backtested execution system and supports no claim
of future profitability. See [limitations.md](limitations.md).
