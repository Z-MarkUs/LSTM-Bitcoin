# Architecture

## Design intent

LSTM Bitcoin Research is a small research pipeline, not a live trading service. Its
architecture makes four boundaries visible:

1. data retrieval is explicit, pinned, and hash-verified;
2. every feature is available by its forecast origin;
3. fitting and model selection precede each outer test period;
4. reported evidence is generated from machine-readable artifacts.

~~~mermaid
flowchart LR
    C[Typed TOML config] --> D[Verified daily prices]
    D --> F[Past-only features and windows]
    F --> S[Annual expanding folds]
    S --> B[Three baselines]
    S --> L[Seeded CPU LSTM]
    B --> M[Forecast metrics]
    L --> M
    M --> T[Secondary signal simulation]
    T --> A[CSV and JSON evidence]
    A --> H[SHA-256 manifest]
    A --> P[Deterministic SVG plots]
~~~

## Component map

| Component | Responsibility |
| --- | --- |
| <code>cli.py</code> | Exposes data, evaluation, reporting, verification, and end-to-end reproduction commands. |
| <code>config.py</code> | Loads and validates immutable dataclass settings from TOML. |
| <code>data.py</code> | Fetches the allowlisted Coin Metrics source, verifies its raw digest, writes the two-column snapshot, and validates its manifest. |
| <code>features.py</code> | Builds price-derived features and three-dimensional windows ending at the forecast origin. |
| <code>splits.py</code> | Creates non-overlapping, annual expanding train/validation/test folds from target dates. |
| <code>models/base.py</code> | Defines the forecast protocol and train-only feature/target standardizers. |
| <code>models/baselines.py</code> | Implements zero-return, historical-mean, and ridge forecasts. |
| <code>models/lstm.py</code> | Implements the deterministic CPU PyTorch LSTM and feature-only prediction boundary. |
| <code>experiment.py</code> | Runs every fold, aggregates seed forecasts, calculates comparisons, and assembles evidence. |
| <code>metrics.py</code> | Calculates forecast error and the moving-block bootstrap comparison. |
| <code>backtest.py</code> | Runs the explicitly idealized, lagged, long/flat signal simulation with turnover costs. |
| <code>artifacts.py</code> | Writes deterministic CSV/JSON files atomically and creates or verifies SHA-256 lists. |
| <code>plotting.py</code> | Produces four deterministic, evidence-first SVGs from a checksum-verified run. |

The package is installed as <code>lstm-bitcoin-research</code>, imported as
<code>lstm_bitcoin</code>, and invoked as <code>lstm-bitcoin</code>.

## Leakage controls

<code>SupervisedData</code> stores origin and target dates explicitly and rejects any
sample whose target is not strictly later than its origin. A feature window includes
data only through the origin. The default target is the next close-to-close log return.

Each annual fold contains:

- expanding training targets from earlier years;
- a later calendar validation year used by the LSTM for early stopping;
- one still-later test year used only for outer evaluation.

LSTM feature and target scalers are fitted on the training partition only. The LSTM's
prediction method accepts only features; there is no decoder input or target-shaped
argument. Fixed baselines are fitted on all observations available before the test
year because they do not use the validation year for model selection.

## Trust and security boundaries

- Network access occurs only in <code>data fetch</code>.
- The source scheme, host, size, commit, and SHA-256 are checked before parsing.
- Existing data or evidence is not overwritten unless <code>--force</code> is used.
- Writes are atomic and checksum entries must be simple, unique filenames.
- The pipeline does not deserialize old pickle, HDF5, Keras, or PyTorch model files.
- Models are retrained from source and configuration; model weights are not release
  inputs.
- The CLI needs no provider credential.

A checksum proves byte integrity relative to <code>SHA256SUMS</code>; it does not by
itself prove that a methodology is sound. Tests, source review, the data manifest, the
run manifest, and clean reproduction provide the other layers of evidence.

## Evidence boundary

The canonical run directory is <code>evaluation/results/reference-v1</code>. It
contains fold definitions, per-model and per-seed predictions, training diagnostics,
forecast metrics, signal-simulation rows, a run manifest, and checksums. Plots live
under <code>docs/assets</code> with a separate checksum list.

No number in a README or portfolio should be maintained independently. It must be
traceable to the reviewed artifacts, and a negative comparison with the baselines must
remain visible.
