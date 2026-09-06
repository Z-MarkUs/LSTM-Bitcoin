# Legacy Coursework and Audit Record

## Historical context

This repository began as APS1052 coursework exploring Bitcoin forecasting with LSTM
models. The rebuild preserves the educational context, not the old empirical claims
or its artifact dump. New source code was written around a narrower, auditable research
question and a leakage-controlled evaluation.

## What the audit found

The old notebook and supporting files were not suitable evidence for a portfolio or a
financial claim:

- test-time decoder input included the first future target value;
- trading decisions were revised using future realized targets;
- the executed notebook reported materially negative values while the following prose
  claimed strong positive CAGR, Sharpe ratio, and profit factor;
- the SHAP section ended in an execution error;
- the described White Reality Check was not implemented;
- a simple persistence baseline beat the notebook's printed LSTM error on the same
  old test windows;
- the presentation described a different one-day model and different results;
- copied or collaborator-authored code and data lacked a complete license and
  attribution chain;
- large model binaries, duplicate data, IDE state, bytecode, checkpoints, personal
  filesystem paths, and a temporary Office file were committed.

These findings invalidate the old performance narrative. They do not imply that LSTMs
can never be useful; they show that this particular evidence could not answer the
question.

## Disposition

The maintained branch and new releases exclude the old notebooks, datasets, scraper,
presentation, environment dump, caches, and serialized Keras/HDF5 models. No legacy
release or tag is created. If an older Git object remains available in a clone or
repository history, it is unsupported historical material and its presence grants no
data or code reuse rights.

The current project replaces that material with:

- a pinned, checksummed Coin Metrics source and explicit CC BY-NC 4.0 attribution;
- features that use information available no later than each forecast origin;
- one-step future-return targets and a feature-only prediction API;
- expanding annual train/validation/test folds;
- zero-return, historical-mean, and ridge baselines alongside the LSTM;
- generated metrics, predictions, fold boundaries, simulation records, and checksums;
- an explicitly secondary, lagged, cost-aware signal simulation;
- automated leakage, integrity, typing, test, dependency, and security checks.

The MIT license applies to the new code and documentation only. See
[DATA_LICENSE.md](../DATA_LICENSE.md) for the separate data terms.

## Secret review

The public history was checked before the rebuild. A checksum-verified Gitleaks 8.30.1
scan covered all seven reachable commits and reported zero findings. Additional
high-confidence checks for common provider tokens, private-key blocks, credential URLs,
and generic credential assignments also found no matches.

GitHub secret scanning was disabled at the time of that audit, so the independent scan
was important. The result means no secret exposure was identified in this repository;
it does not prove whether a credential from another system was active or revoked.

## How to describe the project

The defensible portfolio statement is:

> I audited a leakage-prone course experiment, removed unsupported performance claims
> and unclear artifacts, and rebuilt it as a reproducible walk-forward comparison
> against strong baselines.

Do not describe the old strategy as profitable. Do not present the rebuilt signal
simulation as investment advice or live-trading validation.
