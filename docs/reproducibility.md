# Reproducibility

## What is frozen

The reference evidence binds together:

- the project version and Git revision;
- <code>configs/reference.toml</code> and its SHA-256;
- the Coin Metrics source commit and raw SHA-256;
- the processed data SHA-256 and date range;
- Python, platform, and relevant dependency versions;
- declared LSTM seeds and training settings;
- every result artifact's SHA-256.

Python 3.12 is the canonical reference environment. Python 3.11 and 3.12 are supported
by CI. A committed <code>uv.lock</code> is required for a release-quality run.

## Clean reproduction

Install the locked environment:

~~~bash
uv sync --locked --all-extras --python 3.12
~~~

Validate the tracked reference data:

~~~bash
uv run --frozen lstm-bitcoin data validate
~~~

To deliberately reconstruct both tracked files from the pinned upstream bytes, run:

~~~bash
uv run --frozen lstm-bitcoin data fetch --force
uv run --frozen lstm-bitcoin data validate
~~~

Run the full reference experiment in a scratch directory so the reviewed evidence is
not overwritten accidentally:

~~~bash
uv run --frozen lstm-bitcoin reproduce \
  --config configs/reference.toml \
  --output evaluation/runs/reference-v1 \
  --plots evaluation/runs/reference-v1-plots
uv run --frozen lstm-bitcoin verify \
  --run evaluation/runs/reference-v1
~~~

The first command validates the input manifest, runs all folds, generates plots, and
checks the newly written run checksums. The explicit verify command repeats the
byte-integrity check. It does not substitute for comparing the scratch artifacts with
the reviewed reference bundle.

To exercise the pipeline quickly without producing reference evidence:

~~~bash
uv run --frozen lstm-bitcoin reproduce \
  --config configs/ci.toml \
  --models baselines \
  --output evaluation/runs/ci-smoke \
  --plots evaluation/runs/ci-smoke-plots
~~~

The CI configuration is a smoke test. Its dates, folds, model set, and short training
settings must never be cited as the canonical study.

## Determinism controls

- Source and configuration inputs are hash-bound.
- Input dates must be ordered, unique, complete at daily grain, and schema-valid.
- NumPy and PyTorch receive explicit seeds.
- PyTorch uses one CPU thread and deterministic algorithms.
- The data loader uses a seeded generator and no worker processes.
- CSV numeric formatting is bounded and JSON keys are sorted.
- SVG metadata omits creation time and uses a fixed hash salt.
- All generated writes are atomic.

The run manifest records a dirty-worktree flag. Reviewed release evidence should be
generated from a clean commit with <code>dirty=false</code>.

## Updating reviewed evidence

Do not hand-edit generated outputs. A deliberate update uses:

~~~bash
uv run --frozen lstm-bitcoin reproduce \
  --config configs/reference.toml \
  --output evaluation/results/reference-v1 \
  --plots docs/assets \
  --force
uv run --frozen lstm-bitcoin verify \
  --run evaluation/results/reference-v1
~~~

Review the changed source manifest, fold boundaries, predictions, per-seed results,
forecast metrics, cost sensitivity, limitations, plots, and checksums. Explain the
methodological reason in the pull request and changelog. A changed metric without a
corresponding input, code, or environment explanation is a release blocker.

## Expected variation

The project targets deterministic CPU execution in its locked environment, not
bit-for-bit equivalence across arbitrary hardware, operating systems, BLAS builds, or
future PyTorch versions. Reproduce a release using its exact lockfile and supported
Python version. If results differ, keep both bundles, compare manifests first, and do
not overwrite the reviewed evidence until the cause is understood.

Data and data-derived outputs remain subject to CC BY-NC 4.0; the MIT code license
does not relicense them.
