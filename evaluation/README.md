# Evaluation Evidence

## Directory policy

<code>evaluation/results/reference-v1</code> is the reviewed canonical bundle.
<code>evaluation/runs</code> is ignored scratch space for local reproductions. Generated
files are evidence, not source code, and must not be edited by hand.

Until a complete bundle is generated, reviewed, and checksum-verified, the repository
makes no numerical performance claim.

## Canonical bundle

| File | Purpose |
| --- | --- |
| <code>manifest.json</code> | Experiment configuration, code state, environment, input digests, source lineage, feature policy, fold boundaries, artifact digests, and limitations. |
| <code>metrics.json</code> | Overall forecast metrics, best-baseline comparison, moving-block interval, final-year result, and primary signal-simulation summary. |
| <code>folds.csv</code> | Exact train, validation, and test date bounds and sample counts for every fold. |
| <code>fold_metrics.csv</code> | Per-fold metrics for each released model. |
| <code>predictions.csv</code> | One row per fold, model, origin, and unique out-of-sample target date. |
| <code>seed_predictions.csv</code> | Individual LSTM predictions for seeds 7, 17, and 29. |
| <code>seed_metrics.csv</code> | Combined out-of-sample metrics for each LSTM seed. |
| <code>training.csv</code> | Per-fold and per-seed early-stopping diagnostics. |
| <code>signal_simulation.csv</code> | Self-contained fold, year, origin, target, cost, position, turnover, return, and equity rows at the primary cost. |
| <code>signal_metrics.csv</code> | Model and buy-and-hold summaries at 0, 5, 10, 25, and 50 basis points. |
| <code>SHA256SUMS</code> | GNU-compatible SHA-256 entries for the ten evidence files above; it does not hash itself. |

The four generated figures—forecast MAE, predicted-versus-realized returns, signal
equity, and walk-forward folds—are stored in <code>docs/assets</code> with their own
<code>SHA256SUMS</code>.

## Generate and verify

After fetching and validating the pinned data:

~~~bash
uv run --frozen lstm-bitcoin reproduce \
  --config configs/reference.toml \
  --output evaluation/results/reference-v1 \
  --plots docs/assets
uv run --frozen lstm-bitcoin verify \
  --run evaluation/results/reference-v1
~~~

Use <code>--force</code> only for an intentional reviewed update. The generator
replaces only its known artifact filenames and refuses an accidental overwrite by
default.

Checksum verification establishes that the files match their bundle. Reviewers should
also inspect the manifest, confirm a clean Git revision, validate the source manifest,
and reproduce in the locked environment.

## Interpretation order

1. Read <code>folds.csv</code> and confirm chronological, non-overlapping partitions.
2. Inspect <code>metrics.json</code> and compare LSTM MAE with all three baselines.
3. Review fold and seed variation before interpreting the aggregate.
4. Treat the bootstrap interval as uncertainty around the MAE difference, not as proof
   of market predictability.
5. Read the signal simulation only after the forecast evidence and its limitations.

The <code>beats_best_baseline</code> field must remain visible whether true or false.
The signal output is research-only, assumes idealized execution, and makes no
profitability or investment claim.

Data-derived artifacts remain subject to the CC BY-NC 4.0 terms documented in
[DATA_LICENSE.md](../DATA_LICENSE.md).
