# Data Provenance

## Frozen lineage

The reference run deliberately uses one simple, auditable input rather than combining
datasets with incompatible publication schedules and unclear redistribution terms.

| Stage | Evidence |
| --- | --- |
| Upstream repository | [coinmetrics/data](https://github.com/coinmetrics/data) |
| Immutable revision | <code>f1a36afb962731c387bb03982758ab0103063da5</code> |
| Upstream path | <code>csv/btc.csv</code> |
| Raw SHA-256 | <code>06495ff8e643432e6948b7b4686ce44fc106217287dabdc1b38351d9ddec46c3</code> |
| License at that revision | [CC BY-NC 4.0](https://github.com/coinmetrics/data/blob/f1a36afb962731c387bb03982758ab0103063da5/README.md#license) |
| Time convention | Coin Metrics beginning-of-interval date label; daily close at the 00:00 UTC cutoff |
| Transformation | Select <code>time</code> and positive <code>PriceUSD</code>, rename to <code>date</code> and <code>close_usd</code>, filter complete calendar years 2013–2025 |
| Frozen subset | 4,748 unique, ordered, gap-free daily rows |
| Local evidence | Generated manifest with source identity, transform policy, row count, date bounds, and processed SHA-256 |

The raw digest was independently verified against the pinned source while preparing
the rebuild. The project still verifies it on every authorized fetch; documentation
alone is not a trust boundary.

## Retrieval and transformation controls

The constants in <code>src/lstm_bitcoin/data.py</code> pin the provider, commit, path,
raw digest, allowed HTTPS host, maximum response size, and reference period. Fetching
is an explicit CLI action. The package performs no network work at import time.

Processing is intentionally narrow:

- empty upstream <code>PriceUSD</code> values are skipped before selecting the
  reference period;
- retained prices must be finite and positive;
- retained dates must be unique, strictly increasing, and exactly one day apart;
- output formatting is deterministic and contains only two columns;
- the snapshot and manifest use atomic replacement so an interrupted write cannot
  masquerade as a valid dataset.

<code>lstm-bitcoin data validate</code> recalculates the processed digest, reloads the
strict schema, and checks the observed row count and date bounds against the manifest.
<code>lstm-bitcoin verify --run ...</code> separately checks the evidence produced by
an experiment.

## Why the legacy inputs are excluded

The historical repository mixed a copied Bitcoin dataset with fields attributed to
Glassnode, WRDS, the New York Fed, Google, and other providers. It later added large
Blockchain.com and Look Into Bitcoin exports produced by an undocumented scraper.
There was no complete data card, retrieval manifest, immutable source revision, or
redistribution grant. The base <code>btc_dataset.csv</code> was also byte-identical to
a file in another public repository that carried no license.

Those CSV files, derived proxy file, scraping directory, presentation, environment
dump, and saved models are removed from the maintained project and excluded from new
releases. This rebuild makes no claim that the former files were redistributable.

## License boundary

New project code and documentation are MIT-licensed. Coin Metrics data and materials
derived from its reference subset remain subject to CC BY-NC 4.0. The exact boundary,
attribution, and noncommercial restriction are documented in
[DATA_LICENSE.md](../DATA_LICENSE.md).
