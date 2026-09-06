# Local Data Workspace

This directory is a workspace for verified local inputs. The project does not require
an API key, and importing the package does not access the network.

## Fetch and validate

From a locked development environment:

~~~bash
uv run --frozen lstm-bitcoin data validate
~~~

For a deliberate byte-for-byte rebuild from the pinned upstream source:

~~~bash
uv run --frozen lstm-bitcoin data fetch --force
uv run --frozen lstm-bitcoin data validate
~~~

The reviewed default snapshot is <code>data/btc_usd_daily.csv</code>; its provenance
record is <code>data/source-manifest.json</code>. Both are tracked so the canonical
study remains reproducible without a live network request, and either can be rebuilt
byte-for-byte from the pinned upstream source.

The fetch command:

1. requests the exact Coin Metrics file at the pinned commit over HTTPS;
2. rejects redirects outside <code>raw.githubusercontent.com</code>;
3. refuses a response larger than 5,000,000 bytes;
4. verifies the raw SHA-256 before parsing;
5. selects only <code>time</code> and <code>PriceUSD</code>;
6. keeps 2013-01-01 through 2025-12-31;
7. writes a deterministic <code>date,close_usd</code> CSV and a provenance manifest
   atomically.

The tracked snapshot should normally be validated in place. Unless <code>--force</code>
is explicitly supplied, the fetch command will not overwrite that snapshot or its
manifest. Validation fails on a digest mismatch, an unexpected schema, a non-positive
or non-finite price, a duplicate date, an out-of-order row, or a missing calendar day.

The expected processed snapshot contains 4,748 daily observations. Its SHA-256 is
recorded in the generated manifest because the manifest, rather than an informal
filename, is the evidence that identifies a local copy.

## Version-control policy

- Do not commit the full downloaded Coin Metrics archive or local caches.
- Do not commit credentials, provider exports, WRDS data, scraped website data, model
  weights, or ad hoc notebook copies.
- Tests use small synthetic fixtures and remain offline.
- Only the reviewed two-column snapshot, its source manifest, and the canonical
  evaluation under <code>evaluation/results/reference-v1</code> may be tracked as
  generated evidence.

The data and data-derived artifacts are not covered by the repository's MIT code
license. Read [DATA_LICENSE.md](../DATA_LICENSE.md) before sharing or adapting them.
