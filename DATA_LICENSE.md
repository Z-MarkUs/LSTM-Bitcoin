# Data License and Attribution

## License boundary

The MIT license in [LICENSE](LICENSE) applies only to the new source code and
documentation authored for this rebuild. It does not relicense market data, a processed
data subset, or an artifact derived from third-party data.

The reference experiment uses a minimal subset of Coin Metrics Community Data. That
data remains available under the
[Creative Commons Attribution-NonCommercial 4.0 International license](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0).

## Reference source

| Field | Frozen value |
| --- | --- |
| Provider | Coin Metrics |
| Dataset | Community Data BTC daily archive |
| Repository | [coinmetrics/data](https://github.com/coinmetrics/data) |
| Source revision | <code>f1a36afb962731c387bb03982758ab0103063da5</code> |
| Source file | <code>csv/btc.csv</code> |
| Raw SHA-256 | <code>06495ff8e643432e6948b7b4686ce44fc106217287dabdc1b38351d9ddec46c3</code> |
| Selected fields | <code>time</code> and <code>PriceUSD</code> |
| Selected period | 2013-01-01 through 2025-12-31, inclusive |
| Time convention | Coin Metrics beginning-of-interval date label; daily close at the 00:00 UTC cutoff |
| Processed rows | 4,748 complete daily observations |
| Data license | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

Requested attribution:

> Coin Metrics Community Data, BTC daily PriceUSD.

Modification notice: the project selects the <code>time</code> and
<code>PriceUSD</code> fields, skips source rows without a price, retains the documented
2013-01-01 through 2025-12-31 interval, renames the two output fields, and generates
the derived forecasts, tables, and figures identified by the release manifest.

The upstream license statement is recorded in the
[Coin Metrics README at the pinned revision](https://github.com/coinmetrics/data/blob/f1a36afb962731c387bb03982758ab0103063da5/README.md#license).
The downloader verifies the exact raw bytes before processing; a newer copy from the
same URL or branch is not silently accepted.

## Noncommercial restriction

CC BY-NC 4.0 permits sharing and adaptation with attribution for noncommercial
purposes, subject to its full terms. The MIT license on this project's code does not
remove that restriction. Anyone planning commercial use must obtain suitable data
rights from Coin Metrics or replace the reference dataset with a source whose terms
permit that use.

This repository is published as a non-monetized research portfolio. Creative
Commons explains that whether a particular use is NonCommercial depends on its
purpose and circumstances. This project does not claim that every employment-related
or business-context use automatically qualifies. If a use may be primarily directed
toward commercial advantage, obtain permission from Coin Metrics or substitute data
with suitable rights. See the
[Creative Commons NonCommercial guidance](https://creativecommons.org/faq/#does-my-use-violate-the-noncommercial-clause-of-the-licenses).

The processed two-column snapshot, reference predictions, tables, and figures may be
derived from the licensed data. Treat those materials as CC BY-NC 4.0 unless a
rights-qualified review establishes otherwise. This file is a project licensing
notice, not legal advice.

## Excluded legacy material

The maintained source tree and new release artifacts do not distribute the prior
Glassnode-, WRDS-, Google-, Look Into Bitcoin-, or Blockchain.com-derived datasets,
the old scraping dump, copied legacy code, presentation, or serialized model files.
Their provenance and redistribution rights were not adequately documented. Removing
them does not grant permission to recover or reuse copies from an old clone or commit.

No release or tag is created for the pre-rebuild artifact dump. See
[docs/legacy-coursework.md](docs/legacy-coursework.md) for the historical audit and
[docs/data-provenance.md](docs/data-provenance.md) for the maintained lineage.
