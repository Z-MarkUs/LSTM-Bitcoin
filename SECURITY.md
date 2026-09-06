# Security Policy

## Supported versions

Security fixes are applied to the latest release and the current `main` branch. Older
coursework revisions are retained only as history and are not supported.

## Reporting a vulnerability

Please use GitHub's **Security** tab and **Report a vulnerability** to submit details
privately. Do not open a public issue containing an exploit, credential, private URL,
or sensitive dataset. If private vulnerability reporting is unavailable, open a minimal
issue asking the maintainer to establish a private channel without disclosing the
vulnerability.

Include the affected version or commit, impact, reproduction steps, and any suggested
mitigation. The maintainer will acknowledge reports on a best-effort basis and coordinate
disclosure after a fix is available.

## Security boundaries

- The CLI does not require API keys for the frozen reference experiment.
- Data retrieval must be an explicit command; importing the package must not perform
  network requests.
- Reference inputs are restricted to the documented source and verified against the
  SHA-256 digest in the data manifest.
- The project does not load untrusted pickle, HDF5, Keras, or PyTorch model files.
  Reference models are retrained from configuration rather than distributed as
  executable serialized objects.
- Release distributions and result bundles are checksummed and receive GitHub build
  provenance attestations.
- Continuous integration runs static analysis, dependency auditing, branch-aware tests,
  and CodeQL with minimal workflow permissions.

Market-model error, poor forecast performance, or investment loss is not a software
security vulnerability. This repository is research software and does not provide
financial advice or a production trading service.

