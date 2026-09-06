# Releasing LSTM Bitcoin Research

Releases are built from an annotated tag on `main`. The tag must identify the exact
commit that passed CI; the privileged publishing workflow does not check out or execute
tag-controlled code.

## Prepare

1. Choose a Semantic Versioning release number and update the version in
   `pyproject.toml`.
2. Move the relevant `CHANGELOG.md` entries from `Unreleased` into a dated release
   section.
3. Confirm that the source manifest, data attribution, methodology, metric definitions,
   and limitations still match the reference artifacts.
4. Reproduce and verify the reference evaluation in the locked Python 3.12 environment:

   ```bash
   uv lock --check
   uv sync --locked --extra lstm --python 3.12
   uv run --frozen lstm-bitcoin reproduce \
     --config configs/reference.toml \
     --output evaluation/results/reference-v1 \
     --plots docs/assets \
     --force
   uv run --frozen lstm-bitcoin verify \
     --run evaluation/results/reference-v1
   ```

5. Run the complete contributor quality gate in the locked development and LSTM
   environment, then build the distributions with dependency sources disabled:

   ```bash
   uv sync --locked --extra dev --extra lstm --python 3.12
   uv run --frozen ruff check src tests scripts notebooks
   uv run --frozen ruff format --check src tests scripts notebooks
   uv run --frozen mypy src/lstm_bitcoin
   uv run --frozen pytest -m "not network" --cov-fail-under=90
   uv build --no-sources
   ```

6. Inspect the wheel, source archive, reference-result bundle, and checksums from a
   clean checkout. Install both Python distributions into fresh environments and
   smoke-test `lstm-bitcoin --help`.
7. Merge through a focused pull request only after CI, CodeQL, and the reference
   evaluation are green.

## Tag

Create an annotated tag on the verified `main` commit. Do not move or reuse a release
tag.

```bash
git switch main
git pull --ff-only
version="$(python -c 'import pathlib,tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])')"
tag="v${version}"
git tag --annotate "$tag" --message "LSTM Bitcoin Research $tag"
git push origin "refs/tags/$tag"
```

The tag runs the same lint, typing, test, security, build, reference-integrity, and
packaging gates as `main`, plus a full reference reproduction. The reproduction
compares every canonical JSON and CSV artifact with the checked evidence. Because the
checked bundle was generated on Windows while the release runner uses Linux, numeric
comparisons allow only `1e-5` relative and `1e-7` absolute tolerance; structure,
schemas, dates, identifiers, provenance inputs, and file sets must still match. CI
creates SHA-256 checksums, a CycloneDX SBOM for the locked default runtime, and
provenance attestations. After those jobs pass, the release workflow verifies the
annotated tag, main-branch ancestry, artifacts, checksums, and attestations before
creating the GitHub release.

The reference archive is a deterministic snapshot of the complete tracked tree plus
an internal manifest that binds every file to the verified tag commit and its SHA-256.
It therefore carries the code, lockfile, frozen data, data-license notice, notebook,
and reviewed evidence needed to inspect or rerun the study together.

The reference bundle contains research evidence, not a claim of trading profitability.
Do not publish to PyPI or another registry without a separate maintainer decision and
trusted-publisher review.

## Verify the public release

```bash
tag="vX.Y.Z"
commit="$(git rev-list --max-count=1 "$tag")"
gh release view "$tag" --repo Z-MarkUs/LSTM-Bitcoin
gh release verify "$tag" --repo Z-MarkUs/LSTM-Bitcoin
gh release download "$tag" --repo Z-MarkUs/LSTM-Bitcoin --dir release-download
(cd release-download && sha256sum --check SHA256SUMS)
for artifact in \
  release-download/*.whl \
  release-download/*.tar.gz \
  release-download/*.cdx.json \
  release-download/SHA256SUMS; do
  gh attestation verify "$artifact" \
    --repo Z-MarkUs/LSTM-Bitcoin \
    --source-ref "refs/tags/$tag" \
    --source-digest "$commit" \
    --signer-workflow Z-MarkUs/LSTM-Bitcoin/.github/workflows/ci.yml
done
```

Also install the public wheel and source archive in fresh environments, run
`pip check`, and verify the CLI. If immutable releases are enabled in the repository,
confirm the published release reports that it is immutable. Correct mistakes with a
new patch release rather than replacing assets or moving a tag.
