"""Command-line interface for data, evaluation, reporting, and verification."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .artifacts import verify_checksums
from .config import load_config
from .data import fetch_reference_data, validate_reference_data
from .experiment import RUN_FILES, run_experiment
from .plotting import generate_plots

DEFAULT_CONFIG = Path("configs/reference.toml")
DEFAULT_DATA = Path("data/btc_usd_daily.csv")
DEFAULT_SOURCE_MANIFEST = Path("data/source-manifest.json")
DEFAULT_RUN = Path("evaluation/results/reference-v1")
DEFAULT_PLOTS = Path("docs/assets")


def _verify_result_bundle(run: Path) -> tuple[str, ...]:
    verified = verify_checksums(run)
    required = set(RUN_FILES) - {"SHA256SUMS"}
    if set(verified) != required:
        missing = ", ".join(sorted(required - set(verified))) or "none"
        unexpected = ", ".join(sorted(set(verified) - required)) or "none"
        raise ValueError(f"incomplete result bundle; missing: {missing}; unexpected: {unexpected}")
    return verified


def _shared_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--models", choices=("baselines", "all"), default="all")
    parser.add_argument("--revision", help="Code revision to record instead of auto-detection")
    parser.add_argument("--force", action="store_true", help="Replace known generated files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lstm-bitcoin",
        description="Leakage-controlled Bitcoin return forecasting research.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Fetch or validate the frozen reference data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch", help="Download and hash-check the pinned source")
    fetch.add_argument("--output", type=Path, default=DEFAULT_DATA)
    fetch.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    fetch.add_argument("--force", action="store_true")
    validate = data_commands.add_parser("validate", help="Validate data and provenance locally")
    validate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    validate.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)

    evaluate = commands.add_parser("evaluate", help="Run the chronological benchmark")
    _shared_evaluate_arguments(evaluate)

    report = commands.add_parser("report", help="Generate evidence-first SVG plots")
    report.add_argument("--run", type=Path, default=DEFAULT_RUN)
    report.add_argument("--output", type=Path, default=DEFAULT_PLOTS)
    report.add_argument("--force", action="store_true")

    verify = commands.add_parser("verify", help="Verify a result bundle's SHA-256 checksums")
    verify.add_argument("--run", type=Path, default=DEFAULT_RUN)

    reproduce = commands.add_parser(
        "reproduce", help="Validate inputs, evaluate, generate plots, and verify outputs"
    )
    _shared_evaluate_arguments(reproduce)
    reproduce.add_argument("--plots", type=Path, default=DEFAULT_PLOTS)
    return parser


def _evaluate(arguments: argparse.Namespace) -> None:
    validate_reference_data(arguments.data, arguments.manifest)
    config = load_config(arguments.config)
    run_experiment(
        config=config,
        config_path=arguments.config,
        data_path=arguments.data,
        source_manifest_path=arguments.manifest,
        output=arguments.output,
        models=arguments.models,
        force=arguments.force,
        revision=arguments.revision,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "data" and arguments.data_command == "fetch":
            metadata = fetch_reference_data(
                arguments.output, arguments.manifest, force=arguments.force
            )
            processed = metadata["processed"]
            print(
                f"Wrote {processed['rows']} verified rows "
                f"({processed['start_date']} to {processed['end_date']})."
            )
        elif arguments.command == "data" and arguments.data_command == "validate":
            metadata = validate_reference_data(arguments.data, arguments.manifest)
            print(f"Data verified: {metadata['processed']['sha256']}")
        elif arguments.command == "evaluate":
            _evaluate(arguments)
            print(f"Evaluation written to {arguments.output}")
        elif arguments.command == "report":
            generate_plots(arguments.run, arguments.output, force=arguments.force)
            print(f"Plots written to {arguments.output}")
        elif arguments.command == "verify":
            verified = _verify_result_bundle(arguments.run)
            print(f"Verified {len(verified)} artifacts in {arguments.run}")
        elif arguments.command == "reproduce":
            _evaluate(arguments)
            generate_plots(arguments.output, arguments.plots, force=arguments.force)
            verified = _verify_result_bundle(arguments.output)
            print(f"Reproduced and verified {len(verified)} artifacts in {arguments.output}")
        else:  # pragma: no cover - argparse enforces valid commands
            parser.error("unsupported command")
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
