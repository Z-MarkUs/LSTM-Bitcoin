from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lstm_bitcoin import cli
from lstm_bitcoin.artifacts import write_checksums


def test_parser_has_required_commands_and_defaults() -> None:
    parser = cli.build_parser()
    arguments = parser.parse_args(["evaluate", "--models", "baselines"])
    assert arguments.command == "evaluate"
    assert arguments.config == cli.DEFAULT_CONFIG
    assert arguments.models == "baselines"
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_data_fetch_and_validate_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "fetch_reference_data",
        lambda output, manifest, force: {
            "processed": {
                "rows": 2,
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
            }
        },
    )
    assert cli.main(["data", "fetch", "--force"]) == 0
    assert "Wrote 2 verified rows" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "validate_reference_data",
        lambda data, manifest: {"processed": {"sha256": "abc"}},
    )
    assert cli.main(["data", "validate"]) == 0
    assert "Data verified: abc" in capsys.readouterr().out


def test_evaluate_report_verify_and_reproduce_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(cli, "_evaluate", lambda arguments: calls.append(("evaluate", arguments)))
    monkeypatch.setattr(
        cli,
        "generate_plots",
        lambda run, output, force: calls.append(("report", (run, output, force))),
    )
    monkeypatch.setattr(cli, "_verify_result_bundle", lambda run: ("a", "b"))

    assert cli.main(["evaluate", "--models", "baselines"]) == 0
    assert "Evaluation written" in capsys.readouterr().out
    assert cli.main(["report", "--force"]) == 0
    assert "Plots written" in capsys.readouterr().out
    assert cli.main(["verify"]) == 0
    assert "Verified 2 artifacts" in capsys.readouterr().out
    assert cli.main(["reproduce", "--models", "baselines", "--force"]) == 0
    assert "Reproduced and verified 2 artifacts" in capsys.readouterr().out
    assert [name for name, _ in calls] == ["evaluate", "report", "evaluate", "report"]


def test_evaluate_helper_validates_before_running(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        cli, "validate_reference_data", lambda data, manifest: order.append("validate")
    )
    monkeypatch.setattr(cli, "load_config", lambda path: order.append("config") or object())
    monkeypatch.setattr(cli, "run_experiment", lambda **kwargs: order.append("run"))
    arguments = cli.build_parser().parse_args(["evaluate"])
    cli._evaluate(arguments)
    assert order == ["validate", "config", "run"]


@pytest.mark.parametrize("exception", [FileNotFoundError("missing"), ValueError("bad")])
def test_cli_returns_two_and_prints_concise_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exception: Exception,
) -> None:
    monkeypatch.setattr(
        cli,
        "validate_reference_data",
        lambda data, manifest: (_ for _ in ()).throw(exception),
    )
    assert cli.main(["data", "validate"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_passes_explicit_paths_and_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "validate_reference_data", lambda data, manifest: {})
    monkeypatch.setattr(cli, "load_config", lambda path: "config")
    monkeypatch.setattr(cli, "run_experiment", lambda **kwargs: captured.update(kwargs))
    output = tmp_path / "run"
    assert (
        cli.main(
            [
                "evaluate",
                "--config",
                "custom.toml",
                "--data",
                "custom.csv",
                "--manifest",
                "custom.json",
                "--output",
                str(output),
                "--revision",
                "abc123",
                "--force",
            ]
        )
        == 0
    )
    assert captured["config"] == "config"
    assert captured["output"] == output
    assert captured["revision"] == "abc123"
    assert captured["force"] is True


def test_verify_rejects_an_incomplete_result_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text("{}\n", encoding="utf-8")
    write_checksums(run, ("metrics.json",))

    assert cli.main(["verify", "--run", str(run)]) == 2
    assert "incomplete" in capsys.readouterr().err.lower()
