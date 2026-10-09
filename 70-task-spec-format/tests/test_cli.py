from pathlib import Path

import pytest

from main import BAD_FIXTURE, GOOD_FIXTURE
from main import main as cli_main


def test_cli_validate_good_fixture_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["validate", str(GOOD_FIXTURE)]) == 0

    captured = capsys.readouterr()
    assert "10 valid, 0 invalid" in captured.out
    assert captured.err == ""


def test_cli_validate_bad_fixture_exits_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["validate", str(BAD_FIXTURE)]) == 1

    output = capsys.readouterr().out
    assert "1 valid, 16 invalid" in output
    assert "duplicate_task_id" in output


def test_cli_allow_bad_tasks_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["validate", str(BAD_FIXTURE), "--allow-bad-tasks"]) == 0

    output = capsys.readouterr().out
    assert "16 error(s); continuing because --allow-bad-tasks is set" in output


def test_cli_missing_file_exits_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "missing.jsonl"

    assert cli_main(["validate", str(missing)]) == 2

    captured = capsys.readouterr()
    assert "cannot read" in captured.err


def test_cli_demo_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["demo"]) == 0

    output = capsys.readouterr().out
    assert "valid fixture: 10 tasks, 0 errors" in output
    assert "bad fixture: 1 task(s) accepted, 16 errors" in output
    assert "rules tripped: 16" in output


def test_cli_default_command_runs_demo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main([]) == 0

    assert "valid fixture: 10 tasks, 0 errors" in capsys.readouterr().out


def test_cli_validate_default_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["validate"]) == 0

    assert "10 valid, 0 invalid" in capsys.readouterr().out
