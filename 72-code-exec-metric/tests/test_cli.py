import json
from pathlib import Path
from typing import Any

import pytest

import main as main_module
from main import DEMO_FIXTURE, DemoFixtureError, load_demo_cases, main

PASS_GENERATION = "```python\ndef add(a, b):\n    return a + b\n```"


def _sample(**overrides: Any) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "generation": PASS_GENERATION,
        "expected_exit": "pass",
        "expected_score": 1.0,
    }
    sample.update(overrides)
    return sample


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_id": "task",
        "targets": ["assert add(1, 2) == 3"],
        "samples": [_sample()],
    }
    record.update(overrides)
    return record


def _write_cases(path: Path, records: list[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_demo_fixture_loads_and_covers_every_exit_code() -> None:
    cases = load_demo_cases(DEMO_FIXTURE)

    assert [case.task_id for case in cases] == ["add_two", "failure_modes"]
    assert [len(case.samples) for case in cases] == [5, 6]
    assert {sample.expected_exit for case in cases for sample in case.samples} == {
        "pass",
        "assertion_fail",
        "syntax_error",
        "timeout",
        "error",
    }


def test_demo_cli_verifies_fixture_expectations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo"]) == 0

    captured = capsys.readouterr()
    assert "add_two: n=5 c=3 pass@1=0.600" in captured.out
    assert "add_two: n=5 c=3 pass@3=1.000" in captured.out
    assert "failure_modes: n=6 c=1 pass@3=0.500" in captured.out
    assert "demo: ok" in captured.out
    assert "MISMATCH" not in captured.out


def test_demo_cli_rejects_unknown_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["not-a-command"]) == 2

    captured = capsys.readouterr()
    assert "usage: python main.py [demo]" in captured.err


def test_demo_cli_reports_missing_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_module, "DEMO_FIXTURE", tmp_path / "missing.jsonl")

    assert main_module.main(["demo"]) == 2

    captured = capsys.readouterr()
    assert "cannot read" in captured.err


def test_demo_cli_reports_a_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "mismatch.jsonl"
    _write_cases(fixture, [_record(samples=[_sample(expected_exit="error")])])
    monkeypatch.setattr(main_module, "DEMO_FIXTURE", fixture)

    assert main_module.main(["demo"]) == 1

    captured = capsys.readouterr()
    assert "MISMATCH" in captured.out
    assert "demo: mismatch" in captured.out


def test_loader_skips_blank_lines(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(f"\n{json.dumps(_record())}\n\n", encoding="utf-8")

    cases = load_demo_cases(fixture)

    assert len(cases) == 1
    assert cases[0].task_id == "task"


def test_loader_rejects_an_empty_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "empty.jsonl"
    fixture.write_text("", encoding="utf-8")

    with pytest.raises(DemoFixtureError, match="no demo cases"):
        load_demo_cases(fixture)


def test_loader_rejects_malformed_json(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text("{\n", encoding="utf-8")

    with pytest.raises(DemoFixtureError, match="invalid JSON"):
        load_demo_cases(fixture)


def test_loader_rejects_a_non_object_record(tmp_path: Path) -> None:
    fixture = tmp_path / "list.jsonl"
    fixture.write_text("[]\n", encoding="utf-8")

    with pytest.raises(DemoFixtureError, match="record must be an object"):
        load_demo_cases(fixture)


def test_loader_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    fixture = tmp_path / "duplicate.jsonl"
    _write_cases(fixture, [_record(), _record()])

    with pytest.raises(DemoFixtureError, match="duplicate task_id"):
        load_demo_cases(fixture)


@pytest.mark.parametrize(
    "record",
    [
        _record(extra=1),
        _record(targets=[]),
        _record(targets=["   "]),
        _record(samples=[]),
        _record(samples=[_sample(extra=1)]),
        _record(samples=[_sample(expected_exit="unknown")]),
        _record(samples=[_sample(expected_score=True)]),
        _record(samples=[_sample(expected_score=2.0)]),
        _record(samples=[_sample(generation="   ")]),
    ],
)
def test_loader_rejects_invalid_fixtures(
    tmp_path: Path, record: dict[str, Any]
) -> None:
    fixture = tmp_path / "invalid.jsonl"
    _write_cases(fixture, [record])

    with pytest.raises(DemoFixtureError):
        load_demo_cases(fixture)
