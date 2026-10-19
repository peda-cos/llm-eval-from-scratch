from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_cli_demo_writes_ok_envelope_and_markdown(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    result = _cli(
        "demo",
        "--json-out",
        str(report_path),
        "--bootstrap",
        "50",
        "--seed",
        "7",
    )
    assert result.returncode == 0, result.stderr
    assert "tasks x 3 models" in result.stderr
    assert "| 1 | rule_based |" in result.stdout
    assert "| Model | ECE | Brier |" in result.stdout

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["schema_version"] == 1
    assert payload["summary"]["tasks"] == 10
    assert payload["summary"]["models"] == 3
    assert len(payload["results"]) == 30
    assert set(payload["calibration"]) == {"rule_based", "biased", "noisy"}
    assert all(calibration["n_samples"] == 10 for calibration in payload["calibration"].values())
    assert payload["leaderboard"][0]["model_id"] == "rule_based"
    assert len(payload["pairwise"]) == 3


def test_cli_without_subcommand_runs_demo(tmp_path: Path) -> None:
    report_path = tmp_path / "implicit.json"
    result = _cli("--json-out", str(report_path), "--bootstrap", "20")
    assert result.returncode == 0, result.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_cli_invalid_fixture_writes_structured_error(tmp_path: Path) -> None:
    report_path = tmp_path / "error.json"
    result = _cli(
        "demo",
        "--tasks",
        "fixtures/tasks_invalid.jsonl",
        "--json-out",
        str(report_path),
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "error[validate]" in result.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "error"
    assert payload["error"]["stage"] == "validate"
    assert payload["error"]["type"] == "FixtureValidationError"
    assert len(payload["error"]["details"]) == 3


def test_cli_rejects_zero_workers_with_usage_error(tmp_path: Path) -> None:
    result = _cli(
        "demo",
        "--workers",
        "0",
        "--json-out",
        str(tmp_path / "unused.json"),
    )
    assert result.returncode == 2
    assert "must be a positive integer" in result.stderr
