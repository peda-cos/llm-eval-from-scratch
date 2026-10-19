from __future__ import annotations

import pytest

from eval_runner.code_exec import extract_code, run_candidate, score_code_exec


def test_candidate_passes_all_assertions() -> None:
    result = run_candidate(
        "def add(a, b):\n    return a + b\n",
        ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
    )
    assert result.exit_code == "pass"
    assert result.score == 1.0
    assert result.passed == result.total == 2
    assert result.detail == ""


def test_candidate_partial_assertion_failure() -> None:
    result = run_candidate(
        "def identity(value):\n    return value\n",
        ["assert identity(1) == 1", "assert identity(2) == 3"],
    )
    assert result.exit_code == "assertion_fail"
    assert result.score == 0.5
    assert result.passed == 1
    assert result.total == 2
    assert "1 of 2" in result.detail


def test_candidate_syntax_error_is_normalized() -> None:
    result = run_candidate("def broken(:\n    pass\n", ["assert True"])
    assert result.exit_code == "syntax_error"
    assert result.score == 0.0
    assert "SyntaxError" in result.detail


def test_candidate_timeout() -> None:
    result = run_candidate("while True:\n    pass\n", ["assert True"], timeout_s=0.2)
    assert result.exit_code == "timeout"
    assert result.score == 0.0
    assert "timeout" in result.detail


def test_missing_or_unclosed_fence_scores_zero_without_spawning() -> None:
    missing = score_code_exec("plain text, no fence", ["assert True"])
    assert missing.exit_code == "error"
    assert missing.score == 0.0
    assert "no code block" in missing.detail

    unclosed = score_code_exec("```python\nassert True", ["assert True"])
    assert unclosed.exit_code == "error"
    assert unclosed.score == 0.0

    fenced = score_code_exec(
        "```python\nvalue = 2 + 2\n```",
        ["assert value == 4"],
    )
    assert fenced.exit_code == "pass"
    assert fenced.score == 1.0
    assert extract_code("```python\nx = 1\n```") == "x = 1"


def test_output_over_cap_is_rejected() -> None:
    result = run_candidate('print("x" * 300000)\n', ["assert True"], timeout_s=2.0)
    assert result.exit_code == "error"
    assert result.score == 0.0
    assert "output overflow" in result.detail


def test_denied_import_and_empty_assertions_are_rejected() -> None:
    denied = run_candidate("import socket\n", ["assert True"])
    assert denied.exit_code == "error"
    assert "denied" in denied.detail

    with pytest.raises(ValueError, match="assertions"):
        run_candidate("x = 1", [])
    with pytest.raises(ValueError, match="assertions"):
        run_candidate("x = 1", [""])
    with pytest.raises(ValueError, match="assertions"):
        run_candidate("x = 1", "assert True")  # type: ignore[arg-type]
