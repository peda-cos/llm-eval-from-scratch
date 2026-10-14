import math
import time

import pytest

from main import RESULT_MARKER, run_candidate, score_code_exec


def test_all_assertions_pass() -> None:
    code = "def add(a, b):\n    return a + b\n"
    targets = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]

    result = run_candidate(code, targets)

    assert result.exit_code == "pass"
    assert result.score == 1.0
    assert result.passed == 2
    assert result.total == 2
    assert len(result.assertions) == 2
    assert all(outcome.passed for outcome in result.assertions)
    assert result.detail == ""


def test_partial_assertions_score_a_fraction() -> None:
    code = "def square(x):\n    return x * x\n"
    targets = [
        "assert square(2) == 4",
        "assert square(3) == 9",
        "assert square(0) == 1",
    ]

    result = run_candidate(code, targets)

    assert result.exit_code == "assertion_fail"
    assert result.total == 3
    assert result.passed == 2
    assert math.isclose(result.score, 2 / 3, rel_tol=0.0, abs_tol=1e-12)
    assert [outcome.passed for outcome in result.assertions] == [True, True, False]
    assert "1 of 3" in result.detail
    assert result.assertions[2].detail


def test_top_level_candidate_error_is_an_error_not_an_assertion_failure() -> None:
    result = run_candidate('raise ValueError("boom")', ["assert True"])

    assert result.exit_code == "error"
    assert result.score == 0.0
    assert result.total == 1
    assert "ValueError" in result.detail
    assert "boom" in result.detail


def test_syntax_error_is_a_distinct_exit_code() -> None:
    result = run_candidate("def broken(:\n    pass\n", ["assert True"])

    assert result.exit_code == "syntax_error"
    assert result.score == 0.0
    assert result.detail.startswith("SyntaxError")


def test_syntax_error_during_import_time_exec_is_a_syntax_error() -> None:
    result = run_candidate('exec("def broken(:")\n', ["assert True"])

    assert result.exit_code == "syntax_error"
    assert "SyntaxError" in result.detail


def test_candidate_top_level_assertion_is_an_error() -> None:
    result = run_candidate("assert False", ["assert True"])

    assert result.exit_code == "error"
    assert "AssertionError" in result.detail


def test_static_denied_import_is_blocked() -> None:
    result = run_candidate("import subprocess\n", ["assert True"])

    assert result.exit_code == "error"
    assert result.score == 0.0
    assert "denied" in result.detail


@pytest.mark.parametrize(
    "module",
    [
        "subprocess",
        "socket",
        "requests",
        "urllib.parse",
        "ctypes",
        "shutil",
        "http.client",
        "asyncio.subprocess",
    ],
)
def test_every_denied_module_is_blocked(module: str) -> None:
    result = run_candidate(f"import {module}\n", ["assert True"])

    assert result.exit_code == "error"
    assert result.score == 0.0
    assert "denied" in result.detail


def test_dynamic_denied_import_is_blocked() -> None:
    result = run_candidate('__import__("socket")\n', ["assert True"])

    assert result.exit_code == "error"
    assert "denied" in result.detail


def test_os_system_is_monkey_patched_to_deny() -> None:
    result = run_candidate(
        'from os import system\nsystem("true")\n', ["assert True"]
    )

    assert result.exit_code == "error"
    assert "denied: os.system" in result.detail


def test_wall_clock_timeout() -> None:
    started = time.monotonic()
    result = run_candidate("while True:\n    pass\n", ["assert True"], 0.2)
    elapsed = time.monotonic() - started

    assert result.exit_code == "timeout"
    assert result.score == 0.0
    assert result.assertions == ()
    assert elapsed < 2.0


def test_zero_timeout_is_clamped_upward() -> None:
    started = time.monotonic()
    result = run_candidate("while True:\n    pass\n", ["assert True"], 0.0)
    elapsed = time.monotonic() - started

    assert result.exit_code == "timeout"
    assert elapsed < 2.0


def test_output_overflow_is_killed_with_the_expected_detail() -> None:
    code = "import os\nwhile True:\n    os.write(1, b'x' * 8192)\n"

    result = run_candidate(code, ["assert True"], 2.0)

    assert result.exit_code == "error"
    assert result.score == 0.0
    assert result.detail == "output overflow"


def test_child_that_exits_without_an_envelope_is_normalized() -> None:
    result = run_candidate("import os\nos._exit(0)\n", ["assert True"])

    assert result.exit_code == "error"
    assert result.score == 0.0
    assert result.total == 1
    assert "envelope" in result.detail or "runner exited" in result.detail


def test_stdout_noise_and_atexit_trailing_output_do_not_break_parsing() -> None:
    code = (
        "import atexit\n"
        "print('noise ' * 100)\n"
        f"print({RESULT_MARKER + 'not-json'!r})\n"
        "atexit.register(lambda: print('late noise'))\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )

    result = run_candidate(code, ["assert add(1, 2) == 3"])

    assert result.exit_code == "pass"
    assert result.score == 1.0


def test_targets_with_quotes_newlines_and_non_ascii_round_trip() -> None:
    code = (
        "def greet(name):\n"
        '    return "café " + name\n'
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )
    targets = [
        'assert greet("x") == "café x"',
        'assert add(1, 2) == 3  # __TARGETS_JSON__\nassert "quoted" == "quoted"',
    ]

    result = run_candidate(code, targets)

    assert result.exit_code == "pass"
    assert result.passed == 2
    assert result.total == 2


def test_score_code_exec_extracts_and_runs_the_first_fenced_block() -> None:
    generation = (
        "Here is the implementation:\n"
        "```python\n"
        "def double(x):\n"
        "    return x * 2\n"
        "```\n"
    )

    result = score_code_exec(generation, ["assert double(3) == 6"])

    assert result.exit_code == "pass"
    assert result.score == 1.0


@pytest.mark.parametrize("generation", ["no fence", "```\n\n```", "```\n   \n```"])
def test_score_code_exec_rejects_missing_or_empty_code_blocks(
    generation: str,
) -> None:
    result = score_code_exec(generation, ["assert True"])

    assert result.exit_code == "error"
    assert result.score == 0.0
    assert result.detail == "no code block found"
    assert result.total == 1


def test_empty_code_is_still_a_low_level_candidate() -> None:
    result = run_candidate("", ["assert True"])

    assert result.exit_code == "pass"
    assert result.score == 1.0


@pytest.mark.parametrize("targets", [[], "assert True", [""], ["   "]])
def test_invalid_targets_raise_value_error(targets: object) -> None:
    with pytest.raises(ValueError):
        run_candidate("x = 1", targets)  # type: ignore[arg-type]


def test_syntax_error_inside_a_supplied_assertion_is_an_assertion_failure() -> None:
    result = run_candidate("x = 1", ["def broken(:"])

    assert result.exit_code == "assertion_fail"
    assert result.passed == 0
    assert result.total == 1
    assert "SyntaxError" in result.assertions[0].detail
