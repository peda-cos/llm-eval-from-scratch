"""Code execution scoring with a subprocess sandbox and unbiased pass-at-k.

A generation is reduced to one fenced code block, executed in a fresh Python
interpreter with a wall-clock deadline and an output cap, and scored by the
fraction of supplied assertion strings that pass. The sandbox is portable and
educational, not a security boundary.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Sequence, cast

DEFAULT_TIMEOUT_S: Final = 3.0
MIN_TIMEOUT_S: Final = 0.1
MAX_TIMEOUT_S: Final = 30.0
OUTPUT_CAP_BYTES: Final = 256 * 1024
DETAIL_LIMIT: Final = 400
RESULT_MARKER: Final = "__code_exec_result__:"
DEMO_TIMEOUT_S: Final = 1.0
DEMO_FIXTURE: Final[Path] = Path(__file__).parent / "fixtures" / "demo_cases.jsonl"

DENIED_IMPORTS: Final[tuple[str, ...]] = (
    "os.system",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "urllib.request",
    "urllib.error",
    "urllib.parse",
    "ctypes",
    "shutil",
    "http.client",
    "asyncio.subprocess",
)

ExitCode = Literal["pass", "assertion_fail", "syntax_error", "timeout", "error"]
EXIT_CODES: Final[frozenset[str]] = frozenset(
    {"pass", "assertion_fail", "syntax_error", "timeout", "error"}
)


@dataclass(frozen=True, slots=True)
class AssertionOutcome:
    """One supplied assertion's result, indexed in task order."""

    index: int
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Normalized result for one candidate, whatever the failure mode."""

    exit_code: ExitCode
    score: float
    passed: int
    total: int
    assertions: tuple[AssertionOutcome, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class DemoSample:
    generation: str
    expected_exit: ExitCode
    expected_score: float


@dataclass(frozen=True, slots=True)
class DemoCase:
    task_id: str
    targets: tuple[str, ...]
    samples: tuple[DemoSample, ...]


class DemoFixtureError(ValueError):
    """The demo JSONL does not match its small frozen schema."""


def extract_code(text: str) -> str:
    """Return the first fenced block, matching lesson 70's post-process rule."""
    lines = text.splitlines()
    opening: int | None = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            opening = index
            break
    if opening is None:
        return ""
    for index in range(opening + 1, len(lines)):
        if lines[index].lstrip().startswith("```"):
            return "\n".join(lines[opening + 1 : index])
    return ""


# The child script is passed to a fresh interpreter with ``-c``. The three
# placeholders are replaced with Python literals before spawning; the order in
# ``_build_runner`` prevents inserted target text from being rescanned.
RUNNER_TEMPLATE: Final = r'''
import builtins
import importlib.abc
import json
import os
import sys

DENIED_IMPORTS = __DENIED_IMPORTS__
MARKER = __RESULT_MARKER__
TARGETS = json.loads(__TARGETS_JSON__)
DETAIL_LIMIT = 400
_ORIGINAL_STDOUT = sys.stdout


def _short_exception(exc):
    message = str(exc)
    name = type(exc).__name__
    text = f"{name}: {message}" if message else name
    return text[:DETAIL_LIMIT]


def _is_denied(fullname):
    return any(
        fullname == denied or fullname.startswith(denied + ".")
        for denied in DENIED_IMPORTS
    )


class _DeniedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if _is_denied(fullname):
            raise ImportError(f"denied: {fullname}")
        return None


def _denied_system(*args, **kwargs):
    raise ImportError("denied: os.system")


def _flush_streams():
    for stream in (sys.stdout, sys.stderr, _ORIGINAL_STDOUT):
        try:
            stream.flush()
        except BaseException:
            pass


def _emit(status, assertions=None, error=None):
    _flush_streams()
    envelope = {
        "status": status,
        "assertions": [] if assertions is None else assertions,
        "error": error,
    }
    try:
        os.write(1, (MARKER + json.dumps(envelope) + "\n").encode("utf-8"))
    except OSError:
        raise SystemExit(1)
    raise SystemExit(0)


sys.meta_path.insert(0, _DeniedFinder())
os.system = _denied_system

try:
    source = sys.stdin.buffer.read().decode("utf-8")
except BaseException as exc:
    _emit("error", error=_short_exception(exc))

namespace = {"__name__": "__candidate__", "__builtins__": builtins}

try:
    compiled = compile(source, "<candidate>", "exec")
except SyntaxError as exc:
    _emit("syntax_error", error=_short_exception(exc))
except BaseException as exc:
    _emit("error", error=_short_exception(exc))

try:
    exec(compiled, namespace)
except SyntaxError as exc:
    _emit("syntax_error", error=_short_exception(exc))
except BaseException as exc:
    _emit("error", error=_short_exception(exc))

outcomes = []
for index, assertion in enumerate(TARGETS):
    try:
        exec(compile(assertion, f"<assertion {index}>", "exec"), namespace)
    except BaseException as exc:
        outcomes.append({"passed": False, "detail": _short_exception(exc)})
    else:
        outcomes.append({"passed": True, "detail": ""})

_emit("ok", assertions=outcomes)
'''


def _validate_targets(targets: Sequence[str]) -> list[str]:
    if isinstance(targets, str):
        raise ValueError("targets must be a non-empty sequence of non-empty strings")
    normalized = list(targets)
    if not normalized or any(
        not isinstance(target, str) or not target.strip() for target in normalized
    ):
        raise ValueError("targets must be a non-empty sequence of non-empty strings")
    return normalized


def _build_runner(targets: Sequence[str]) -> str:
    runner = RUNNER_TEMPLATE.replace(
        "__DENIED_IMPORTS__", repr(list(DENIED_IMPORTS))
    )
    runner = runner.replace("__RESULT_MARKER__", repr(RESULT_MARKER))
    return runner.replace(
        "__TARGETS_JSON__", repr(json.dumps(list(targets)))
    )


def _failure(exit_code: ExitCode, total: int, detail: str) -> ExecResult:
    return ExecResult(
        exit_code=exit_code,
        score=0.0,
        passed=0,
        total=total,
        assertions=(),
        detail=detail,
    )


def _kill_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        proc.wait()
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    else:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def _parse_envelope(output: str) -> dict[str, object] | None:
    marker_index = output.rfind(RESULT_MARKER)
    if marker_index < 0:
        return None
    start = marker_index + len(RESULT_MARKER)
    try:
        payload, _ = json.JSONDecoder().raw_decode(output, start)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _result_from_envelope(
    payload: dict[str, object], targets: Sequence[str]
) -> ExecResult:
    total = len(targets)
    status = payload.get("status")
    error = payload.get("error")
    if status not in ("ok", "syntax_error", "error"):
        return _failure("error", total, "malformed result envelope")
    if error is not None and not isinstance(error, str):
        return _failure("error", total, "malformed result envelope")

    if status == "syntax_error":
        detail = error if isinstance(error, str) and error else "syntax error"
        return _failure("syntax_error", total, detail)
    if status == "error":
        detail = error if isinstance(error, str) and error else "candidate error"
        return _failure("error", total, detail)

    raw_assertions = payload.get("assertions")
    if not isinstance(raw_assertions, list) or len(raw_assertions) != total:
        return _failure("error", total, "malformed result envelope")

    outcomes: list[AssertionOutcome] = []
    for index, item in enumerate(raw_assertions):
        if not isinstance(item, dict):
            return _failure("error", total, "malformed result envelope")
        passed = item.get("passed")
        assertion_detail = item.get("detail")
        if not isinstance(passed, bool) or not isinstance(assertion_detail, str):
            return _failure("error", total, "malformed result envelope")
        outcomes.append(
            AssertionOutcome(index=index, passed=passed, detail=assertion_detail)
        )

    passed_count = sum(1 for outcome in outcomes if outcome.passed)
    if passed_count == total:
        exit_code: ExitCode = "pass"
        result_detail = ""
    else:
        exit_code = "assertion_fail"
        result_detail = (
            f"{total - passed_count} of {total} assertion(s) failed"
        )
    return ExecResult(
        exit_code=exit_code,
        score=passed_count / total,
        passed=passed_count,
        total=total,
        assertions=tuple(outcomes),
        detail=result_detail,
    )


def run_candidate(
    code: str,
    targets: Sequence[str],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecResult:
    """Run candidate code in a fresh interpreter and score its assertions."""
    normalized = _validate_targets(targets)
    total = len(normalized)
    timeout = min(MAX_TIMEOUT_S, max(MIN_TIMEOUT_S, timeout_s))
    runner = _build_runner(normalized)

    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", runner],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()

    try:
        try:
            data = code.encode("utf-8")
            while data:
                written = proc.stdin.write(data)
                if not written:
                    break
                data = data[written:]
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

        selector.register(proc.stdout, selectors.EVENT_READ)
        chunks: list[bytes] = []
        size = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process(proc)
                return _failure(
                    "timeout", total, f"timeout after {timeout:.1f}s"
                )
            if not selector.select(remaining):
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            size += len(chunk)
            if size > OUTPUT_CAP_BYTES:
                _kill_process(proc)
                return _failure("error", total, "output overflow")
            chunks.append(chunk)

        wait_remaining = deadline - time.monotonic()
        try:
            proc.wait(timeout=max(wait_remaining, 0.01))
        except subprocess.TimeoutExpired:
            _kill_process(proc)
            return _failure("timeout", total, f"timeout after {timeout:.1f}s")

        output = b"".join(chunks).decode("utf-8", errors="replace")
        if proc.returncode != 0:
            tail = output[-300:].strip()
            detail = f"runner exited with status {proc.returncode}"
            if tail:
                detail = f"{detail}: {tail}"
            return _failure("error", total, detail[:DETAIL_LIMIT])

        payload = _parse_envelope(output)
        if payload is None:
            if RESULT_MARKER not in output:
                return _failure("error", total, "missing result envelope")
            return _failure("error", total, "malformed result envelope")
        return _result_from_envelope(payload, normalized)
    finally:
        selector.close()
        try:
            proc.stdout.close()
        except OSError:
            pass
        if proc.poll() is None:
            _kill_process(proc)


def score_code_exec(
    generation: str,
    targets: Sequence[str],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecResult:
    """Extract the first fenced block and score it against ``targets``."""
    normalized = _validate_targets(targets)
    code = extract_code(generation)
    if not code.strip():
        return _failure("error", len(normalized), "no code block found")
    return run_candidate(code, normalized, timeout_s)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return the unbiased pass@k estimator for n samples and c passes."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= c <= n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _fixture_error(path: Path, line: int, reason: str) -> DemoFixtureError:
    return DemoFixtureError(f"{path}:{line}: {reason}")


def _string_list(value: object, path: Path, line: int, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _fixture_error(path, line, f"{field} must be a non-empty list")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _fixture_error(
                path, line, f"{field} entries must be non-empty strings"
            )
        strings.append(item)
    return tuple(strings)


def load_demo_cases(path: Path) -> tuple[DemoCase, ...]:
    """Load and strictly validate the handcrafted demo JSONL fixture."""
    raw_lines = path.read_bytes().splitlines()
    cases: list[DemoCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fixture_error(path, line_number, f"invalid UTF-8: {exc}") from None
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _fixture_error(
                path, line_number, f"invalid JSON: {exc.msg}"
            ) from None
        if not isinstance(record, dict):
            raise _fixture_error(path, line_number, "record must be an object")
        if set(record) != {"task_id", "targets", "samples"}:
            raise _fixture_error(
                path, line_number, "record must have exactly task_id, targets, samples"
            )

        task_id = record["task_id"]
        if not isinstance(task_id, str) or not task_id.strip():
            raise _fixture_error(path, line_number, "task_id must be a non-empty string")
        if task_id in seen:
            raise _fixture_error(path, line_number, f"duplicate task_id {task_id!r}")
        seen.add(task_id)
        targets = _string_list(record["targets"], path, line_number, "targets")

        raw_samples = record["samples"]
        if not isinstance(raw_samples, list) or not raw_samples:
            raise _fixture_error(
                path, line_number, "samples must be a non-empty list"
            )
        samples: list[DemoSample] = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                raise _fixture_error(path, line_number, "sample must be an object")
            if set(sample) != {"generation", "expected_exit", "expected_score"}:
                raise _fixture_error(
                    path,
                    line_number,
                    "sample must have exactly generation, expected_exit, expected_score",
                )
            generation = sample["generation"]
            expected_exit = sample["expected_exit"]
            expected_score = sample["expected_score"]
            if not isinstance(generation, str) or not generation.strip():
                raise _fixture_error(
                    path, line_number, "generation must be a non-empty string"
                )
            if not isinstance(expected_exit, str) or expected_exit not in EXIT_CODES:
                raise _fixture_error(
                    path, line_number, "expected_exit is not a known exit code"
                )
            if isinstance(expected_score, bool) or not isinstance(
                expected_score, (int, float)
            ):
                raise _fixture_error(
                    path, line_number, "expected_score must be a number in [0, 1]"
                )
            try:
                score = float(expected_score)
            except (OverflowError, ValueError):
                raise _fixture_error(
                    path, line_number, "expected_score must be a number in [0, 1]"
                ) from None
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise _fixture_error(
                    path, line_number, "expected_score must be a number in [0, 1]"
                )
            samples.append(
                DemoSample(
                    generation=generation,
                    expected_exit=cast(ExitCode, expected_exit),
                    expected_score=score,
                )
            )
        cases.append(
            DemoCase(
                task_id=task_id,
                targets=targets,
                samples=tuple(samples),
            )
        )
    if not cases:
        raise DemoFixtureError(f"{path}: no demo cases")
    return tuple(cases)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic reference demo."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args != ["demo"]:
        print("usage: python main.py [demo]", file=sys.stderr)
        return 2

    try:
        cases = load_demo_cases(DEMO_FIXTURE)
    except OSError as exc:
        print(f"error: cannot read {DEMO_FIXTURE}: {exc}", file=sys.stderr)
        return 2
    except DemoFixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    all_ok = True
    for case in cases:
        sample_count = len(case.samples)
        pass_count = 0
        for index, sample in enumerate(case.samples):
            result = score_code_exec(
                sample.generation, case.targets, DEMO_TIMEOUT_S
            )
            if result.exit_code == "pass":
                pass_count += 1
            matches = result.exit_code == sample.expected_exit and math.isclose(
                result.score,
                sample.expected_score,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            if not matches:
                all_ok = False
                print(
                    f"MISMATCH {case.task_id} sample={index}: "
                    f"expected exit={sample.expected_exit} "
                    f"score={sample.expected_score:.3f}, "
                    f"got exit={result.exit_code} score={result.score:.3f} "
                    f"detail={result.detail!r}"
                )
            print(
                f"{case.task_id} sample={index} exit={result.exit_code} "
                f"score={result.score:.3f} passed={result.passed}/{result.total} "
                f"detail={result.detail!r}"
            )
        for k in sorted({1, min(3, sample_count), sample_count}):
            print(
                f"{case.task_id}: n={sample_count} c={pass_count} "
                f"pass@{k}={pass_at_k(sample_count, pass_count, k):.3f}"
            )

    print("demo: ok" if all_ok else "demo: mismatch")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
