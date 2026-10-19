"""Code execution scoring in a subprocess.

The sandbox is an educational isolation layer: a fresh interpreter, a denied
import list, a wall-clock deadline, and an output cap. It is not a security
boundary for untrusted code.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Final, Literal, Sequence, cast

DEFAULT_TIMEOUT_S: Final[float] = 2.0
MIN_TIMEOUT_S: Final[float] = 0.1
MAX_TIMEOUT_S: Final[float] = 30.0
OUTPUT_CAP_BYTES: Final[int] = 256 * 1024
DETAIL_LIMIT: Final[int] = 400
RESULT_MARKER: Final[str] = "__eval_runner_result__:"

DENIED_IMPORTS: Final[tuple[str, ...]] = (
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "ctypes",
    "shutil",
    "http.client",
    "asyncio.subprocess",
)

ExitCode = Literal["pass", "assertion_fail", "syntax_error", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Normalized result for one candidate, whatever the failure mode."""

    exit_code: ExitCode
    score: float
    passed: int
    total: int
    detail: str


CHILD_TEMPLATE: Final[str] = r'''
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


def extract_code(text: str) -> str:
    """Return the first fenced block, matching the task post-process rule."""
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


def _validate_assertions(assertions: Sequence[str]) -> list[str]:
    if isinstance(assertions, str):
        raise ValueError("assertions must be a non-empty sequence of non-empty strings")
    normalized = list(assertions)
    if not normalized or any(
        not isinstance(assertion, str) or not assertion.strip() for assertion in normalized
    ):
        raise ValueError("assertions must be a non-empty sequence of non-empty strings")
    return normalized


def _build_child(assertions: Sequence[str]) -> str:
    child = CHILD_TEMPLATE.replace("__DENIED_IMPORTS__", repr(list(DENIED_IMPORTS)))
    child = child.replace("__RESULT_MARKER__", repr(RESULT_MARKER))
    return child.replace("__TARGETS_JSON__", repr(json.dumps(list(assertions))))


def _failure(exit_code: ExitCode, total: int, detail: str) -> ExecResult:
    return ExecResult(
        exit_code=exit_code,
        score=0.0,
        passed=0,
        total=total,
        detail=detail[:DETAIL_LIMIT],
    )


def _kill_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
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
        try:
            proc.kill()
        except ProcessLookupError:
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
    return cast(dict[str, object], payload) if isinstance(payload, dict) else None


def _result_from_envelope(payload: dict[str, object], assertions: Sequence[str]) -> ExecResult:
    total = len(assertions)
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

    passed = 0
    failures = 0
    for item in raw_assertions:
        if not isinstance(item, dict):
            return _failure("error", total, "malformed result envelope")
        item_passed = item.get("passed")
        item_detail = item.get("detail")
        if not isinstance(item_passed, bool) or not isinstance(item_detail, str):
            return _failure("error", total, "malformed result envelope")
        if item_passed:
            passed += 1
        else:
            failures += 1

    if passed == total:
        return ExecResult(
            exit_code="pass",
            score=1.0,
            passed=passed,
            total=total,
            detail="",
        )
    return ExecResult(
        exit_code="assertion_fail",
        score=passed / total,
        passed=passed,
        total=total,
        detail=f"{failures} of {total} assertion(s) failed",
    )


def run_candidate(
    code: str,
    assertions: Sequence[str],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecResult:
    """Execute candidate code in a fresh interpreter and score its assertions."""
    normalized = _validate_assertions(assertions)
    total = len(normalized)
    timeout = min(MAX_TIMEOUT_S, max(MIN_TIMEOUT_S, float(timeout_s)))
    child = _build_child(normalized)

    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", child],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=os.name == "posix",
    )
    try:
        output, _ = proc.communicate(input=code.encode("utf-8"), timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process(proc)
        return _failure("timeout", total, f"timeout after {timeout:.1f}s")
    except OSError as exc:
        _kill_process(proc)
        return _failure("error", total, f"could not run candidate: {exc}")

    if len(output) > OUTPUT_CAP_BYTES:
        return _failure("error", total, "output overflow")
    text = output.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = text[-300:].strip()
        detail = f"runner exited with status {proc.returncode}"
        if tail:
            detail = f"{detail}: {tail}"
        return _failure("error", total, detail)

    payload = _parse_envelope(text)
    if payload is None:
        if RESULT_MARKER not in text:
            return _failure("error", total, "missing result envelope")
        return _failure("error", total, "malformed result envelope")
    return _result_from_envelope(payload, normalized)


def score_code_exec(
    generation: str,
    assertions: Sequence[str],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecResult:
    """Extract the first fenced block and score it against the assertions."""
    normalized = _validate_assertions(assertions)
    code = extract_code(generation)
    if not code.strip():
        return _failure("error", len(normalized), "no code block found")
    return run_candidate(code, normalized, timeout_s)
