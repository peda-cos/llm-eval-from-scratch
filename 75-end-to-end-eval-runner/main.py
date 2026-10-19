"""End-to-end eval demo: three deterministic mock adapters over ten tasks.

Run ``python main.py demo`` to validate the fixture, execute every adapter,
calibrate, aggregate the leaderboard, write the JSON envelope, and print the
markdown tables. A clean run exits 0; validation, runtime, or gate failures
write a structured error envelope and exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from eval_runner import Generation, ModelAdapter, RunnerError, run_eval
from eval_runner.tasks import (
    DEFAULT_FIXTURE,
    FixtureValidationError,
    TaskSpec,
    load_fixtures,
)

SCHEMA_VERSION: Final[int] = 1
DEMO_ADAPTER_COUNT: Final[int] = 3
DEFAULT_JSON_OUT: Final[Path] = Path("artifacts/eval_report.json")
ARITHMETIC_RE: Final[re.Pattern[str]] = re.compile(
    r"Question:\s*(-?\d+)\s*([+-])\s*(-?\d+)"
)


def _token_fields(text: str, confidence: float) -> tuple[float, int]:
    count = max(1, len(text.split()))
    return -math.log(confidence) * count, count


def _generation(text: str, confidence: float) -> Generation:
    token_nll, token_count = _token_fields(text, confidence)
    return Generation(
        text=text,
        confidence=confidence,
        token_nll=token_nll,
        token_count=token_count,
    )


def _digest(task: TaskSpec, salt: str, modulus: int) -> int:
    payload = f"{salt}:{task['task_id']}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def _arithmetic_answer(prompt: str) -> str | None:
    matches = list(ARITHMETIC_RE.finditer(prompt))
    if not matches:
        return None
    left, operator, right = matches[-1].groups()
    if operator == "+":
        return str(int(left) + int(right))
    return str(int(left) - int(right))


def _classification_answer(prompt: str, task: TaskSpec) -> str:
    lowered = prompt.lower()
    if "battery" in lowered:
        return "positive"
    if "broke" in lowered:
        return "negative"
    if "invoice" in lowered or "charged" in lowered:
        return "billing"
    return task["targets"][0]


def _canonical_code(prompt: str) -> str:
    if "add(" in prompt:
        body = "def add(a, b):\n    return a + b"
    elif "reverse(" in prompt:
        body = "def reverse(text):\n    return text[::-1]"
    else:
        body = "def answer():\n    return None"
    return f"```python\n{body}\n```"


def _wrong_code(prompt: str) -> str:
    if "add(" in prompt:
        body = "def add(a, b):\n    return a - b"
    elif "reverse(" in prompt:
        body = "def reverse(text):\n    return 'wrong'"
    else:
        body = "def answer():\n    return 'wrong'"
    return f"```python\n{body}\n```"


def _wrong_text(task: TaskSpec, salt: str) -> str:
    category = task["category"]
    targets = task["targets"]
    if category == "arithmetic":
        try:
            base = int(targets[0])
        except ValueError:
            base = 0
        return str(base + 1 + _digest(task, salt, 3))
    if category == "mcq":
        candidates = [letter for letter in "ABCDE" if letter not in targets]
        return candidates[_digest(task, salt, len(candidates))] if candidates else "A"
    if category == "classification":
        candidates = [label for label in ("technical", "negative", "positive", "billing") if label not in targets]
        return candidates[_digest(task, salt, len(candidates))] if candidates else "other"
    if category == "summary":
        options = (
            "An unrelated answer.",
            "No summary is available.",
            "This covers something else entirely.",
        )
        return options[_digest(task, salt, len(options))]
    if category == "code_exec":
        return _wrong_code(task["prompt"])
    return "unrelated"


class RuleBasedAdapter:
    """Deterministic oracle-style mock for non-programmatic fixture categories."""

    model_id = "rule_based"
    confidence = 0.97

    def generate(self, prompt: str, task: TaskSpec) -> Generation:
        category = task["category"]
        if category == "arithmetic":
            answer = _arithmetic_answer(prompt) or task["targets"][0]
        elif category == "classification":
            answer = _classification_answer(prompt, task)
        elif category == "code_exec":
            answer = _canonical_code(prompt)
        else:
            answer = task["targets"][0]
        return _generation(answer, self.confidence)


class NoisyAdapter:
    """Overconfident, deterministic wrong answers; the random/noisy baseline."""

    model_id = "noisy"
    confidence = 0.93

    def generate(self, prompt: str, task: TaskSpec) -> Generation:
        return _generation(_wrong_text(task, self.model_id), self.confidence)


class BiasedAdapter:
    """Good on classification/summary, overconfident and wrong elsewhere."""

    model_id = "biased"
    strong_confidence = 0.95
    weak_confidence = 0.90

    def generate(self, prompt: str, task: TaskSpec) -> Generation:
        if task["category"] in ("classification", "summary"):
            return _generation(task["targets"][0], self.strong_confidence)
        return _generation(_wrong_text(task, self.model_id), self.weak_confidence)


def demo_adapters() -> list[ModelAdapter]:
    return [RuleBasedAdapter(), BiasedAdapter(), NoisyAdapter()]


def check_demo_clean(report: object, task_count: int) -> None:
    """Enforce the demo's clean-run criteria, raising a gate error otherwise."""
    from eval_runner.runner import EvalReport

    if not isinstance(report, EvalReport):
        raise RunnerError("gate", "demo report has an unexpected type")
    expected = DEMO_ADAPTER_COUNT * task_count
    if len(report.results) != expected:
        raise RunnerError(
            "gate",
            f"expected {expected} scored results, got {len(report.results)}",
        )
    pairs = {(result.model_id, result.task_id) for result in report.results}
    if len(pairs) != expected:
        raise RunnerError("gate", "duplicate or missing model/task result")
    for model_id in report.model_ids:
        calibration = report.calibration.get(model_id)
        if calibration is None or calibration.n_samples != task_count:
            raise RunnerError(
                "gate", f"calibration missing or incomplete for {model_id!r}",
                model_id=model_id,
            )
    means = {row.model_id: row.mean_score for row in report.leaderboard}
    if not report.leaderboard or report.leaderboard[0].model_id != "rule_based":
        raise RunnerError("gate", "leaderboard rank 1 is not rule_based")
    if "rule_based" not in means or "noisy" not in means:
        raise RunnerError("gate", "leaderboard is missing rule_based or noisy")
    if not means["rule_based"] > means["noisy"]:
        raise RunnerError(
            "gate",
            "rule_based must rank strictly above noisy",
        )


def _error_payload(exc: BaseException) -> dict[str, object]:
    if isinstance(exc, FixtureValidationError):
        error: dict[str, object] = {
            "stage": "validate",
            "type": type(exc).__name__,
            "message": str(exc),
            "details": [item.to_dict() for item in exc.errors],
        }
    elif isinstance(exc, RunnerError):
        error = exc.to_dict()
    else:
        error = {
            "stage": "runtime",
            "type": type(exc).__name__,
            "message": str(exc),
        }
    return {"schema_version": SCHEMA_VERSION, "status": "error", "error": error}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _run_demo(args: argparse.Namespace) -> int:
    tasks_path: Path = args.tasks
    json_out: Path = args.json_out
    try:
        tasks = load_fixtures(tasks_path)
    except (FixtureValidationError, OSError) as exc:
        payload = _error_payload(exc)
        try:
            _write_json(json_out, payload)
        except OSError as write_exc:
            print(f"error: could not write {json_out}: {write_exc}", file=sys.stderr)
        print(f"error[{payload['error']['stage']}]: {exc}", file=sys.stderr)  # type: ignore[index]
        return 1

    try:
        report = run_eval(
            demo_adapters(),
            tasks,
            parallel=args.parallel,
            workers=args.workers,
            bootstrap_samples=args.bootstrap,
            seed=args.seed,
            code_timeout_s=args.code_timeout,
        )
        check_demo_clean(report, len(tasks))
    except RunnerError as exc:
        payload = _error_payload(exc)
        try:
            _write_json(json_out, payload)
        except OSError as write_exc:
            print(f"error: could not write {json_out}: {write_exc}", file=sys.stderr)
        print(f"error[{exc.stage}]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        payload = _error_payload(exc)
        try:
            _write_json(json_out, payload)
        except OSError as write_exc:
            print(f"error: could not write {json_out}: {write_exc}", file=sys.stderr)
        print(f"error[runtime]: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        _write_json(json_out, report.to_dict())
    except OSError as exc:
        print(f"error: could not write {json_out}: {exc}", file=sys.stderr)
        return 1

    print(report.render_markdown())
    print(
        f"{report.tasks} tasks x {len(report.model_ids)} models in "
        f"{report.wall_seconds:.3f}s; JSON -> {json_out}",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the end-to-end eval demo over the bundled fixture.",
    )
    subparsers = parser.add_subparsers(dest="command")
    demo = subparsers.add_parser("demo", help="run the three mock adapters")
    demo.add_argument("--tasks", type=Path, default=DEFAULT_FIXTURE)
    demo.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    demo.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        default=True,
        help="run tasks sequentially in fixture order",
    )
    demo.add_argument("--workers", type=_positive_int, default=None)
    demo.add_argument("--bootstrap", type=_positive_int, default=500)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--code-timeout", type=_positive_float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["demo"]
    elif raw[0].startswith("-"):
        raw = ["demo", *raw]
    args = _build_parser().parse_args(raw)
    if args.command == "demo":
        return _run_demo(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
