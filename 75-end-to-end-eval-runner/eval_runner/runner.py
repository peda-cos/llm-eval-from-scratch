"""End-to-end runner: adapter calls, scoring, calibration, and aggregation."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from eval_runner.adapter import Generation, ModelAdapter
from eval_runner.aggregation import (
    EvalRun,
    LeaderboardRow,
    PairwiseDiff,
    build_report,
    render_markdown,
    render_pairwise_markdown,
)
from eval_runner.calibration import CalibrationReport
from eval_runner.code_exec import ExecResult, run_candidate
from eval_runner.metrics import score as score_text
from eval_runner.tasks import (
    BINARY_METRICS,
    GRADED_METRICS,
    TaskSpec,
    post_process,
    render,
    validate_task,
)

SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_SEED = 7
DEFAULT_CODE_TIMEOUT_S = 2.0
DEFAULT_WORKERS = 8


class RunnerError(RuntimeError):
    """A structured pipeline failure with the stage and task context."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        model_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.model_id = model_id
        self.task_id = task_id

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "type": "RunnerError",
            "message": str(self),
            "model_id": self.model_id,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One model's scored result on one task, including calibration signals."""

    model_id: str
    task_id: str
    category: str
    metric_name: str
    score: float
    correct: bool
    confidence: float
    prediction: str
    duration_s: float
    token_nll: float | None
    token_count: int | None
    metric_detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "task_id": self.task_id,
            "category": self.category,
            "metric_name": self.metric_name,
            "score": self.score,
            "correct": self.correct,
            "confidence": self.confidence,
            "prediction": self.prediction,
            "duration_ms": round(self.duration_s * 1000.0, 3),
            "token_nll": self.token_nll,
            "token_count": self.token_count,
            "metric_detail": dict(self.metric_detail),
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The complete single-pass run: results, leaderboard, pairwise, calibration."""

    results: tuple[TaskResult, ...]
    leaderboard: tuple[LeaderboardRow, ...]
    pairwise: tuple[PairwiseDiff, ...]
    calibration: dict[str, CalibrationReport]
    tasks: int
    model_ids: tuple[str, ...]
    wall_seconds: float
    parallel: bool
    workers: int
    bootstrap_samples: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        leaderboard: list[dict[str, object]] = []
        for rank, row in enumerate(self.leaderboard, start=1):
            payload = row.to_dict()
            payload["rank"] = rank
            leaderboard.append(payload)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "summary": {
                "tasks": self.tasks,
                "models": len(self.model_ids),
                "wall_seconds": self.wall_seconds,
                "parallel": self.parallel,
                "workers": self.workers,
                "bootstrap_samples": self.bootstrap_samples,
                "seed": self.seed,
                "model_ids": list(self.model_ids),
            },
            "leaderboard": leaderboard,
            "pairwise": [pair.to_dict() for pair in self.pairwise],
            "calibration": {
                model_id: self.calibration[model_id].to_dict()
                for model_id in self.model_ids
            },
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, allow_nan=False
        )

    def render_markdown(self) -> str:
        calibration_lines = [
            "| Model | ECE | Brier | Populated bins | Samples |",
            "|-------|-----|-------|----------------|---------|",
        ]
        for model_id in self.model_ids:
            report = self.calibration[model_id]
            calibration_lines.append(
                f"| {_model_cell(model_id)} | {report.ece:.3f} | "
                f"{report.brier:.3f} | {report.populated_bins} | {report.n_samples} |"
            )
        return "\n\n".join(
            [
                render_markdown(self.leaderboard),
                render_pairwise_markdown(self.pairwise),
                "\n".join(calibration_lines),
            ]
        )


def _model_cell(model_id: str) -> str:
    return " ".join(model_id.split())[:20].replace("|", r"\|")


def _correct_from_score(metric_name: str, score: float) -> bool:
    """Map a metric score to the binary signal used for calibration."""
    if metric_name in BINARY_METRICS:
        return score >= 1.0
    if metric_name in GRADED_METRICS:
        return score >= 0.5
    raise ValueError(f"unknown metric_name: {metric_name!r}")


def _validate_adapters(adapters: Sequence[ModelAdapter]) -> list[ModelAdapter]:
    materialized = list(adapters)
    if not materialized:
        raise RunnerError("validate", "at least one adapter is required")
    seen: set[str] = set()
    for adapter in materialized:
        model_id = getattr(adapter, "model_id", None)
        if not isinstance(model_id, str) or not model_id.strip():
            raise RunnerError("validate", "adapter model_id must be a non-empty string")
        if model_id in seen:
            raise RunnerError("validate", f"duplicate adapter model_id: {model_id!r}")
        if not callable(getattr(adapter, "generate", None)):
            raise RunnerError(
                "validate",
                "adapter must define generate(prompt, task)",
                model_id=model_id,
            )
        seen.add(model_id)
    return materialized


def _validate_tasks(tasks: Sequence[TaskSpec]) -> list[TaskSpec]:
    materialized = list(tasks)
    if not materialized:
        raise RunnerError("validate", "at least one task is required")
    for index, task in enumerate(materialized, start=1):
        errors = validate_task(task, line=index)
        if errors:
            first = errors[0]
            raise RunnerError(
                "validate",
                f"invalid task {index}: {first.rule} ({first.field}): {first.message}",
                task_id=str(task.get("task_id", "")),
            )
    return materialized


def _validate_options(
    parallel: bool,
    workers: int | None,
    bootstrap_samples: int,
    seed: int,
    task_count: int,
) -> int:
    if not isinstance(parallel, bool):
        raise RunnerError("validate", "parallel must be a boolean")
    requested = DEFAULT_WORKERS if workers is None else workers
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise RunnerError("validate", "workers must be a positive integer")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise RunnerError("validate", "bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RunnerError("validate", "seed must be an integer")
    if not parallel:
        return 1
    return min(requested, task_count)


def _score_one(
    adapter: ModelAdapter,
    task: TaskSpec,
    code_timeout_s: float,
) -> TaskResult:
    task_id = task["task_id"]
    prompt = render(task)
    started = time.perf_counter()
    try:
        generation = adapter.generate(prompt, task)
    except Exception as exc:
        raise RunnerError(
            "generate",
            f"{type(exc).__name__}: {exc}",
            model_id=adapter.model_id,
            task_id=task_id,
        ) from exc
    duration_s = time.perf_counter() - started

    if not isinstance(generation, Generation):
        raise RunnerError(
            "generate",
            "adapter.generate must return Generation",
            model_id=adapter.model_id,
            task_id=task_id,
        )

    try:
        prediction = post_process(generation.text, task["post_process"])
    except ValueError as exc:
        raise RunnerError(
            "score",
            str(exc),
            model_id=adapter.model_id,
            task_id=task_id,
        ) from exc

    metric_name = task["metric_name"]
    metric_detail: dict[str, object] = {}
    if metric_name == "code_exec":
        if not prediction.strip():
            result = ExecResult(
                exit_code="error",
                score=0.0,
                passed=0,
                total=len(task["targets"]),
                detail="no code block found",
            )
        else:
            try:
                result = run_candidate(prediction, task["targets"], code_timeout_s)
            except ValueError as exc:
                raise RunnerError(
                    "score",
                    str(exc),
                    model_id=adapter.model_id,
                    task_id=task_id,
                ) from exc
        score = result.score
        metric_detail = {
            "exit_code": result.exit_code,
            "passed": result.passed,
            "total": result.total,
            "detail": result.detail,
        }
    else:
        try:
            score = score_text(metric_name, prediction, task["targets"])
        except ValueError as exc:
            raise RunnerError(
                "score",
                str(exc),
                model_id=adapter.model_id,
                task_id=task_id,
            ) from exc

    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RunnerError(
            "score",
            f"metric returned an invalid score: {score!r}",
            model_id=adapter.model_id,
            task_id=task_id,
        )
    try:
        correct = _correct_from_score(metric_name, score)
    except ValueError as exc:
        raise RunnerError(
            "score",
            str(exc),
            model_id=adapter.model_id,
            task_id=task_id,
        ) from exc

    return TaskResult(
        model_id=adapter.model_id,
        task_id=task_id,
        category=task["category"],
        metric_name=metric_name,
        score=score,
        correct=correct,
        confidence=float(generation.confidence),
        prediction=prediction,
        duration_s=duration_s,
        token_nll=None if generation.token_nll is None else float(generation.token_nll),
        token_count=generation.token_count,
        metric_detail=metric_detail,
    )


def _run_adapter_tasks(
    adapter: ModelAdapter,
    tasks: Sequence[TaskSpec],
    parallel: bool,
    workers: int,
    code_timeout_s: float,
) -> list[TaskResult]:
    if not parallel:
        return [_score_one(adapter, task, code_timeout_s) for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_score_one, adapter, task, code_timeout_s) for task in tasks
        ]
        return [future.result() for future in futures]


def run_eval(
    adapters: Sequence[ModelAdapter],
    tasks: Sequence[TaskSpec],
    parallel: bool = True,
    *,
    workers: int | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    code_timeout_s: float = DEFAULT_CODE_TIMEOUT_S,
) -> EvalReport:
    """Run one single-pass eval per adapter, then calibrate and aggregate."""
    checked_adapters = _validate_adapters(adapters)
    checked_tasks = _validate_tasks(tasks)
    effective_workers = _validate_options(
        parallel, workers, bootstrap_samples, seed, len(checked_tasks)
    )
    started = time.perf_counter()

    results: list[TaskResult] = []
    for adapter in checked_adapters:
        results.extend(
            _run_adapter_tasks(
                adapter,
                checked_tasks,
                parallel,
                effective_workers,
                code_timeout_s,
            )
        )

    model_ids = tuple(adapter.model_id for adapter in checked_adapters)
    runs = [
        EvalRun(
            model_id=result.model_id,
            task_id=result.task_id,
            metric_name=result.metric_name,
            score=result.score,
            category=result.category,
        )
        for result in results
    ]
    try:
        leaderboard_report = build_report(
            runs, b=bootstrap_samples, alpha=0.05, seed=seed
        )
    except ValueError as exc:
        raise RunnerError("aggregate", str(exc)) from exc

    calibration: dict[str, CalibrationReport] = {}
    for model_id in model_ids:
        model_results = [result for result in results if result.model_id == model_id]
        try:
            calibration[model_id] = CalibrationReport.from_predictions(
                [result.confidence for result in model_results],
                [1.0 if result.correct else 0.0 for result in model_results],
            )
        except ValueError as exc:
            raise RunnerError("calibrate", str(exc), model_id=model_id) from exc

    return EvalReport(
        results=tuple(results),
        leaderboard=leaderboard_report.rows,
        pairwise=leaderboard_report.pairwise,
        calibration=calibration,
        tasks=len(checked_tasks),
        model_ids=model_ids,
        wall_seconds=time.perf_counter() - started,
        parallel=parallel,
        workers=effective_workers,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
