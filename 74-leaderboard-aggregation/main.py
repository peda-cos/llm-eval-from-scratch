"""Aggregate per-task eval scores into a ranked leaderboard with uncertainty.

The input is one ``EvalRun`` per ``(model, task)`` pair, already normalized to
``[0, 1]`` by the upstream metric layer. This module validates that contract,
pivots the runs into a model x task view, and reports means, category means,
task-level win rates, and percentile bootstrap intervals. Pairwise comparisons
use the paired bootstrap over common tasks. It never calls a model or a metric.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

SCHEMA_VERSION = 1
PUBLIC_BOOTSTRAP_SAMPLES = 500
HELPER_BOOTSTRAP_SAMPLES = 1000
MODEL_COLUMN_WIDTH = 20


@dataclass(frozen=True, slots=True)
class EvalRun:
    """One model's score on one task, before aggregation."""

    model_id: str
    task_id: str
    metric_name: str
    score: float
    category: str = ""
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    """One model's aggregate row."""

    model_id: str
    mean_score: float
    mean_ci_lo: float
    mean_ci_hi: float
    win_rate: float
    tasks_completed: int
    categories: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PairwiseDiff:
    """Paired score difference between two models on their shared tasks."""

    model_a: str
    model_b: str
    mean_diff: float
    ci_lo: float
    ci_hi: float
    tasks_compared: int
    significant: bool


@dataclass(frozen=True, slots=True)
class LeaderboardReport:
    """Leaderboard rows and pairwise comparisons with their settings."""

    rows: tuple[LeaderboardRow, ...]
    pairwise: tuple[PairwiseDiff, ...]
    alpha: float
    b: int
    seed: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report payload."""
        return {
            "schema_version": SCHEMA_VERSION,
            "alpha": float(self.alpha),
            "bootstrap_samples": int(self.b),
            "seed": self.seed,
            "models": [_model_dict(row) for row in self.rows],
            "pairwise": [_pair_dict(pair) for pair in self.pairwise],
        }

    def to_json(self) -> str:
        """Serialize the report as stable, strict JSON."""
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, allow_nan=False
        )


def _model_dict(row: LeaderboardRow) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_id": row.model_id,
        "mean_score": float(row.mean_score),
        "mean_ci_lo": float(row.mean_ci_lo),
        "mean_ci_hi": float(row.mean_ci_hi),
        "win_rate": float(row.win_rate),
        "tasks_completed": int(row.tasks_completed),
    }
    if row.categories:
        payload["categories"] = {
            name: float(value) for name, value in sorted(row.categories.items())
        }
    return payload


def _pair_dict(pair: PairwiseDiff) -> dict[str, object]:
    return {
        "model_a": pair.model_a,
        "model_b": pair.model_b,
        "mean_diff": float(pair.mean_diff),
        "ci_lo": float(pair.ci_lo),
        "ci_hi": float(pair.ci_hi),
        "tasks_compared": int(pair.tasks_compared),
        "significant": bool(pair.significant),
    }


def _validate_alpha(alpha: float) -> float:
    """Return alpha as a float, requiring a finite value in (0, 1)."""
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be a finite number in (0, 1)")
    return value


def _validate_bootstrap(b: int, alpha: float) -> float:
    """Validate bootstrap settings and return the normalized alpha."""
    if isinstance(b, bool) or not isinstance(b, int) or b < 1:
        raise ValueError("b must be a positive integer")
    return _validate_alpha(alpha)


def _score_vector(values: ArrayLike, name: str) -> FloatArray:
    """Convert a one-dimensional score sequence to validated float64 values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    if bool(np.any((array < 0.0) | (array > 1.0))):
        raise ValueError(f"{name} must be in [0, 1]")
    return array


def _bootstrap_means(values: FloatArray, b: int, seed: int | None) -> FloatArray:
    """Draw b bootstrap means by resampling values with replacement."""
    rng = np.random.default_rng(seed)
    n = int(values.size)
    indices = rng.integers(0, n, size=(b, n))
    return values[indices].mean(axis=1)


def _percentile_interval(
    values: FloatArray, b: int, alpha: float, seed: int | None
) -> tuple[float, float]:
    """Return the alpha percentile bootstrap interval for the mean."""
    alpha_value = _validate_bootstrap(b, alpha)
    bootstrap_means = _bootstrap_means(values, b, seed)
    lo, hi = np.quantile(
        bootstrap_means,
        [alpha_value / 2.0, 1.0 - alpha_value / 2.0],
        method="linear",
    )
    return float(lo), float(hi)


def bootstrap_mean_ci(
    scores: ArrayLike,
    b: int = HELPER_BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-task scores.

    Tasks are the resampling unit. ``b`` defaults to 1000 here; the public
    aggregators default to 500 so the demo and tests stay quick.
    """
    values = _score_vector(scores, "scores")
    return _percentile_interval(values, b, alpha, seed)


def bootstrap_pairwise_diff(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    b: int = HELPER_BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired mean difference A - B.

    The two score arrays must describe the same tasks, in the same order. The
    bootstrap resamples their differences, preserving the pairing.
    """
    a = _score_vector(scores_a, "scores_a")
    b_scores = _score_vector(scores_b, "scores_b")
    if a.size != b_scores.size:
        raise ValueError("scores_a and scores_b must have the same number of values")
    return _percentile_interval(a - b_scores, b, alpha, seed)


def _validated_runs(runs: Sequence[EvalRun]) -> list[EvalRun]:
    """Validate every run at the aggregation trust boundary."""
    materialized = list(runs)
    seen: set[tuple[str, str]] = set()
    for run in materialized:
        for label, value in (
            ("model_id", run.model_id),
            ("task_id", run.task_id),
            ("metric_name", run.metric_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must not be empty")
        try:
            score = float(run.score)
        except (TypeError, ValueError) as exc:
            raise ValueError("score must be a finite number in [0, 1]") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"score must be finite and in [0, 1], got {run.score!r}"
            )
        try:
            weight = float(run.weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("weight must be a finite positive number") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight must be a finite positive number")
        key = (run.model_id, run.task_id)
        if key in seen:
            raise ValueError(
                f"duplicate run for model={run.model_id!r}, task={run.task_id!r}"
            )
        seen.add(key)
    return materialized


def _group_runs(
    runs: Sequence[EvalRun],
) -> tuple[dict[str, list[EvalRun]], dict[str, dict[str, EvalRun]]]:
    """Build task-first and model-first views of validated runs."""
    by_task: dict[str, list[EvalRun]] = {}
    by_model: dict[str, dict[str, EvalRun]] = {}
    for run in runs:
        by_task.setdefault(run.task_id, []).append(run)
        by_model.setdefault(run.model_id, {})[run.task_id] = run
    return by_task, by_model


def _ordered_scores(task_map: Mapping[str, EvalRun]) -> FloatArray:
    """Scores in lexicographic task order, for reproducible bootstrap draws."""
    return np.asarray(
        [task_map[task_id].score for task_id in sorted(task_map)], dtype=np.float64
    )


def win_rate(
    model_id: str,
    runs_by_task: Mapping[str, Sequence[EvalRun]],
    all_models: Collection[str],
) -> float:
    """Fraction of completed tasks where the model ties for the best score.

    Every model tied at the task maximum gets a full win. Tasks the model did
    not run are excluded from both numerator and denominator, so the rate is
    conditional on the model's own coverage.
    """
    wins = 0
    total = 0
    for runs in runs_by_task.values():
        scores = {
            run.model_id: run.score
            for run in runs
            if run.model_id in all_models
        }
        if model_id not in scores:
            continue
        total += 1
        if scores[model_id] >= max(scores.values()):
            wins += 1
    return wins / total if total else 0.0


def aggregate(
    runs: Sequence[EvalRun],
    b: int = PUBLIC_BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int | None = None,
) -> list[LeaderboardRow]:
    """Aggregate runs into rows sorted by mean score, then model id.

    Every completed task counts once; ``EvalRun.weight`` is validated but
    deliberately ignored. The per-model CI resamples that model's tasks.
    """
    checked = _validated_runs(runs)
    _validate_bootstrap(b, alpha)
    if not checked:
        return []
    by_task, by_model = _group_runs(checked)
    all_models = frozenset(by_model)
    rows: list[LeaderboardRow] = []
    for model_id in sorted(by_model):
        task_map = by_model[model_id]
        scores = _ordered_scores(task_map)
        mean_score = float(scores.mean())
        mean_ci_lo, mean_ci_hi = bootstrap_mean_ci(
            scores, b=b, alpha=alpha, seed=seed
        )
        category_scores: dict[str, list[float]] = {}
        for task_id in sorted(task_map):
            category = task_map[task_id].category
            if category:
                category_scores.setdefault(category, []).append(
                    task_map[task_id].score
                )
        categories = {
            name: float(np.mean(values))
            for name, values in sorted(category_scores.items())
        }
        rows.append(
            LeaderboardRow(
                model_id=model_id,
                mean_score=mean_score,
                mean_ci_lo=mean_ci_lo,
                mean_ci_hi=mean_ci_hi,
                win_rate=win_rate(model_id, by_task, all_models),
                tasks_completed=len(task_map),
                categories=categories,
            )
        )
    rows.sort(key=lambda row: (-row.mean_score, row.model_id))
    return rows


def pairwise_diffs(
    runs: Sequence[EvalRun],
    b: int = PUBLIC_BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int | None = None,
) -> list[PairwiseDiff]:
    """Paired pairwise differences in leaderboard order.

    Models are compared only on shared tasks. Pairs with fewer than two shared
    tasks are omitted: a one-task bootstrap has a zero-width interval and would
    overstate certainty. Intervals are marginal per comparison.
    """
    checked = _validated_runs(runs)
    _validate_bootstrap(b, alpha)
    if not checked:
        return []
    _, by_model = _group_runs(checked)
    means = {
        model_id: float(_ordered_scores(task_map).mean())
        for model_id, task_map in by_model.items()
    }
    order = sorted(by_model, key=lambda model_id: (-means[model_id], model_id))
    pairs: list[PairwiseDiff] = []
    for index, model_a in enumerate(order):
        tasks_a = by_model[model_a]
        for model_b in order[index + 1 :]:
            tasks_b = by_model[model_b]
            shared = sorted(set(tasks_a) & set(tasks_b))
            if len(shared) < 2:
                continue
            differences = np.asarray(
                [
                    tasks_a[task_id].score - tasks_b[task_id].score
                    for task_id in shared
                ],
                dtype=np.float64,
            )
            ci_lo, ci_hi = _percentile_interval(differences, b, alpha, seed)
            pairs.append(
                PairwiseDiff(
                    model_a=model_a,
                    model_b=model_b,
                    mean_diff=float(differences.mean()),
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    tasks_compared=len(shared),
                    significant=ci_lo > 0.0 or ci_hi < 0.0,
                )
            )
    return pairs


def build_report(
    runs: Sequence[EvalRun],
    b: int = PUBLIC_BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int | None = None,
) -> LeaderboardReport:
    """Build the rows, pairwise comparisons, and settings for one report."""
    rows = aggregate(runs, b=b, alpha=alpha, seed=seed)
    pairs = pairwise_diffs(runs, b=b, alpha=alpha, seed=seed)
    return LeaderboardReport(
        rows=tuple(rows),
        pairwise=tuple(pairs),
        alpha=alpha,
        b=b,
        seed=seed,
    )


def _ci_label(alpha: float) -> str:
    return f"{100.0 * (1.0 - alpha):g}% CI"


def _markdown_model(model_id: str) -> str:
    """Normalize, truncate to 20 characters, and escape table separators."""
    cleaned = " ".join(model_id.split())
    return cleaned[:MODEL_COLUMN_WIDTH].replace("|", r"\|")


def render_markdown(
    rows: Sequence[LeaderboardRow], alpha: float = 0.05
) -> str:
    """Render the leaderboard as a markdown table sorted by mean score."""
    alpha_value = _validate_alpha(alpha)
    ordered = sorted(rows, key=lambda row: (-row.mean_score, row.model_id))
    lines = [
        f"| Rank | Model | Mean | {_ci_label(alpha_value)} | Win rate | Tasks |",
        "|------|-------|------|--------|----------|-------|",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | {_markdown_model(row.model_id)} | "
            f"{row.mean_score:.2f} | "
            f"{row.mean_ci_lo:.2f}-{row.mean_ci_hi:.2f} | "
            f"{row.win_rate:.2f} | {row.tasks_completed} |"
        )
    return "\n".join(lines)


def render_pairwise_markdown(
    pairs: Sequence[PairwiseDiff], alpha: float = 0.05
) -> str:
    """Render pairwise differences as a markdown table."""
    alpha_value = _validate_alpha(alpha)
    lines = [
        f"| Model A | Model B | Mean diff | {_ci_label(alpha_value)} | Significant | Tasks |",
        "|---------|---------|-----------|--------|-------------|-------|",
    ]
    for pair in pairs:
        lines.append(
            f"| {_markdown_model(pair.model_a)} | "
            f"{_markdown_model(pair.model_b)} | "
            f"{pair.mean_diff:.3f} | "
            f"{pair.ci_lo:.3f} to {pair.ci_hi:.3f} | "
            f"{'yes' if pair.significant else 'no'} | "
            f"{pair.tasks_compared} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _DemoTask:
    task_id: str
    category: str
    metric_name: str
    scores: tuple[float, float, float]


_DEMO_MODELS = ("atlas-7b", "beacon-13b", "helios-2026-experimental")
_DEMO_TASKS = (
    _DemoTask("math_arithmetic", "math", "accuracy", (0.90, 0.95, 0.55)),
    _DemoTask("math_algebra", "math", "accuracy", (0.85, 0.79, 0.50)),
    _DemoTask("math_geometry", "math", "accuracy", (0.80, 0.76, 0.45)),
    _DemoTask("code_python", "code", "pass_rate", (0.82, 0.87, 0.40)),
    _DemoTask("code_debug", "code", "pass_rate", (0.88, 0.82, 0.60)),
    _DemoTask("code_refactor", "code", "pass_rate", (0.76, 0.79, 0.35)),
    _DemoTask("reasoning_logic", "reasoning", "exact_match", (0.80, 0.75, 0.50)),
    _DemoTask("reasoning_plan", "reasoning", "exact_match", (0.70, 0.75, 0.42)),
    _DemoTask(
        "reasoning_analysis", "reasoning", "exact_match", (0.74, 0.70, 0.38)
    ),
    _DemoTask(
        "safety_refusal", "safety", "policy_pass_rate", (0.86, 0.80, 0.65)
    ),
    _DemoTask("safety_bias", "safety", "policy_pass_rate", (0.90, 0.86, 0.58)),
    _DemoTask(
        "safety_injection", "safety", "policy_pass_rate", (0.82, 0.78, 0.48)
    ),
)


def _demo_runs() -> list[EvalRun]:
    return [
        EvalRun(
            model_id=model_id,
            task_id=task.task_id,
            metric_name=task.metric_name,
            score=task.scores[index],
            category=task.category,
        )
        for task in _DEMO_TASKS
        for index, model_id in enumerate(_DEMO_MODELS)
    ]


def _print_demo() -> None:
    report = build_report(_demo_runs(), b=500, alpha=0.05, seed=7)
    print("## Leaderboard")
    print(render_markdown(report.rows, report.alpha))
    print()
    print("## Pairwise differences")
    print(render_pairwise_markdown(report.pairwise, report.alpha))
    print()
    print("## JSON report")
    print(report.to_json())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic synthetic demo or print usage."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in ([], ["demo"]):
        print("usage: python main.py [demo]", file=sys.stderr)
        return 2
    _print_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
