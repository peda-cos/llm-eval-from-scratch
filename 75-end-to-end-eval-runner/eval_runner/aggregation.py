"""Aggregate per-task eval scores into a ranked leaderboard with uncertainty.

The input is one ``EvalRun`` per ``(model, task)`` pair. Scores are already in
``[0, 1]``. Bootstrap intervals resample tasks with a seeded stdlib RNG so the
JSON report is reproducible on the same fixture.
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

PUBLIC_BOOTSTRAP_SAMPLES: Final[int] = 500
DEFAULT_ALPHA: Final[float] = 0.05
MODEL_COLUMN_WIDTH: Final[int] = 20


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

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model_id": self.model_id,
            "mean_score": self.mean_score,
            "mean_ci_lo": self.mean_ci_lo,
            "mean_ci_hi": self.mean_ci_hi,
            "win_rate": self.win_rate,
            "tasks_completed": self.tasks_completed,
        }
        if self.categories:
            payload["categories"] = {
                name: value for name, value in sorted(self.categories.items())
            }
        return payload


@dataclass(frozen=True, slots=True)
class PairwiseDiff:
    """Paired score difference between two models on shared tasks."""

    model_a: str
    model_b: str
    mean_diff: float
    ci_lo: float
    ci_hi: float
    tasks_compared: int
    significant: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "mean_diff": self.mean_diff,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "tasks_compared": self.tasks_compared,
            "significant": self.significant,
        }


@dataclass(frozen=True, slots=True)
class LeaderboardReport:
    """Leaderboard rows and pairwise comparisons with their settings."""

    rows: tuple[LeaderboardRow, ...]
    pairwise: tuple[PairwiseDiff, ...]
    alpha: float
    b: int
    seed: int | None


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be a finite number in (0, 1)")
    return value


def _validate_bootstrap(b: int, alpha: float) -> float:
    if isinstance(b, bool) or not isinstance(b, int) or b < 1:
        raise ValueError("b must be a positive integer")
    return _validate_alpha(alpha)


def _score_vector(values: Sequence[float], name: str) -> list[float]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError(f"{name} must contain at least one value")
    for value in materialized:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    return materialized


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _bootstrap_interval(
    values: Sequence[float], b: int, alpha: float, seed: int | None
) -> tuple[float, float]:
    alpha_value = _validate_bootstrap(b, alpha)
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(b)]
    means.sort()
    return _percentile(means, alpha_value / 2.0), _percentile(
        means, 1.0 - alpha_value / 2.0
    )


def bootstrap_mean_ci(
    scores: Sequence[float],
    b: int = PUBLIC_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-task scores."""
    values = _score_vector(scores, "scores")
    return _bootstrap_interval(values, b, alpha, seed)


def bootstrap_pairwise_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    b: int = PUBLIC_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired mean difference A - B."""
    a_values = _score_vector(scores_a, "scores_a")
    b_values = _score_vector(scores_b, "scores_b")
    if len(a_values) != len(b_values):
        raise ValueError("scores_a and scores_b must have the same number of values")
    differences = [a - b for a, b in zip(a_values, b_values, strict=True)]
    return _bootstrap_interval(differences, b, alpha, seed)


def _validated_runs(runs: Sequence[EvalRun]) -> list[EvalRun]:
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
            raise ValueError(f"score must be finite and in [0, 1], got {run.score!r}")
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
    by_task: dict[str, list[EvalRun]] = {}
    by_model: dict[str, dict[str, EvalRun]] = {}
    for run in runs:
        by_task.setdefault(run.task_id, []).append(run)
        by_model.setdefault(run.model_id, {})[run.task_id] = run
    return by_task, by_model


def _ordered_scores(task_map: Mapping[str, EvalRun]) -> list[float]:
    return [task_map[task_id].score for task_id in sorted(task_map)]


def win_rate(
    model_id: str,
    runs_by_task: Mapping[str, Sequence[EvalRun]],
    all_models: Collection[str],
) -> float:
    """Fraction of completed tasks where the model ties for the best score."""
    wins = 0
    total = 0
    for runs in runs_by_task.values():
        scores = {
            run.model_id: run.score for run in runs if run.model_id in all_models
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
    alpha: float = DEFAULT_ALPHA,
    seed: int | None = None,
) -> list[LeaderboardRow]:
    """Aggregate runs into rows sorted by mean score, then model id."""
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
        mean_score = sum(scores) / len(scores)
        mean_ci_lo, mean_ci_hi = bootstrap_mean_ci(scores, b=b, alpha=alpha, seed=seed)
        category_scores: dict[str, list[float]] = {}
        for task_id in sorted(task_map):
            category = task_map[task_id].category
            if category:
                category_scores.setdefault(category, []).append(task_map[task_id].score)
        categories = {
            name: sum(values) / len(values)
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
    alpha: float = DEFAULT_ALPHA,
    seed: int | None = None,
) -> list[PairwiseDiff]:
    """Paired differences on shared tasks, in leaderboard order."""
    checked = _validated_runs(runs)
    _validate_bootstrap(b, alpha)
    if not checked:
        return []
    _, by_model = _group_runs(checked)
    means = {
        model_id: sum(_ordered_scores(task_map)) / len(task_map)
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
            differences = [tasks_a[task_id].score - tasks_b[task_id].score for task_id in shared]
            ci_lo, ci_hi = _bootstrap_interval(differences, b, alpha, seed)
            pairs.append(
                PairwiseDiff(
                    model_a=model_a,
                    model_b=model_b,
                    mean_diff=sum(differences) / len(differences),
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
    alpha: float = DEFAULT_ALPHA,
    seed: int | None = None,
) -> LeaderboardReport:
    """Build leaderboard rows, pairwise comparisons, and their settings."""
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
    cleaned = " ".join(model_id.split())
    return cleaned[:MODEL_COLUMN_WIDTH].replace("|", r"\|")


def render_markdown(rows: Sequence[LeaderboardRow], alpha: float = DEFAULT_ALPHA) -> str:
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
    pairs: Sequence[PairwiseDiff], alpha: float = DEFAULT_ALPHA
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
