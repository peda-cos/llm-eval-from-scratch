import math
import random

import pytest

from main import EvalRun, aggregate, build_report, win_rate


def make_run(
    model: str,
    task: str,
    score: float,
    category: str = "",
    metric: str = "accuracy",
    weight: float = 1.0,
) -> EvalRun:
    return EvalRun(model, task, metric, score, category, weight)


def test_means_categories_and_win_rates_are_hand_computable() -> None:
    runs = [
        make_run("model-a", "a1", 0.8, "math"),
        make_run("model-a", "a2", 0.6, "math"),
        make_run("model-a", "a3", 1.0, "code"),
        make_run("model-b", "a1", 0.7, "math"),
        make_run("model-b", "a2", 0.9, "math"),
        make_run("model-b", "a3", 0.5, "code"),
    ]

    rows = aggregate(runs, b=200, seed=1)

    assert [row.model_id for row in rows] == ["model-a", "model-b"]
    first, second = rows
    assert first.mean_score == pytest.approx(0.8)
    assert first.tasks_completed == 3
    assert first.categories == pytest.approx({"math": 0.7, "code": 1.0})
    assert first.win_rate == pytest.approx(2 / 3)
    assert second.mean_score == pytest.approx(0.7)
    assert second.categories == pytest.approx({"math": 0.8, "code": 0.5})
    assert second.win_rate == pytest.approx(1 / 3)
    assert first.mean_ci_lo <= first.mean_score <= first.mean_ci_hi


def test_ties_give_a_full_win_to_every_tied_model() -> None:
    runs = [
        make_run("a", "t1", 0.9),
        make_run("b", "t1", 0.9),
        make_run("a", "t2", 0.5),
        make_run("b", "t2", 0.5),
        make_run("c", "t1", 0.2),
        make_run("c", "t2", 0.1),
    ]
    rows = {row.model_id: row for row in aggregate(runs, b=100, seed=2)}

    assert rows["a"].win_rate == 1.0
    assert rows["b"].win_rate == 1.0
    assert rows["c"].win_rate == 0.0


def test_win_rate_denominator_counts_only_the_models_own_tasks() -> None:
    runs = [
        make_run("a", "t1", 0.9),
        make_run("a", "t2", 0.8),
        make_run("b", "t1", 0.7),
        make_run("b", "t2", 0.9),
        make_run("c", "t1", 0.9),
    ]
    rows = {row.model_id: row for row in aggregate(runs, b=100, seed=3)}

    assert rows["a"].win_rate == 0.5
    assert rows["a"].tasks_completed == 2
    assert rows["b"].win_rate == 0.5
    assert rows["c"].win_rate == 1.0
    assert rows["c"].tasks_completed == 1


@pytest.mark.parametrize(
    "score",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_scores_outside_the_unit_interval_are_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="score"):
        aggregate([make_run("m", "t", score)])


def test_duplicate_model_task_pairs_are_rejected() -> None:
    runs = [
        make_run("m", "t", 0.5),
        make_run("m", "t", 0.6),
    ]
    with pytest.raises(ValueError, match="duplicate run"):
        aggregate(runs)


@pytest.mark.parametrize("field", ["model_id", "task_id", "metric_name"])
def test_empty_identifiers_are_rejected(field: str) -> None:
    kwargs = {
        "model_id": "m",
        "task_id": "t",
        "metric_name": "accuracy",
        "score": 0.5,
        "category": "",
        "weight": 1.0,
    }
    kwargs[field] = "   "
    with pytest.raises(ValueError, match=field):
        aggregate([EvalRun(**kwargs)])


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_weights_are_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        aggregate([make_run("m", "t", 0.5, weight=weight)])


def test_aggregate_is_independent_of_run_order() -> None:
    runs = [
        make_run("a", "t1", 0.9, "math"),
        make_run("a", "t2", 0.7, "code"),
        make_run("a", "t3", 0.5, "code"),
        make_run("b", "t1", 0.8, "math"),
        make_run("b", "t2", 0.6, "code"),
        make_run("b", "t3", 0.9, "code"),
    ]
    expected = aggregate(runs, b=150, seed=4)

    shuffled = list(runs)
    random.Random(0).shuffle(shuffled)
    assert aggregate(shuffled, b=150, seed=4) == expected
    assert aggregate(list(reversed(runs)), b=150, seed=4) == expected


def test_weight_is_reserved_and_does_not_change_aggregation() -> None:
    light = [
        make_run("a", "t1", 0.9, weight=0.01),
        make_run("a", "t2", 0.7, weight=0.01),
        make_run("b", "t1", 0.8, weight=0.01),
        make_run("b", "t2", 0.6, weight=0.01),
    ]
    heavy = [
        make_run("a", "t1", 0.9, weight=100.0),
        make_run("a", "t2", 0.7, weight=100.0),
        make_run("b", "t1", 0.8, weight=100.0),
        make_run("b", "t2", 0.6, weight=100.0),
    ]

    assert aggregate(light, b=100, seed=5) == aggregate(heavy, b=100, seed=5)


def test_empty_runs_and_single_model_suites() -> None:
    assert aggregate([], b=50, seed=0) == []
    report = build_report([], b=50, seed=0)
    assert report.rows == ()
    assert report.pairwise == ()

    one_model = [
        make_run("solo", "t1", 0.4),
        make_run("solo", "t2", 0.6),
    ]
    rows = aggregate(one_model, b=50, seed=0)
    assert len(rows) == 1
    assert rows[0].win_rate == 1.0
    assert rows[0].tasks_completed == 2
    assert not math.isnan(rows[0].mean_score)


def test_win_rate_helper_accepts_the_spec_shape() -> None:
    runs = [
        make_run("a", "t1", 0.8),
        make_run("b", "t1", 0.7),
        make_run("a", "t2", 0.2),
        make_run("b", "t2", 0.9),
    ]
    by_task: dict[str, list[EvalRun]] = {}
    for run in runs:
        by_task.setdefault(run.task_id, []).append(run)

    assert win_rate("a", by_task, {"a", "b"}) == 0.5
    assert win_rate("b", by_task, {"a", "b"}) == 0.5
    assert win_rate("missing", by_task, {"a", "b"}) == 0.0
