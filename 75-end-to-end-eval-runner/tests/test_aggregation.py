from __future__ import annotations

import pytest

from eval_runner.aggregation import (
    EvalRun,
    LeaderboardRow,
    aggregate,
    bootstrap_mean_ci,
    bootstrap_pairwise_diff,
    build_report,
    pairwise_diffs,
    render_markdown,
    render_pairwise_markdown,
)


def _runs() -> list[EvalRun]:
    return [
        EvalRun("alpha", "t1", "exact_match", 1.0, "math"),
        EvalRun("alpha", "t2", "f1", 0.5, "code"),
        EvalRun("alpha", "t3", "rouge_l", 0.5, "math"),
        EvalRun("beta", "t1", "exact_match", 1.0, "math"),
        EvalRun("beta", "t2", "f1", 1.0, "code"),
        EvalRun("beta", "t3", "rouge_l", 0.0, "math"),
        EvalRun("gamma", "t1", "exact_match", 0.0, "math"),
    ]


def test_aggregate_means_categories_win_rates_and_order() -> None:
    rows = aggregate(_runs(), b=50, seed=7)
    assert [row.model_id for row in rows] == ["alpha", "beta", "gamma"]
    assert rows[0].mean_score == pytest.approx(2.0 / 3.0)
    assert rows[1].mean_score == pytest.approx(2.0 / 3.0)
    assert rows[0].categories == {"code": 0.5, "math": 0.75}
    assert rows[0].win_rate == pytest.approx(2.0 / 3.0)
    assert rows[1].win_rate == pytest.approx(2.0 / 3.0)
    assert rows[2].win_rate == 0.0
    assert rows[2].tasks_completed == 1
    assert rows[0].to_dict()["categories"] == {"code": 0.5, "math": 0.75}


def test_bootstrap_is_reproducible_and_pairwise_is_symmetric() -> None:
    first = bootstrap_mean_ci([0.0, 0.5, 1.0], b=100, seed=3)
    second = bootstrap_mean_ci([0.0, 0.5, 1.0], b=100, seed=3)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0

    ab = bootstrap_pairwise_diff([0.0, 0.5, 1.0], [0.0, 1.0, 0.0], b=100, seed=4)
    ba = bootstrap_pairwise_diff([0.0, 1.0, 0.0], [0.0, 0.5, 1.0], b=100, seed=4)
    assert ab[0] == pytest.approx(-ba[1])
    assert ab[1] == pytest.approx(-ba[0])


def test_pairwise_uses_shared_tasks_and_detects_constant_difference() -> None:
    pairs = pairwise_diffs(_runs(), b=50, seed=7)
    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair.model_a, pair.model_b) == ("alpha", "beta")
    assert pair.tasks_compared == 3
    assert pair.mean_diff == pytest.approx(0.0)

    constant = [
        EvalRun("high", "t1", "exact_match", 1.0),
        EvalRun("high", "t2", "exact_match", 1.0),
        EvalRun("low", "t1", "exact_match", 0.0),
        EvalRun("low", "t2", "exact_match", 0.0),
    ]
    constant_pair = pairwise_diffs(constant, b=50, seed=7)[0]
    assert constant_pair.mean_diff == 1.0
    assert (constant_pair.ci_lo, constant_pair.ci_hi) == (1.0, 1.0)
    assert constant_pair.significant is True
    assert constant_pair.to_dict()["significant"] is True


def test_insufficient_shared_tasks_and_single_task_interval() -> None:
    runs = [
        EvalRun("one", "t1", "exact_match", 0.5),
        EvalRun("two", "t2", "exact_match", 1.0),
        EvalRun("three", "t1", "exact_match", 0.5),
    ]
    assert pairwise_diffs(runs, b=20, seed=1) == []
    ci = bootstrap_mean_ci([0.5], b=20, seed=1)
    assert ci == (0.5, 0.5)


def test_aggregate_rejects_invalid_runs_and_empty_input() -> None:
    assert aggregate([]) == []
    assert pairwise_diffs([]) == []
    report = build_report([])
    assert report.rows == () and report.pairwise == ()

    with pytest.raises(ValueError, match="duplicate"):
        aggregate(
            [
                EvalRun("a", "t", "exact_match", 1.0),
                EvalRun("a", "t", "exact_match", 0.0),
            ]
        )
    with pytest.raises(ValueError, match="score"):
        aggregate([EvalRun("a", "t", "exact_match", 1.5)])
    with pytest.raises(ValueError, match="weight"):
        aggregate([EvalRun("a", "t", "exact_match", 1.0, weight=0.0)])
    with pytest.raises(ValueError, match="model_id"):
        aggregate([EvalRun(" ", "t", "exact_match", 1.0)])
    with pytest.raises(ValueError, match="alpha"):
        aggregate([EvalRun("a", "t", "exact_match", 1.0)], alpha=1.0)
    with pytest.raises(ValueError, match="b must"):
        aggregate([EvalRun("a", "t", "exact_match", 1.0)], b=True)  # type: ignore[arg-type]


def test_markdown_tables_include_rank_truncation_and_escaping() -> None:
    rows = aggregate(_runs(), b=20, seed=7)
    table = render_markdown(rows, alpha=0.05)
    assert table.splitlines()[0] == "| Rank | Model | Mean | 95% CI | Win rate | Tasks |"
    assert "| 1 | alpha |" in table
    assert "| 2 | beta |" in table

    long_row = LeaderboardRow("x" * 25, 0.5, 0.4, 0.6, 0.5, 2)
    rendered = render_markdown([long_row])
    assert "x" * 20 in rendered
    escaped = render_markdown([LeaderboardRow("a|b", 0.5, 0.4, 0.6, 0.5, 2)])
    assert r"a\|b" in escaped

    pairs = pairwise_diffs(_runs(), b=20, seed=7)
    pair_table = render_pairwise_markdown(pairs)
    assert pair_table.splitlines()[0].startswith("| Model A | Model B | Mean diff | 95% CI |")
    assert "alpha" in pair_table and "beta" in pair_table
