import json

import pytest

from main import (
    EvalRun,
    LeaderboardRow,
    PairwiseDiff,
    build_report,
    render_markdown,
    render_pairwise_markdown,
)


def make_row(
    model: str,
    mean: float,
    low: float,
    high: float,
    win_rate: float,
    tasks: int,
    categories: dict[str, float] | None = None,
) -> LeaderboardRow:
    return LeaderboardRow(
        model_id=model,
        mean_score=mean,
        mean_ci_lo=low,
        mean_ci_hi=high,
        win_rate=win_rate,
        tasks_completed=tasks,
        categories={} if categories is None else categories,
    )


def test_markdown_sorts_rounds_and_truncates() -> None:
    short = make_row("a|b", 0.80, 0.70, 0.90, 0.50, 3)
    long = make_row("x" * 25, 0.90, 0.85, 0.95, 0.75, 12)

    rendered = render_markdown([short, long])

    assert rendered == "\n".join(
        [
            "| Rank | Model | Mean | 95% CI | Win rate | Tasks |",
            "|------|-------|------|--------|----------|-------|",
            f"| 1 | {'x' * 20} | 0.90 | 0.85-0.95 | 0.75 | 12 |",
            "| 2 | a\\|b | 0.80 | 0.70-0.90 | 0.50 | 3 |",
        ]
    )


def test_markdown_ci_label_tracks_alpha() -> None:
    row = make_row("m", 0.5, 0.4, 0.6, 1.0, 2)
    rendered = render_markdown([row], alpha=0.10)
    assert "| Rank | Model | Mean | 90% CI | Win rate | Tasks |" in rendered


def test_pairwise_markdown_formats_negative_intervals_and_flags() -> None:
    pairs = [
        PairwiseDiff("a", "b", -0.05, -0.10, -0.01, 4, True),
        PairwiseDiff("a", "c", 0.02, -0.01, 0.05, 4, False),
    ]

    rendered = render_pairwise_markdown(pairs)

    assert rendered == "\n".join(
        [
            "| Model A | Model B | Mean diff | 95% CI | Significant | Tasks |",
            "|---------|---------|-----------|--------|-------------|-------|",
            "| a | b | -0.050 | -0.100 to -0.010 | yes | 4 |",
            "| a | c | 0.020 | -0.010 to 0.050 | no | 4 |",
        ]
    )


def test_json_report_schema_and_optional_categories() -> None:
    runs = [
        EvalRun("a", "t1", "accuracy", 0.9, "math"),
        EvalRun("a", "t2", "accuracy", 0.7, "code"),
        EvalRun("a", "t3", "accuracy", 0.8),
        EvalRun("b", "t1", "accuracy", 0.6),
        EvalRun("b", "t2", "accuracy", 0.5),
        EvalRun("b", "t3", "accuracy", 0.4),
    ]
    report = build_report(runs, b=100, alpha=0.05, seed=7)

    payload = report.to_dict()
    assert set(payload) == {
        "schema_version",
        "alpha",
        "bootstrap_samples",
        "seed",
        "models",
        "pairwise",
    }
    assert payload["schema_version"] == 1
    assert payload["alpha"] == 0.05
    assert payload["bootstrap_samples"] == 100
    assert payload["seed"] == 7

    models = payload["models"]
    assert isinstance(models, list)
    assert models[0]["model_id"] == "a"
    assert models[0]["categories"] == pytest.approx(
        {"math": 0.9, "code": 0.7}
    )
    assert "rank" not in models[0]
    assert "categories" not in models[1]
    pairs = payload["pairwise"]
    assert isinstance(pairs, list)
    assert pairs[0]["model_a"] == "a"
    assert pairs[0]["significant"] is True

    assert json.loads(report.to_json()) == payload
    assert report.to_json() == report.to_json()
    json.dumps(payload, allow_nan=False)


def test_empty_report_renders_header_only() -> None:
    report = build_report([], b=100)
    payload = report.to_dict()
    assert payload["models"] == []
    assert payload["pairwise"] == []
    assert payload["seed"] is None

    header = "| Rank | Model | Mean | 95% CI | Win rate | Tasks |"
    separator = "|------|-------|------|--------|----------|-------|"
    assert render_markdown([]) == header + "\n" + separator

    pair_header = "| Model A | Model B | Mean diff | 95% CI | Significant | Tasks |"
    pair_separator = "|---------|---------|-----------|--------|-------------|-------|"
    assert render_pairwise_markdown([]) == pair_header + "\n" + pair_separator
