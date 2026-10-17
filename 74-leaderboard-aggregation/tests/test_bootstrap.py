import pytest

from main import (
    EvalRun,
    bootstrap_mean_ci,
    bootstrap_pairwise_diff,
    pairwise_diffs,
)


def make_run(model: str, task: str, score: float) -> EvalRun:
    return EvalRun(model, task, "accuracy", score)


def test_constant_scores_have_a_zero_width_interval() -> None:
    assert bootstrap_mean_ci([0.7] * 8, b=101, seed=3) == pytest.approx(
        (0.7, 0.7), abs=1e-12
    )


def test_bootstrap_is_reproducible_and_uses_the_default_b() -> None:
    scores = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.9, 0.65, 0.3, 0.5, 0.75, 0.95]

    first = bootstrap_mean_ci(scores, seed=5)
    second = bootstrap_mean_ci(scores, seed=5)

    assert first == second
    assert first == bootstrap_mean_ci(scores, b=1000, seed=5)
    assert first[0] <= sum(scores) / len(scores) <= first[1]


def test_pairwise_bootstrap_is_symmetric() -> None:
    a = [0.9, 0.7, 0.8, 0.6, 0.75, 0.85]
    b = [0.6, 0.65, 0.5, 0.55, 0.7, 0.4]

    forward = bootstrap_pairwise_diff(a, b, b=300, seed=11)
    reverse = bootstrap_pairwise_diff(b, a, b=300, seed=11)

    assert forward[0] == pytest.approx(-reverse[1])
    assert forward[1] == pytest.approx(-reverse[0])


def test_constant_pairwise_difference_is_significant_only_when_nonzero() -> None:
    assert bootstrap_pairwise_diff(
        [0.9, 0.8, 0.7], [0.6, 0.5, 0.4], b=100, seed=7
    ) == pytest.approx((0.3, 0.3))
    assert bootstrap_pairwise_diff(
        [0.5, 0.6, 0.7], [0.5, 0.6, 0.7], b=100, seed=7
    ) == pytest.approx((0.0, 0.0))

    positive = pairwise_diffs(
        [
            make_run("a", "t1", 0.9),
            make_run("a", "t2", 0.8),
            make_run("a", "t3", 0.7),
            make_run("b", "t1", 0.6),
            make_run("b", "t2", 0.5),
            make_run("b", "t3", 0.4),
        ],
        b=100,
        seed=7,
    )
    tied = pairwise_diffs(
        [
            make_run("a", "t1", 0.6),
            make_run("a", "t2", 0.5),
            make_run("b", "t1", 0.6),
            make_run("b", "t2", 0.5),
        ],
        b=100,
        seed=7,
    )

    assert positive[0].significant is True
    assert tied[0].significant is False


def test_tighter_alpha_nests_inside_the_wider_interval() -> None:
    scores = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.55, 0.4, 0.75, 0.3]

    wide = bootstrap_mean_ci(scores, b=500, alpha=0.01, seed=2)
    narrow = bootstrap_mean_ci(scores, b=500, alpha=0.10, seed=2)

    assert wide[0] <= narrow[0] <= narrow[1] <= wide[1]


def test_pairwise_diffs_use_only_shared_tasks() -> None:
    runs = [
        make_run("a", "t1", 0.99),
        make_run("a", "t2", 0.90),
        make_run("a", "t3", 0.80),
        make_run("a", "t4", 0.70),
        make_run("b", "t2", 0.60),
        make_run("b", "t3", 0.50),
        make_run("b", "t4", 0.40),
        make_run("c", "t1", 0.10),
    ]

    pairs = pairwise_diffs(runs, b=100, seed=8)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.model_a == "a"
    assert pair.model_b == "b"
    assert pair.tasks_compared == 3
    assert pair.mean_diff == pytest.approx((0.90 - 0.60 + 0.80 - 0.50 + 0.70 - 0.40) / 3)
    assert all(pair.model_a != "c" and pair.model_b != "c" for pair in pairs)


def test_pairs_with_fewer_than_two_shared_tasks_are_omitted() -> None:
    runs = [
        make_run("a", "t1", 0.9),
        make_run("a", "t2", 0.8),
        make_run("b", "t1", 0.1),
    ]
    assert pairwise_diffs(runs, b=50, seed=1) == []


@pytest.mark.parametrize(
    "scores",
    [[], [0.5, 1.1], [0.5, float("nan")], [[0.1], [0.2]]],
)
def test_bootstrap_rejects_invalid_score_vectors(scores: object) -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci(scores)  # type: ignore[arg-type]


@pytest.mark.parametrize("b", [0, -1, True, 1.5])
def test_bootstrap_rejects_invalid_b(b: object) -> None:
    with pytest.raises(ValueError, match="b must be"):
        bootstrap_mean_ci([0.5], b=b)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_bootstrap_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_mean_ci([0.5], alpha=alpha)


def test_pairwise_bootstrap_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same number"):
        bootstrap_pairwise_diff([0.1, 0.2], [0.1])
