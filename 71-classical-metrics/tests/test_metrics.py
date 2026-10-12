import math

import pytest

from main import (
    _reference_examples,
    accuracy,
    bleu4,
    exact_match,
    f1_score,
    lcs_length,
    main as cli_main,
    rouge_l,
    score,
    tokenize,
)

REFERENCE_METRICS = {"exact_match", "f1", "bleu_4", "rouge_l", "accuracy"}


def test_tokenize_lowercases_and_keeps_word_runs() -> None:
    assert tokenize("The cat's 42_fish, café!") == [
        "the",
        "cat",
        "s",
        "42_fish",
        "café",
    ]


def test_tokenize_punctuation_only_is_empty() -> None:
    assert tokenize("!!! --- ...") == []


def test_exact_match_strips_surrounding_whitespace() -> None:
    assert exact_match(" 41 ", ["41"]) == 1.0


def test_exact_match_is_case_and_punctuation_sensitive() -> None:
    assert exact_match("Paris!", ["Paris"]) == 0.0
    assert exact_match("Paris", ["paris"]) == 0.0


def test_exact_match_accepts_any_target() -> None:
    assert exact_match("b", ["a", "b"]) == 1.0


@pytest.mark.parametrize(
    ("prediction", "targets", "expected"),
    [
        (" 41 ", ["41"], 1.0),
        ("Positive", ["positive"], 0.0),
        ("positive", ["negative", "positive"], 1.0),
    ],
)
def test_accuracy_matches_exact_match_semantics(
    prediction: str, targets: list[str], expected: float
) -> None:
    assert accuracy(prediction, targets) == expected


def test_f1_perfect_match_after_tokenization() -> None:
    assert f1_score("The cat!", "the cat") == pytest.approx(1.0)


def test_f1_partial_overlap_worked_example() -> None:
    assert f1_score("the cat sat", "the cat is sat") == pytest.approx(6 / 7)


def test_f1_uses_multiset_intersection_not_set() -> None:
    assert f1_score("a a a", "a a") == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("prediction", "target", "expected"),
    [
        ("", "", 1.0),
        ("", "a", 0.0),
        ("a", "", 0.0),
        ("a b", "c d", 0.0),
    ],
)
def test_f1_empty_and_disjoint_cases(
    prediction: str, target: str, expected: float
) -> None:
    assert f1_score(prediction, target) == expected


def test_bleu4_identical_prediction_scores_one() -> None:
    sentence = "the cat sat on the mat"
    assert bleu4(sentence, sentence) == pytest.approx(1.0)


def test_bleu4_clips_repeated_candidate_ngrams() -> None:
    # p = 3/5, 1/4, 1/3, 1/2; BP = exp(-0.5); BLEU = exp(-0.5) * (1/40)^(1/4).
    assert bleu4("the the the the", "the cat sat on the mat") == pytest.approx(
        0.24117803988461298, abs=1e-12
    )


def test_bleu4_short_candidate_applies_brevity_penalty() -> None:
    assert bleu4(
        "the cat sat", "the cat sat on the mat"
    ) == pytest.approx(math.exp(-1), abs=1e-12)


def test_bleu4_partial_match_worked_example() -> None:
    # p = 6/7, 4/6, 3/5, 2/4; BP = 1; product = 6/35.
    assert bleu4(
        "the cat sat on the mat", "the cat sat on a mat"
    ) == pytest.approx(0.6434588841607617, abs=1e-12)


def test_bleu4_add_one_smoothing_keeps_disjoint_pair_positive() -> None:
    # p = 1/5, 1/4, 1/3, 1/2; BP = 1; product = 1/120.
    assert bleu4(
        "alpha beta gamma delta", "one two three four"
    ) == pytest.approx(0.3021375397356768, abs=1e-12)


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("", "", 0.0),
        ("", "a", 0.0),
        ("a", "", 0.0),
    ],
)
def test_bleu4_empty_inputs_score_zero(
    prediction: str, reference: str, expected: float
) -> None:
    assert bleu4(prediction, reference) == expected


def test_lcs_length_worked_example() -> None:
    assert lcs_length(list("ABCBDAB"), list("BDCABA")) == 4


def test_lcs_length_empty_and_identical_cases() -> None:
    assert lcs_length([], []) == 0
    assert lcs_length(["a", "b"], ["a", "b"]) == 2


def test_rouge_l_uses_non_contiguous_longest_common_subsequence() -> None:
    assert rouge_l(
        "the quick brown fox jumps",
        "the quick fox jumps over the dog",
    ) == pytest.approx(2 / 3)


def test_rouge_l_combines_precision_and_recall() -> None:
    assert rouge_l(
        "police the gunman", "police killed the gunman"
    ) == pytest.approx(6 / 7)


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("", "", 1.0),
        ("", "a", 0.0),
        ("a", "", 0.0),
        ("a b", "c d", 0.0),
    ],
)
def test_rouge_l_empty_and_disjoint_cases(
    prediction: str, reference: str, expected: float
) -> None:
    assert rouge_l(prediction, reference) == expected


@pytest.mark.parametrize(
    ("metric_name", "prediction", "targets", "expected"),
    [
        ("exact_match", " 41 ", ["41"], 1.0),
        ("accuracy", "positive", ["negative", "positive"], 1.0),
        ("f1", "the cat sat", ["the cat is sat"], 6 / 7),
        (
            "bleu_4",
            "the cat sat",
            ["the cat sat on the mat"],
            math.exp(-1),
        ),
        (
            "rouge_l",
            "the quick brown fox jumps",
            ["the quick fox jumps over the dog"],
            2 / 3,
        ),
    ],
)
def test_score_dispatches_every_metric(
    metric_name: str,
    prediction: str,
    targets: list[str],
    expected: float,
) -> None:
    assert score(metric_name, prediction, targets) == pytest.approx(expected)


def test_score_takes_best_target_for_multi_target_metrics() -> None:
    assert score("f1", "a b c d", ["a b e", "a b c d"]) == 1.0
    assert score(
        "bleu_4",
        "the cat sat on the mat",
        ["a dog slept on the rug", "the cat sat on the mat"],
    ) == pytest.approx(1.0)
    assert score(
        "rouge_l",
        "the quick brown fox jumps",
        ["the quick fox jumps over the dog", "the quick brown fox jumps"],
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("metric_name", ["code_exec", "bogus"])
def test_score_rejects_unknown_metric(metric_name: str) -> None:
    with pytest.raises(ValueError) as error:
        score(metric_name, "prediction", ["target"])
    assert str(error.value) == f"unknown metric_name: {metric_name}"


@pytest.mark.parametrize(
    "metric_name",
    ["exact_match", "accuracy", "f1", "bleu_4", "rouge_l"],
)
def test_score_rejects_empty_targets(metric_name: str) -> None:
    with pytest.raises(ValueError) as error:
        score(metric_name, "prediction", [])
    assert str(error.value) == "targets must not be empty"


def test_reference_example_scores_stay_in_unit_interval() -> None:
    for example in _reference_examples:
        value = score(
            example["metric_name"],
            example["prediction"],
            example["targets"],
        )
        assert 0.0 <= value <= 1.0


def test_reference_examples_are_eight_and_cover_every_metric() -> None:
    assert len(_reference_examples) == 8
    assert {example["metric_name"] for example in _reference_examples} == (
        REFERENCE_METRICS
    )


def test_reference_examples_match_hand_derived_values() -> None:
    for example in _reference_examples:
        actual = score(
            example["metric_name"],
            example["prediction"],
            example["targets"],
        )
        assert math.isclose(
            actual, example["expected"], rel_tol=0.0, abs_tol=1e-12
        ), example["note"]


def test_demo_prints_one_line_per_reference_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["demo"]) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 8
    assert all("score=" in line and "expected=" in line for line in lines)
    assert captured.err == ""


def test_cli_rejects_unknown_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["bogus"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage: python main.py [demo]" in captured.err
