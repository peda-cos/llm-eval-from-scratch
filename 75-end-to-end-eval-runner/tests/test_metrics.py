from __future__ import annotations

import math

import pytest

from eval_runner.metrics import (
    accuracy,
    bleu4,
    exact_match,
    f1_score,
    rouge_l,
    score,
    tokenize,
)


def test_exact_match_and_accuracy_strip_and_accept_any_target() -> None:
    assert tokenize("Hello, WORLD_42") == ["hello", "world_42"]
    assert exact_match(" 41 ", ["41"]) == 1.0
    assert exact_match("paris", ["Berlin", "paris"]) == 1.0
    assert exact_match("forty one", ["41"]) == 0.0
    assert accuracy("positive", ["negative", "positive"]) == 1.0
    assert accuracy("", []) == 0.0


def test_f1_score_reference_and_empty_cases() -> None:
    assert f1_score("the cat sat", "the cat is sat") == pytest.approx(0.8571428571428571)
    assert f1_score("", "") == 1.0
    assert f1_score("cat", "") == 0.0
    assert f1_score("", "cat") == 0.0
    assert f1_score("cat", "dog") == 0.0


def test_bleu4_reference_values() -> None:
    assert bleu4("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert bleu4("the cat sat", "the cat sat on the mat") == pytest.approx(
        0.36787944117144233
    )
    assert bleu4("the cat sat on the mat", "the cat sat on a mat") == pytest.approx(
        0.6434588841607617
    )
    assert bleu4("", "reference") == 0.0
    assert bleu4("candidate", "") == 0.0
    assert math.isfinite(bleu4("one", "one two three four"))


def test_rouge_l_reference_and_empty_cases() -> None:
    assert rouge_l("the quick brown fox jumps", "the quick fox jumps over the dog") == pytest.approx(
        0.6666666666666666
    )
    assert rouge_l("same text", "same text") == pytest.approx(1.0)
    assert rouge_l("", "") == 1.0
    assert rouge_l("cat", "") == 0.0
    assert rouge_l("cat", "dog") == 0.0


def test_score_dispatches_and_takes_the_best_target() -> None:
    assert score("f1", "billing", ["payments", "billing"]) == 1.0
    bleu_targets = ["the cat sat on the mat", "unrelated"]
    assert score("bleu_4", "the cat sat", bleu_targets) == pytest.approx(
        max(bleu4("the cat sat", target) for target in bleu_targets)
    )
    assert score("rouge_l", "the cat", ["the cat slept", "unrelated"]) == pytest.approx(0.8)
    assert score("exact_match", " B ", ["A", "B"]) == 1.0
    assert score("accuracy", "wrong", ["right"]) == 0.0
    with pytest.raises(ValueError, match="unknown text metric"):
        score("code_exec", "x", ["y"])
    with pytest.raises(ValueError, match="targets"):
        score("f1", "x", [])
