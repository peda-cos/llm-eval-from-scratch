import math
import warnings

import pytest

from main import PerplexityResult, perplexity


def test_perplexity_aggregates_token_nll_over_sequences() -> None:
    nll = [0.5, 1.0, 1.5, 2.0]
    counts = [1, 3]
    expected_mean = 5.0 / 4.0
    assert math.isclose(
        perplexity(nll, counts), math.exp(expected_mean), rel_tol=0.0, abs_tol=1e-12
    )


def test_perplexity_result_exposes_totals_and_mean() -> None:
    result = PerplexityResult.from_token_nll([0.5, 1.5], [2])
    assert result.total_tokens == 2
    assert math.isclose(result.total_nll, 2.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.mean_nll, 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.perplexity, math.e, rel_tol=0.0, abs_tol=1e-12)


def test_uniform_model_over_fifty_tokens_has_perplexity_fifty() -> None:
    nll = [math.log(50.0)] * 20
    assert math.isclose(perplexity(nll, [7, 13]), 50.0, rel_tol=0.0, abs_tol=1e-12)


def test_zero_tokens_return_nan_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(perplexity([], []))
        assert math.isnan(perplexity([], [0, 0]))
        result = PerplexityResult.from_token_nll([], [])
    assert math.isnan(result.perplexity)
    assert math.isnan(result.mean_nll)
    assert result.total_tokens == 0


def test_negative_nll_is_a_contract_violation() -> None:
    with pytest.raises(ValueError, match="log p"):
        perplexity([-0.25], [1])


def test_malformed_counts_and_shapes_are_rejected() -> None:
    for bad_counts in ([-1], [1.5], [2]):
        with pytest.raises(ValueError):
            perplexity([1.0], bad_counts)
    with pytest.raises(ValueError):
        perplexity([[1.0]], [1])
    with pytest.raises(ValueError):
        perplexity([1.0], [[1]])


def test_certain_tokens_have_perplexity_one() -> None:
    assert perplexity([0.0] * 20, [20]) == 1.0
