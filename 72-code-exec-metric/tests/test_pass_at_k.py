import math

import pytest

from main import pass_at_k


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (10, 3, 1, 0.3),
        (10, 3, 5, 231 / 252),
        (10, 3, 10, 1.0),
        (10, 0, 5, 0.0),
        (10, 10, 1, 1.0),
        (5, 1, 3, 0.6),
        (3, 1, 2, 2 / 3),
        (3, 2, 2, 1.0),
        (2, 1, 2, 1.0),
        (1, 0, 1, 0.0),
        (1, 1, 1, 1.0),
    ],
)
def test_pass_at_k_worked_vectors(
    n: int, c: int, k: int, expected: float
) -> None:
    assert math.isclose(pass_at_k(n, c, k), expected, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("n,c", [(1, 0), (2, 1), (10, 3), (10, 10)])
def test_pass_at_1_is_the_empirical_pass_rate(n: int, c: int) -> None:
    assert math.isclose(pass_at_k(n, c, 1), c / n, rel_tol=0.0, abs_tol=1e-12)


def test_pass_at_k_saturates_when_the_remaining_failures_are_fewer_than_k() -> None:
    # C(n - c, k) is zero by convention when n - c < k.
    assert pass_at_k(5, 3, 3) == 1.0
    assert pass_at_k(10, 8, 5) == 1.0
    assert pass_at_k(10, 10, 10) == 1.0


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [
        (0, 0, 1),
        (-1, 0, 1),
        (10, -1, 1),
        (10, 11, 1),
        (10, 3, 0),
        (10, 3, 11),
    ],
)
def test_pass_at_k_rejects_out_of_domain_inputs(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)
