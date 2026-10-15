import math
import warnings

import numpy as np
import pytest

from main import (
    BrierDecomposition,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)


def test_perfectly_calibrated_bins_have_zero_ece() -> None:
    confidences: list[float] = []
    correct: list[int] = []
    for hits in range(5, 100, 10):
        confidences.extend([hits / 100.0] * 100)
        correct.extend([1] * hits + [0] * (100 - hits))
    assert math.isclose(
        expected_calibration_error(confidences, correct),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_all_wrong_predictions_report_average_confidence() -> None:
    confidences = [0.9] * 100
    correct = [0] * 100
    assert math.isclose(
        expected_calibration_error(confidences, correct),
        0.9,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        brier_score(confidences, correct), 0.81, rel_tol=0.0, abs_tol=1e-12
    )


def test_all_correct_predictions_keep_the_confidence_gap() -> None:
    confidences = [0.95] * 100
    correct = [1] * 100
    assert math.isclose(
        expected_calibration_error(confidences, correct),
        0.05,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        brier_score(confidences, correct), 0.0025, rel_tol=0.0, abs_tol=1e-12
    )


def test_uncertain_predictions_hide_in_ece_but_not_brier() -> None:
    confidences = [0.5] * 4
    correct = [1, 0, 1, 0]
    assert expected_calibration_error(confidences, correct) == 0.0
    assert math.isclose(
        brier_score(confidences, correct), 0.25, rel_tol=0.0, abs_tol=1e-12
    )


def test_ece_changes_when_the_same_predictions_change_bins() -> None:
    confidences = [0.52, 0.58]
    correct = [1, 0]
    assert math.isclose(
        expected_calibration_error(confidences, correct, bins=10),
        0.05,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        expected_calibration_error(confidences, correct, bins=15),
        0.53,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_reliability_arrays_expose_bin_means_and_counts() -> None:
    mean_confidence, mean_accuracy, counts = reliability_diagram(
        [0.05, 0.05, 0.55], [1, 0, 1], bins=10
    )
    assert counts.sum() == 3
    assert counts[0] == 2
    assert counts[5] == 1
    assert math.isclose(mean_confidence[0], 0.05, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mean_accuracy[0], 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mean_confidence[5], 0.55, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mean_accuracy[5], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert counts[1] == 0
    assert mean_confidence[1] == 0.0
    assert mean_accuracy[1] == 0.0


def test_confidence_extremes_land_in_first_and_last_bins() -> None:
    _, _, counts = reliability_diagram([0.0, 1.0], [0, 1], bins=10)
    assert counts[0] == 1
    assert counts[9] == 1
    assert counts.sum() == 2


def test_empty_reliability_returns_zero_arrays_of_requested_length() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean_confidence, mean_accuracy, counts = reliability_diagram([], [], bins=3)
    assert mean_confidence.shape == (3,)
    assert mean_accuracy.shape == (3,)
    assert counts.shape == (3,)
    assert counts.dtype == np.int64
    assert np.all(mean_confidence == 0.0)
    assert np.all(mean_accuracy == 0.0)
    assert np.all(counts == 0)


def test_brier_score_hand_computed_vector() -> None:
    expected = (0.9 - 1.0) ** 2 + (0.8 - 0.0) ** 2 + (0.2 - 0.0) ** 2
    assert math.isclose(
        brier_score([0.9, 0.8, 0.2], [1, 0, 0]),
        expected / 3.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_brier_decomposition_hand_computed_terms() -> None:
    confidences = [0.9] * 10 + [0.2] * 10
    correct = [1] * 6 + [0] * 4 + [1] * 2 + [0] * 8
    terms = brier_decomposition(confidences, correct)
    assert isinstance(terms, BrierDecomposition)
    assert math.isclose(terms.reliability, 0.045, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(terms.resolution, 0.04, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(terms.uncertainty, 0.24, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(terms.base_rate, 0.4, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        brier_score(confidences, correct), 0.245, rel_tol=0.0, abs_tol=1e-12
    )


def test_brier_decomposition_identity_holds_on_repeated_confidences() -> None:
    confidences = [0.1, 0.1, 0.2, 0.2, 0.2, 0.7, 0.7, 0.8, 0.9, 0.9]
    correct = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    terms = brier_decomposition(confidences, correct)
    decomposed = terms.reliability - terms.resolution + terms.uncertainty
    assert math.isclose(
        brier_score(confidences, correct),
        decomposed,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_invalid_calibration_inputs_are_rejected() -> None:
    for confidences, correct in (
        ([-0.1], [1]),
        ([1.1], [1]),
        ([float("nan")], [1]),
        ([0.5], [1, 0]),
        ([0.5], [2]),
        ([0.5], [-1]),
    ):
        with pytest.raises(ValueError):
            expected_calibration_error(confidences, correct)
    for bad_bins in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [1], bins=bad_bins)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        reliability_diagram([[0.5]], [1])


def test_empty_calibration_metrics_are_zero_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert expected_calibration_error([], []) == 0.0
        assert brier_score([], []) == 0.0
        mean_confidence, mean_accuracy, counts = reliability_diagram([], [])
        terms = brier_decomposition([], [])
    assert mean_confidence.shape == (10,)
    assert mean_accuracy.shape == (10,)
    assert counts.shape == (10,)
    assert np.all(mean_confidence == 0.0)
    assert np.all(mean_accuracy == 0.0)
    assert np.all(counts == 0)
    assert terms == BrierDecomposition(0.0, 0.0, 0.0, 0.0)
