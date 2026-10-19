from __future__ import annotations

import math

import pytest

from eval_runner.calibration import (
    CalibrationReport,
    brier_score,
    expected_calibration_error,
)


def test_hand_computed_ece_and_brier() -> None:
    report = CalibrationReport.from_predictions([0.9, 0.8, 0.4], [1, 0, 0])
    assert report.ece == pytest.approx(1.3 / 3)
    assert report.brier == pytest.approx(0.27)
    assert report.populated_bins == 3
    assert report.bins == 10
    assert report.n_samples == 3
    assert report.counts[4] == report.counts[8] == report.counts[9] == 1
    assert sum(report.counts) == 3
    assert expected_calibration_error([0.9, 0.8, 0.4], [True, False, False]) == pytest.approx(
        report.ece
    )
    assert brier_score([0.9, 0.8, 0.4], [1, 0, 0]) == pytest.approx(report.brier)


def test_one_bin_identity_and_reliability_arrays() -> None:
    report = CalibrationReport.from_predictions([0.2, 0.8], [0, 1], bins=1)
    assert report.ece == pytest.approx(0.0)
    assert report.brier == pytest.approx(0.04)
    assert report.mean_confidence == (pytest.approx(0.5),)
    assert report.mean_accuracy == (pytest.approx(0.5),)
    assert report.counts == (2,)

    payload = report.to_dict()
    assert payload["populated_bins"] == 1
    assert payload["reliability"] == {
        "mean_confidence": [pytest.approx(0.5)],
        "mean_accuracy": [pytest.approx(0.5)],
        "counts": [2],
    }


def test_perfect_calibration_reports_zero_error() -> None:
    report = CalibrationReport.from_predictions([1.0, 0.0, 1.0], [1, 0, 1])
    assert report.ece == pytest.approx(0.0)
    assert report.brier == pytest.approx(0.0)
    assert report.populated_bins == 2
    assert report.mean_confidence[9] == pytest.approx(1.0)
    assert report.mean_confidence[0] == pytest.approx(0.0)


def test_calibration_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CalibrationReport.from_predictions([], [])
    with pytest.raises(ValueError, match="same length"):
        CalibrationReport.from_predictions([0.5], [1, 0])
    with pytest.raises(ValueError, match="confidences"):
        CalibrationReport.from_predictions([1.1], [1])
    with pytest.raises(ValueError, match="confidences"):
        CalibrationReport.from_predictions([math.nan], [1])
    with pytest.raises(ValueError, match="correct"):
        CalibrationReport.from_predictions([0.5], [2])
    with pytest.raises(ValueError, match="bins"):
        CalibrationReport.from_predictions([0.5], [1], bins=0)
    with pytest.raises(ValueError, match="bins"):
        CalibrationReport.from_predictions([0.5], [1], bins=True)
    with pytest.raises(ValueError, match="bins"):
        CalibrationReport.from_predictions([0.5], [1], bins=1.5)  # type: ignore[arg-type]
