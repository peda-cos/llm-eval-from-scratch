"""Calibration reports: expected calibration error and Brier score.

The adapter supplies one confidence and one binary correctness value per
prediction. This module validates and aggregates them without NumPy.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

DEFAULT_BINS: Final[int] = 10


def _validate_bins(bins: int) -> int:
    if isinstance(bins, bool):
        raise ValueError("bins must be a positive integer")
    try:
        value = operator.index(bins)
    except TypeError as exc:
        raise ValueError("bins must be a positive integer") from exc
    if value < 1:
        raise ValueError("bins must be a positive integer")
    return value


def _validated_predictions(
    confidences: Sequence[float], correct: Sequence[bool | int | float]
) -> tuple[list[float], list[float]]:
    p_values = [float(value) for value in confidences]
    y_values = [float(value) for value in correct]
    if not p_values:
        raise ValueError("confidences and correct must not be empty")
    if len(p_values) != len(y_values):
        raise ValueError("confidences and correct must have the same length")
    for value in p_values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("confidences must be finite values in [0, 1]")
    for value in y_values:
        if value not in (0.0, 1.0):
            raise ValueError("correct entries must be 0 or 1")
    return p_values, y_values


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """One model's pooled calibration numbers and reliability bins."""

    ece: float
    brier: float
    populated_bins: int
    bins: int
    n_samples: int
    mean_confidence: tuple[float, ...]
    mean_accuracy: tuple[float, ...]
    counts: tuple[int, ...]

    @classmethod
    def from_predictions(
        cls,
        confidences: Sequence[float],
        correct: Sequence[bool | int | float],
        bins: int = DEFAULT_BINS,
    ) -> CalibrationReport:
        """Pool all predictions into one count-weighted reliability report."""
        p_values, y_values = _validated_predictions(confidences, correct)
        validated_bins = _validate_bins(bins)
        n = len(p_values)
        counts = [0] * validated_bins
        confidence_sums = [0.0] * validated_bins
        correct_sums = [0.0] * validated_bins

        for confidence, outcome in zip(p_values, y_values, strict=True):
            index = min(int(confidence * validated_bins), validated_bins - 1)
            counts[index] += 1
            confidence_sums[index] += confidence
            correct_sums[index] += outcome

        mean_confidence = [0.0] * validated_bins
        mean_accuracy = [0.0] * validated_bins
        for index, count in enumerate(counts):
            if count:
                mean_confidence[index] = confidence_sums[index] / count
                mean_accuracy[index] = correct_sums[index] / count

        ece = sum(
            (count / n) * abs(mean_accuracy[index] - mean_confidence[index])
            for index, count in enumerate(counts)
            if count
        )
        brier = sum((confidence - outcome) ** 2 for confidence, outcome in zip(p_values, y_values, strict=True)) / n
        return cls(
            ece=float(ece),
            brier=float(brier),
            populated_bins=sum(1 for count in counts if count),
            bins=validated_bins,
            n_samples=n,
            mean_confidence=tuple(mean_confidence),
            mean_accuracy=tuple(mean_accuracy),
            counts=tuple(counts),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ece": self.ece,
            "brier": self.brier,
            "reliability": {
                "mean_confidence": list(self.mean_confidence),
                "mean_accuracy": list(self.mean_accuracy),
                "counts": list(self.counts),
            },
            "populated_bins": self.populated_bins,
            "bins": self.bins,
            "n_samples": self.n_samples,
        }


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool | int | float],
    bins: int = DEFAULT_BINS,
) -> float:
    """Return the count-weighted mean confidence/accuracy gap."""
    return CalibrationReport.from_predictions(confidences, correct, bins=bins).ece


def brier_score(
    confidences: Sequence[float], correct: Sequence[bool | int | float]
) -> float:
    """Return the mean squared error between confidence and binary outcome."""
    return CalibrationReport.from_predictions(confidences, correct).brier
