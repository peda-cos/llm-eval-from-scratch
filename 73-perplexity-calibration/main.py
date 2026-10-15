"""Perplexity and calibration metrics for an LLM eval harness.

Perplexity is the exponentiated average negative log-likelihood per token.
Calibration is measured with ECE, the Brier score, its reliability /
resolution / uncertainty decomposition, and reliability-diagram bin data.
The adapter supplies every log-probability and confidence; this module only
validates and aggregates the numbers.
"""

from __future__ import annotations

import math
import operator
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ReliabilityData = tuple[FloatArray, FloatArray, IntArray]


def _float_vector(values: ArrayLike, name: str) -> FloatArray:
    """Convert a one-dimensional array-like to float64 without guessing shape."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _validated_nll(neg_log_probs: ArrayLike) -> FloatArray:
    """Validate per-token negative log-probabilities in nats."""
    nll = _float_vector(neg_log_probs, "neg_log_probs")
    if not bool(np.all(np.isfinite(nll))):
        raise ValueError("neg_log_probs must be finite")
    if bool(np.any(nll < 0.0)):
        raise ValueError(
            "neg_log_probs must be non-negative; "
            "the adapter probably returned log p instead of -log p"
        )
    return nll


def _validated_token_counts(token_counts: ArrayLike) -> IntArray:
    """Validate per-sequence token counts, rejecting truncating float input."""
    raw = np.asarray(token_counts)
    if raw.ndim != 1:
        raise ValueError("token_counts must be one-dimensional")
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64)
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("token_counts must contain non-negative integers")
    counts = raw.astype(np.int64)
    if bool(np.any(counts < 0)):
        raise ValueError("token_counts must be non-negative")
    return counts


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """Corpus perplexity and the per-token negative log-likelihood behind it."""

    perplexity: float
    mean_nll: float
    total_nll: float
    total_tokens: int

    @classmethod
    def from_token_nll(
        cls, neg_log_probs: ArrayLike, token_counts: ArrayLike
    ) -> PerplexityResult:
        """Aggregate per-token NLL, requiring one entry per scored token.

        ``token_counts`` is per sequence; the adapter passes one NLL entry for
        every token across all sequences. A corpus with no scored tokens is
        undefined, so its perplexity and mean NLL are ``nan``.
        """
        nll = _validated_nll(neg_log_probs)
        counts = _validated_token_counts(token_counts)
        total_tokens = int(counts.sum())
        if total_tokens != int(nll.size):
            raise ValueError(
                f"token_counts sum to {total_tokens}, "
                f"but {nll.size} per-token nll values were supplied"
            )
        total_nll = float(np.sum(nll))
        if total_tokens == 0:
            return cls(
                perplexity=float("nan"),
                mean_nll=float("nan"),
                total_nll=total_nll,
                total_tokens=0,
            )
        mean_nll = total_nll / total_tokens
        return cls(
            perplexity=math.exp(mean_nll),
            mean_nll=mean_nll,
            total_nll=total_nll,
            total_tokens=total_tokens,
        )


def perplexity(neg_log_probs: ArrayLike, token_counts: ArrayLike) -> float:
    """Return ``exp(sum(nll) / sum(token_counts))`` for a held-out corpus.

    The NLL values are per token in nats. A negative value violates the adapter
    contract because a log-probability cannot be positive; returning ``log p``
    instead of ``-log p`` would produce a perplexity below one.
    """
    return PerplexityResult.from_token_nll(neg_log_probs, token_counts).perplexity


def _calibration_vectors(
    confidences: ArrayLike, correct: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Validate one confidence and one binary correctness value per prediction."""
    p = _float_vector(confidences, "confidences")
    y = _float_vector(correct, "correct")
    if p.size != y.size:
        raise ValueError("confidences and correct must have the same length")
    if not bool(np.all(np.isfinite(p))):
        raise ValueError("confidences must be finite")
    if bool(np.any((p < 0.0) | (p > 1.0))):
        raise ValueError("confidences must lie in [0, 1]")
    if not bool(np.all((y == 0.0) | (y == 1.0))):
        raise ValueError("correct entries must be 0 or 1")
    return p, y


def _validated_bins(bins: int) -> int:
    """Accept any positive integer-like bin count; reject bool and floats."""
    if isinstance(bins, bool):
        raise ValueError("bins must be a positive integer")
    try:
        value = operator.index(bins)
    except TypeError as exc:
        raise ValueError("bins must be a positive integer") from exc
    if value < 1:
        raise ValueError("bins must be a positive integer")
    return value


def _reliability_from_arrays(
    p: FloatArray, y: FloatArray, bins: int
) -> ReliabilityData:
    """Compute per-bin confidence, accuracy, and count without empty-bin warnings."""
    indices = np.minimum((p * bins).astype(np.int64), bins - 1)
    counts = np.bincount(indices, minlength=bins)
    confidence_sums = np.bincount(indices, weights=p, minlength=bins)
    correct_sums = np.bincount(indices, weights=y, minlength=bins)
    mean_confidence = np.zeros(bins, dtype=np.float64)
    mean_accuracy = np.zeros(bins, dtype=np.float64)
    populated = counts > 0
    np.divide(confidence_sums, counts, out=mean_confidence, where=populated)
    np.divide(correct_sums, counts, out=mean_accuracy, where=populated)
    return mean_confidence, mean_accuracy, counts.astype(np.int64)


def _ece_from_stats(
    mean_confidence: FloatArray, mean_accuracy: FloatArray, counts: IntArray
) -> float:
    n = int(counts.sum())
    if n == 0:
        return 0.0
    populated = counts > 0
    weights = counts[populated].astype(np.float64) / float(n)
    gaps = np.abs(mean_accuracy[populated] - mean_confidence[populated])
    return float(np.sum(weights * gaps))


def _ece_from_arrays(p: FloatArray, y: FloatArray, bins: int) -> float:
    if p.size == 0:
        return 0.0
    return _ece_from_stats(*_reliability_from_arrays(p, y, bins))


def expected_calibration_error(
    confidences: ArrayLike, correct: ArrayLike, bins: int = 10
) -> float:
    """Return the count-weighted mean confidence/accuracy gap over equal-width bins.

    Confidence and correctness must be one value per prediction. Empty input is
    ``0.0``; use :class:`CalibrationReport` when the populated-bin count matters.
    """
    p, y = _calibration_vectors(confidences, correct)
    return _ece_from_arrays(p, y, _validated_bins(bins))


def brier_score(confidences: ArrayLike, correct: ArrayLike) -> float:
    """Return the mean squared error between confidence and binary outcome."""
    p, y = _calibration_vectors(confidences, correct)
    return _brier_from_arrays(p, y)


def _brier_from_arrays(p: FloatArray, y: FloatArray) -> float:
    if p.size == 0:
        return 0.0
    return float(np.mean(np.square(p - y)))


def reliability_diagram(
    confidences: ArrayLike, correct: ArrayLike, bins: int = 10
) -> ReliabilityData:
    """Return per-bin mean confidence, mean accuracy, and count.

    Unpopulated bins have zero mean confidence, zero mean accuracy, and count
    zero; plotting code should mask on ``counts > 0``. Empty input returns
    zero-filled arrays of length ``bins``.
    """
    p, y = _calibration_vectors(confidences, correct)
    return _reliability_from_arrays(p, y, _validated_bins(bins))


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    """Murphy decomposition: brier = reliability - resolution + uncertainty."""

    reliability: float
    resolution: float
    uncertainty: float
    base_rate: float


def _decomposition_from_arrays(p: FloatArray, y: FloatArray) -> BrierDecomposition:
    if p.size == 0:
        return BrierDecomposition(
            reliability=0.0, resolution=0.0, uncertainty=0.0, base_rate=0.0
        )
    unique, inverse = np.unique(p, return_inverse=True)
    counts = np.bincount(inverse, minlength=int(unique.size)).astype(np.float64)
    accuracy = (
        np.bincount(inverse, weights=y, minlength=int(unique.size)) / counts
    )
    base_rate = float(np.mean(y))
    weights = counts / float(p.size)
    reliability = float(np.sum(weights * np.square(unique - accuracy)))
    resolution = float(np.sum(weights * np.square(accuracy - base_rate)))
    uncertainty = base_rate * (1.0 - base_rate)
    return BrierDecomposition(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        base_rate=base_rate,
    )


def brier_decomposition(
    confidences: ArrayLike, correct: ArrayLike
) -> BrierDecomposition:
    """Return Murphy's reliability, resolution, and uncertainty terms.

    Groups share an exact confidence value, not an equal-width ECE bin. Using a
    bin's mean confidence would leave a within-bin variance term out of the
    identity and make the decomposition disagree with the Brier score.
    """
    p, y = _calibration_vectors(confidences, correct)
    return _decomposition_from_arrays(p, y)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """One model's pooled calibration numbers and bin data."""

    ece: float
    brier: float
    reliability: ReliabilityData
    decomposition: BrierDecomposition
    populated_bins: int
    bins: int
    n_samples: int

    @classmethod
    def from_predictions(
        cls, confidences: ArrayLike, correct: ArrayLike, bins: int = 10
    ) -> CalibrationReport:
        """Pool all predictions into one report for a model."""
        p, y = _calibration_vectors(confidences, correct)
        validated_bins = _validated_bins(bins)
        mean_confidence, mean_accuracy, counts = _reliability_from_arrays(
            p, y, validated_bins
        )
        return cls(
            ece=_ece_from_stats(mean_confidence, mean_accuracy, counts),
            brier=_brier_from_arrays(p, y),
            reliability=(mean_confidence, mean_accuracy, counts),
            decomposition=_decomposition_from_arrays(p, y),
            populated_bins=int(np.count_nonzero(counts)),
            bins=validated_bins,
            n_samples=int(p.size),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report body; non-finite values are not expected here."""
        mean_confidence, mean_accuracy, counts = self.reliability
        return {
            "ece": float(self.ece),
            "brier": float(self.brier),
            "reliability": {
                "mean_confidence": [float(value) for value in mean_confidence],
                "mean_accuracy": [float(value) for value in mean_accuracy],
                "counts": [int(value) for value in counts],
            },
            "decomposition": {
                "reliability": float(self.decomposition.reliability),
                "resolution": float(self.decomposition.resolution),
                "uncertainty": float(self.decomposition.uncertainty),
                "base_rate": float(self.decomposition.base_rate),
            },
            "populated_bins": int(self.populated_bins),
            "bins": int(self.bins),
            "n_samples": int(self.n_samples),
        }


@dataclass(frozen=True, slots=True)
class ModelReport:
    """The runner-facing report: calibration plus held-out perplexity."""

    model: str
    calibration: CalibrationReport | None = None
    perplexity: PerplexityResult | None = None

    def metrics(self) -> dict[str, float]:
        """Return exactly the scalars a leaderboard row needs."""
        metrics: dict[str, float] = {}
        if self.calibration is not None:
            metrics["ece"] = self.calibration.ece
            metrics["brier"] = self.calibration.brier
        if self.perplexity is not None:
            metrics["perplexity"] = self.perplexity.perplexity
        return metrics

    def to_dict(self) -> dict[str, object]:
        """Serialize the full report; a zero-token perplexity remains ``nan``."""
        calibration: dict[str, object] | None = None
        if self.calibration is not None:
            calibration = self.calibration.to_dict()
        perplexity: dict[str, object] | None = None
        if self.perplexity is not None:
            perplexity = {
                "perplexity": float(self.perplexity.perplexity),
                "mean_nll": float(self.perplexity.mean_nll),
                "total_nll": float(self.perplexity.total_nll),
                "total_tokens": int(self.perplexity.total_tokens),
            }
        return {
            "model": self.model,
            "metrics": self.metrics(),
            "calibration": calibration,
            "perplexity": perplexity,
        }


def _well_calibrated_predictions() -> tuple[list[float], list[bool]]:
    confidences: list[float] = []
    correct: list[bool] = []
    for hits in range(5, 100, 10):
        confidence = hits / 100.0
        confidences.extend([confidence] * 100)
        correct.extend([True] * hits)
        correct.extend([False] * (100 - hits))
    return confidences, correct


def _demo_reports() -> tuple[ModelReport, ...]:
    held_out = PerplexityResult.from_token_nll([1.0] * 30, [7, 11, 12])
    well_confidences, well_correct = _well_calibrated_predictions()
    return tuple(
        ModelReport(
            model=name,
            calibration=CalibrationReport.from_predictions(confidences, correct),
            perplexity=held_out,
        )
        for name, confidences, correct in (
            ("well_calibrated", well_confidences, well_correct),
            ("overconfident", [0.9] * 100, [True] * 60 + [False] * 40),
            ("underconfident", [0.6] * 100, [True] * 90 + [False] * 10),
        )
    )


def _print_demo(reports: Sequence[ModelReport]) -> None:
    for report in reports:
        calibration = report.calibration
        if calibration is None:
            raise RuntimeError("demo reports must include calibration data")
        terms = calibration.decomposition
        print(
            f"model={report.model} n={calibration.n_samples} "
            f"ece={calibration.ece:.6f} brier={calibration.brier:.6f} "
            f"populated_bins={calibration.populated_bins} "
            f"reliability={terms.reliability:.6f} "
            f"resolution={terms.resolution:.6f} "
            f"uncertainty={terms.uncertainty:.6f}"
        )
    if not reports or reports[0].perplexity is None:
        raise RuntimeError("demo reports must include a held-out perplexity")
    held_out = reports[0].perplexity
    print(
        f"perplexity corpus=held_out_synthetic tokens={held_out.total_tokens} "
        f"mean_nll={held_out.mean_nll:.6f} ppl={held_out.perplexity:.6f}"
    )
    print("demo: ok")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic synthetic demo."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args != ["demo"]:
        print("usage: python main.py [demo]", file=sys.stderr)
        return 2
    _print_demo(_demo_reports())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
