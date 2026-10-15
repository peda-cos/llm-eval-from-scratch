"""Perplexity and calibration metrics for an LLM eval harness.

Perplexity is the exponentiated average negative log-likelihood per token.
Calibration is measured with ECE, the Brier score, its reliability /
resolution / uncertainty decomposition, and reliability-diagram bin data.
The adapter supplies every log-probability and confidence; this module only
validates and aggregates the numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


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
