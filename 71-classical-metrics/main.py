"""Five classical text metrics implemented from first principles.

Every metric consumes the same lowercase word-run tokenizer. The dispatcher is
the only metric-name branch in the module, so a runner can stay metric-agnostic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Final, Sequence

TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase text and return runs of Unicode word characters."""
    return TOKEN_RE.findall(text.lower())


def exact_match(prediction: str, targets: Sequence[str]) -> float:
    """Return 1.0 when the stripped prediction equals any stripped target."""
    return float(any(prediction.strip() == target.strip() for target in targets))


def accuracy(prediction: str, targets: Sequence[str]) -> float:
    """Classification accuracy is exact match over the target list."""
    return exact_match(prediction, targets)


def f1_score(prediction: str, target: str) -> float:
    """Token-level F1 using the multiset intersection of both token lists."""
    prediction_tokens = tokenize(prediction)
    target_tokens = tokenize(target)
    if not prediction_tokens or not target_tokens:
        return float(prediction_tokens == target_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(target_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _ngram_counts(
    tokens: Sequence[str], n: int
) -> Counter[tuple[str, ...]]:
    """Count each contiguous n-gram in a token sequence."""
    return Counter(
        tuple(tokens[start : start + n])
        for start in range(len(tokens) - n + 1)
    )


def bleu4(prediction: str, reference: str) -> float:
    """Sentence-level BLEU-4 with add-one smoothing and a brevity penalty."""
    candidate_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    log_precisions: list[float] = []
    for n in (1, 2, 3, 4):
        candidate_counts = _ngram_counts(candidate_tokens, n)
        reference_counts = _ngram_counts(reference_tokens, n)
        clipped = sum(
            min(count, reference_counts[gram])
            for gram, count in candidate_counts.items()
        )
        total = max(len(candidate_tokens) - n + 1, 0)
        # The capstone's Lin-Och rule adds one to every order, including n=1.
        log_precisions.append(math.log((clipped + 1) / (total + 1)))

    if len(candidate_tokens) >= len(reference_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(
            1 - len(reference_tokens) / len(candidate_tokens)
        )
    return brevity_penalty * math.exp(
        sum(log_precisions) / len(log_precisions)
    )
