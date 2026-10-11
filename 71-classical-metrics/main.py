"""Five classical text metrics implemented from first principles.

Every metric consumes the same lowercase word-run tokenizer. The dispatcher is
the only metric-name branch in the module, so a runner can stay metric-agnostic.
"""

from __future__ import annotations

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
