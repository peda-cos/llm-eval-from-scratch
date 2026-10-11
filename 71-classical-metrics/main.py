"""Five classical text metrics implemented from first principles.

Every metric consumes the same lowercase word-run tokenizer. The dispatcher is
the only metric-name branch in the module, so a runner can stay metric-agnostic.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from typing import Final, Sequence, TypedDict

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


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Return the longest common subsequence length via dynamic programming."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(prediction: str, reference: str) -> float:
    """Token-level ROUGE-L using the symmetric LCS F1 (beta = 1)."""
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    lcs = lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score(metric_name: str, prediction: str, targets: Sequence[str]) -> float:
    """Dispatch one prediction to one metric and return a float in [0, 1]."""
    if not targets:
        raise ValueError("targets must not be empty")
    if metric_name == "exact_match":
        return exact_match(prediction, targets)
    if metric_name == "accuracy":
        return accuracy(prediction, targets)
    if metric_name == "f1":
        return max(f1_score(prediction, target) for target in targets)
    if metric_name == "bleu_4":
        return max(bleu4(prediction, target) for target in targets)
    if metric_name == "rouge_l":
        return max(rouge_l(prediction, target) for target in targets)
    raise ValueError(f"unknown metric_name: {metric_name}")


class ReferenceExample(TypedDict):
    metric_name: str
    prediction: str
    targets: list[str]
    expected: float
    note: str


_reference_examples: Final[tuple[ReferenceExample, ...]] = (
    {
        "metric_name": "exact_match",
        "prediction": " 41 ",
        "targets": ["41"],
        "expected": 1.0,
        "note": "strip only; no lowercasing",
    },
    {
        "metric_name": "exact_match",
        "prediction": "forty one",
        "targets": ["41"],
        "expected": 0.0,
        "note": "exact string comparison is format-sensitive",
    },
    {
        "metric_name": "accuracy",
        "prediction": "positive",
        "targets": ["negative", "positive"],
        "expected": 1.0,
        "note": "accuracy is exact match over targets",
    },
    {
        "metric_name": "f1",
        "prediction": "the cat sat",
        "targets": ["the cat is sat"],
        "expected": 0.8571428571428571,
        "note": "3/4 recall, 1 precision, F1 = 6/7",
    },
    {
        "metric_name": "bleu_4",
        "prediction": "the cat sat on the mat",
        "targets": ["the cat sat on the mat"],
        "expected": 1.0,
        "note": "identical tokens",
    },
    {
        "metric_name": "bleu_4",
        "prediction": "the cat sat",
        "targets": ["the cat sat on the mat"],
        "expected": 0.36787944117144233,
        "note": "all precisions 1; brevity penalty = exp(-1)",
    },
    {
        "metric_name": "bleu_4",
        "prediction": "the cat sat on the mat",
        "targets": ["the cat sat on a mat"],
        "expected": 0.6434588841607617,
        "note": "fourth root of 6/35",
    },
    {
        "metric_name": "rouge_l",
        "prediction": "the quick brown fox jumps",
        "targets": ["the quick fox jumps over the dog"],
        "expected": 0.6666666666666666,
        "note": "LCS 4; precision 4/5, recall 4/7",
    },
)


def _run_demo() -> None:
    for example in _reference_examples:
        actual = score(
            example["metric_name"],
            example["prediction"],
            example["targets"],
        )
        expected = example["expected"]
        if not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(
                f"reference example failed: {example['metric_name']} "
                f"scored {actual!r}, expected {expected!r}"
            )
        print(
            f"{example['metric_name']:<11} "
            f"score={actual:.6f} expected={expected:.6f} ok  "
            f"{example['note']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reference demo; only the optional `demo` argument is accepted."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args != ["demo"]:
        print("usage: python main.py [demo]", file=sys.stderr)
        return 2
    _run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
