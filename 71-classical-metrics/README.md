# Classical Metrics

Five metrics that still account for most published LLM evaluation numbers, implemented from
first principles: exact match, token-level F1, sentence-level BLEU-4, ROUGE-L, and accuracy.
Every metric uses one fixed tokenizer and one dispatcher, so a runner only passes a
`metric_name` and never branches on metric semantics.

This module is part of the `llm-eval-from-scratch` track but has no imports from sibling
modules. It does not call a model, download a dataset, execute generated code, or compute
confidence intervals.

## Why reimplement

BLEU 28.3 and BLEU 0.283 may be the same system measured under different tokenizers,
normalization rules, or smoothing. ROUGE-L implementations also diverge on casing,
tokenization, and multi-reference aggregation. Rewriting the metrics makes those choices
visible in code: `TOKEN_RE` fixes the tokenizer, `bleu4` owns clipping and smoothing, and
`rouge_l` owns the LCS recurrence.

The module implements the capstone's stated formulas, not another library's interpretation.
That is the point: scores are reproducible here and are not claimed to be directly comparable
with NLTK, sacreBLEU, or ROUGE package scores.

## Quickstart

Prerequisites: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). From a fresh clone:

```bash
cd llm-eval-from-scratch/71-classical-metrics
./verify.sh
```

`verify.sh` installs the pinned dev toolchain into the ignored `.venv`, runs the test suite,
runs strict mypy, and executes the eight-example demo. It is safe to run repeatedly. The first
run may download the pinned dev dependencies; tests and the demo perform no network I/O.

Manual commands:

```bash
uv sync --frozen
uv run python main.py demo
uv run pytest -q
uv run mypy main.py
```

## Architecture

```text
prediction + targets
        |
        v
score(metric_name, prediction, targets)
        |
        +-- exact_match --> strip equality over targets
        +-- accuracy    --> strip equality over targets
        +-- f1          --> tokenize -> multiset intersection -> best target
        +-- bleu_4      --> tokenize -> clipped n-grams -> smoothing + BP -> best target
        +-- rouge_l     --> tokenize -> LCS DP -> precision/recall F1 -> best target
        |
        v
float in [0, 1]
```

| Path | Role |
|---|---|
| `main.py` | Fixed tokenizer, five metrics, dispatcher, eight reference examples, demo CLI. |
| `tests/test_metrics.py` | 49 collected cases: worked vectors, edge cases, dispatch errors, demo CLI. |
| `verify.sh` | Frozen install, pytest, strict mypy, demo smoke test. |
| `pyproject.toml`, `uv.lock` | Pinned dev toolchain; no runtime dependencies. |
| `README.md`, `LICENSE` | Documentation and MIT license. |

## Tokenizer contract

Every metric except exact match and accuracy uses exactly this tokenizer:

```python
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
```

Lowercase, keep runs of Unicode word characters (letters, digits, underscore), discard
everything else. Thus `"The cat's 42_fish, café!"` becomes
`["the", "cat", "s", "42_fish", "café"]`. The tokenizer is a contract, not a per-task knob.

## Metrics

### Exact match and accuracy

`exact_match(prediction, targets)` returns `1.0` when the stripped prediction equals any
stripped target, else `0.0`. It does not lowercase, remove punctuation, or tokenize;
normalization belongs in the upstream post-process rules from lesson 70. `accuracy` is a
separate dispatch symbol that delegates to exact match. Dataset-level exact match or accuracy
is the mean of these per-task values.

### Token-level F1

Tokenize the prediction and target, then take the multiset intersection:

```text
precision = |pred multiset intersect target multiset| / |pred|
recall    = |pred multiset intersect target multiset| / |target|
F1        = 2 * precision * recall / (precision + recall)
```

Multiset counts prevent one repeated token from matching an unlimited number of target
tokens. Both-empty is `1.0`; exactly one empty or no shared token is `0.0`. For multi-target
tasks the dispatcher takes the best F1 over targets, following the SQuAD convention.

### BLEU-4

For each candidate-reference pair:

1. Count candidate and reference n-grams for `n = 1, 2, 3, 4`.
2. Clip every candidate count by the maximum count of that n-gram in the reference.
3. Compute smoothed modified precision `p_n = (clipped + 1) / (total + 1)` for every order.
4. Return `BP * (p_1 * p_2 * p_3 * p_4) ** (1/4)`, where
   `BP = 1` if candidate length is at least reference length, else
   `exp(1 - reference_length / candidate_length)`.

The add-one rule is applied to unigrams too, exactly as the capstone states. This deliberately
differs from libraries that smooth only `n >= 2`; it also means BLEU+1 can award a positive
score to a disjoint pair. The dispatcher takes the best pairwise score over targets. This is
sentence-level BLEU, not corpus BLEU. Any empty candidate or reference returns `0.0`.

### ROUGE-L

Tokenize both texts, compute the longest common subsequence (LCS) length with a standard
dynamic-programming table, then combine precision and recall with `beta = 1`:

```text
recall    = LCS / |reference tokens|
precision = LCS / |candidate tokens|
ROUGE-L   = 2 * precision * recall / (precision + recall)
```

Both-empty is `1.0`; exactly one empty or an LCS of zero is `0.0`. This is the symmetric F1
form, not the recall-heavy configuration used in some original DUC evaluations.

## Dispatch contract

```python
score(metric_name, prediction, targets) -> float
```

| `metric_name` | Behavior |
|---|---|
| `exact_match` | Exact match over targets. |
| `accuracy` | Alias with exact-match semantics. |
| `f1` | Best token F1 over targets. |
| `bleu_4` | Best sentence BLEU-4 over targets. |
| `rouge_l` | Best ROUGE-L over targets. |

`code_exec` is intentionally unknown here and raises `ValueError`; lesson 72 owns that
metric. Unknown names raise `ValueError(f"unknown metric_name: {metric_name}")`. An empty
`targets` sequence raises `ValueError("targets must not be empty")`. Every successful call
returns a float in `[0, 1]`.

## Reference examples and demo

The eight hand-derived vectors live in `_reference_examples` at the bottom of `main.py`.
`python main.py demo` dispatches each one, asserts it against the pinned value, and prints it.

| # | Metric | Prediction | Target | Score |
|---|---|---|---|---|
| 1 | `exact_match` | `" 41 "` | `"41"` | 1.000000 |
| 2 | `exact_match` | `"forty one"` | `"41"` | 0.000000 |
| 3 | `accuracy` | `"positive"` | `"negative"`, `"positive"` | 1.000000 |
| 4 | `f1` | `"the cat sat"` | `"the cat is sat"` | 0.857143 (6/7) |
| 5 | `bleu_4` | `"the cat sat on the mat"` | same | 1.000000 |
| 6 | `bleu_4` | `"the cat sat"` | `"the cat sat on the mat"` | 0.367879 (`exp(-1)`) |
| 7 | `bleu_4` | `"the cat sat on the mat"` | `"the cat sat on a mat"` | 0.643459 (`(6/35)^(1/4)`) |
| 8 | `rouge_l` | `"the quick brown fox jumps"` | `"the quick fox jumps over the dog"` | 0.666667 (2/3) |

## Offline boundary and real-provider/GPU mode

After the one-time dev dependency install, everything here runs offline: tokenization,
metrics, tests, type checking, and the demo. There is no model call, paid API, GPU kernel,
dataset download, or cloud sandbox. No external service is simulated because no external
service is part of the metric computation.

A caller can pass generations from any real or stub provider into `score` upstream; that
provider adapter is outside this module. `score` is a pure function of its three arguments.

## Design decisions

- **Stdlib only.** The LCS table is a pure Python list of lists; numpy is unnecessary and not
  declared. This keeps runtime dependencies at zero.
- **One tokenizer.** `TOKEN_RE` is not configurable. Changing lowercase or punctuation
  handling would change every metric and invalidate the reference vectors.
- **Exact match is strip-only.** Case folding and punctuation normalization belong to the
  post-process rules, so this module does not silently apply a second normalization layer.
- **Multiset F1.** Token counts matter; repeated candidate tokens cannot all match one target
  occurrence.
- **Best of targets.** F1, BLEU-4, and ROUGE-L use the maximum pairwise score; a task with
  several acceptable answers is not penalized for matching only one.
- **Capstone smoothing.** Add one to numerator and denominator for all four orders, including
  unigrams. This is not NLTK/sacreBLEU parity and is not meant to be.
- **Symmetric ROUGE-L.** `beta = 1` makes precision and recall equally weighted.
- **Explicit edge conventions.** Both-empty is perfect for exact match, F1, and ROUGE-L;
  BLEU-4 returns zero whenever either side is empty.
- **Hand-derived vectors.** Expected values come from the worked arithmetic in the test
  comments and reference block, not from a third-party metric library.
- **No `code_exec`.** The dispatcher accepts exactly the five classical metric names; code
  execution scoring is a later lesson.

## Measured results

Measured locally with `./verify.sh`:

- **49 pytest cases passed.**
- **mypy strict:** `Success: no issues found in 1 source file`.
- **Demo:** eight reference examples printed with scores
  `1.000000, 0.000000, 1.000000, 0.857143, 1.000000, 0.367879, 0.643459, 0.666667`.
- **CLI error path:** an unknown command prints usage and exits 2.

No model accuracy, BLEU benchmark, or dataset-level score is reported because this module
does not call a model or evaluate a corpus. The numbers above are the pinned local vectors.

## Limitations

- The tokenizer is a deliberate simplification: CJK segmentation, contractions, and code
  identifiers are not handled the way production metrics often handle them.
- Sentence-level BLEU-4 is not corpus BLEU. Averaging sentence scores is not the same as
  corpus-level micro-averaging.
- Add-one smoothing gives positive scores to disjoint sentences and is known to favor shorter
  outputs; the brevity penalty mitigates but does not remove that bias.
- Best-of-targets aggregation can inflate scores when many references are available, and it
  differs from combining all references into one BLEU or ROUGE computation.
- Exact match and accuracy are strictly string-based after stripping; case or punctuation
  mismatches must be normalized upstream.
- F1, BLEU, and ROUGE-L reward surface overlap only. They do not capture entailment,
  correctness, or meaning, and they use no stemming or synonyms.
- LCS uses `O(n*m)` time and memory; very long documents should use a streaming or
  memory-reduced variant.
- No corpus aggregation, confidence intervals, BLEURT, BERTScore, or code execution.

## References

Sources consulted for the formulas, smoothing caveat, tokenizer/reporting discipline, and
SQuAD-style best-of-targets convention. No code was copied from them.

1. AI Engineering From Scratch, Classical Metrics capstone:
   <https://aiengineeringfromscratch.com/learn/capstone-projects/classical-metrics>
2. Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). *BLEU: a Method for Automatic
   Evaluation of Machine Translation.* ACL 2002.
   <https://aclanthology.org/P02-1040/>
3. Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* Text
   Summarization Branches Out (ACL workshop).
   <https://aclanthology.org/W04-1013/>
4. Chen, B., and Cherry, C. (2014). *A Systematic Comparison of Smoothing Techniques for
   Sentence-Level BLEU.* WMT 2014.
   <https://aclanthology.org/W14-3346/>
5. Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.* WMT 2018.
   <https://aclanthology.org/W18-6319/>
