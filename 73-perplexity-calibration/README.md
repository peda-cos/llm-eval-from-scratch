# Perplexity and Calibration

A model report needs two independent answers: does the model find held-out text plausible,
and do its stated probabilities match reality? This module implements both surfaces for an
LLM eval harness.

- **Perplexity** aggregates adapter-supplied per-token negative log-likelihoods over a
  held-out corpus. It is the exponentiated average NLL per token; lower is better.
- **Calibration** pools `(confidence, correct)` pairs into expected calibration error (ECE),
  the Brier score and its reliability/resolution/uncertainty decomposition, and the arrays
  needed for a reliability diagram.
- **ModelReport** attaches `perplexity`, `ece`, and `brier` to one serializable report.

The adapter owns tokenization, context windows, softmax, and confidence extraction. This
module validates and aggregates numbers; it does not call a model, plot a diagram, or fit a
calibration curve.

## Quickstart

Prerequisites: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). From a fresh clone:

```bash
cd llm-eval-from-scratch/73-perplexity-calibration
./verify.sh
```

`verify.sh` installs the pinned toolchain into the ignored `.venv`, runs the tests and strict
mypy, then runs the deterministic demo. It is safe to run repeatedly. The first run may
download pinned packages; the tests and demo themselves perform no network I/O.

Manual commands:

```bash
uv sync --frozen
uv run python main.py demo
uv run pytest -q
uv run mypy main.py
```

Minimal usage:

```python
from main import CalibrationReport, ModelReport, PerplexityResult

held_out = PerplexityResult.from_token_nll([1.0] * 30, [7, 11, 12])
calibration = CalibrationReport.from_predictions(
    [0.9] * 100,
    [True] * 60 + [False] * 40,
    bins=10,
)
report = ModelReport("model-a", calibration, held_out)

print(report.metrics())   # {'ece': 0.3, 'brier': 0.33, 'perplexity': 2.718...}
print(report.to_dict())   # JSON-safe nested report
```

## Architecture

```text
held-out token nll (nats) + per-sequence token counts
                          |
                          v
                 PerplexityResult.from_token_nll
                          |
                          +--> perplexity = exp(total_nll / total_tokens)

(confidence, correct) pairs pooled over the eval
                          |
                          v
             CalibrationReport.from_predictions
                          |
                          +--> ECE + populated_bins
                          +--> Brier score + decomposition
                          +--> (mean confidence, mean accuracy, counts) per bin
                          |
                          v
                    ModelReport
                          |
                          +--> metrics(): perplexity, ece, brier
                          +--> to_dict(): JSON-safe report
```

| Path | Role |
|---|---|
| `main.py` | Validation, perplexity, calibration metrics, reports, demo CLI. |
| `tests/test_perplexity.py` | NLL/count contract, zero tokens, uniform model, adapter mistakes. |
| `tests/test_calibration.py` | ECE edge cases, bin sensitivity, Brier decomposition, reliability bins. |
| `tests/test_reports.py` | Report/primitive agreement, metrics keys, JSON serialization. |
| `tests/test_cli.py` | Deterministic demo output and usage errors. |
| `verify.sh` | Frozen install, 27 tests, strict mypy, demo gate. |
| `pyproject.toml`, `uv.lock` | Pinned runtime and dev toolchain. NumPy is the only runtime dependency. |

## API

### Perplexity

```python
PerplexityResult.from_token_nll(neg_log_probs, token_counts)
perplexity(neg_log_probs, token_counts)
```

`neg_log_probs` is one NLL value per scored token, in nats. `token_counts` is per sequence;
the values must sum to the number of NLL values. The result exposes `perplexity`, `mean_nll`,
`total_nll`, and `total_tokens`.

```text
perplexity = exp( sum(nll_i) / sum(token_counts_j) )
```

Validation rejects non-finite or negative NLL, non-integer or negative token counts, shape
errors, and count/NLL length disagreements. A negative NLL means the adapter returned `log p`
instead of `-log p`; accepting it would produce an impossible perplexity below one.

A corpus with zero scored tokens returns `nan` for perplexity and mean NLL rather than
raising or emitting a warning. A uniform model over a vocabulary of size `V` has expected
perplexity `V`; a model that assigns probability one to every actual token has perplexity
one.

### ECE

```python
expected_calibration_error(confidences, correct, bins=10)
CalibrationReport.from_predictions(confidences, correct, bins=10).populated_bins
```

Confidences must be finite and in `[0, 1]`; correctness must be exactly `0` or `1` per
prediction. Bins are equal width on `[0, 1]`. Confidence `p` lands in
`min(int(p * bins), bins - 1)`, so `p = 1.0` stays in the last bin.

```text
ECE = sum over populated bins of (count_b / N) * |accuracy_b - confidence_b|
```

Empty input returns `0.0`. The report also returns the number of populated bins, so a runner
can refuse to publish ECE from a sample that occupies too few bins. This module does not
implement adaptive or sweep ECE; the returned reliability arrays are the extension point.

### Brier score and decomposition

```python
brier_score(confidences, correct)
brier_decomposition(confidences, correct)
```

```text
Brier = mean((confidence_i - correct_i)^2)
```

Brier is a strictly proper scoring rule for binary outcomes. ECE averages gaps inside bins,
so overconfidence and underconfidence within one bin can cancel to zero. The Brier score
penalizes each prediction separately. The test `[0.5, 0.5, 0.5, 0.5]` with outcomes
`[1, 0, 1, 0]` has ECE `0.0` but Brier `0.25`.

The decomposition is:

```text
Brier = reliability - resolution + uncertainty

reliability = sum_k (n_k / N) * (forecast_k - accuracy_k)^2
resolution  = sum_k (n_k / N) * (accuracy_k - base_rate)^2
uncertainty = base_rate * (1 - base_rate)
```

Groups share an **exact confidence value**, not an equal-width ECE bin. Grouping by a bin's
mean confidence would omit within-bin confidence variance and break the identity. Empty input
returns four zeros.

### Reliability diagram data

```python
mean_confidence, mean_accuracy, counts = reliability_diagram(confidences, correct, bins=10)
```

The arrays have length `bins` in bin order. Unpopulated bins return confidence `0.0`, accuracy
`0.0`, and count `0`; plotting code should mask on `counts > 0`. Empty input returns three
zero-filled arrays of the requested length. A plotting layer can draw `mean_confidence`
against `mean_accuracy`, with the diagonal as perfect calibration.

### Reports

`CalibrationReport.from_predictions` computes all calibration fields once from the pooled
predictions. `ModelReport.metrics()` returns exactly the available keys among `perplexity`,
`ece`, and `brier`. `to_dict()` converts every NumPy scalar and array to plain Python values;
`json.dumps` accepts it. A zero-token perplexity is serialized as `NaN`, so JSON consumers
must treat non-finite numbers as unavailable.

## Design decisions

- **NumPy arrays leave the module.** The spec calls for arrays, and downstream plotting and
  custom ECE variants should not have to convert Python lists.
- **ECE is equal-width and fixed by a `bins` parameter.** This is the standard reporting
  convention. Adaptive binning is deliberately left to the caller.
- **Decomposition groups exact confidence values.** This keeps the additive Brier identity
  exact. The ECE bins and the decomposition answer different questions.
- **Populated-bin count is part of the report.** ECE is biased by bin count and sample size;
  a scalar alone hides that. The count is a guard, not a confidence interval.
- **No softmax, confidence estimation, temperature scaling, or plotting.** Those belong to the
  adapter or a later calibration-fitting layer. The harness consumes numbers.
- **No sibling imports.** The module is independently runnable and duplicates the small
  validation helpers it needs.

## What runs offline

Everything in `verify.sh` after dependency installation runs offline and deterministically:
27 tests, strict mypy on `main.py`, and the synthetic demo. There is no model, dataset
download, network call, RNG, or sleep. `verify.sh` installs the pinned dependencies into the
ignored `.venv`; that install step is the only network access.

## Optional real-model adapter / GPU mode

The module is provider-neutral. A real adapter can run on GPU outside this project and feed
the validated functions:

1. Tokenize held-out text with the model's own tokenizer.
2. Run the model with its real context window. For fixed-length models, use a strided sliding
   window so most tokens are scored with context rather than in isolated chunks; the
   [Hugging Face perplexity guide](https://huggingface.co/docs/transformers/en/perplexity)
   shows the window/label-masking pattern.
3. Collect one NLL value per scored token in nats and one token count per sequence, then call
   `PerplexityResult.from_token_nll`.
4. For each eval prediction, compute confidence in the adapter: a softmax over option
   log-likelihoods for multiple choice, or a self-reported probability / exponential of mean
   log-probability for free text. Pass `(confidence, correct)` pairs to
   `CalibrationReport.from_predictions`.

Perplexity numbers are only comparable when tokenizer, context strategy, and corpus match.
This module does not add a live provider, GPU dependency, or network test; the adapter
contract above is the integration point.

## Measured demo results

Measured locally with `uv run python main.py demo` on the synthetic fixture in `main.py`.
These are deterministic fixture values, **not benchmark results** and not claims about any
model.

| Predictor | N | ECE | Brier | Populated bins | Reliability | Resolution | Uncertainty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Well calibrated | 1000 | 0.000000 | 0.167500 | 10 | 0.000000 | 0.082500 | 0.250000 |
| Overconfident | 100 | 0.300000 | 0.330000 | 1 | 0.090000 | 0.000000 | 0.240000 |
| Underconfident | 100 | 0.300000 | 0.180000 | 1 | 0.090000 | 0.000000 | 0.090000 |

Held-out synthetic corpus: 30 tokens, mean NLL `1.000000`, perplexity `2.718282`.

Both biased predictors report ECE `0.3`, but Brier separates them: the overconfident model
scores `0.33` and the underconfident model `0.18`. The well-calibrated fixture occupies every
bin and has zero reliability error. `verify.sh` currently reports 27 passed tests plus a clean
mypy run.

## Limitations

- ECE is sensitive to the number and placement of bins, and is biased by sample size. Nixon
  et al. show that calibration rankings can change with bin count and recommend adaptive
  binning; this module stays equal-width for the spec's reporting convention and exposes the
  raw arrays for other variants. `populated_bins` is not a confidence interval.
- Brier is implemented for binary outcomes only. Multi-class scoring needs a per-class or
  one-vs-rest formulation outside this API.
- Calibration is pooled per model; it does not stratify by task, class, or slice.
- There is no post-hoc calibration fitting (temperature scaling or Platt scaling). The report
  measures the model as supplied.
- Perplexity is only as trustworthy as the adapter's tokenization, context handling, and
  `-log p` conversion. Token-level NLL must be in nats.
- The demo fixture is synthetic. No WikiText or model perplexity figures are reported here
  because none were measured.

## References

1. AI Engineering from Scratch, "Perplexity and Calibration" capstone spec.
   <https://aiengineeringfromscratch.com/learn/capstone-projects/perplexity-calibration>
2. Jeremy Nixon, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel, Dustin Tran,
   "Measuring Calibration in Deep Learning", CVPR Workshops 2019.
   <https://arxiv.org/abs/1904.01685>
3. Allan H. Murphy, "A New Vector Partition of the Probability Score", *Journal of Applied
   Meteorology* 12(4), 1973.
   <https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2>
4. Glenn W. Brier, "Verification of Forecasts Expressed in Terms of Probability", *Monthly
   Weather Review* 78(1), 1950.
   <https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2>
5. Hugging Face Transformers, "Perplexity of fixed-length models".
   <https://huggingface.co/docs/transformers/en/perplexity>
