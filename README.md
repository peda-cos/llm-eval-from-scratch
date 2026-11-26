# LLM Evaluation From Scratch

A collection of 6 tested projects built from scratch, one module per capstone lesson.

Each module has its own `README.md`, `verify.sh`, tests, and a deterministic offline demo.

## Modules

| # | Module | What it demonstrates | Verify |
|---:|---|---|---|
| 70 | [Task Spec Format](70-task-spec-format/) | An eval harness is only as good as the contract its tasks honour. Freeze the JSONL shape and the metric vocabulary before you write a single scoring function. | `cd 70-task-spec-format && ./verify.sh` |
| 71 | [Classical Metrics](71-classical-metrics/) | BLEU, ROUGE-L, F1, exact-match, accuracy. Five metrics that still account for most published LLM eval numbers. Implement each from first principles so you know what the number means. | `cd 71-classical-metrics && ./verify.sh` |
| 72 | [Code Exec Metric](72-code-exec-metric/) | Generated code is right when it passes the tests. The eval harness has to extract code, run it without crashing the host, and tally pass-rates honestly. This lesson builds that surface. | `cd 72-code-exec-metric && ./verify.sh` |
| 73 | [Perplexity and Calibration](73-perplexity-calibration/) | If your model says 90 percent confident on a thousand answers and gets six hundred right, it is not well calibrated. Calibration is half of trustworthy eval. The other half is perplexity, which tells you whether the model thinks the held-out text is plausible at all. | `cd 73-perplexity-calibration && ./verify.sh` |
| 74 | [Leaderboard Aggregation](74-leaderboard-aggregation/) | Per-task scores are easy. Per-model rankings across heterogeneous tasks are harder. Statistical significance on a thousand-prediction leaderboard is the part everyone skips. This lesson does not skip it. | `cd 74-leaderboard-aggregation && ./verify.sh` |
| 75 | [End-to-End Eval Runner](75-end-to-end-eval-runner/) | Five lessons of plumbing, one lesson to glue them. The runner reads the task spec from lesson 70, calls a model through an adapter, scores with lessons 71 and 72, attaches the calibration report from lesson 73, and emits the leaderboard from lesson 74. Demo self-terminates. | `cd 75-end-to-end-eval-runner && ./verify.sh` |

## Requirements

- Python 3.12+ and `uv` for Python modules
- Node 22+ / Bun 1.4 for TypeScript modules
- No API keys, GPU, or network needed for the test suites

## Verify all

```bash
make verify
```

## References

- Capstone source: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects
