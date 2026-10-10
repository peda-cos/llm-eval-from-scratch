# Task Spec Format

A frozen JSONL task contract and a strict validator for an LLM eval harness. A task is one
JSON object per line; the validator decides whether it is legal before any runner, model, or
metric sees it. This module does not score, call a model, or execute code.

## Why a frozen spec

Without one schema, every eval notebook grows its own JSON shape and every metric gets
reimplemented. This module fixes the shape, the category vocabulary, the metric names, the
few-shot rendering path, and the post-processing rules. Unknown fields fail. A bad line is
reported and skipped; it never aborts the rest of the file.

## Quickstart

From a checkout of the track repository:

```bash
cd llm-eval-from-scratch/70-task-spec-format
./verify.sh
```

`verify.sh` installs the pinned dev toolchain into the ignored `.venv`, runs the test suite and
`mypy`, then smoke-tests the CLI. It is safe to run repeatedly. The first run may download the
pinned dev dependencies; tests themselves never touch the network.

Manual commands:

```bash
uv sync --frozen
uv run python main.py demo
uv run python main.py validate fixtures/tasks.jsonl
uv run pytest -q
```

Expected fixture results: the valid file prints `10 valid, 0 invalid`; the deliberately broken
file prints `1 valid, 16 invalid` and exits 1. Add `--allow-bad-tasks` to make the gate exit 0
while still reporting every error.

## Architecture

```text
fixtures/tasks.jsonl
        |
        v
  validate_file  ---->  (validated tasks, TaskError list)
        |                              |
        v                              v
  load_fixtures                 CLI gate / demo
        |
        v
  render(task) -> provider (later lesson) -> post_process(text, rule) -> metric (later lesson)
```

| Path | Role |
|---|---|
| `main.py` | Vocabulary, `TaskSpec`, validator, loader, renderer, post-process rules, CLI. |
| `fixtures/tasks.jsonl` | 10 valid tasks: two per category, covering every metric and post-process rule. |
| `fixtures/tasks_bad.jsonl` | 17 lines: one valid seed and exactly 16 single-rule failures. |
| `tests/test_spec.py` | Validation rules, category matrix, fixture counts, duplicate and malformed-line behavior. |
| `tests/test_pipeline.py` | Few-shot rendering and all post-process semantics. |
| `tests/test_cli.py` | Exit codes and the `--allow-bad-tasks` gate. |
| `verify.sh` | Frozen install, pytest, strict mypy, CLI smoke checks. |
| `pyproject.toml`, `uv.lock` | Pinned dev toolchain; no runtime dependencies. |

## Record contract

A task is one JSON object on one line. Required fields:

| Field | Type | Rule |
|---|---|---|
| `task_id` | string | Non-empty, no whitespace, unique across the file. |
| `category` | string | One of `arithmetic`, `mcq`, `code_exec`, `classification`, `summary`. |
| `prompt` | string | Non-empty, no trailing whitespace on any line, no blank line. |
| `targets` | list of strings | Non-empty. |
| `metric_name` | string | One of the closed metric vocabulary below. |
| `post_process` | string | Exactly one closed post-process rule below. |

Optional fields:

| Field | Type | Rule |
|---|---|---|
| `few_shot_examples` | list of `{prompt, completion}` objects | At most 8 entries; both values non-empty strings; no extra keys. |
| `metadata` | object | Any keys and values; omitted means absent, not `null`. |

Unknown top-level fields fail validation. A record must be a JSON object, not an array, scalar,
or `null`.

Example line (the `\n` sequences are JSON escapes, not literal line breaks):

```json
{"task_id":"arith_001","category":"arithmetic","prompt":"Compute the result.\nQuestion: 17 + 24\nAnswer:","targets":["41"],"metric_name":"exact_match","few_shot_examples":[{"prompt":"Question: 2 + 2\nAnswer:","completion":"4"}],"post_process":"strip_whitespace","metadata":{"difficulty":"easy"}}
```

## Closed vocabularies

Categories:

```text
arithmetic  mcq  code_exec  classification  summary
```

Metrics:

```text
exact_match  f1  bleu_4  rouge_l  accuracy  code_exec
```

Post-process rules:

```text
none  strip_whitespace  lower  extract_letter  extract_code_block  extract_first_line
```

The category constrains the legal metric and post-process pair:

| Category | Allowed metrics | Allowed post-process |
|---|---|---|
| `arithmetic` | `exact_match` | `none`, `strip_whitespace` |
| `mcq` | `exact_match` | `extract_letter` |
| `code_exec` | `code_exec` | `extract_code_block` |
| `classification` | `accuracy`, `f1` | `none`, `lower`, `strip_whitespace` |
| `summary` | `rouge_l`, `bleu_4` | `none`, `strip_whitespace`, `extract_first_line` |

An `mcq` task must have exactly one target and it must be one of `A`, `B`, `C`, `D`, `E`.
A `code_exec` task must use `code_exec`; a summary task cannot use `code_exec`; and so on.

`metric_name` is a dispatch key only. This lesson does not implement the metric semantics:
later lessons consume the name to dispatch exact match, F1, BLEU-4, ROUGE-L, aggregate
accuracy, or a code-execution judge. The `targets` list is opaque to this module; the
`code_exec` fixture uses Python `assert` expressions as the convention a later runner will
consume, but the validator only requires non-empty strings.

## Validator behavior

`validate_task(record)` collects every record-local error and returns `TaskError` values with
`line`, `rule`, `field`, and `message`. Errors are ordered deterministically: unknown fields,
missing required fields, field-local rules, then cross-field rules.

`validate_file(path)` reads a JSONL file line by line as strict UTF-8, parses each line
independently, adds the 1-based line number to errors, and enforces `task_id` uniqueness after a
record passes its local rules. A malformed line, invalid UTF-8 line, non-object JSON value, or
duplicate id becomes an error for that line; the rest of the file continues. Blank lines are
parse errors, as required by JSON Lines.

Rule ids:

| Rule | Field | Trigger |
|---|---|---|
| `record_not_object` | `$` | Parsed JSON value is not an object. |
| `parse_error` | `$` | Invalid UTF-8 or invalid JSON on a line. |
| `unknown_field` | offending key | Top-level key outside the schema. |
| `missing_field` | missing key | Required field absent. |
| `invalid_task_id` | `task_id` | Empty, non-string, or contains whitespace. |
| `duplicate_task_id` | `task_id` | Id already validated on an earlier line. |
| `invalid_category` | `category` | Not in the category vocabulary. |
| `invalid_prompt` | `prompt` | Empty, non-string, trailing whitespace, or a blank line. |
| `invalid_targets` | `targets` | Not a non-empty list of strings. |
| `invalid_metric` | `metric_name` | Not in the metric vocabulary. |
| `invalid_post_process` | `post_process` | Not exactly one rule in the vocabulary. |
| `invalid_few_shot_examples` | `few_shot_examples` or `few_shot_examples[i]` | Wrong list shape, more than 8 entries, or a bad pair. |
| `invalid_metadata` | `metadata` | Present but not an object. |
| `metric_category_mismatch` | `metric_name` | Metric not legal for the category. |
| `post_process_category_mismatch` | `post_process` | Rule not legal for the category. |
| `invalid_mcq_targets` | `targets` | MCQ target is not exactly one letter A-E. |

`load_fixtures(path)` is the strict runner entry point: it returns validated tasks or raises
`FixtureValidationError` carrying the full error list. The CLI is the gate:

```bash
uv run python main.py validate fixtures/tasks.jsonl
uv run python main.py validate fixtures/tasks_bad.jsonl                  # exit 1
uv run python main.py validate fixtures/tasks_bad.jsonl --allow-bad-tasks # exit 0
```

Missing files exit 2. `python main.py demo` validates both bundled fixtures and prints a
summary.

## Few-shot rendering and post-processing

`render(task)` puts each example in front of the prompt, joining with a blank line:

```python
parts = [f"{example['prompt']} {example['completion']}" for example in examples]
parts.append(task["prompt"])
return "\n\n".join(parts)
```

No examples returns the prompt unchanged. The same code path runs for every model, so only the
model's generation varies.

`post_process(text, rule)` is deterministic and stateless:

| Rule | Behavior |
|---|---|
| `none` | Return the string unchanged. |
| `strip_whitespace` | Strip leading and trailing whitespace. |
| `lower` | Lowercase the string. |
| `extract_letter` | First uppercase `A`-`E` character, else `""`. |
| `extract_code_block` | Body of the first line-fenced triple-backtick block, else `""`. |
| `extract_first_line` | First non-empty line, stripped, else `""`. |

The blank-line ban in `prompt` is the deterministic marker for an already-rendered few-shot
block. It is conservative: it also rejects multi-paragraph prompts. The code block extractor is
line-based; it ignores language tags, preserves internal blank lines, and requires a closing
fence. `extract_letter` is case-sensitive because the spec's vocabulary is `[A-E]`.

## Fixtures

The 10 valid tasks:

| Task | Category | Metric | Post-process |
|---|---|---|---|
| `arith_001` | `arithmetic` | `exact_match` | `strip_whitespace` |
| `arith_002` | `arithmetic` | `exact_match` | `none` |
| `mcq_001` | `mcq` | `exact_match` | `extract_letter` |
| `mcq_002` | `mcq` | `exact_match` | `extract_letter` |
| `code_001` | `code_exec` | `code_exec` | `extract_code_block` |
| `code_002` | `code_exec` | `code_exec` | `extract_code_block` |
| `class_001` | `classification` | `accuracy` | `lower` |
| `class_002` | `classification` | `f1` | `none` |
| `summary_001` | `summary` | `rouge_l` | `extract_first_line` |
| `summary_002` | `summary` | `bleu_4` | `extract_first_line` |

The broken fixture has one valid seed and one line for each failure: duplicate id, malformed
JSON, non-object record, missing field, unknown field, invalid task id, invalid category,
invalid prompt, invalid targets, invalid metric, invalid post-process, over-cap few-shot list,
invalid metadata, metric-category mismatch, post-process-category mismatch, and invalid MCQ
targets. It therefore yields exactly 16 errors and one validated task.

## Offline boundary and real-provider/GPU mode

Everything in this module runs offline after the one-time dev dependency install: validation,
rendering, post-processing, the CLI, and tests. There are no model calls, no code execution, no
paid API, no GPU, no dataset download, and no cloud sandbox. No external service is part of this
lesson, so no adapter or simulation is needed.

A downstream runner imports `load_fixtures`, `render`, and `post_process`, calls a provider of
its choice, and dispatches on `metric_name`. Swapping providers does not change the record
shape or the normalization path.

## Design decisions

- **Python only.** The track brief ranks Python first and forbids parallel ports; the
  capstone's TypeScript mention is not implemented as a second project.
- **Stdlib runtime.** Only `pytest` and `mypy` are dev dependencies, pinned in `uv.lock`.
- **Strict unknown-field rejection.** Typos and experimental fields fail instead of silently
  changing meaning.
- **Closed matrix.** Every category has an explicit metric and post-process allowlist; every
  vocabulary value is legal somewhere and exercised by the fixture or tests.
- **Collect all record-local errors.** Authors see every shape problem at once; uniqueness is
  file-level and checked only after local validation.
- **Strict loader plus gate.** `load_fixtures` raises on any invalid record; the CLI exposes
  `--allow-bad-tasks` for deliberate partial runs.
- **No version field.** The contract is frozen in code and tests. A schema change requires a
  code change, test change, and fixture change in one review.
- **No JSON Schema dependency.** Custom cross-field rules and line/field error reporting need
  code; the typed literals, category matrix, and tests are the single source of truth.

## Measured results

Measured locally with `./verify.sh`:

- **107 pytest cases passed.**
- **mypy strict:** `Success: no issues found in 1 source file`.
- **Valid fixture:** 10 tasks, 0 errors.
- **Broken fixture:** 1 task accepted, 16 errors, covering 16 distinct rule ids.
- **CLI exit codes:** valid file 0; broken file 1; broken file with `--allow-bad-tasks` 0.

No model accuracy or benchmark numbers are reported because this module does not run models or
metrics.

## Limitations

- The validator checks shape and vocabulary, not whether a target is semantically correct.
- `code_exec` targets are not executed; no scoring is implemented.
- `extract_letter` only matches uppercase `A`-`E`; prompts must ask for uppercase output.
- Prompts cannot contain blank lines, which also rejects multi-paragraph prompts.
- `metadata` values are unconstrained and not validated.
- `extract_code_block` handles line-fenced blocks, not inline triple-backtick spans.
- Only local JSONL files are loaded; there is no remote fixture support.
- There is no TypeScript port.

## Prior art

BIG-bench's JSON task documentation keeps task-level metadata and a closed metric list, and
separates multiple-choice scoring from text-to-text targets. LM Evaluation Harness separates
prompt and target configuration, centralizes `metric_list`, uses `\n\n` as its few-shot
delimiter, and records task versions in metadata. HELM argues for standardized scenarios and
multi-metric measurement; this lesson borrows the fixed-vocabulary discipline, not HELM's
per-scenario aggregation. JSON Lines requires UTF-8, one valid JSON value per line, and treats
blank lines as invalid. The field names and category matrix here are ours.

## References

1. AI Engineering From Scratch — Task Spec Format:
   https://aiengineeringfromscratch.com/learn/capstone-projects/task-spec-format
2. BIG-bench JSON task documentation:
   https://github.com/google/BIG-bench/blob/main/docs/doc.md
3. LM Evaluation Harness task configuration reference:
   https://lm-evaluation-harness.readthedocs.io/writing_tasks/task_config_reference/
4. Holistic Evaluation of Language Models (HELM):
   https://arxiv.org/abs/2211.09110
5. JSON Lines:
   https://jsonlines.org/
