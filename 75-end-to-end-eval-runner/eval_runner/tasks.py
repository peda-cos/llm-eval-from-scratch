"""Frozen JSONL task contract and strict validator for the eval runner.

This is the runner's local copy of the lesson-70 contract. The task shape is
closed: unknown fields fail, and category-specific metric/post-process pairs
are enforced before any model or metric sees a record.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, NotRequired, TypedDict, cast, get_args

Category = Literal["arithmetic", "mcq", "code_exec", "classification", "summary"]
MetricName = Literal["exact_match", "f1", "bleu_4", "rouge_l", "accuracy", "code_exec"]
PostProcessRule = Literal[
    "none",
    "strip_whitespace",
    "lower",
    "extract_letter",
    "extract_code_block",
    "extract_first_line",
]

CATEGORIES: Final[frozenset[str]] = frozenset(get_args(Category))
METRICS: Final[frozenset[str]] = frozenset(get_args(MetricName))
POST_PROCESS_RULES: Final[frozenset[str]] = frozenset(get_args(PostProcessRule))
GRADED_METRICS: Final[frozenset[str]] = frozenset({"f1", "bleu_4", "rouge_l"})
BINARY_METRICS: Final[frozenset[str]] = frozenset({"exact_match", "accuracy", "code_exec"})

MAX_FEW_SHOT_EXAMPLES: Final[int] = 8
FEW_SHOT_KEYS: Final[frozenset[str]] = frozenset({"prompt", "completion"})
MCQ_TARGET_RE: Final[re.Pattern[str]] = re.compile(r"^[A-E]$")

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "category",
    "prompt",
    "targets",
    "metric_name",
    "post_process",
)
OPTIONAL_FIELDS: Final[tuple[str, ...]] = ("few_shot_examples", "metadata")
ALLOWED_FIELDS: Final[frozenset[str]] = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)

FIXTURES_DIR: Final[Path] = Path(__file__).parent.parent / "fixtures"
DEFAULT_FIXTURE: Final[Path] = FIXTURES_DIR / "tasks.jsonl"


class CategoryRule(NamedTuple):
    metrics: frozenset[str]
    post_process: frozenset[str]


CATEGORY_RULES: Final[dict[str, CategoryRule]] = {
    "arithmetic": CategoryRule(
        metrics=frozenset({"exact_match"}),
        post_process=frozenset({"none", "strip_whitespace"}),
    ),
    "mcq": CategoryRule(
        metrics=frozenset({"exact_match"}),
        post_process=frozenset({"extract_letter"}),
    ),
    "code_exec": CategoryRule(
        metrics=frozenset({"code_exec"}),
        post_process=frozenset({"extract_code_block"}),
    ),
    "classification": CategoryRule(
        metrics=frozenset({"accuracy", "f1"}),
        post_process=frozenset({"none", "lower", "strip_whitespace"}),
    ),
    "summary": CategoryRule(
        metrics=frozenset({"rouge_l", "bleu_4"}),
        post_process=frozenset({"none", "strip_whitespace", "extract_first_line"}),
    ),
}


class FewShotExample(TypedDict):
    prompt: str
    completion: str


class TaskSpec(TypedDict):
    task_id: str
    category: Category
    prompt: str
    targets: list[str]
    metric_name: MetricName
    post_process: PostProcessRule
    few_shot_examples: NotRequired[list[FewShotExample]]
    metadata: NotRequired[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TaskError:
    line: int
    rule: str
    field: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "line": self.line,
            "rule": self.rule,
            "field": self.field,
            "message": self.message,
        }


class FixtureValidationError(ValueError):
    """One or more JSONL records violate the frozen task contract."""

    def __init__(self, path: Path, errors: list[TaskError]) -> None:
        self.path = path
        self.errors = errors
        first = errors[0]
        super().__init__(
            f"{path}: {len(errors)} invalid record(s); "
            f"first error on line {first.line}: {first.rule} ({first.field})"
        )


def _error(line: int, rule: str, field: str, message: str) -> TaskError:
    return TaskError(line=line, rule=rule, field=field, message=message)


def _choices(values: frozenset[str]) -> str:
    return ", ".join(sorted(values))


def validate_task(record: object, *, line: int = 1) -> list[TaskError]:
    """Return every record-local contract violation, never raising."""
    if not isinstance(record, dict):
        return [_error(line, "record_not_object", "$", "task record must be a JSON object")]

    errors: list[TaskError] = []

    unknown = [
        key for key in record if not isinstance(key, str) or key not in ALLOWED_FIELDS
    ]
    for key in sorted(unknown, key=str):
        errors.append(
            _error(line, "unknown_field", str(key), f"unknown top-level field {key!r}")
        )

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(
                _error(line, "missing_field", field, f"missing required field {field!r}")
            )

    task_id = record.get("task_id")
    task_id_ok = (
        isinstance(task_id, str)
        and bool(task_id)
        and not any(character.isspace() for character in task_id)
    )
    if "task_id" in record and not task_id_ok:
        errors.append(
            _error(
                line,
                "invalid_task_id",
                "task_id",
                "task_id must be a non-empty string with no whitespace",
            )
        )

    category = record.get("category")
    category_ok = isinstance(category, str) and category in CATEGORIES
    if "category" in record and not category_ok:
        errors.append(
            _error(
                line,
                "invalid_category",
                "category",
                f"category must be one of: {_choices(CATEGORIES)}",
            )
        )

    prompt = record.get("prompt")
    if "prompt" in record:
        if not isinstance(prompt, str):
            errors.append(_error(line, "invalid_prompt", "prompt", "prompt must be a string"))
        elif not prompt.strip():
            errors.append(_error(line, "invalid_prompt", "prompt", "prompt must not be empty"))
        elif prompt != prompt.rstrip() or any(
            part != part.rstrip() for part in prompt.splitlines()
        ):
            errors.append(
                _error(
                    line,
                    "invalid_prompt",
                    "prompt",
                    "prompt must not contain trailing whitespace",
                )
            )
        elif "\n\n" in prompt:
            errors.append(
                _error(
                    line,
                    "invalid_prompt",
                    "prompt",
                    "prompt must not contain a blank-line few-shot block",
                )
            )

    targets = record.get("targets")
    targets_ok = (
        isinstance(targets, list)
        and len(targets) > 0
        and all(isinstance(target, str) for target in targets)
    )
    if "targets" in record and not targets_ok:
        errors.append(
            _error(
                line,
                "invalid_targets",
                "targets",
                "targets must be a non-empty list of strings",
            )
        )

    metric = record.get("metric_name")
    metric_ok = isinstance(metric, str) and metric in METRICS
    if "metric_name" in record and not metric_ok:
        errors.append(
            _error(
                line,
                "invalid_metric",
                "metric_name",
                f"metric_name must be one of: {_choices(METRICS)}",
            )
        )

    post_rule = record.get("post_process")
    post_ok = isinstance(post_rule, str) and post_rule in POST_PROCESS_RULES
    if "post_process" in record and not post_ok:
        errors.append(
            _error(
                line,
                "invalid_post_process",
                "post_process",
                f"post_process must be one of: {_choices(POST_PROCESS_RULES)}",
            )
        )

    if "few_shot_examples" in record:
        examples = record["few_shot_examples"]
        if not isinstance(examples, list):
            errors.append(
                _error(
                    line,
                    "invalid_few_shot_examples",
                    "few_shot_examples",
                    "few_shot_examples must be a list",
                )
            )
        elif len(examples) > MAX_FEW_SHOT_EXAMPLES:
            errors.append(
                _error(
                    line,
                    "invalid_few_shot_examples",
                    "few_shot_examples",
                    f"few_shot_examples must contain at most "
                    f"{MAX_FEW_SHOT_EXAMPLES} entries; got {len(examples)}",
                )
            )
        else:
            for index, example in enumerate(examples):
                field = f"few_shot_examples[{index}]"
                if not isinstance(example, dict):
                    errors.append(
                        _error(
                            line,
                            "invalid_few_shot_examples",
                            field,
                            "few-shot example must be an object",
                        )
                    )
                    continue
                if set(example) != FEW_SHOT_KEYS:
                    errors.append(
                        _error(
                            line,
                            "invalid_few_shot_examples",
                            field,
                            "few-shot example must contain exactly prompt and completion",
                        )
                    )
                    continue
                example_prompt = example["prompt"]
                completion = example["completion"]
                if not isinstance(example_prompt, str) or not example_prompt.strip():
                    errors.append(
                        _error(
                            line,
                            "invalid_few_shot_examples",
                            field,
                            "few-shot prompt must be a non-empty string",
                        )
                    )
                    continue
                if not isinstance(completion, str) or not completion.strip():
                    errors.append(
                        _error(
                            line,
                            "invalid_few_shot_examples",
                            field,
                            "few-shot completion must be a non-empty string",
                        )
                    )

    if "metadata" in record and not isinstance(record["metadata"], dict):
        errors.append(
            _error(line, "invalid_metadata", "metadata", "metadata must be an object")
        )

    if category_ok and metric_ok and isinstance(category, str) and category in CATEGORY_RULES:
        allowed_metrics = CATEGORY_RULES[category].metrics
        if metric not in allowed_metrics:
            errors.append(
                _error(
                    line,
                    "metric_category_mismatch",
                    "metric_name",
                    f"category {category!r} does not allow metric {metric!r}; "
                    f"allowed: {_choices(allowed_metrics)}",
                )
            )

    if category_ok and post_ok and isinstance(category, str) and category in CATEGORY_RULES:
        allowed_rules = CATEGORY_RULES[category].post_process
        if post_rule not in allowed_rules:
            errors.append(
                _error(
                    line,
                    "post_process_category_mismatch",
                    "post_process",
                    f"category {category!r} does not allow post_process {post_rule!r}; "
                    f"allowed: {_choices(allowed_rules)}",
                )
            )

    if category == "mcq" and metric == "exact_match" and targets_ok and isinstance(targets, list):
        if len(targets) != 1 or not isinstance(targets[0], str) or MCQ_TARGET_RE.fullmatch(targets[0]) is None:
            errors.append(
                _error(
                    line,
                    "invalid_mcq_targets",
                    "targets",
                    "mcq targets must be exactly one letter from A to E",
                )
            )

    return errors


def validate_file(path: str | Path) -> tuple[list[TaskSpec], list[TaskError]]:
    """Validate every JSONL line independently; bad lines never abort the file."""
    fixture_path = Path(path)
    validated: list[TaskSpec] = []
    errors: list[TaskError] = []
    seen: dict[str, int] = {}

    with fixture_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(
                    _error(line_number, "parse_error", "$", f"invalid UTF-8: {exc}")
                )
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(
                    _error(
                        line_number,
                        "parse_error",
                        "$",
                        f"invalid JSON: {exc.msg} at column {exc.colno}",
                    )
                )
                continue

            record_errors = validate_task(record, line=line_number)
            if record_errors:
                errors.extend(record_errors)
                continue

            task = cast(TaskSpec, record)
            task_id = task["task_id"]
            if task_id in seen:
                errors.append(
                    _error(
                        line_number,
                        "duplicate_task_id",
                        "task_id",
                        f"task_id {task_id!r} already used on line {seen[task_id]}",
                    )
                )
                continue

            seen[task_id] = line_number
            validated.append(task)

    return validated, errors


def load_fixtures(path: str | Path = DEFAULT_FIXTURE) -> list[TaskSpec]:
    """Load a JSONL fixture, raising if any record is invalid."""
    fixture_path = Path(path)
    tasks, errors = validate_file(fixture_path)
    if errors:
        raise FixtureValidationError(fixture_path, errors)
    return tasks


def render(task: TaskSpec) -> str:
    """Render few-shot examples ahead of the prompt with blank-line separators."""
    examples = task["few_shot_examples"] if "few_shot_examples" in task else []
    parts = [f"{example['prompt']} {example['completion']}" for example in examples]
    parts.append(task["prompt"])
    return "\n\n".join(parts)


def post_process(text: str, rule: PostProcessRule) -> str:
    """Apply exactly one deterministic, stateless normalization rule."""
    if rule == "none":
        return text
    if rule == "strip_whitespace":
        return text.strip()
    if rule == "lower":
        return text.lower()
    if rule == "extract_letter":
        match = re.search(r"[A-E]", text)
        return match.group(0) if match is not None else ""
    if rule == "extract_code_block":
        lines = text.splitlines()
        opening: int | None = None
        for index, line in enumerate(lines):
            if line.lstrip().startswith("```"):
                opening = index
                break
        if opening is None:
            return ""
        for index in range(opening + 1, len(lines)):
            if lines[index].lstrip().startswith("```"):
                return "\n".join(lines[opening + 1 : index])
        return ""
    if rule == "extract_first_line":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""
    raise ValueError(f"unknown post_process rule: {rule!r}")
