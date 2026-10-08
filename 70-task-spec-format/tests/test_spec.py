import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

from main import (
    BAD_FIXTURE,
    GOOD_FIXTURE,
    METRICS,
    POST_PROCESS_RULES,
    REQUIRED_FIELDS,
    FixtureValidationError,
    load_fixtures,
    validate_file,
    validate_task,
)


def write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_valid_fixture_has_ten_tasks_and_no_errors() -> None:
    tasks, errors = validate_file(GOOD_FIXTURE)

    assert len(tasks) == 10
    assert errors == []


def test_valid_fixture_has_two_tasks_per_category() -> None:
    tasks, _ = validate_file(GOOD_FIXTURE)
    counts = Counter(task["category"] for task in tasks)

    assert counts == Counter(
        {
            "arithmetic": 2,
            "mcq": 2,
            "code_exec": 2,
            "classification": 2,
            "summary": 2,
        }
    )


def test_valid_fixture_covers_every_metric_and_post_process() -> None:
    tasks, _ = validate_file(GOOD_FIXTURE)

    assert {task["metric_name"] for task in tasks} == METRICS
    assert {task["post_process"] for task in tasks} == POST_PROCESS_RULES


def test_bad_fixture_has_exactly_one_validated_task() -> None:
    tasks, _ = validate_file(BAD_FIXTURE)

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "bad_seed"


def test_bad_fixture_returns_exactly_sixteen_errors() -> None:
    _, errors = validate_file(BAD_FIXTURE)

    assert len(errors) == 16


def test_bad_fixture_error_rules_and_fields_in_line_order() -> None:
    _, errors = validate_file(BAD_FIXTURE)

    assert [(error.rule, error.field) for error in errors] == [
        ("duplicate_task_id", "task_id"),
        ("parse_error", "$"),
        ("record_not_object", "$"),
        ("missing_field", "prompt"),
        ("unknown_field", "temperature"),
        ("invalid_task_id", "task_id"),
        ("invalid_category", "category"),
        ("invalid_prompt", "prompt"),
        ("invalid_targets", "targets"),
        ("invalid_metric", "metric_name"),
        ("invalid_post_process", "post_process"),
        ("invalid_few_shot_examples", "few_shot_examples"),
        ("invalid_metadata", "metadata"),
        ("metric_category_mismatch", "metric_name"),
        ("post_process_category_mismatch", "post_process"),
        ("invalid_mcq_targets", "targets"),
    ]
    assert [error.line for error in errors] == list(range(2, 18))


def test_validate_file_continues_after_malformed_json(
    tmp_path: Path, task_factory: Callable[..., dict[str, Any]]
) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(task_factory()) + "\n"
        + '{"task_id": "broken",' + "\n"
        + json.dumps(task_factory(task_id="base_002")) + "\n",
        encoding="utf-8",
    )

    tasks, errors = validate_file(path)

    assert [task["task_id"] for task in tasks] == ["base_001", "base_002"]
    assert len(errors) == 1
    assert errors[0].line == 2
    assert errors[0].rule == "parse_error"


def test_duplicate_task_id_reports_second_line_and_first_occurrence(
    tmp_path: Path, task_factory: Callable[..., dict[str, Any]]
) -> None:
    path = tmp_path / "duplicate.jsonl"
    write_jsonl(path, [task_factory(), task_factory()])

    tasks, errors = validate_file(path)

    assert len(tasks) == 1
    assert len(errors) == 1
    assert errors[0].line == 2
    assert errors[0].rule == "duplicate_task_id"
    assert errors[0].field == "task_id"
    assert "line 1" in errors[0].message


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_each_required_field_is_required(
    task_factory: Callable[..., dict[str, Any]], field: str
) -> None:
    task = task_factory()
    task.pop(field)

    errors = validate_task(task)

    assert [(error.rule, error.field) for error in errors] == [
        ("missing_field", field)
    ]


@pytest.mark.parametrize("record", [[], "x", None, 1])
def test_record_must_be_object(record: object) -> None:
    errors = validate_task(record)

    assert [(error.rule, error.field) for error in errors] == [
        ("record_not_object", "$")
    ]


@pytest.mark.parametrize("task_id", ["bad id", "", 7, "a\tb"])
def test_invalid_task_ids(
    task_factory: Callable[..., dict[str, Any]], task_id: object
) -> None:
    errors = validate_task(task_factory(task_id=task_id))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_task_id", "task_id")
    ]


def test_invalid_category(task_factory: Callable[..., dict[str, Any]]) -> None:
    errors = validate_task(task_factory(category="math"))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_category", "category")
    ]


def test_invalid_metric(task_factory: Callable[..., dict[str, Any]]) -> None:
    errors = validate_task(task_factory(metric_name="exact_match_v2"))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_metric", "metric_name")
    ]


def test_combined_post_process_is_rejected(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    errors = validate_task(task_factory(post_process="strip_whitespace+lower"))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_post_process", "post_process")
    ]


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
        "Question: 1 + 1 \nAnswer:",
        "Question: 1 + 1\nAnswer:\n",
        "Question: 1 + 1\n\nAnswer:",
        7,
    ],
)
def test_invalid_prompt_variants(
    task_factory: Callable[..., dict[str, Any]], prompt: object
) -> None:
    errors = validate_task(task_factory(prompt=prompt))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_prompt", "prompt")
    ]


@pytest.mark.parametrize("targets", [[], "x", [1], [None]])
def test_invalid_targets(
    task_factory: Callable[..., dict[str, Any]], targets: object
) -> None:
    errors = validate_task(task_factory(targets=targets))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_targets", "targets")
    ]


def test_valid_targets_are_accepted(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    assert validate_task(task_factory(targets=["ok"])) == []


@pytest.mark.parametrize(
    ("examples", "field"),
    [
        ("not-a-list", "few_shot_examples"),
        ([{"prompt": "Q"}], "few_shot_examples[0]"),
        ([{"prompt": "Q", "completion": "A", "extra": 1}], "few_shot_examples[0]"),
        ([{"prompt": "", "completion": "A"}], "few_shot_examples[0]"),
    ],
)
def test_invalid_few_shot_examples(
    task_factory: Callable[..., dict[str, Any]], examples: object, field: str
) -> None:
    errors = validate_task(task_factory(few_shot_examples=examples))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_few_shot_examples", field)
    ]


def test_few_shot_examples_are_capped_at_eight(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    pair = {"prompt": "Question: 1 + 1\nAnswer:", "completion": "2"}

    errors = validate_task(task_factory(few_shot_examples=[pair] * 9))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_few_shot_examples", "few_shot_examples")
    ]


def test_eight_few_shot_examples_are_accepted(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    pair = {"prompt": "Question: 1 + 1\nAnswer:", "completion": "2"}

    assert validate_task(task_factory(few_shot_examples=[pair] * 8)) == []


@pytest.mark.parametrize("metadata", [[], "x", None])
def test_metadata_must_be_object(
    task_factory: Callable[..., dict[str, Any]], metadata: object
) -> None:
    errors = validate_task(task_factory(metadata=metadata))

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_metadata", "metadata")
    ]


def test_empty_metadata_is_accepted(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    assert validate_task(task_factory(metadata={})) == []


def test_unknown_top_level_field(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    errors = validate_task(task_factory(temperature=0.2))

    assert [(error.rule, error.field) for error in errors] == [
        ("unknown_field", "temperature")
    ]


@pytest.mark.parametrize(
    ("overrides", "rule", "field"),
    [
        ({"metric_name": "rouge_l"}, "metric_category_mismatch", "metric_name"),
        ({"post_process": "lower"}, "post_process_category_mismatch", "post_process"),
        (
            {"category": "mcq", "targets": ["A"], "post_process": "lower"},
            "post_process_category_mismatch",
            "post_process",
        ),
        (
            {
                "category": "code_exec",
                "metric_name": "exact_match",
                "post_process": "extract_code_block",
                "targets": ["assert True"],
            },
            "metric_category_mismatch",
            "metric_name",
        ),
        (
            {
                "category": "classification",
                "metric_name": "code_exec",
                "post_process": "lower",
            },
            "metric_category_mismatch",
            "metric_name",
        ),
        (
            {
                "category": "summary",
                "metric_name": "exact_match",
                "post_process": "extract_first_line",
            },
            "metric_category_mismatch",
            "metric_name",
        ),
        (
            {
                "category": "summary",
                "metric_name": "rouge_l",
                "post_process": "lower",
            },
            "post_process_category_mismatch",
            "post_process",
        ),
    ],
)
def test_category_metric_matrix_rejects_mismatch(
    task_factory: Callable[..., dict[str, Any]],
    overrides: dict[str, object],
    rule: str,
    field: str,
) -> None:
    errors = validate_task(task_factory(**overrides))

    assert [(error.rule, error.field) for error in errors] == [(rule, field)]


@pytest.mark.parametrize(
    ("category", "metric", "post_process"),
    [
        ("arithmetic", "exact_match", "none"),
        ("arithmetic", "exact_match", "strip_whitespace"),
        ("mcq", "exact_match", "extract_letter"),
        ("code_exec", "code_exec", "extract_code_block"),
        ("classification", "accuracy", "none"),
        ("classification", "accuracy", "lower"),
        ("classification", "accuracy", "strip_whitespace"),
        ("classification", "f1", "none"),
        ("classification", "f1", "lower"),
        ("classification", "f1", "strip_whitespace"),
        ("summary", "rouge_l", "none"),
        ("summary", "rouge_l", "strip_whitespace"),
        ("summary", "rouge_l", "extract_first_line"),
        ("summary", "bleu_4", "none"),
        ("summary", "bleu_4", "strip_whitespace"),
        ("summary", "bleu_4", "extract_first_line"),
    ],
)
def test_all_legal_category_pairs_accepted(
    task_factory: Callable[..., dict[str, Any]],
    category: str,
    metric: str,
    post_process: str,
) -> None:
    overrides: dict[str, object] = {
        "category": category,
        "metric_name": metric,
        "post_process": post_process,
    }
    if category == "mcq":
        overrides["targets"] = ["A"]
    if category == "code_exec":
        overrides["targets"] = ["assert True"]

    assert validate_task(task_factory(**overrides)) == []


@pytest.mark.parametrize("targets", [["A", "B"], ["F"], ["a"], [""]])
def test_mcq_requires_single_a_to_e_target(
    task_factory: Callable[..., dict[str, Any]], targets: list[str]
) -> None:
    errors = validate_task(
        task_factory(
            category="mcq",
            targets=targets,
            post_process="extract_letter",
        )
    )

    assert [(error.rule, error.field) for error in errors] == [
        ("invalid_mcq_targets", "targets")
    ]


@pytest.mark.parametrize("letter", ["A", "B", "C", "D", "E"])
def test_mcq_accepts_a_to_e(
    task_factory: Callable[..., dict[str, Any]], letter: str
) -> None:
    assert (
        validate_task(
            task_factory(
                category="mcq",
                targets=[letter],
                post_process="extract_letter",
            )
        )
        == []
    )


def test_validate_task_reports_all_record_local_errors(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    task = task_factory()
    del task["prompt"]
    task["targets"] = []
    task["metric_name"] = "nope"

    errors = validate_task(task)

    assert [(error.rule, error.field) for error in errors] == [
        ("missing_field", "prompt"),
        ("invalid_targets", "targets"),
        ("invalid_metric", "metric_name"),
    ]


def test_load_fixtures_returns_ten_tasks() -> None:
    assert len(load_fixtures()) == 10


def test_load_fixtures_raises_on_bad_fixture() -> None:
    with pytest.raises(FixtureValidationError) as excinfo:
        load_fixtures(BAD_FIXTURE)

    assert len(excinfo.value.errors) == 16
