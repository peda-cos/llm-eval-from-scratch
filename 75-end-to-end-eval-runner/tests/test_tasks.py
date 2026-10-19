from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_runner.tasks import (
    FIXTURES_DIR,
    FixtureValidationError,
    load_fixtures,
    post_process,
    render,
    validate_file,
    validate_task,
)


def _base_task() -> dict[str, object]:
    return {
        "task_id": "unit_001",
        "category": "arithmetic",
        "prompt": "Question: 1 + 1\nAnswer:",
        "targets": ["2"],
        "metric_name": "exact_match",
        "post_process": "strip_whitespace",
    }


def test_valid_fixture_loads_ten_unique_tasks_across_all_categories() -> None:
    tasks = load_fixtures()
    assert len(tasks) == 10
    assert len({task["task_id"] for task in tasks}) == 10
    counts = {category: 0 for category in ("arithmetic", "mcq", "code_exec", "classification", "summary")}
    for task in tasks:
        counts[task["category"]] += 1
    assert set(counts.values()) == {2}


def test_task_field_validation_reports_all_violations() -> None:
    task = _base_task()
    task["unexpected"] = True
    task["task_id"] = "has space"
    task["targets"] = []
    task["metadata"] = []
    errors = validate_task(task, line=4)
    assert {error.rule for error in errors} >= {
        "unknown_field",
        "invalid_task_id",
        "invalid_targets",
        "invalid_metadata",
    }
    assert all(error.line == 4 for error in errors)
    assert errors[0].to_dict()["rule"] == errors[0].rule


def test_category_metric_matrix_and_mcq_target_rules() -> None:
    task = _base_task()
    task["metric_name"] = "f1"
    errors = validate_task(task)
    assert "metric_category_mismatch" in {error.rule for error in errors}

    mcq = _base_task()
    mcq["category"] = "mcq"
    mcq["post_process"] = "extract_letter"
    mcq["targets"] = ["F"]
    errors = validate_task(mcq)
    assert "invalid_mcq_targets" in {error.rule for error in errors}

    wrong_post = _base_task()
    wrong_post["post_process"] = "lower"
    errors = validate_task(wrong_post)
    assert "post_process_category_mismatch" in {error.rule for error in errors}


def test_validate_file_reports_parse_errors_duplicates_and_bad_records(tmp_path: Path) -> None:
    valid = json.dumps(_base_task())
    bad_unknown = dict(_base_task(), task_id="unit_002", extra=True)
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                valid,
                "{not-json",
                valid,
                json.dumps(bad_unknown),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tasks, errors = validate_file(path)
    assert [task["task_id"] for task in tasks] == ["unit_001"]
    rules = {error.rule for error in errors}
    assert {"parse_error", "duplicate_task_id", "unknown_field"} <= rules


def test_bundled_invalid_fixture_reports_three_distinct_failures() -> None:
    tasks, errors = validate_file(FIXTURES_DIR / "tasks_invalid.jsonl")
    assert tasks == []
    assert len(errors) == 3
    assert {error.rule for error in errors} == {"unknown_field", "metric_category_mismatch", "parse_error"}


def test_load_fixtures_raises_with_structured_errors() -> None:
    with pytest.raises(FixtureValidationError) as caught:
        load_fixtures(FIXTURES_DIR / "tasks_invalid.jsonl")
    assert caught.value.path.name == "tasks_invalid.jsonl"
    assert len(caught.value.errors) == 3
    assert "first error on line" in str(caught.value)


def test_few_shot_limits_and_shape() -> None:
    task = _base_task()
    task["few_shot_examples"] = [
        {"prompt": f"q{index}", "completion": f"a{index}"} for index in range(9)
    ]
    errors = validate_task(task)
    assert "invalid_few_shot_examples" in {error.rule for error in errors}

    task["few_shot_examples"] = [{"prompt": "q"}]
    errors = validate_task(task)
    assert "invalid_few_shot_examples" in {error.rule for error in errors}

    task["few_shot_examples"] = [{"prompt": "q", "completion": ""}]
    errors = validate_task(task)
    assert "invalid_few_shot_examples" in {error.rule for error in errors}


def test_render_and_all_post_process_rules() -> None:
    task = load_fixtures()[0]
    rendered = render(task)
    assert rendered.startswith("Question: 2 + 2\nAnswer: 4\n\n")
    assert rendered.endswith("Compute the result.\nQuestion: 17 + 24\nAnswer:")

    assert post_process("  value  ", "strip_whitespace") == "value"
    assert post_process("HeLLo", "lower") == "hello"
    assert post_process("I think C.", "extract_letter") == "C"
    assert post_process("no letter here", "extract_letter") == ""
    assert post_process("```python\nprint(1)\n```", "extract_code_block") == "print(1)"
    assert post_process("```python\nprint(1)", "extract_code_block") == ""
    assert post_process("\n\n  first line  \nsecond", "extract_first_line") == "first line"
    assert post_process("unchanged", "none") == "unchanged"
    with pytest.raises(ValueError, match="unknown post_process"):
        post_process("x", "not-a-rule")  # type: ignore[arg-type]


def test_prompt_and_target_validation() -> None:
    task = _base_task()
    task["prompt"] = "Question: 1 + 1  \nAnswer:"
    assert "invalid_prompt" in {error.rule for error in validate_task(task)}

    task = _base_task()
    task["prompt"] = "First block\n\nSecond block"
    assert "invalid_prompt" in {error.rule for error in validate_task(task)}

    task = _base_task()
    task["targets"] = ["2", 3]
    assert "invalid_targets" in {error.rule for error in validate_task(task)}

    assert validate_task("not-an-object")[0].rule == "record_not_object"
