from typing import Any, Callable, cast

import pytest

from main import TaskSpec, post_process, render


def test_render_without_few_shot_examples(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    task = task_factory()

    assert render(cast(TaskSpec, task)) == task["prompt"]


def test_render_with_few_shot_examples_uses_blank_lines(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    task = task_factory(
        few_shot_examples=[{"prompt": "Question: 2 + 2\nAnswer:", "completion": "4"}]
    )

    assert render(cast(TaskSpec, task)) == "Question: 2 + 2\nAnswer: 4\n\n" + task["prompt"]


def test_render_preserves_example_order(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    task = task_factory(
        few_shot_examples=[
            {"prompt": "Q1", "completion": "A1"},
            {"prompt": "Q2", "completion": "A2"},
        ]
    )

    assert render(cast(TaskSpec, task)) == "Q1 A1\n\nQ2 A2\n\n" + task["prompt"]


@pytest.mark.parametrize(
    ("rule", "text", "expected"),
    [
        ("none", "  A b \n", "  A b \n"),
        ("strip_whitespace", "  A b \n", "A b"),
        ("lower", "AbC", "abc"),
        ("extract_letter", "I think C.", "C"),
        ("extract_letter", "no letter", ""),
        ("extract_letter", "lowercase c", ""),
        (
            "extract_code_block",
            "before\n```python\nprint(1)\n```\nafter",
            "print(1)",
        ),
        ("extract_code_block", "```\n\nx = 1\n\n```", "\nx = 1\n"),
        ("extract_code_block", "no fence", ""),
        ("extract_code_block", "```python\nprint(1)\n", ""),
        ("extract_first_line", "\n  \n  Hello world  \nsecond", "Hello world"),
        ("extract_first_line", "", ""),
    ],
)
def test_post_process_rules(rule: str, text: str, expected: str) -> None:
    assert post_process(text, cast(Any, rule)) == expected


def test_unknown_post_process_rule_raises() -> None:
    with pytest.raises(ValueError, match="unknown post_process"):
        post_process("x", cast(Any, "not_a_rule"))


def test_strip_whitespace_normalizes_equivalent_generations() -> None:
    normalized = post_process("  41 \n", "strip_whitespace")

    assert normalized == "41"
    assert normalized == post_process("41", "strip_whitespace")


def test_rendered_prompt_and_mcq_post_process_are_deterministic(
    task_factory: Callable[..., dict[str, Any]]
) -> None:
    task = task_factory(
        category="mcq",
        targets=["B"],
        post_process="extract_letter",
        prompt="Pick one.\nA. one\nB. two\nAnswer:",
    )

    typed_task = cast(TaskSpec, task)
    first_render = render(typed_task)

    assert first_render == render(typed_task)
    assert post_process("I choose B.", task["post_process"]) == "B"
